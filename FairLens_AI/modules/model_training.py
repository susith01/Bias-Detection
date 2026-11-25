"""Machine learning model training, evaluation, and Joblib persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import CONFIG, MODELS_DIR
from modules.preprocessing import (
    apply_age_grouping,
    build_preprocessor,
    detect_sensitive_columns,
    prepare_binary_target,
)


@dataclass
class TrainingBundle:
    """Container holding a trained model pipeline and shared test evaluation split."""

    model_name: str
    algorithm_key: str
    pipeline: Pipeline
    feature_columns: List[str]
    target_column: str
    sensitive_columns: Dict[str, Optional[str]]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_sensitive_df: pd.DataFrame
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: Dict[str, float]
    confusion: List[List[int]]
    model_path: Optional[Path] = None


def create_estimator(algorithm_name: str, random_state: int = 42, **params: Any) -> Any:
    """Instantiate one of the three supported Scikit-learn classifiers."""
    if algorithm_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=int(params.get("max_iter", 1000)),
            C=float(params.get("C", 1.0)),
            random_state=random_state,
        )
    if algorithm_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 6)),
            min_samples_leaf=int(params.get("min_samples_leaf", 15)),
            random_state=random_state,
        )
    if algorithm_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 120)),
            max_depth=int(params.get("max_depth", 10)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm_name}")


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1-score, and ROC-AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_column: str,
    algorithms: List[str],
    test_size: float = 0.25,
    random_state: int = CONFIG.random_seed,
    include_sensitive_in_features: bool = True,
    save_artifacts: bool = True,
    hyperparams_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, TrainingBundle]:
    """
    Train selected models on the exact same stratified train/test split to prevent
    data leakage and enable fair model-to-model comparison.
    """
    working = df.dropna(subset=[target_column]).copy().reset_index(drop=True)
    if len(working) < 10:
        raise ValueError("Dataset must contain at least 10 non-null target rows for training.")
    sens_map = detect_sensitive_columns(working)
    if sens_map.get("age"):
        working = apply_age_grouping(working, age_col=sens_map["age"])  # preserves raw age

    y_all, _ = prepare_binary_target(working[target_column])
    X_all = working.drop(columns=[target_column])

    # Shared stratified split across all models
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all if y_all.nunique() > 1 else None,
    )

    # Preserve protected attributes for fairness evaluation even if excluded from model training
    protected_cols = []
    if sens_map.get("gender"):
        protected_cols.append(sens_map["gender"])
    if sens_map.get("race"):
        protected_cols.append(sens_map["race"])
    if "age_group" in X_test_full.columns:
        protected_cols.append("age_group")
    if sens_map.get("age") and sens_map["age"] in X_test_full.columns:
        protected_cols.append(sens_map["age"])

    test_sensitive_df = X_test_full[list(dict.fromkeys(protected_cols))].copy()

    exclude_cols: List[str] = ["age_group"]
    if not include_sensitive_in_features:
        for col in (sens_map.get("gender"), sens_map.get("race"), sens_map.get("age")):
            if col:
                exclude_cols.append(col)

    bundles: Dict[str, TrainingBundle] = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for algo in algorithms:
        preprocessor, feature_cols, _, _ = build_preprocessor(
            X_train_full, exclude_columns=exclude_cols
        )
        algo_params = (hyperparams_map or {}).get(algo, {})
        estimator = create_estimator(algo, random_state=random_state, **algo_params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        X_train_model = X_train_full[feature_cols].copy()
        X_test_model = X_test_full[feature_cols].copy()

        pipeline.fit(X_train_model, y_train)
        y_pred = pipeline.predict(X_test_model)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test_model)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        metrics = compute_classification_metrics(y_test.to_numpy(), y_pred, y_prob)
        cm = confusion_matrix(y_test.to_numpy(), y_pred).tolist()

        model_path: Optional[Path] = None
        if save_artifacts:
            slug = CONFIG.supported_models.get(algo, algo.lower().replace(" ", "_"))
            model_path = MODELS_DIR / f"{slug}_{timestamp}.joblib"
            joblib.dump(
                {
                    "pipeline": pipeline,
                    "feature_columns": feature_cols,
                    "target_column": target_column,
                    "algorithm": algo,
                    "sensitive_columns": sens_map,
                },
                model_path,
            )

        bundles[algo] = TrainingBundle(
            model_name=algo,
            algorithm_key=CONFIG.supported_models.get(algo, algo.lower().replace(" ", "_")),
            pipeline=pipeline,
            feature_columns=feature_cols,
            target_column=target_column,
            sensitive_columns=sens_map,
            X_train=X_train_full,
            X_test=X_test_full,
            y_train=y_train,
            y_test=y_test,
            test_sensitive_df=test_sensitive_df,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=metrics,
            confusion=cm,
            model_path=model_path,
        )

    return bundles


def load_saved_model(model_path: Path) -> Dict[str, Any]:
    """Load a persisted Joblib model artifact."""
    return joblib.load(model_path)


def extract_feature_names(pipeline: Pipeline) -> List[str]:
    """Return expanded post-preprocessing feature names for explainability."""
    preprocessor = pipeline.named_steps["preprocessor"]
    names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "num":
            names.extend(list(cols))
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            names.extend(list(ohe.get_feature_names_out(cols)))
    return names
