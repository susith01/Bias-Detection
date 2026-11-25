"""Bias mitigation algorithms and baseline-vs-mitigated trade-off evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modules.fairness_metrics import compute_fairness_metrics
from modules.model_training import TrainingBundle, compute_classification_metrics

MITIGATION_DISCLAIMER = (
    "Notice: Post-processing and in-processing bias mitigation reduce measured statistical "
    "disparities on the evaluation split, but do not guarantee fairness across unseen populations, "
    "unmeasured subgroups, or downstream operational contexts."
)


def compute_reweighing_weights(
    y_train: np.ndarray, sensitive_train: pd.Series
) -> np.ndarray:
    """Kamiran & Calders reweighing weights W(S, Y) = P(S)*P(Y) / P(S, Y)."""
    s = sensitive_train.fillna("Missing").astype(str).to_numpy()
    y = np.asarray(y_train, dtype=int)
    n = len(y)
    weights = np.ones(n, dtype=float)

    unique_s = np.unique(s)
    unique_y = np.unique(y)
    for sv in unique_s:
        p_s = np.mean(s == sv)
        for yv in unique_y:
            p_y = np.mean(y == yv)
            mask = (s == sv) & (y == yv)
            p_sy = np.mean(mask)
            if p_sy > 0:
                weights[mask] = (p_s * p_y) / p_sy
    return weights


def apply_group_threshold_adjustment(
    y_prob: np.ndarray,
    sensitive_test: pd.Series,
    y_train_prob: np.ndarray,
    sensitive_train: pd.Series,
    target_selection_rate: Optional[float] = None,
) -> np.ndarray:
    """Group-specific probability threshold tuning calibrated on training probabilities."""
    s_train = sensitive_train.fillna("Missing").astype(str).to_numpy()
    s_test = sensitive_test.fillna("Missing").astype(str).to_numpy()
    target_rate = (
        float(target_selection_rate)
        if target_selection_rate is not None
        else float(np.mean(y_train_prob >= 0.5))
    )
    target_rate = float(np.clip(target_rate, 0.05, 0.95))

    thresholds: Dict[str, float] = {}
    for grp in np.unique(s_train):
        grp_probs = y_train_prob[s_train == grp]
        if len(grp_probs) > 0:
            thresholds[grp] = float(np.quantile(grp_probs, 1.0 - target_rate))
        else:
            thresholds[grp] = 0.5

    y_pred_adj = np.zeros(len(y_prob), dtype=int)
    for idx, (prob, grp) in enumerate(zip(y_prob, s_test)):
        thr = thresholds.get(grp, 0.5)
        y_pred_adj[idx] = 1 if prob >= thr else 0
    return y_pred_adj


def run_mitigation_comparison(
    bundle: TrainingBundle,
    sensitive_column: str,
    strategies: Optional[List[str]] = None,
    constraint_type: str = "DemographicParity",
) -> Dict[str, Any]:
    """
    Evaluate baseline vs mitigated models on the exact same held-out test records.

    Supported strategies:
      - Reweighing
      - Threshold Adjustment
      - ExponentiatedGradient
      - ThresholdOptimizer
    """
    selected = strategies or [
        "Reweighing",
        "Threshold Adjustment",
        "ExponentiatedGradient",
        "ThresholdOptimizer",
    ]
    if sensitive_column not in bundle.X_train.columns:
        raise ValueError(f"Sensitive column '{sensitive_column}' not found in training data.")

    s_train = bundle.X_train[sensitive_column].fillna("Missing").astype(str)
    s_test = bundle.X_test[sensitive_column].fillna("Missing").astype(str)
    y_train = bundle.y_train.to_numpy()
    y_test = bundle.y_test.to_numpy()

    X_train_model = bundle.X_train[bundle.feature_columns].copy()
    X_test_model = bundle.X_test[bundle.feature_columns].copy()

    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    X_train_enc = preprocessor.transform(X_train_model)
    X_test_enc = preprocessor.transform(X_test_model)

    base_fairness = compute_fairness_metrics(y_test, bundle.y_pred, s_test, compute_ci=False)
    rows: List[Dict[str, Any]] = [
        {
            "strategy": "Baseline (Unmitigated)",
            "accuracy": bundle.metrics["accuracy"],
            "f1_score": bundle.metrics["f1_score"],
            "roc_auc": bundle.metrics["roc_auc"],
            "demographic_parity_diff": base_fairness["prediction_only_metrics"][
                "demographic_parity_difference"
            ],
            "demographic_parity_ratio": base_fairness["prediction_only_metrics"][
                "demographic_parity_ratio"
            ],
            "equal_opportunity_diff": base_fairness["outcome_based_metrics"][
                "equal_opportunity_difference"
            ],
            "equalized_odds_diff": base_fairness["outcome_based_metrics"][
                "equalized_odds_difference"
            ],
            "y_pred": bundle.y_pred,
        }
    ]

    for strategy in selected:
        try:
            if strategy == "Reweighing":
                weights = compute_reweighing_weights(y_train, s_train)
                clf = clone(bundle.pipeline.named_steps["classifier"])
                clf.fit(X_train_enc, y_train, sample_weight=weights)
                y_pred_m = clf.predict(X_test_enc)
                y_prob_m = (
                    clf.predict_proba(X_test_enc)[:, 1]
                    if hasattr(clf, "predict_proba")
                    else y_pred_m.astype(float)
                )
            elif strategy == "Threshold Adjustment":
                y_train_prob = (
                    bundle.pipeline.predict_proba(X_train_model)[:, 1]
                    if hasattr(bundle.pipeline, "predict_proba")
                    else bundle.pipeline.predict(X_train_model).astype(float)
                )
                y_pred_m = apply_group_threshold_adjustment(
                    bundle.y_prob, s_test, y_train_prob, s_train
                )
                y_prob_m = bundle.y_prob
            elif strategy == "ExponentiatedGradient":
                constraint = (
                    DemographicParity()
                    if constraint_type == "DemographicParity"
                    else EqualizedOdds()
                )
                base_est = LogisticRegression(max_iter=1000, random_state=42)
                eg = ExponentiatedGradient(
                    estimator=base_est,
                    constraints=constraint,
                    eps=0.02,
                    max_iter=25,
                )
                eg.fit(X_train_enc, y_train, sensitive_features=s_train)
                y_pred_m = eg.predict(X_test_enc)
                y_prob_m = y_pred_m.astype(float)
            elif strategy == "ThresholdOptimizer":
                base_pipe = Pipeline(
                    steps=[
                        ("preprocessor", clone(preprocessor)),
                        ("classifier", clone(bundle.pipeline.named_steps["classifier"])),
                    ]
                )
                base_pipe.fit(X_train_model, y_train)
                constraint_str = (
                    "demographic_parity"
                    if constraint_type == "DemographicParity"
                    else "equalized_odds"
                )
                to = ThresholdOptimizer(
                    estimator=base_pipe,
                    constraints=constraint_str,
                    prefit=True,
                    predict_method="predict_proba",
                )
                to.fit(X_train_model, y_train, sensitive_features=s_train)
                y_pred_m = to.predict(X_test_model, sensitive_features=s_test)
                y_prob_m = y_pred_m.astype(float)
            else:
                continue

            perf = compute_classification_metrics(y_test, y_pred_m, y_prob_m)
            fair = compute_fairness_metrics(y_test, y_pred_m, s_test, compute_ci=False)
            rows.append(
                {
                    "strategy": strategy,
                    "accuracy": perf["accuracy"],
                    "f1_score": perf["f1_score"],
                    "roc_auc": perf["roc_auc"],
                    "demographic_parity_diff": fair["prediction_only_metrics"][
                        "demographic_parity_difference"
                    ],
                    "demographic_parity_ratio": fair["prediction_only_metrics"][
                        "demographic_parity_ratio"
                    ],
                    "equal_opportunity_diff": fair["outcome_based_metrics"][
                        "equal_opportunity_difference"
                    ],
                    "equalized_odds_diff": fair["outcome_based_metrics"][
                        "equalized_odds_difference"
                    ],
                    "y_pred": y_pred_m,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "strategy": strategy,
                    "accuracy": np.nan,
                    "f1_score": np.nan,
                    "roc_auc": np.nan,
                    "demographic_parity_diff": np.nan,
                    "demographic_parity_ratio": np.nan,
                    "equal_opportunity_diff": np.nan,
                    "equalized_odds_diff": np.nan,
                    "error": str(exc),
                }
            )

    return {
        "sensitive_attribute": sensitive_column,
        "constraint": constraint_type,
        "comparison_table": pd.DataFrame(
            [{k: v for k, v in r.items() if k != "y_pred"} for r in rows]
        ),
        "raw_results": rows,
        "disclaimer": MITIGATION_DISCLAIMER,
    }
