"""SHAP explainability engine with global, local, group, and sensitive-impact views."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from modules.model_training import TrainingBundle, extract_feature_names

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

CAUSATION_NOTE = (
    "Explainability Note: SHAP values measure statistical feature attribution within the "
    "trained model. High SHAP importance indicates predictive association, NOT causal proof "
    "of real-world merit or demographic causation."
)


def _compute_shap_matrix(
    bundle: TrainingBundle, sample_size: int = 200
) -> tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Compute SHAP values on a deterministic subset of the test split."""
    X_test_model = bundle.X_test[bundle.feature_columns].iloc[:sample_size].copy()
    X_train_model = bundle.X_train[bundle.feature_columns].iloc[: min(150, len(bundle.X_train))].copy()

    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    classifier = bundle.pipeline.named_steps["classifier"]

    X_test_enc = preprocessor.transform(X_test_model)
    X_train_enc = preprocessor.transform(X_train_model)
    feature_names = extract_feature_names(bundle.pipeline)

    if shap is None:
        # Fallback to coefficient / feature importance approximation if shap missing
        if hasattr(classifier, "coef_"):
            weights = classifier.coef_[0]
            shap_vals = (X_test_enc - X_train_enc.mean(axis=0)) * weights
        elif hasattr(classifier, "feature_importances_"):
            weights = classifier.feature_importances_
            shap_vals = (X_test_enc - X_train_enc.mean(axis=0)) * weights
        else:
            shap_vals = np.zeros_like(X_test_enc)
        return shap_vals, pd.DataFrame(X_test_enc, columns=feature_names), feature_names

    if hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, X_train_enc)
        raw_vals = explainer.shap_values(X_test_enc)
    elif hasattr(classifier, "tree_") or hasattr(classifier, "estimators_"):
        explainer = shap.TreeExplainer(classifier)
        raw_vals = explainer.shap_values(X_test_enc)
    else:
        explainer = shap.KernelExplainer(classifier.predict_proba, X_train_enc[:50])
        raw_vals = explainer.shap_values(X_test_enc[:50])

    if isinstance(raw_vals, list):
        shap_vals = np.asarray(raw_vals[1] if len(raw_vals) > 1 else raw_vals[0])
    elif getattr(raw_vals, "ndim", 2) == 3:
        shap_vals = np.asarray(raw_vals[:, :, 1])
    else:
        shap_vals = np.asarray(raw_vals)

    return shap_vals, pd.DataFrame(X_test_enc, columns=feature_names), feature_names


def compute_explainability_summary(
    bundle: TrainingBundle,
    sample_index: int = 0,
    group_column: Optional[str] = None,
    sample_size: int = 200,
) -> Dict[str, Any]:
    """Generate global, individual, group-level, and sensitive-attribute SHAP analyses."""
    shap_vals, encoded_df, feature_names = _compute_shap_matrix(bundle, sample_size=sample_size)
    n_eval = len(encoded_df)
    safe_idx = int(max(0, min(sample_index, n_eval - 1)))

    # 1. Global feature importance (mean absolute SHAP)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    global_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # 2. Individual explanation
    indiv_vals = shap_vals[safe_idx]
    indiv_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "encoded_value": encoded_df.iloc[safe_idx].to_numpy(),
                "shap_value": indiv_vals,
                "abs_shap": np.abs(indiv_vals),
            }
        )
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # 3. Sensitive attribute impact
    sensitive_tokens = [
        tok.lower()
        for tok in bundle.sensitive_columns.values()
        if tok is not None
    ]
    sens_mask = [
        any(feat.lower().startswith(f"{tok}_") or feat.lower() == tok for tok in sensitive_tokens)
        for feat in feature_names
    ]
    sensitive_impact_df = global_df[
        global_df["feature"].isin([f for f, flag in zip(feature_names, sens_mask) if flag])
    ].reset_index(drop=True)

    total_imp = float(global_df["mean_abs_shap"].sum()) or 1.0
    sensitive_share_pct = float(sensitive_impact_df["mean_abs_shap"].sum() / total_imp * 100.0)

    # 4. Group-level explanations
    group_df = pd.DataFrame()
    if group_column and group_column in bundle.test_sensitive_df.columns:
        grp_series = bundle.test_sensitive_df[group_column].iloc[:n_eval].astype(str).reset_index(drop=True)
        top_feats = global_df["feature"].head(8).tolist()
        grp_rows: List[Dict[str, Any]] = []
        shap_frame = pd.DataFrame(shap_vals, columns=feature_names)
        for grp in grp_series.unique():
            mask = grp_series == grp
            row: Dict[str, Any] = {"group": grp, "samples": int(mask.sum())}
            for feat in top_feats:
                row[feat] = float(shap_frame.loc[mask, feat].mean())
            grp_rows.append(row)
        group_df = pd.DataFrame(grp_rows)

    return {
        "global_importance": global_df,
        "individual_explanation": indiv_df,
        "individual_index": safe_idx,
        "individual_prob": float(bundle.y_prob[safe_idx]),
        "individual_pred": int(bundle.y_pred[safe_idx]),
        "sensitive_impact": sensitive_impact_df,
        "sensitive_share_pct": sensitive_share_pct,
        "group_explanations": group_df,
        "causation_note": CAUSATION_NOTE,
    }
