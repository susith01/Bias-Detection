"""Configurable PASS / WARN / FAIL bias detection across protected attributes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import CONFIG, FairnessThresholds
from modules.fairness_metrics import compute_fairness_metrics


def classify_disparity(value: float, pass_limit: float, warn_limit: float) -> str:
    if value <= pass_limit:
        return "PASS"
    if value <= warn_limit:
        return "WARN"
    return "FAIL"


def classify_ratio(value: float, pass_limit: float, warn_limit: float) -> str:
    if value >= pass_limit:
        return "PASS"
    if value >= warn_limit:
        return "WARN"
    return "FAIL"


def evaluate_attribute_bias(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    sensitive_series: pd.Series,
    attribute_label: str,
    thresholds: FairnessThresholds = CONFIG.thresholds,
) -> Dict[str, Any]:
    """Run fairness evaluation and assign PASS / WARN / FAIL statuses."""
    metrics = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series,
        min_group_size=thresholds.min_group_size,
    )
    pred_only = metrics["prediction_only_metrics"]
    dp_diff_status = classify_disparity(
        pred_only["demographic_parity_difference"],
        thresholds.dp_diff_pass,
        thresholds.dp_diff_warn,
    )
    dp_ratio_status = classify_ratio(
        pred_only["demographic_parity_ratio"],
        thresholds.dp_ratio_pass,
        thresholds.dp_ratio_warn,
    )

    statuses = [dp_diff_status, dp_ratio_status]
    metric_statuses = {
        "demographic_parity_difference": dp_diff_status,
        "demographic_parity_ratio": dp_ratio_status,
    }

    outcome = metrics["outcome_based_metrics"]
    if outcome is not None:
        eo_status = classify_disparity(
            outcome["equal_opportunity_difference"],
            thresholds.eo_diff_pass,
            thresholds.eo_diff_warn,
        )
        eodds_status = classify_disparity(
            outcome["equalized_odds_difference"],
            thresholds.eodds_diff_pass,
            thresholds.eodds_diff_warn,
        )
        statuses.extend([eo_status, eodds_status])
        metric_statuses["equal_opportunity_difference"] = eo_status
        metric_statuses["equalized_odds_difference"] = eodds_status

    overall = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    return {
        "attribute": attribute_label,
        "overall_status": overall,
        "metric_statuses": metric_statuses,
        **metrics,
    }


def detect_bias_across_attributes(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
    thresholds: FairnessThresholds = CONFIG.thresholds,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate Gender, grouped Age (`age_group`), and Race when present."""
    results: Dict[str, Dict[str, Any]] = {}
    col_map: List[tuple[str, str]] = []
    for col in sensitive_df.columns:
        low = col.lower()
        if low in ("gender", "sex"):
            col_map.append(("Gender", col))
        elif low == "age_group":
            col_map.append(("Age Group", col))
        elif low in ("race", "ethnicity"):
            col_map.append(("Race", col))

    for label, col in col_map:
        results[label] = evaluate_attribute_bias(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_series=sensitive_df[col],
            attribute_label=label,
            thresholds=thresholds,
        )
    return results
