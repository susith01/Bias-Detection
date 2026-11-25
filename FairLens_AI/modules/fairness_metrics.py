"""Fairness metrics engine using Fairlearn MetricFrame and statistical intervals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

from config import CONFIG


def wilson_confidence_interval(
    successes: int, total: int, z: float = 1.96
) -> Tuple[float, float]:
    """Compute 95% Wilson score interval for group proportions (e.g. selection rate)."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric_fn: Any,
    n_boot: int = 120,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval for disparity summary metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_pred)
    if n < 20 or len(np.unique(sensitive)) < 2:
        val = float(metric_fn(y_true, y_pred, sensitive_features=sensitive))
        return (val, val)
    samples: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(sensitive[idx])) < 2 or len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(
                float(metric_fn(y_true[idx], y_pred[idx], sensitive_features=sensitive[idx]))
            )
        except Exception:
            continue
    if not samples:
        val = float(metric_fn(y_true, y_pred, sensitive_features=sensitive))
        return (val, val)
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def compute_fairness_metrics(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    min_group_size: int = CONFIG.thresholds.min_group_size,
    compute_ci: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate prediction-only and outcome-based fairness metrics using Fairlearn MetricFrame.

    When `y_true` is None (e.g., unlabeled production predictions), outcome-based metrics
    (Equal Opportunity, Equalized Odds, FPR, FNR) are omitted and explicitly marked unavailable.
    """
    sens = sensitive_features.fillna("Missing").astype(str).reset_index(drop=True)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    group_counts = sens.value_counts().to_dict()
    small_group_warnings = [
        f"Group '{grp}' has only {cnt} samples (< {min_group_size}); estimates have wide uncertainty."
        for grp, cnt in group_counts.items()
        if cnt < min_group_size
    ]

    # Prediction-only metrics (do not require ground truth)
    dp_diff = float(
        demographic_parity_difference(
            y_true=y_pred_arr, y_pred=y_pred_arr, sensitive_features=sens
        )
    )
    dp_ratio = float(
        demographic_parity_ratio(
            y_true=y_pred_arr, y_pred=y_pred_arr, sensitive_features=sens
        )
    )

    pred_frame = MetricFrame(
        metrics={"selection_rate": selection_rate},
        y_true=y_pred_arr,
        y_pred=y_pred_arr,
        sensitive_features=sens,
    )

    group_records: List[Dict[str, Any]] = []
    for grp, row in pred_frame.by_group.iterrows():
        mask = (sens == grp).to_numpy()
        n_grp = int(mask.sum())
        n_pos_pred = int(y_pred_arr[mask].sum())
        ci_low, ci_high = wilson_confidence_interval(n_pos_pred, n_grp)
        group_records.append(
            {
                "group": str(grp),
                "count": n_grp,
                "selection_rate": float(row["selection_rate"]),
                "selection_rate_ci_low": ci_low,
                "selection_rate_ci_high": ci_high,
                "true_positive_rate": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
            }
        )

    result: Dict[str, Any] = {
        "has_ground_truth": y_true is not None,
        "prediction_only_metrics": {
            "demographic_parity_difference": dp_diff,
            "demographic_parity_ratio": dp_ratio,
        },
        "outcome_based_metrics": None,
        "group_metrics": group_records,
        "small_group_warnings": small_group_warnings,
        "confidence_intervals": {},
    }

    if compute_ci:
        dummy_y = np.asarray(y_true, dtype=int) if y_true is not None else y_pred_arr
        dp_ci = bootstrap_metric_ci(
            dummy_y, y_pred_arr, sens.to_numpy(), demographic_parity_difference
        )
        result["confidence_intervals"]["demographic_parity_difference"] = {
            "low": dp_ci[0],
            "high": dp_ci[1],
        }

    # Outcome-based metrics (only when ground truth is available)
    if y_true is not None:
        y_true_arr = np.asarray(y_true, dtype=int)
        outcome_frame = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            },
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            sensitive_features=sens,
        )
        def _safe_float(val: Any, default: Optional[float] = 0.0) -> Optional[float]:
            try:
                f = float(val)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except (TypeError, ValueError):
                return default

        tpr_series = outcome_frame.by_group["true_positive_rate"].dropna()
        eo_diff = float(tpr_series.max() - tpr_series.min()) if len(tpr_series) > 0 else 0.0
        eodds_diff = _safe_float(
            equalized_odds_difference(
                y_true=y_true_arr, y_pred=y_pred_arr, sensitive_features=sens
            ),
            default=0.0,
        ) or 0.0
        result["outcome_based_metrics"] = {
            "equal_opportunity_difference": eo_diff,
            "equalized_odds_difference": eodds_diff,
        }

        for rec in group_records:
            grp = rec["group"]
            if grp in outcome_frame.by_group.index:
                rec["true_positive_rate"] = _safe_float(
                    outcome_frame.by_group.loc[grp, "true_positive_rate"], default=0.0
                )
                rec["false_positive_rate"] = _safe_float(
                    outcome_frame.by_group.loc[grp, "false_positive_rate"], default=0.0
                )
                rec["false_negative_rate"] = _safe_float(
                    outcome_frame.by_group.loc[grp, "false_negative_rate"], default=0.0
                )

        if compute_ci:
            eodds_ci = bootstrap_metric_ci(
                y_true_arr, y_pred_arr, sens.to_numpy(), equalized_odds_difference
            )
            result["confidence_intervals"]["equalized_odds_difference"] = {
                "low": eodds_ci[0],
                "high": eodds_ci[1],
            }

    return result
