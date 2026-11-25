"""Intersectional Fairness Analysis across combined protected attributes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import CONFIG
from modules.fairness_metrics import compute_fairness_metrics


def run_intersectional_analysis(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
    attributes: List[str],
    min_group_size: int = CONFIG.thresholds.min_group_size,
) -> Dict[str, Any]:
    """Evaluate disparities across intersections such as Gender x Age Group x Race."""
    valid_attrs = [a for a in attributes if a in sensitive_df.columns]
    if not valid_attrs:
        raise ValueError("No valid sensitive attributes found for intersectional analysis.")

    combined = (
        sensitive_df[valid_attrs]
        .fillna("Missing")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    metrics = compute_fairness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=combined,
        min_group_size=min_group_size,
        compute_ci=False,
    )

    subgroups_df = pd.DataFrame(metrics["group_metrics"]).sort_values(
        "selection_rate", ascending=True
    ).reset_index(drop=True)

    lowest = subgroups_df.iloc[0].to_dict() if not subgroups_df.empty else {}
    highest = subgroups_df.iloc[-1].to_dict() if not subgroups_df.empty else {}

    return {
        "attributes": valid_attrs,
        "intersectional_table": subgroups_df,
        "prediction_only_metrics": metrics["prediction_only_metrics"],
        "outcome_based_metrics": metrics["outcome_based_metrics"],
        "lowest_selection_subgroup": lowest,
        "highest_selection_subgroup": highest,
        "small_group_warnings": metrics["small_group_warnings"],
    }
