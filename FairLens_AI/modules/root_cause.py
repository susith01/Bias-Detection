"""Root Cause Attribution engine diagnosing structural drivers of measured bias."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from modules.fairness_metrics import compute_fairness_metrics
from modules.proxy_analysis import compute_proxy_leakage


def attribute_root_causes(
    df: pd.DataFrame,
    target_column: str,
    sensitive_column: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_test: pd.Series,
) -> Dict[str, Any]:
    """
    Quantify four distinct structural sources of disparity:
      1. Representation Imbalance (sample share skew in dataset)
      2. Historical Label Disparity (ground-truth positive rate gap)
      3. Proxy Amplification (non-sensitive features strongly correlated with sensitive group)
      4. Model Error / Threshold Disparity (FPR / FNR gap beyond base rates)
    """
    sens_all = df[sensitive_column].fillna("Missing").astype(str)
    shares = sens_all.value_counts(normalize=True)
    min_share = float(shares.min()) if len(shares) > 0 else 1.0
    max_share = float(shares.max()) if len(shares) > 0 else 1.0
    rep_skew = float(min(100.0, (1.0 - min_share / max(max_share, 1e-6)) * 100.0))

    # Historical label base-rate gap
    y_series = pd.Series(y_true, index=sensitive_test.index)
    base_rates = y_series.groupby(sensitive_test.fillna("Missing").astype(str)).mean()
    label_gap = float(base_rates.max() - base_rates.min()) if len(base_rates) > 1 else 0.0
    label_bias_score = float(min(100.0, label_gap * 250.0))

    # Proxy amplification
    proxy_res = compute_proxy_leakage(
        df, sensitive_column=sensitive_column, target_column=target_column
    )
    proxy_score = float(proxy_res["max_leakage_score"])

    # Model error / threshold disparity
    fair_res = compute_fairness_metrics(y_true, y_pred, sensitive_test, compute_ci=False)
    eodds_diff = float(fair_res["outcome_based_metrics"]["equalized_odds_difference"])
    dp_diff = float(fair_res["prediction_only_metrics"]["demographic_parity_difference"])
    amplification_gap = max(0.0, dp_diff - label_gap)
    model_disparity_score = float(min(100.0, (eodds_diff * 200.0 + amplification_gap * 180.0)))

    total = rep_skew + label_bias_score + proxy_score + model_disparity_score
    norm = total if total > 0 else 1.0

    drivers: List[Dict[str, Any]] = [
        {
            "driver": "Historical Label Disparity",
            "raw_score": round(label_bias_score, 2),
            "contribution_pct": round(label_bias_score / norm * 100.0, 1),
            "evidence": f"Ground-truth positive rate differs by {label_gap:.1%} across {sensitive_column} groups.",
        },
        {
            "driver": "Model Error & Threshold Disparity",
            "raw_score": round(model_disparity_score, 2),
            "contribution_pct": round(model_disparity_score / norm * 100.0, 1),
            "evidence": f"Equalized Odds diff is {eodds_diff:.3f}; model amplifies DP gap by {amplification_gap:.3f} over base rates.",
        },
        {
            "driver": "Proxy Feature Leakage",
            "raw_score": round(proxy_score, 2),
            "contribution_pct": round(proxy_score / norm * 100.0, 1),
            "evidence": (
                f"Top proxy features: {', '.join(proxy_res['high_risk_features'][:3])}"
                if proxy_res["high_risk_features"]
                else f"Peak proxy leakage score is {proxy_score:.1f}/100."
            ),
        },
        {
            "driver": "Representation Imbalance",
            "raw_score": round(rep_skew, 2),
            "contribution_pct": round(rep_skew / norm * 100.0, 1),
            "evidence": f"Smallest group share is {min_share:.1%} vs largest {max_share:.1%}.",
        },
    ]

    drivers_df = (
        pd.DataFrame(drivers)
        .sort_values("contribution_pct", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "sensitive_attribute": sensitive_column,
        "drivers_table": drivers_df,
        "primary_driver": str(drivers_df.iloc[0]["driver"]),
        "base_rates": base_rates.to_dict(),
        "group_shares": shares.to_dict(),
        "proxy_details": proxy_res,
    }
