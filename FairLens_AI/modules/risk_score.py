"""Composite Bias Risk Score (0-100) grounded in calculated fairness, proxy, and drift results."""

from __future__ import annotations

from typing import Any, Dict, Optional


def compute_bias_risk_score(
    bias_results: Dict[str, Dict[str, Any]],
    max_proxy_score: float = 0.0,
    drift_psi: float = 0.0,
) -> Dict[str, Any]:
    """
    Aggregate measured disparities across protected attributes into a transparent
    0-100 Bias Risk Score with explicit component weights:
      - 35%: Peak Demographic Parity Difference
      - 35%: Peak Equalized Odds Difference (or Equal Opportunity if unavailable)
      - 20%: Peak Proxy Feature Leakage Score
      - 10%: Score/Fairness Drift PSI & Small-Group Penalty
    """
    dp_diffs = []
    eodds_diffs = []
    small_group_count = 0

    for attr_res in bias_results.values():
        dp_diffs.append(
            float(attr_res["prediction_only_metrics"]["demographic_parity_difference"])
        )
        outcome = attr_res.get("outcome_based_metrics")
        if outcome:
            eodds_diffs.append(float(outcome["equalized_odds_difference"]))
        small_group_count += len(attr_res.get("small_group_warnings", []))

    peak_dp = max(dp_diffs) if dp_diffs else 0.0
    peak_eodds = max(eodds_diffs) if eodds_diffs else peak_dp

    dp_component = min(100.0, (peak_dp / 0.25) * 100.0) * 0.35
    eodds_component = min(100.0, (peak_eodds / 0.25) * 100.0) * 0.35
    proxy_component = min(100.0, max_proxy_score) * 0.20
    drift_penalty = min(100.0, (drift_psi / 0.25) * 70.0 + small_group_count * 10.0) * 0.10

    total_score = round(
        float(min(100.0, dp_component + eodds_component + proxy_component + drift_penalty)), 2
    )

    if total_score >= 70.0:
        level = "CRITICAL"
    elif total_score >= 45.0:
        level = "HIGH"
    elif total_score >= 20.0:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "bias_risk_score": total_score,
        "risk_level": level,
        "breakdown": {
            "demographic_parity_component": round(dp_component, 2),
            "equalized_odds_component": round(eodds_component, 2),
            "proxy_leakage_component": round(proxy_component, 2),
            "drift_and_sample_component": round(drift_penalty, 2),
        },
        "peak_dp_diff": round(peak_dp, 4),
        "peak_eodds_diff": round(peak_eodds, 4),
    }
