"""Mitigation Recommender & Fairness Metric Conflict Explainer."""

from __future__ import annotations

from typing import Any, Dict, List


def recommend_mitigations(
    bias_results: Dict[str, Dict[str, Any]],
    proxy_max_score: float = 0.0,
    high_risk_proxies: Optional[List[str]] = None,  # type: ignore[name-defined]
) -> Dict[str, Any]:
    """Generate prioritized, metric-grounded mitigation recommendations."""
    recs: List[Dict[str, str]] = []
    worst_attr = None
    worst_dp = 0.0
    worst_eo = 0.0
    worst_eodds = 0.0

    for attr, res in bias_results.items():
        dp = float(res["prediction_only_metrics"]["demographic_parity_difference"])
        outcome = res.get("outcome_based_metrics") or {}
        eo = float(outcome.get("equal_opportunity_difference", 0.0))
        eodds = float(outcome.get("equalized_odds_difference", 0.0))
        if max(dp, eodds) >= max(worst_dp, worst_eodds):
            worst_attr = attr
            worst_dp = dp
            worst_eo = eo
            worst_eodds = eodds

    if worst_dp > 0.08:
        recs.append(
            {
                "priority": "High",
                "strategy": "ExponentiatedGradient (DemographicParity) or Reweighing",
                "rationale": (
                    f"Measured Demographic Parity Difference on {worst_attr} is {worst_dp:.3f} (> 0.08). "
                    "In-processing reweighing or ExponentiatedGradient directly aligns selection rates."
                ),
            }
        )

    if worst_eodds > 0.08 or worst_eo > 0.08:
        recs.append(
            {
                "priority": "High",
                "strategy": "ThresholdOptimizer (EqualizedOdds)",
                "rationale": (
                    f"Measured Equalized Odds Difference on {worst_attr} is {worst_eodds:.3f} "
                    f"(Equal Opportunity {worst_eo:.3f}). Post-processing ThresholdOptimizer balances TPR/FPR."
                ),
            }
        )

    if proxy_max_score >= 30.0:
        proxy_list = ", ".join(high_risk_proxies or []) or "correlated features"
        recs.append(
            {
                "priority": "Medium",
                "strategy": "Proxy Feature Auditing & Regularization",
                "rationale": (
                    f"Peak proxy leakage score is {proxy_max_score:.1f}/100 ({proxy_list}). "
                    "Drop or regularize proxy columns before retraining; removing sensitive attributes alone is insufficient."
                ),
            }
        )

    if not recs:
        recs.append(
            {
                "priority": "Low",
                "strategy": "Continuous Batch Drift Monitoring",
                "rationale": "Measured baseline disparities are within moderate bounds; maintain periodic batch audits.",
            }
        )

    conflict_analysis = explain_fairness_metric_conflicts(bias_results)
    return {
        "recommendations": recs,
        "conflict_explanation": conflict_analysis,
        "advisory_disclaimer": (
            "Advisory Notice: Automated recommendations are based on observed statistical trade-offs "
            "and require domain, legal, and ethical review before deployment."
        ),
    }


def explain_fairness_metric_conflicts(
    bias_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze tensions between Demographic Parity, Equalized Odds, and Calibration
    grounded in actual base-rate differences (Chouldechova / Kleinberg impossibility).
    """
    conflicts: List[str] = []
    for attr, res in bias_results.items():
        dp = float(res["prediction_only_metrics"]["demographic_parity_difference"])
        outcome = res.get("outcome_based_metrics")
        if outcome is None:
            continue
        eo = float(outcome["equal_opportunity_difference"])
        eodds = float(outcome["equalized_odds_difference"])

        if dp > 0.08 and eodds <= 0.06:
            conflicts.append(
                f"{attr}: Demographic Parity fails ({dp:.3f}) while Equalized Odds is relatively tight ({eodds:.3f}). "
                "This indicates differing ground-truth base rates across groups—enforcing equal selection rates will force unequal error rates."
            )
        elif eodds > 0.08 and dp <= 0.06:
            conflicts.append(
                f"{attr}: Selection rates appear balanced (DP Diff={dp:.3f}), yet Equalized Odds shows disparity ({eodds:.3f}). "
                "Equal approval volume masks unequal False Positive / False Negative error burdens."
            )
        elif dp > 0.08 and eodds > 0.08:
            conflicts.append(
                f"{attr}: Both Demographic Parity ({dp:.3f}) and Equalized Odds ({eodds:.3f}) exceed thresholds "
                f"(Equal Opportunity={eo:.3f}). Expect an explicit accuracy-vs-fairness trade-off during mitigation."
            )

    if not conflicts:
        conflicts.append(
            "Across evaluated attributes, Demographic Parity and Equalized Odds remain broadly aligned on this split; "
            "nevertheless, when group base rates differ, satisfying Demographic Parity, Equalized Odds, and Calibration simultaneously is mathematically impossible."
        )

    return {
        "has_detected_conflict": len(conflicts) > 0,
        "details": conflicts,
        "theoretical_note": (
            "By the Chouldechova / Kleinberg impossibility theorems, when underlying positive base rates differ across groups, "
            "a non-trivial classifier cannot simultaneously achieve Demographic Parity, Equalized Odds, and Predictive Parity."
        ),
    }
