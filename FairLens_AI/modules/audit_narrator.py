"""AI Audit Narrator: translates calculated fairness metrics into plain-English reports."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from modules.gemini_client import GeminiClient

ADVISORY_DISCLAIMER = (
    "Advisory Disclaimer: Automated explanations and recommendations are derived from "
    "observed statistical metrics and do not constitute proof of causation or legal compliance. "
    "Human domain oversight is required before modifying production decision systems."
)

SYSTEM_PROMPT = (
    "You are a senior AI Fairness Auditor writing concise, rigorous, plain-English explanations "
    "for technical and non-technical stakeholders. You MUST ONLY cite the exact numerical metrics "
    "provided in the JSON payload. NEVER invent or extrapolate accuracy, precision, recall, "
    "fairness scores, or demographic figures. Explicitly treat statistical associations as "
    "non-causal observations and note that recommendations are advisory."
)


def _sanitized_metrics_payload(
    dataset_name: str,
    model_name: str,
    model_metrics: Dict[str, float],
    bias_results: Dict[str, Dict[str, Any]],
    risk_summary: Optional[Dict[str, Any]] = None,
    mitigation_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a strictly PII-free, aggregate-only JSON summary for Gemini."""
    attr_summary = {}
    for attr, res in bias_results.items():
        attr_summary[attr] = {
            "status": res["overall_status"],
            "prediction_only": res["prediction_only_metrics"],
            "outcome_based": res["outcome_based_metrics"],
            "group_rates": [
                {
                    "group": g["group"],
                    "count": g["count"],
                    "selection_rate": round(g["selection_rate"], 4),
                    "fpr": round(g["false_positive_rate"], 4)
                    if g["false_positive_rate"] is not None
                    else None,
                    "fnr": round(g["false_negative_rate"], 4)
                    if g["false_negative_rate"] is not None
                    else None,
                }
                for g in res["group_metrics"]
            ],
        }
    return {
        "dataset": dataset_name,
        "model": model_name,
        "performance": {k: round(float(v), 4) for k, v in model_metrics.items()},
        "fairness_by_attribute": attr_summary,
        "bias_risk_score": risk_summary or {},
        "mitigation": mitigation_summary or {},
    }


def generate_offline_audit_summary(payload: Dict[str, Any]) -> str:
    """Deterministic template-based fallback summary grounded strictly in calculated metrics."""
    perf = payload.get("performance", {})
    risk = payload.get("bias_risk_score", {})
    fairness = payload.get("fairness_by_attribute", {})

    lines = [
        f"Audit Summary for {payload.get('dataset')} using {payload.get('model')}:",
        (
            f"- Model Performance: Accuracy={perf.get('accuracy', 0):.3f}, "
            f"Precision={perf.get('precision', 0):.3f}, Recall={perf.get('recall', 0):.3f}, "
            f"F1={perf.get('f1_score', 0):.3f}, ROC-AUC={perf.get('roc_auc', 0):.3f}."
        ),
    ]
    if risk:
        lines.append(
            f"- Composite Bias Risk Score: {risk.get('bias_risk_score', 0)}/100 "
            f"({risk.get('risk_level', 'N/A')})."
        )
    for attr, info in fairness.items():
        pred_m = info.get("prediction_only", {})
        out_m = info.get("outcome_based") or {}
        lines.append(
            f"- {attr} [{info.get('status')}]: Demographic Parity Diff="
            f"{pred_m.get('demographic_parity_difference', 0):.3f}, "
            f"DP Ratio={pred_m.get('demographic_parity_ratio', 0):.3f}, "
            f"Equal Opportunity Diff={out_m.get('equal_opportunity_difference', 0):.3f}, "
            f"Equalized Odds Diff={out_m.get('equalized_odds_difference', 0):.3f}."
        )
    lines.append(
        "- Interpretation: Disparities reflect statistical differences in model outputs and "
        "historical outcome distributions rather than causal proof. Review proxy features and "
        "evaluate ThresholdOptimizer or ExponentiatedGradient trade-offs before deployment."
    )
    lines.append(ADVISORY_DISCLAIMER)
    return "\n".join(lines)


def generate_audit_narrative(
    task_type: str,
    dataset_name: str,
    model_name: str,
    model_metrics: Dict[str, float],
    bias_results: Dict[str, Dict[str, Any]],
    risk_summary: Optional[Dict[str, Any]] = None,
    mitigation_summary: Optional[Dict[str, Any]] = None,
    gemini_client: Optional[GeminiClient] = None,
) -> Dict[str, str]:
    """
    Generate plain-English fairness audit narrative via Gemini or offline fallback.

    `task_type` options:
      - "audit_summary"
      - "demographic_disparity"
      - "metric_comparison"
      - "mitigation_advice"
      - "executive_summary"
      - "academic_report"
    """
    client = gemini_client or GeminiClient()
    payload = _sanitized_metrics_payload(
        dataset_name=dataset_name,
        model_name=model_name,
        model_metrics=model_metrics,
        bias_results=bias_results,
        risk_summary=risk_summary,
        mitigation_summary=mitigation_summary,
    )
    fallback = generate_offline_audit_summary(payload)

    task_prompts = {
        "audit_summary": "Provide a clear 3-paragraph AI fairness audit summary highlighting model performance, demographic disparities, and top risks.",
        "demographic_disparity": "Explain the detected disparities across Gender, grouped Age, and Race in plain English for non-technical decision makers.",
        "metric_comparison": "Explain the differences between Demographic Parity, Equal Opportunity, and Equalized Odds using the exact calculated values below.",
        "mitigation_advice": "Recommend practical bias mitigation next steps based strictly on the measured gaps and trade-offs.",
        "executive_summary": "Write an executive summary suitable for a formal PDF fairness governance report.",
        "academic_report": "Write an academic methodology and findings discussion suitable for a university machine learning fairness project report.",
    }
    instruction = task_prompts.get(task_type, task_prompts["audit_summary"])
    prompt = (
        f"{instruction}\n\n"
        f"Calculated Metrics JSON (strict source of truth):\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        f"End with a short advisory disclaimer noting these findings are statistical, not causal proof."
    )

    result = client.generate_text(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        fallback_text=fallback,
    )
    if ADVISORY_DISCLAIMER not in result["text"]:
        result["text"] = f"{result['text']}\n\n{ADVISORY_DISCLAIMER}"
    return result


def explain_single_prediction_narrative(
    model_name: str,
    predicted_label: int,
    predicted_prob: float,
    top_shap_features: list[dict[str, Any]],
    gemini_client: Optional[GeminiClient] = None,
) -> Dict[str, str]:
    """Explain an individual prediction result in plain English without sending PII."""
    client = gemini_client or GeminiClient()
    decision_str = "Positive (Approved / Selected)" if predicted_label == 1 else "Negative (Declined / Not Selected)"
    feat_summary = ", ".join(
        f"{row['feature']} (SHAP={row['shap_value']:+.3f})" for row in top_shap_features[:5]
    )
    fallback = (
        f"Model {model_name} produced prediction '{decision_str}' with probability {predicted_prob:.1%}. "
        f"Top statistical contributors: {feat_summary}. "
        "Note: Feature attributions reflect model associations, not causal guarantees."
    )
    prompt = (
        f"Explain this single model decision in 4-5 clear sentences using only these values:\n"
        f"Model: {model_name}\nDecision: {decision_str}\nProbability: {predicted_prob:.4f}\n"
        f"Top SHAP Contributors: {json.dumps(top_shap_features[:5])}"
    )
    return client.generate_text(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        fallback_text=fallback,
    )
