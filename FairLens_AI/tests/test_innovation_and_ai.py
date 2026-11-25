"""Tests for SHAP explainability, innovation modules, and Gemini offline fallback."""

from __future__ import annotations

from modules.audit_narrator import generate_audit_narrative
from modules.bias_detection import detect_bias_across_attributes
from modules.drift_monitor import simulate_drift_batches
from modules.explainability import compute_explainability_summary
from modules.gemini_client import GeminiClient
from modules.individual_fairness import evaluate_individual_fairness
from modules.intersectional_analysis import run_intersectional_analysis
from modules.mitigation_recommender import recommend_mitigations
from modules.proxy_analysis import compute_proxy_leakage
from modules.risk_score import compute_bias_risk_score
from modules.root_cause import attribute_root_causes


def test_explainability_global_local_group(trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    exp = compute_explainability_summary(
        bundle=bundle, sample_index=0, group_column="gender", sample_size=60
    )
    assert not exp["global_importance"].empty
    assert not exp["individual_explanation"].empty
    assert not exp["group_explanations"].empty


def test_proxy_and_root_cause(sample_loan_df, trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    proxy = compute_proxy_leakage(
        sample_loan_df.head(500), sensitive_column="race", target_column="loan_approved"
    )
    assert not proxy["proxy_table"].empty
    assert "zip_risk_tier" in proxy["proxy_table"]["feature"].tolist()

    rc = attribute_root_causes(
        df=sample_loan_df.head(500),
        target_column="loan_approved",
        sensitive_column="gender",
        y_true=bundle.y_test.to_numpy(),
        y_pred=bundle.y_pred,
        sensitive_test=bundle.test_sensitive_df["gender"],
    )
    assert len(rc["drivers_table"]) == 4
    assert abs(rc["drivers_table"]["contribution_pct"].sum() - 100.0) <= 1.0


def test_drift_individual_intersectional_and_risk(trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    bias = detect_bias_across_attributes(
        bundle.y_test.to_numpy(), bundle.y_pred, bundle.test_sensitive_df
    )
    risk = compute_bias_risk_score(bias, max_proxy_score=32.0, drift_psi=0.10)
    assert 0.0 <= risk["bias_risk_score"] <= 100.0

    drift_df = simulate_drift_batches(
        baseline_probs=bundle.y_prob,
        baseline_dp_diff=bias["Gender"]["prediction_only_metrics"]["demographic_parity_difference"],
        X_reference=bundle.X_test,
        pipeline=bundle.pipeline,
        feature_columns=bundle.feature_columns,
        sensitive_column="gender",
        n_batches=3,
    )
    assert len(drift_df) == 3

    indiv = evaluate_individual_fairness(bundle, sensitive_column="gender", max_samples=60)
    assert 0.0 <= indiv["knn_consistency_score"] <= 1.0

    inter = run_intersectional_analysis(
        y_true=bundle.y_test.to_numpy(),
        y_pred=bundle.y_pred,
        sensitive_df=bundle.test_sensitive_df,
        attributes=["gender", "age_group"],
    )
    assert not inter["intersectional_table"].empty

    recs = recommend_mitigations(bias, proxy_max_score=35.0, high_risk_proxies=["zip_risk_tier"])
    assert len(recs["recommendations"]) >= 1
    assert recs["conflict_explanation"]["theoretical_note"]


def test_gemini_offline_fallback(trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    bias = detect_bias_across_attributes(
        bundle.y_test.to_numpy(), bundle.y_pred, bundle.test_sensitive_df
    )
    client = GeminiClient(api_key="")
    assert client.is_available is False
    nar = generate_audit_narrative(
        task_type="audit_summary",
        dataset_name="Synthetic Loan Approval",
        model_name=bundle.model_name,
        model_metrics=bundle.metrics,
        bias_results=bias,
        gemini_client=client,
    )
    assert nar["source"] == "offline_fallback"
    assert "Advisory Disclaimer" in nar["text"]
