"""Page 6: Bias Mitigation & Fairness-Accuracy Trade-Offs."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from modules.bias_mitigation import MITIGATION_DISCLAIMER, run_mitigation_comparison
from modules.mitigation_recommender import recommend_mitigations
from modules.ui_helpers import (
    get_db,
    refresh_bias_and_risk_for_bundle,
    render_sidebar_status,
    require_active_bundle,
)

st.set_page_config(page_title="Bias Mitigation | FairLens AI", layout="wide")
render_sidebar_status()

st.title("6. Bias Mitigation & Trade-Off Evaluation")
st.caption(
    "Compare Reweighing, Threshold Adjustment, Fairlearn ExponentiatedGradient, and "
    "ThresholdOptimizer against the unmitigated baseline on identical held-out test records."
)

bundle = require_active_bundle()
if bundle is None:
    st.stop()

bias_results = st.session_state.get("bias_results") or refresh_bias_and_risk_for_bundle(bundle)
proxy_res = st.session_state.get("proxy_results") or {}

rec_data = recommend_mitigations(
    bias_results=bias_results,
    proxy_max_score=float(proxy_res.get("max_leakage_score", 0.0)),
    high_risk_proxies=proxy_res.get("high_risk_features", []),
)

with st.expander("Mitigation Recommender & Fairness Metric Conflict Explainer", expanded=True):
    for r in rec_data["recommendations"]:
        st.write(f"- **[{r['priority']}] {r['strategy']}**: {r['rationale']}")
    st.divider()
    st.write("**Fairness Metric Conflict Analysis:**")
    for detail in rec_data["conflict_explanation"]["details"]:
        st.write(f"- {detail}")
    st.caption(rec_data["conflict_explanation"]["theoretical_note"])

candidate_sens = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"]
with st.form("mitigation_form"):
    c1, c2 = st.columns(2)
    with c1:
        sens_col = st.selectbox("Protected Attribute to Mitigate", candidate_sens)
    with c2:
        constraint = st.selectbox("Fairlearn Constraint", ["DemographicParity", "EqualizedOdds"])
    strategies = st.multiselect(
        "Mitigation Algorithms",
        ["Reweighing", "Threshold Adjustment", "ExponentiatedGradient", "ThresholdOptimizer"],
        default=["Reweighing", "Threshold Adjustment", "ExponentiatedGradient", "ThresholdOptimizer"],
    )
    run_btn = st.form_submit_button("Run Mitigation Comparison", type="primary")

if run_btn:
    with st.spinner("Running baseline vs mitigated models on identical held-out test split..."):
        mit_res = run_mitigation_comparison(
            bundle=bundle,
            sensitive_column=sens_col,
            strategies=strategies,
            constraint_type=constraint,
        )
        st.session_state["mitigation_results"] = mit_res

        db = get_db()
        comp = mit_res["comparison_table"]
        base_row = comp.iloc[0]
        for _, row in comp.iloc[1:].iterrows():
            if row.isna().any():
                continue
            db.save_mitigation_result(
                {
                    "analysis_id": st.session_state.get("active_analysis_id"),
                    "mitigation_strategy": str(row["strategy"]),
                    "sensitive_attribute": sens_col,
                    "baseline_accuracy": float(base_row["accuracy"]),
                    "mitigated_accuracy": float(row["accuracy"]),
                    "baseline_dp_diff": float(base_row["demographic_parity_diff"]),
                    "mitigated_dp_diff": float(row["demographic_parity_diff"]),
                    "baseline_eodds_diff": float(base_row["equalized_odds_diff"]),
                    "mitigated_eodds_diff": float(row["equalized_odds_diff"]),
                    "tradeoff_notes": f"Constraint={constraint}",
                }
            )

mit_res = st.session_state.get("mitigation_results")
if mit_res:
    st.warning(MITIGATION_DISCLAIMER)
    comp_df = mit_res["comparison_table"]
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    valid_df = comp_df.dropna(subset=["accuracy", "demographic_parity_diff"])
    if not valid_df.empty:
        fig = px.scatter(
            valid_df,
            x="demographic_parity_diff",
            y="accuracy",
            text="strategy",
            size="f1_score",
            color="strategy",
            title="Fairness-Accuracy Trade-Off (Top-Left = Higher Accuracy & Lower Disparity)",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
