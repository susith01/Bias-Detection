"""Page 5: Comprehensive Fairness Visualization Suite (10+ Interactive Charts)."""

from __future__ import annotations

import streamlit as st

from modules.audit_narrator import generate_audit_narrative
from modules.charts import (
    chart_disparity_summary,
    chart_error_rates,
    chart_group_selection_rates,
    chart_intersectional_fairness,
    chart_prediction_distribution,
)
from modules.intersectional_analysis import run_intersectional_analysis
from modules.ui_helpers import (
    refresh_bias_and_risk_for_bundle,
    render_sidebar_status,
    require_active_bundle,
)

st.set_page_config(page_title="Fairness Analysis | FairLens AI", layout="wide")
render_sidebar_status()

st.title("5. Fairness Visualization & Metric Analysis")
st.caption("Interactive Plotly visualizations built directly from calculated Fairlearn MetricFrame outputs.")

bundle = require_active_bundle()
if bundle is None:
    st.stop()

bias_results = st.session_state.get("bias_results") or refresh_bias_and_risk_for_bundle(bundle)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Selection & Parity", "Error Rates (FPR/FNR)", "Distributions & Intersectional", "AI Metric Explainer"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        if "Gender" in bias_results:
            st.plotly_chart(
                chart_group_selection_rates(bias_results["Gender"]["group_metrics"], "Gender"),
                use_container_width=True,
            )
    with c2:
        if "Age Group" in bias_results:
            st.plotly_chart(
                chart_group_selection_rates(bias_results["Age Group"]["group_metrics"], "Age Group", order_age=True),
                use_container_width=True,
            )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            chart_disparity_summary(bias_results, "demographic_parity_difference", "Demographic Parity Difference"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            chart_disparity_summary(bias_results, "equal_opportunity_difference", "Equal Opportunity Difference"),
            use_container_width=True,
        )
    st.plotly_chart(
        chart_disparity_summary(bias_results, "equalized_odds_difference", "Equalized Odds Difference"),
        use_container_width=True,
    )

with tab2:
    attr_choice = st.selectbox("Select Attribute for Error-Rate Charts", list(bias_results.keys()))
    grp_metrics = bias_results[attr_choice]["group_metrics"]
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(
            chart_error_rates(grp_metrics, "false_positive_rate", attr_choice, f"False Positive Rate by {attr_choice}"),
            use_container_width=True,
        )
    with e2:
        st.plotly_chart(
            chart_error_rates(grp_metrics, "false_negative_rate", attr_choice, f"False Negative Rate by {attr_choice}"),
            use_container_width=True,
        )

with tab3:
    dist_attr = st.selectbox("Select Attribute for Prediction Distribution", list(bundle.test_sensitive_df.columns))
    st.plotly_chart(
        chart_prediction_distribution(bundle.y_prob, bundle.test_sensitive_df[dist_attr], dist_attr),
        use_container_width=True,
    )
    inter_cols = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"][:2]
    if inter_cols:
        inter_res = run_intersectional_analysis(
            y_true=bundle.y_test.to_numpy(),
            y_pred=bundle.y_pred,
            sensitive_df=bundle.test_sensitive_df,
            attributes=inter_cols,
        )
        st.plotly_chart(
            chart_intersectional_fairness(inter_res["intersectional_table"]),
            use_container_width=True,
        )

with tab4:
    st.write("Generate a plain-English comparison of Demographic Parity, Equal Opportunity, and Equalized Odds for these metrics.")
    if st.button("Explain Fairness Metrics with AI"):
        res = generate_audit_narrative(
            task_type="metric_comparison",
            dataset_name=st.session_state.get("active_dataset_name") or "Dataset",
            model_name=bundle.model_name,
            model_metrics=bundle.metrics,
            bias_results=bias_results,
            risk_summary=st.session_state.get("risk_summary"),
        )
        st.info(f"Source: {res['source']}")
        st.write(res["text"])
