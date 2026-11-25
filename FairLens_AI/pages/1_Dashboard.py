"""Page 1: Executive Fairness & Audit Dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.audit_narrator import generate_audit_narrative
from modules.charts import (
    chart_disparity_summary,
    chart_group_selection_rates,
    chart_selection_vs_confidence_bar,
    chart_selection_vs_confidence_scatter,
)
from modules.ui_helpers import (
    load_builtin_dataset_into_state,
    refresh_bias_and_risk_for_bundle,
    render_sidebar_status,
    require_active_bundle,
    require_active_dataset,
)

st.set_page_config(page_title="Dashboard | FairLens AI", layout="wide")
render_sidebar_status()

st.title("1. Executive Fairness Dashboard")
st.caption("Consolidated view of active dataset quality, model performance, bias risk score, and AI audit narrative.")

df = require_active_dataset()
if df is None:
    st.stop()

bundle = require_active_bundle()
if bundle is None:
    st.info("Dataset loaded. Train at least one model on 'Model Training' to populate fairness metrics.")
    st.stop()

bias_results = st.session_state.get("bias_results") or refresh_bias_and_risk_for_bundle(bundle)
risk = st.session_state.get("risk_summary") or {}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model Accuracy", f"{bundle.metrics['accuracy']:.1%}")
c2.metric("ROC-AUC", f"{bundle.metrics['roc_auc']:.3f}")
c3.metric("Bias Risk Score", f"{risk.get('bias_risk_score', 0):.1f}/100", risk.get("risk_level", "N/A"))
c4.metric("Peak DP Diff", f"{risk.get('peak_dp_diff', 0):.3f}")
c5.metric("Peak EqOdds Diff", f"{risk.get('peak_eodds_diff', 0):.3f}")

st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(
        chart_selection_vs_confidence_scatter(
            bundle.y_prob, bundle.y_pred, bundle.test_sensitive_df
        ),
        use_container_width=True,
    )
with col_b:
    avail_cols = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"]
    bar_attr = st.selectbox("Group Comparison Attribute", avail_cols, key="dash_conf_bar_attr")
    st.plotly_chart(
        chart_selection_vs_confidence_bar(
            bundle.y_prob,
            bundle.y_pred,
            bundle.test_sensitive_df[bar_attr],
            "Age Group" if bar_attr == "age_group" else bar_attr.title(),
        ),
        use_container_width=True,
    )

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(
        chart_disparity_summary(
            bias_results, "demographic_parity_difference", "Demographic Parity Difference Across Attributes"
        ),
        use_container_width=True,
    )
with col_d:
    if "Age Group" in bias_results:
        st.plotly_chart(
            chart_group_selection_rates(
                bias_results["Age Group"]["group_metrics"], "Age Group", order_age=True
            ),
            use_container_width=True,
        )
    elif "Gender" in bias_results:
        st.plotly_chart(
            chart_group_selection_rates(
                bias_results["Gender"]["group_metrics"], "Gender"
            ),
            use_container_width=True,
        )

st.subheader("AI Audit Summary (Google Gemini / Offline Fallback)")
if st.button("Generate Plain-English Audit Summary", type="primary"):
    with st.spinner("Generating audit summary from actual calculated metrics..."):
        narrative = generate_audit_narrative(
            task_type="audit_summary",
            dataset_name=st.session_state.get("active_dataset_name") or "Active Dataset",
            model_name=bundle.model_name,
            model_metrics=bundle.metrics,
            bias_results=bias_results,
            risk_summary=risk,
        )
        st.session_state["audit_narrative"] = narrative

if st.session_state.get("audit_narrative"):
    nar = st.session_state["audit_narrative"]
    st.info(f"Source: {nar['source']} ({nar['model']})")
    st.write(nar["text"])

with st.expander("Switch Built-In Dataset"):
    ds_choice = st.selectbox(
        "Dataset",
        ["Synthetic Loan Approval", "Synthetic Recruitment", "UCI Adult Income"],
        key="dash_ds_switch",
    )
    if st.button("Switch Dataset"):
        load_builtin_dataset_into_state(ds_choice)
        st.rerun()
