"""FairLens AI — Streamlit Application Entry Point."""

from __future__ import annotations

import streamlit as st

from config import CONFIG
from modules.ui_helpers import (
    get_db,
    init_session_state,
    load_builtin_dataset_into_state,
    render_sidebar_status,
)

st.set_page_config(
    page_title="FairLens AI — Bias Detection in an AI Decision System",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar_status()

st.title(CONFIG.app_name)
st.subheader(CONFIG.app_subtitle)
st.write(
    "FairLens AI is a modular fairness auditing, explainability, bias mitigation, "
    "and continuous drift monitoring platform built with Scikit-learn, Fairlearn, SHAP, "
    "Supabase PostgreSQL, ReportLab, and Google Gemini."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Dataset", st.session_state.get("active_dataset_name") or "None")
col2.metric("Trained Models", len(st.session_state.get("trained_bundles") or {}))
risk = st.session_state.get("risk_summary") or {}
col3.metric("Bias Risk Score", f"{risk.get('bias_risk_score', '—')}/100" if risk else "—")
db = get_db()
col4.metric("Persistence", "Supabase" if db.is_configured else "Local Session")

st.divider()
st.subheader("Quick Start Workflow")

c_left, c_right = st.columns([2, 1])
with c_left:
    st.write(
        "1. Select or upload a dataset on **Upload Dataset** (Synthetic Loan, Synthetic Recruitment, or UCI Adult Income).\n"
        "2. Train Logistic Regression, Decision Tree, and Random Forest models on **Model Training**.\n"
        "3. Inspect PASS/WARN/FAIL thresholds, confidence intervals, and 10+ charts across **Bias Detection** and **Fairness Analysis**.\n"
        "4. Apply Reweighing, Threshold Adjustment, ExponentiatedGradient, and ThresholdOptimizer in **Bias Mitigation**.\n"
        "5. Explore SHAP attributions, Proxy Leakage, Root Cause, Drift, Individual & Intersectional Fairness, Gemini Audit Summaries, and PDF Reports."
    )

with c_right:
    with st.form("quick_load_form"):
        choice = st.selectbox(
            "Quick-load built-in dataset",
            ["Synthetic Loan Approval", "Synthetic Recruitment", "UCI Adult Income"],
        )
        submitted = st.form_submit_button("Load Selected Dataset", use_container_width=True)
        if submitted:
            df = load_builtin_dataset_into_state(choice)
            st.success(f"Loaded {choice} ({len(df)} records). Navigate to Model Training or Dashboard.")

st.divider()
st.caption(
    "Use the Streamlit sidebar page list (1_Dashboard through 16_About_Project) to navigate across all modules."
)
