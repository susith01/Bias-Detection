"""Page 4: Bias Detection with Configurable PASS/WARN/FAIL Thresholds."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import FairnessThresholds
from modules.bias_detection import detect_bias_across_attributes
from modules.risk_score import compute_bias_risk_score
from modules.ui_helpers import (
    get_db,
    render_sidebar_status,
    require_active_bundle,
)

st.set_page_config(page_title="Bias Detection | FairLens AI", layout="wide")
render_sidebar_status()

st.title("4. Bias Detection & Threshold Auditing")
st.caption(
    "Evaluates Gender, centralized Age Groups (≤25, 26–35, 36–45, 46–55, 56+), and Race using Fairlearn MetricFrame, "
    "95% confidence intervals, small-group warnings, and configurable PASS / WARN / FAIL rules."
)

bundle = require_active_bundle()
if bundle is None:
    st.stop()

with st.expander("Configure PASS / WARN / FAIL Thresholds"):
    c1, c2, c3, c4 = st.columns(4)
    dp_pass = c1.number_input("DP Diff PASS <=", value=0.05, min_value=0.01, max_value=0.30, step=0.01)
    dp_warn = c2.number_input("DP Diff WARN <=", value=0.10, min_value=0.02, max_value=0.40, step=0.01)
    eodds_pass = c3.number_input("EqOdds PASS <=", value=0.05, min_value=0.01, max_value=0.30, step=0.01)
    min_grp = c4.number_input("Min Group Size", value=30, min_value=5, max_value=200, step=5)

thresholds = FairnessThresholds(
    dp_diff_pass=float(dp_pass),
    dp_diff_warn=float(dp_warn),
    eo_diff_pass=float(eodds_pass),
    eo_diff_warn=float(dp_warn),
    eodds_diff_pass=float(eodds_pass),
    eodds_diff_warn=float(dp_warn),
    min_group_size=int(min_grp),
)

bias_results = detect_bias_across_attributes(
    y_true=bundle.y_test.to_numpy(),
    y_pred=bundle.y_pred,
    sensitive_df=bundle.test_sensitive_df,
    thresholds=thresholds,
)
st.session_state["bias_results"] = bias_results
proxy_max = (st.session_state.get("proxy_results") or {}).get("max_leakage_score", 0.0)
risk = compute_bias_risk_score(bias_results, max_proxy_score=float(proxy_max))
st.session_state["risk_summary"] = risk

if st.button("Persist Fairness Audit Results to Supabase"):
    db = get_db()
    for attr, res in bias_results.items():
        pm = res["prediction_only_metrics"]
        om = res["outcome_based_metrics"] or {}
        db.save_fairness_result(
            {
                "analysis_id": st.session_state.get("active_analysis_id"),
                "sensitive_attribute": attr,
                "demographic_parity_diff": pm["demographic_parity_difference"],
                "demographic_parity_ratio": pm["demographic_parity_ratio"],
                "equal_opportunity_diff": om.get("equal_opportunity_difference"),
                "equalized_odds_diff": om.get("equalized_odds_difference"),
                "overall_status": res["overall_status"],
                "group_metrics": res["group_metrics"],
                "confidence_intervals": res["confidence_intervals"],
                "small_group_warnings": res["small_group_warnings"],
            }
        )
    st.success("Fairness audit results saved.")

for attr, res in bias_results.items():
    with st.container(border=True):
        status = res["overall_status"]
        if status == "PASS":
            st.success(f"{attr} — Overall Status: PASS")
        elif status == "WARN":
            st.warning(f"{attr} — Overall Status: WARN")
        else:
            st.error(f"{attr} — Overall Status: FAIL")

        for warn in res["small_group_warnings"]:
            st.warning(warn)

        pm = res["prediction_only_metrics"]
        om = res["outcome_based_metrics"] or {}
        ci = res["confidence_intervals"]

        col_pred, col_out = st.columns(2)
        with col_pred:
            st.write("**Prediction-Only Metrics (No Ground Truth Needed)**")
            st.metric("Demographic Parity Difference", f"{pm['demographic_parity_difference']:.4f}", res["metric_statuses"]["demographic_parity_difference"])
            st.metric("Demographic Parity Ratio (4/5ths >= 0.80)", f"{pm['demographic_parity_ratio']:.4f}", res["metric_statuses"]["demographic_parity_ratio"])
            if "demographic_parity_difference" in ci:
                dp_ci = ci["demographic_parity_difference"]
                st.caption(f"95% Bootstrap CI (DP Diff): [{dp_ci['low']:.4f}, {dp_ci['high']:.4f}]")

        with col_out:
            st.write("**Outcome-Based Metrics (Requires Ground-Truth Labels)**")
            st.metric("Equal Opportunity Difference", f"{om.get('equal_opportunity_difference', 0.0):.4f}", res["metric_statuses"].get("equal_opportunity_difference"))
            st.metric("Equalized Odds Difference", f"{om.get('equalized_odds_difference', 0.0):.4f}", res["metric_statuses"].get("equalized_odds_difference"))
            if "equalized_odds_difference" in ci:
                eo_ci = ci["equalized_odds_difference"]
                st.caption(f"95% Bootstrap CI (EqOdds Diff): [{eo_ci['low']:.4f}, {eo_ci['high']:.4f}]")

        st.dataframe(pd.DataFrame(res["group_metrics"]), use_container_width=True, hide_index=True)
