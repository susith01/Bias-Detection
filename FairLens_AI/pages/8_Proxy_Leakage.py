"""Page 8: Proxy Leakage Scorer."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from modules.proxy_analysis import compute_proxy_leakage
from modules.ui_helpers import render_sidebar_status, require_active_dataset

st.set_page_config(page_title="Proxy Leakage | FairLens AI", layout="wide")
render_sidebar_status()

st.title("8. Proxy Leakage Analysis")
st.caption(
    "Quantifies how strongly non-sensitive features encode protected demographic attributes "
    "using Mutual Information and Cramér's V / Eta-squared."
)

df = require_active_dataset()
if df is None:
    st.stop()

sens_map = st.session_state.get("sensitive_map") or {}
target_col = st.session_state.get("target_column")
candidates = [v for k, v in sens_map.items() if v is not None and v in df.columns]
if "age_group" in df.columns:
    candidates.append("age_group")
candidates = list(dict.fromkeys(candidates))

if not candidates:
    st.warning("No standard Gender, Age, or Race column detected in the active dataset.")
    st.stop()

sens_choice = st.selectbox("Protected Attribute to Audit for Proxy Leakage", candidates)
proxy_res = compute_proxy_leakage(
    df=df,
    sensitive_column=sens_choice,
    target_column=target_col,
)
st.session_state["proxy_results"] = proxy_res

c1, c2 = st.columns(2)
c1.metric("Peak Proxy Leakage Score", f"{proxy_res['max_leakage_score']}/100")
c2.metric("High-Risk Proxy Features", len(proxy_res["high_risk_features"]))

if proxy_res["high_risk_features"]:
    st.warning(
        f"High proxy leakage detected in: {', '.join(proxy_res['high_risk_features'])}. "
        "Dropping the protected column alone will not prevent indirect discrimination."
    )

proxy_df = proxy_res["proxy_table"]
fig = px.bar(
    proxy_df,
    x="proxy_leakage_score",
    y="feature",
    orientation="h",
    color="risk_tier",
    color_discrete_map={"HIGH": "#DC2626", "MODERATE": "#D97706", "LOW": "#1D4ED8"},
    title=f"Proxy Leakage Score by Feature (Target Attribute: {sens_choice})",
)
fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)
st.dataframe(proxy_df, use_container_width=True, hide_index=True)
