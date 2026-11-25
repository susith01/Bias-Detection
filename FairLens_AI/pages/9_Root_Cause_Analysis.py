"""Page 9: Root Cause Attribution."""

from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from modules.root_cause import attribute_root_causes
from modules.ui_helpers import render_sidebar_status, require_active_bundle, require_active_dataset

st.set_page_config(page_title="Root Cause Analysis | FairLens AI", layout="wide")
render_sidebar_status()

st.title("9. Root Cause Attribution")
st.caption(
    "Decomposes observed disparities into Historical Label Disparity, Model Error/Threshold Disparity, "
    "Proxy Feature Leakage, and Representation Imbalance."
)

df = require_active_dataset()
if df is None:
    st.stop()
bundle = require_active_bundle()
if bundle is None:
    st.stop()

sens_cols = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age" and c in df.columns]
if not sens_cols:
    st.warning("No protected column available for root-cause attribution.")
    st.stop()

sens_choice = st.selectbox("Protected Attribute", sens_cols)
rc = attribute_root_causes(
    df=df,
    target_column=bundle.target_column,
    sensitive_column=sens_choice,
    y_true=bundle.y_test.to_numpy(),
    y_pred=bundle.y_pred,
    sensitive_test=bundle.test_sensitive_df[sens_choice],
)
st.session_state["root_cause_results"] = rc

st.metric("Primary Disparity Driver", rc["primary_driver"])

drivers_df = rc["drivers_table"]
fig = px.bar(
    drivers_df,
    x="contribution_pct",
    y="driver",
    orientation="h",
    text="contribution_pct",
    color_discrete_sequence=["#1D4ED8"],
    title=f"Estimated Structural Contribution (%) — {sens_choice}",
)
fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)
st.dataframe(drivers_df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Ground-Truth Positive Base Rates")
    st.dataframe(
        pd.DataFrame([{"Group": k, "Positive Base Rate": round(v, 4)} for k, v in rc["base_rates"].items()]),
        use_container_width=True,
        hide_index=True,
    )
with c2:
    st.subheader("Dataset Group Representation")
    st.dataframe(
        pd.DataFrame([{"Group": k, "Dataset Share": round(v, 4)} for k, v in rc["group_shares"].items()]),
        use_container_width=True,
        hide_index=True,
    )
