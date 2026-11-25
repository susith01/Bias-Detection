"""Page 12: Intersectional Fairness Analysis."""

from __future__ import annotations

import streamlit as st

from modules.charts import chart_intersectional_fairness
from modules.intersectional_analysis import run_intersectional_analysis
from modules.ui_helpers import render_sidebar_status, require_active_bundle

st.set_page_config(page_title="Intersectional Analysis | FairLens AI", layout="wide")
render_sidebar_status()

st.title("12. Intersectional Fairness Analysis")
st.caption(
    "Evaluates compounded disparities across combined protected attributes such as Gender × Age Group × Race."
)

bundle = require_active_bundle()
if bundle is None:
    st.stop()

available = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"]
selected_attrs = st.multiselect(
    "Select Protected Attributes to Intersect",
    available,
    default=available[:2] if len(available) >= 2 else available,
)

if not selected_attrs:
    st.warning("Select at least one protected attribute.")
    st.stop()

inter = run_intersectional_analysis(
    y_true=bundle.y_test.to_numpy(),
    y_pred=bundle.y_pred,
    sensitive_df=bundle.test_sensitive_df,
    attributes=selected_attrs,
)

pm = inter["prediction_only_metrics"]
om = inter["outcome_based_metrics"] or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Intersectional DP Diff", f"{pm['demographic_parity_difference']:.4f}")
c2.metric("Intersectional DP Ratio", f"{pm['demographic_parity_ratio']:.4f}")
c3.metric("Intersectional EqOdds Diff", f"{om.get('equalized_odds_difference', 0.0):.4f}")
c4.metric("Subgroups Evaluated", len(inter["intersectional_table"]))

for warn in inter["small_group_warnings"][:6]:
    st.warning(warn)

st.plotly_chart(chart_intersectional_fairness(inter["intersectional_table"]), use_container_width=True)
st.dataframe(inter["intersectional_table"], use_container_width=True, hide_index=True)
