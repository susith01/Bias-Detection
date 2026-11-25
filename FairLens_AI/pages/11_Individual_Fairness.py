"""Page 11: Individual Fairness (k-NN Consistency & Counterfactual Flips)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from modules.individual_fairness import evaluate_individual_fairness
from modules.ui_helpers import render_sidebar_status, require_active_bundle

st.set_page_config(page_title="Individual Fairness | FairLens AI", layout="wide")
render_sidebar_status()

st.title("11. Individual Fairness & Counterfactual Testing")
st.caption(
    "Tests whether similar individuals receive similar probabilities (k-NN Lipschitz consistency) "
    "and whether changing only a protected attribute flips the model's decision."
)

bundle = require_active_bundle()
if bundle is None:
    st.stop()

sens_cols = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"]
c1, c2 = st.columns(2)
with c1:
    sens_col = st.selectbox("Sensitive Attribute for Counterfactual Flip Test", sens_cols)
with c2:
    k_val = st.slider("k-Nearest Neighbors (Non-Sensitive Merit Space)", min_value=3, max_value=15, value=5)

indiv_res = evaluate_individual_fairness(
    bundle=bundle,
    sensitive_column=sens_col,
    k_neighbors=int(k_val),
    max_samples=300,
)

m1, m2, m3 = st.columns(3)
m1.metric("k-NN Consistency Score", f"{indiv_res['knn_consistency_score']:.4f}")
m2.metric("Counterfactual Flip Rate", f"{indiv_res['counterfactual_flip_rate']:.2%}")
m3.metric("Flipped Records", f"{indiv_res['counterfactual_flip_count']} / {indiv_res['evaluated_records']}")

cf_df = indiv_res["counterfactual_samples"]
if not cf_df.empty:
    fig = px.histogram(
        cf_df,
        x="max_prob_delta",
        color="decision_flipped",
        nbins=20,
        color_discrete_map={True: "#DC2626", False: "#1D4ED8"},
        title=f"Probability Sensitivity to Counterfactual {sens_col} Swap",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cf_df, use_container_width=True, hide_index=True)
