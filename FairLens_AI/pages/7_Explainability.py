"""Page 7: SHAP Explainable AI (Global, Individual, Group, and Sensitive Impact)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from modules.audit_narrator import explain_single_prediction_narrative
from modules.charts import matplotlib_shap_bar
from modules.explainability import CAUSATION_NOTE, compute_explainability_summary
from modules.ui_helpers import render_sidebar_status, require_active_bundle

st.set_page_config(page_title="Explainability | FairLens AI", layout="wide")
render_sidebar_status()

st.title("7. Explainable AI (SHAP Attributions)")
st.info(CAUSATION_NOTE)

bundle = require_active_bundle()
if bundle is None:
    st.stop()

group_options = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"]
c1, c2 = st.columns(2)
with c1:
    sample_idx = st.number_input(
        "Test Record Index for Individual Explanation",
        min_value=0,
        max_value=max(0, min(199, len(bundle.X_test) - 1)),
        value=0,
    )
with c2:
    group_col = st.selectbox("Group-Level Explanation Attribute", group_options) if group_options else None

with st.spinner("Computing SHAP attributions..."):
    shap_summary = compute_explainability_summary(
        bundle=bundle,
        sample_index=int(sample_idx),
        group_column=group_col,
        sample_size=200,
    )

tab_global, tab_indiv, tab_group, tab_sens = st.tabs(
    ["Global Importance", "Individual Explanation", "Group Explanations", "Sensitive Attribute Impact"]
)

with tab_global:
    fig_mpl = matplotlib_shap_bar(shap_summary["global_importance"], top_n=12)
    st.pyplot(fig_mpl)
    st.dataframe(shap_summary["global_importance"].head(20), use_container_width=True, hide_index=True)

with tab_indiv:
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Record Index", shap_summary["individual_index"])
    m2.metric("Predicted Decision", "Positive (1)" if shap_summary["individual_pred"] == 1 else "Negative (0)")
    m3.metric("Predicted Probability", f"{shap_summary['individual_prob']:.1%}")

    indiv_top = shap_summary["individual_explanation"].head(12)
    fig_ind = px.bar(
        indiv_top.iloc[::-1],
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale=["#DC2626", "#E2E8F0", "#1D4ED8"],
        title="Local SHAP Feature Contributions for Selected Record",
    )
    fig_ind.update_layout(template="plotly_white")
    st.plotly_chart(fig_ind, use_container_width=True)

    if st.button("Explain This Prediction in Plain English"):
        exp = explain_single_prediction_narrative(
            model_name=bundle.model_name,
            predicted_label=shap_summary["individual_pred"],
            predicted_prob=shap_summary["individual_prob"],
            top_shap_features=indiv_top.to_dict(orient="records"),
        )
        st.write(exp["text"])

with tab_group:
    if not shap_summary["group_explanations"].empty:
        st.write(f"Mean SHAP Attribution by **{group_col}**:")
        st.dataframe(shap_summary["group_explanations"], use_container_width=True, hide_index=True)
    else:
        st.info("No grouping attribute available for group-level SHAP breakdown.")

with tab_sens:
    st.metric(
        "Direct Sensitive Attribute Share of Total |SHAP|",
        f"{shap_summary['sensitive_share_pct']:.2f}%",
    )
    if not shap_summary["sensitive_impact"].empty:
        st.dataframe(shap_summary["sensitive_impact"], use_container_width=True, hide_index=True)
    else:
        st.info(
            "Sensitive attributes were excluded from direct model inputs. "
            "Check Proxy Leakage to see if correlated proxy features encode demographic signals."
        )
