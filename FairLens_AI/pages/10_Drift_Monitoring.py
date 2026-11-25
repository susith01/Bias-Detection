"""Page 10: Fairness Drift Monitoring."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from modules.drift_monitor import simulate_drift_batches
from modules.fairness_metrics import compute_fairness_metrics
from modules.ui_helpers import get_db, render_sidebar_status, require_active_bundle

st.set_page_config(page_title="Drift Monitoring | FairLens AI", layout="wide")
render_sidebar_status()

st.title("10. Fairness Drift Monitoring")
st.caption(
    "Tracks Demographic Parity shifts and Population Stability Index (PSI) across sequential batches. "
    "Simulated monitoring batches are clearly labeled to distinguish them from live production traffic."
)

bundle = require_active_bundle()
if bundle is None:
    st.stop()

sens_cols = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"]
c1, c2 = st.columns(2)
with c1:
    sens_col = st.selectbox("Monitored Protected Attribute", sens_cols)
with c2:
    n_batches = st.slider("Number of Simulated Batches", min_value=3, max_value=10, value=5)

if st.button("Run Batch Drift Simulation", type="primary"):
    base_fair = compute_fairness_metrics(
        bundle.y_test.to_numpy(), bundle.y_pred, bundle.test_sensitive_df[sens_col], compute_ci=False
    )
    base_dp = float(base_fair["prediction_only_metrics"]["demographic_parity_difference"])
    drift_df = simulate_drift_batches(
        baseline_probs=bundle.y_prob,
        baseline_dp_diff=base_dp,
        X_reference=bundle.X_test,
        pipeline=bundle.pipeline,
        feature_columns=bundle.feature_columns,
        sensitive_column=sens_col,
        n_batches=int(n_batches),
    )
    st.session_state["drift_results"] = drift_df

    db = get_db()
    for _, row in drift_df.iterrows():
        db.save_monitoring_batch(
            {
                "analysis_id": st.session_state.get("active_analysis_id"),
                "batch_label": str(row["batch_label"]),
                "data_origin": "simulated",
                "sample_size": int(row["sample_size"]),
                "dp_diff": float(row["dp_diff"]),
                "selection_rate": float(row["selection_rate"]),
                "feature_drift_psi": float(row["feature_drift_psi"]),
                "alert_triggered": bool(row["alert_triggered"]),
                "alert_message": str(row["alert_message"]),
            }
        )

drift_df = st.session_state.get("drift_results")
if drift_df is not None:
    alerts = drift_df[drift_df["alert_triggered"]]
    if not alerts.empty:
        st.error(f"{len(alerts)} batch(es) triggered fairness or PSI drift alerts (Simulated Data).")
    else:
        st.success("All simulated batches remained within drift tolerances.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drift_df["batch_label"],
            y=drift_df["dp_diff"],
            mode="lines+markers",
            name="Demographic Parity Diff",
            line=dict(color="#1D4ED8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=drift_df["batch_label"],
            y=drift_df["feature_drift_psi"],
            mode="lines+markers",
            name="Score Drift (PSI)",
            line=dict(color="#DC2626", width=2, dash="dash"),
        )
    )
    fig.update_layout(title="Fairness & Distribution Drift Trajectory (Simulated)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(drift_df, use_container_width=True, hide_index=True)
