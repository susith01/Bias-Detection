"""Page 13: Real-Time Prediction & Simulated Incoming Stream Monitoring."""

from __future__ import annotations

import random
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.fairness_metrics import compute_fairness_metrics
from modules.preprocessing import apply_age_grouping, categorize_age
from modules.ui_helpers import get_db, render_sidebar_status, require_active_bundle

st.set_page_config(page_title="Live Predictions | FairLens AI", layout="wide")
render_sidebar_status()

st.title("13. Live Predictions & Batch Stream Monitoring")
st.info(
    "Data Origin Notice: Incoming stream records generated below are explicitly labeled SIMULATED "
    "to separate synthetic stress-testing from real production applicants. Models are scored in "
    "inference mode without retraining."
)

bundle = require_active_bundle()
if bundle is None:
    st.stop()


def _generate_incoming_batch(reference_df: pd.DataFrame, n_items: int, seed: int) -> pd.DataFrame:
    """Sample/synthesize realistic incoming records matching active dataset schema."""
    sampled = reference_df.sample(n=n_items, replace=True, random_state=seed).copy().reset_index(drop=True)
    rng = random.Random(seed)
    for col in sampled.columns:
        if col == bundle.target_column or col.lower().endswith("_id") or col == "age_group":
            continue
        if pd.api.types.is_numeric_dtype(sampled[col]) and col.lower() != "age":
            std = float(reference_df[col].std() or 1.0)
            sampled[col] = sampled[col].apply(
                lambda v: round(max(0.0, float(v) + rng.gauss(0, std * 0.08)), 3)
            )
    if "age" in sampled.columns:
        sampled = apply_age_grouping(sampled, age_col="age")
    return sampled


with st.form("live_batch_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.slider("Simulated Incoming Batch Size", min_value=10, max_value=200, value=40)
    with c2:
        seed_val = st.number_input("Simulation Seed", min_value=1, max_value=99999, value=101)
    with c3:
        include_sim_labels = st.checkbox(
            "Include Explicitly Simulated Ground-Truth Labels (for Outcome Metrics)",
            value=False,
        )
    score_btn = st.form_submit_button("Score Incoming Batch", type="primary")

if score_btn:
    incoming = _generate_incoming_batch(bundle.X_test, int(batch_size), int(seed_val))
    X_in = incoming[bundle.feature_columns]
    preds = bundle.pipeline.predict(X_in)
    probs = (
        bundle.pipeline.predict_proba(X_in)[:, 1]
        if hasattr(bundle.pipeline, "predict_proba")
        else preds.astype(float)
    )

    incoming["predicted_label"] = preds
    incoming["predicted_probability"] = probs.round(4)
    incoming["data_origin"] = "simulated"

    if include_sim_labels:
        # Simulated ground-truth labels derived independently via Bernoulli draw around noisy score
        rng = random.Random(int(seed_val) + 7)
        sim_truth = [1 if rng.random() < float(p) else 0 for p in probs]
        incoming["simulated_ground_truth"] = sim_truth

    db = get_db()
    records_to_append: List[Dict[str, Any]] = []
    for idx, row in incoming.iterrows():
        sens_payload = {
            k: str(row[k])
            for k in bundle.test_sensitive_df.columns
            if k in row.index
        }
        rec_meta = {
            "model_id": None,
            "analysis_id": st.session_state.get("active_analysis_id"),
            "data_origin": "simulated",
            "input_features": {c: (float(row[c]) if isinstance(row[c], (int, float)) else str(row[c])) for c in bundle.feature_columns},
            "protected_attributes": sens_payload,
            "predicted_label": int(row["predicted_label"]),
            "predicted_probability": float(row["predicted_probability"]),
            "actual_label": int(row["simulated_ground_truth"]) if include_sim_labels else None,
        }
        if idx < 15:
            db.save_prediction(rec_meta)
        records_to_append.append(rec_meta)

    st.session_state["live_batch_df"] = incoming
    st.session_state["live_has_sim_truth"] = include_sim_labels
    st.session_state["live_predictions"] = (
        records_to_append + st.session_state.get("live_predictions", [])
    )[:200]
    st.success(f"Scored {len(incoming)} simulated records through {bundle.model_name} without retraining.")

live_df = st.session_state.get("live_batch_df")
if live_df is not None:
    has_truth = bool(st.session_state.get("live_has_sim_truth", False))
    sens_cols = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age" and c in live_df.columns]
    if sens_cols:
        eval_col = st.selectbox("Audit Incoming Batch Fairness By", sens_cols)
        y_true_batch = live_df["simulated_ground_truth"].to_numpy() if has_truth else None
        batch_fair = compute_fairness_metrics(
            y_true=y_true_batch,
            y_pred=live_df["predicted_label"].to_numpy(),
            sensitive_features=live_df[eval_col],
            compute_ci=False,
        )
        pm = batch_fair["prediction_only_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Batch Selection Rate", f"{live_df['predicted_label'].mean():.1%}")
        c2.metric("Prediction-Only DP Diff", f"{pm['demographic_parity_difference']:.4f}")
        if batch_fair["outcome_based_metrics"] is not None:
            c3.metric(
                "Simulated-Truth EqOdds Diff",
                f"{batch_fair['outcome_based_metrics']['equalized_odds_difference']:.4f}",
            )
        else:
            c3.metric("Outcome Metrics", "Unavailable (No Ground Truth)")

        fig = px.bar(
            pd.DataFrame(batch_fair["group_metrics"]),
            x="group",
            y="selection_rate",
            title=f"Simulated Incoming Batch Selection Rate by {eval_col}",
            color_discrete_sequence=["#1D4ED8"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Scored Records (SIMULATED)")
    st.dataframe(live_df.head(50), use_container_width=True)
