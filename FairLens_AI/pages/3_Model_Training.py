"""Page 3: ML Model Training & Shared Test-Split Comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.charts import chart_model_fairness_comparison
from modules.fairness_metrics import compute_fairness_metrics
from modules.model_training import train_and_evaluate_models
from modules.ui_helpers import (
    get_db,
    refresh_bias_and_risk_for_bundle,
    render_sidebar_status,
    require_active_dataset,
)

st.set_page_config(page_title="Model Training | FairLens AI", layout="wide")
render_sidebar_status()

st.title("3. AI Model Training & Evaluation")
st.caption(
    "Train Logistic Regression, Decision Tree, and Random Forest models on an identical "
    "stratified train/test split with leakage-safe preprocessing and Joblib persistence."
)

df = require_active_dataset()
if df is None:
    st.stop()

target_col = st.session_state.get("target_column")

with st.form("training_config_form"):
    selected_algos = st.multiselect(
        "Select Algorithms to Train & Compare",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
        default=["Logistic Regression", "Decision Tree", "Random Forest"],
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        test_ratio = st.slider("Held-out Test Split Ratio", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    with c2:
        include_sensitive = st.checkbox("Include Sensitive Attributes in Training Features", value=True)
    with c3:
        random_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)

    train_btn = st.form_submit_button("Train Selected Models", type="primary")

if train_btn:
    if not selected_algos:
        st.error("Select at least one algorithm.")
    else:
        progress = st.progress(0, text="Initializing leakage-free training pipeline...")
        bundles = train_and_evaluate_models(
            df=df,
            target_column=target_col,
            algorithms=selected_algos,
            test_size=float(test_ratio),
            random_state=int(random_seed),
            include_sensitive_in_features=include_sensitive,
            save_artifacts=True,
        )
        progress.progress(70, text="Saving model artifacts and computing baseline fairness...")
        st.session_state["trained_bundles"] = bundles
        first_model = next(iter(bundles.keys()))
        st.session_state["selected_model_name"] = first_model
        refresh_bias_and_risk_for_bundle(bundles[first_model])

        db = get_db()
        analysis_res = db.save_analysis(
            {
                "dataset_id": st.session_state.get("active_dataset_id"),
                "analysis_name": f"{st.session_state.get('active_dataset_name')} Audit",
                "protected_attributes": list(bundles[first_model].test_sensitive_df.columns),
                "bias_risk_score": st.session_state["risk_summary"].get("bias_risk_score"),
                "risk_level": st.session_state["risk_summary"].get("risk_level"),
            }
        )
        analysis_id = analysis_res["data"].get("id")
        st.session_state["active_analysis_id"] = analysis_id

        for algo_name, b in bundles.items():
            if b.model_path:
                db.upload_file("fairlens-models", b.model_path, b.model_path.name)
            db.save_model_metadata(
                {
                    "analysis_id": analysis_id,
                    "dataset_id": st.session_state.get("active_dataset_id"),
                    "model_name": f"{algo_name} ({st.session_state.get('active_dataset_name')})",
                    "algorithm": algo_name,
                    "accuracy": round(b.metrics["accuracy"], 4),
                    "precision_score": round(b.metrics["precision"], 4),
                    "recall_score": round(b.metrics["recall"], 4),
                    "f1_score": round(b.metrics["f1_score"], 4),
                    "roc_auc": round(b.metrics["roc_auc"], 4),
                    "train_size": len(b.y_train),
                    "test_size": len(b.y_test),
                    "storage_path": str(b.model_path) if b.model_path else None,
                }
            )
        progress.progress(100, text="Training and evaluation complete.")
        st.success(f"Trained {len(bundles)} model(s) on {len(bundles[first_model].y_test)} shared test records.")

bundles = st.session_state.get("trained_bundles") or {}
if bundles:
    st.divider()
    active_choice = st.selectbox(
        "Select Active Model for Downstream Pages",
        list(bundles.keys()),
        index=list(bundles.keys()).index(st.session_state.get("selected_model_name", next(iter(bundles.keys())))),
    )
    if active_choice != st.session_state.get("selected_model_name"):
        st.session_state["selected_model_name"] = active_choice
        refresh_bias_and_risk_for_bundle(bundles[active_choice])

    comp_rows = []
    for name, b in bundles.items():
        max_dp = 0.0
        max_eodds = 0.0
        for col in b.test_sensitive_df.columns:
            if col.lower() == "age":
                continue
            f_res = compute_fairness_metrics(
                b.y_test.to_numpy(), b.y_pred, b.test_sensitive_df[col], compute_ci=False
            )
            max_dp = max(max_dp, f_res["prediction_only_metrics"]["demographic_parity_difference"])
            max_eodds = max(max_eodds, f_res["outcome_based_metrics"]["equalized_odds_difference"])
        comp_rows.append(
            {
                "Model": name,
                "Accuracy": round(b.metrics["accuracy"], 4),
                "Precision": round(b.metrics["precision"], 4),
                "Recall": round(b.metrics["recall"], 4),
                "F1-Score": round(b.metrics["f1_score"], 4),
                "ROC-AUC": round(b.metrics["roc_auc"], 4),
                "Max DP Diff": round(max_dp, 4),
                "Max EqOdds Diff": round(max_eodds, 4),
            }
        )

    comp_df = pd.DataFrame(comp_rows)
    st.subheader("Side-by-Side Model Comparison (Same Test Records)")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.plotly_chart(chart_model_fairness_comparison(comp_df), use_container_width=True)
