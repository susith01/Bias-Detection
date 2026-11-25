"""Shared Streamlit session-state and native UI helpers for FairLens AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from dataset.generate_datasets import DATASET_CATALOG, ensure_all_datasets
from modules.bias_detection import detect_bias_across_attributes
from modules.preprocessing import apply_age_grouping, detect_sensitive_columns
from modules.risk_score import compute_bias_risk_score
from modules.supabase_client import SupabaseManager


def init_session_state() -> None:
    """Initialize required session keys once across all Streamlit pages."""
    defaults: Dict[str, Any] = {
        "active_dataset_name": None,
        "active_dataset_source": None,
        "active_dataset_id": None,
        "active_df": None,
        "target_column": None,
        "sensitive_map": {},
        "trained_bundles": {},
        "selected_model_name": None,
        "active_analysis_id": None,
        "bias_results": {},
        "risk_summary": {},
        "proxy_results": {},
        "root_cause_results": {},
        "mitigation_results": {},
        "drift_results": None,
        "live_predictions": [],
        "audit_narrative": None,
        "local_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_db() -> SupabaseManager:
    if "supabase_manager" not in st.session_state:
        st.session_state["supabase_manager"] = SupabaseManager()
    return st.session_state["supabase_manager"]


def load_builtin_dataset_into_state(dataset_label: str) -> pd.DataFrame:
    """Load one of the three built-in datasets and initialize session metadata."""
    paths = ensure_all_datasets(try_remote_uci=False)
    path = Path(paths[dataset_label])
    df = pd.read_csv(path)
    meta = DATASET_CATALOG[dataset_label]
    sens = detect_sensitive_columns(df)
    if sens.get("age"):
        df = apply_age_grouping(df, age_col=sens["age"])

    st.session_state["active_dataset_name"] = dataset_label
    st.session_state["active_dataset_source"] = meta["source_type"]
    st.session_state["active_df"] = df
    st.session_state["target_column"] = meta["target_column"]
    st.session_state["sensitive_map"] = sens
    # Reset downstream analysis state when switching datasets
    st.session_state["trained_bundles"] = {}
    st.session_state["selected_model_name"] = None
    st.session_state["bias_results"] = {}
    st.session_state["risk_summary"] = {}
    st.session_state["proxy_results"] = {}
    st.session_state["root_cause_results"] = {}
    st.session_state["mitigation_results"] = {}
    st.session_state["drift_results"] = None
    return df


def render_sidebar_status() -> None:
    """Render clean native Streamlit sidebar navigation status."""
    init_session_state()
    with st.sidebar:
        st.title("FairLens AI")
        st.caption("Bias Detection in an AI Decision System")
        st.divider()
        ds_name = st.session_state.get("active_dataset_name")
        ds_src = st.session_state.get("active_dataset_source")
        model_name = st.session_state.get("selected_model_name")
        if ds_name:
            st.info(f"Dataset: {ds_name} ({str(ds_src).upper()})")
        else:
            st.warning("No dataset loaded yet. Visit Upload Dataset.")
        if model_name:
            st.success(f"Active Model: {model_name}")
        risk = st.session_state.get("risk_summary") or {}
        if risk:
            st.metric("Bias Risk Score", f"{risk.get('bias_risk_score', 0)}/100", risk.get("risk_level"))
        db = get_db()
        st.caption(
            "Supabase: Connected" if db.is_configured else "Supabase: Local Offline Mode"
        )


def require_active_dataset() -> Optional[pd.DataFrame]:
    """Ensure a dataset is loaded before rendering downstream analysis pages."""
    init_session_state()
    df = st.session_state.get("active_df")
    if df is None:
        st.warning("Please load or upload a dataset on the 'Upload Dataset' page first.")
        if st.button("Quick-Load Synthetic Loan Approval Dataset"):
            df = load_builtin_dataset_into_state("Synthetic Loan Approval")
            st.rerun()
        return None
    return df


def require_active_bundle():
    """Ensure at least one trained model bundle exists."""
    bundles = st.session_state.get("trained_bundles") or {}
    if not bundles:
        st.warning("No trained model found. Train a model on the 'Model Training' page first.")
        return None
    sel = st.session_state.get("selected_model_name")
    if sel not in bundles:
        sel = next(iter(bundles.keys()))
        st.session_state["selected_model_name"] = sel
    return bundles[sel]


def refresh_bias_and_risk_for_bundle(bundle) -> Dict[str, Any]:
    """Compute and cache bias detection and composite risk score for the active bundle."""
    bias = detect_bias_across_attributes(
        y_true=bundle.y_test.to_numpy(),
        y_pred=bundle.y_pred,
        sensitive_df=bundle.test_sensitive_df,
    )
    st.session_state["bias_results"] = bias
    proxy_max = (st.session_state.get("proxy_results") or {}).get("max_leakage_score", 0.0)
    risk = compute_bias_risk_score(bias, max_proxy_score=float(proxy_max))
    st.session_state["risk_summary"] = risk
    return bias
