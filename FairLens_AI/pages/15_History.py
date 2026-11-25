"""Page 15: Persistent Audit History (Supabase PostgreSQL & Local Session)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.ui_helpers import get_db, render_sidebar_status

st.set_page_config(page_title="History | FairLens AI", layout="wide")
render_sidebar_status()

st.title("15. Audit, Model & Prediction History")
db = get_db()

if db.is_configured:
    st.success("Connected to Supabase PostgreSQL. Displaying persisted audit history.")
else:
    st.info(
        "Supabase credentials are not configured in .env. Displaying current session activity; "
        "configure SUPABASE_URL and SUPABASE_KEY for persistent PostgreSQL history."
    )

tables = [
    "analyses",
    "datasets",
    "models",
    "fairness_results",
    "mitigation_results",
    "monitoring_batches",
    "predictions",
    "reports",
]

selected_table = st.selectbox("Select Supabase Table to Inspect", tables)
rows = db.fetch_records(selected_table, limit=50)

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.write(f"No persisted rows retrieved from `{selected_table}`.")
    if selected_table == "predictions" and st.session_state.get("live_predictions"):
        st.write("**Session Simulated Predictions:**")
        st.dataframe(pd.DataFrame(st.session_state["live_predictions"]), use_container_width=True)
    elif selected_table == "models" and st.session_state.get("trained_bundles"):
        st.write("**Session Trained Models:**")
        model_rows = [
            {"algorithm": k, **v.metrics}
            for k, v in st.session_state["trained_bundles"].items()
        ]
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True)
