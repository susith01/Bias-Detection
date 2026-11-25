"""Page 16: Project Architecture, Methodology, and Academic Documentation."""

from __future__ import annotations

import streamlit as st

from config import AGE_GROUP_LABELS, CONFIG
from modules.gemini_client import GeminiClient
from modules.ui_helpers import render_sidebar_status

st.set_page_config(page_title="About Project | FairLens AI", layout="wide")
render_sidebar_status()

st.title("16. About FairLens AI")
st.subheader("Bias Detection in an AI Decision System")

st.write(
    "FairLens AI is an academic and practical machine learning fairness platform designed "
    "to audit binary AI decision systems across lending, recruitment, and census income prediction."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Development & AI Architecture")
    st.write(
        "- **Visual Studio Code**: Primary Python, Scikit-learn, Fairlearn, SHAP, Streamlit, "
        "Supabase, ReportLab, and Pytest engineering workspace.\n"
        "- **Google AI Studio (Gemini SDK)**: Integrated server-side through `modules/gemini_client.py` "
        "and `modules/audit_narrator.py` using the official `google-genai` Python SDK. "
        "Receives only aggregated calculated metrics (never API keys or personal identifiers) "
        "and automatically degrades to an offline template engine when unavailable."
    )
with c2:
    st.subheader("Standardized Age Grouping Policy")
    st.write(
        "All fairness calculations and visualizations use the centralized `categorize_age()` function "
        "with fixed age groups: " + ", ".join(AGE_GROUP_LABELS) + ". "
        "Missing and invalid ages are classified separately while preserving raw numeric `age` values."
    )

st.subheader("Gemini SDK Connectivity Check")
if st.button("Verify Configured Gemini Model Availability"):
    client = GeminiClient()
    status = client.verify_model_availability()
    if status["available"]:
        st.success(f"Model `{status['model']}` verified ({status['reason']}).")
    else:
        st.warning(f"Offline template fallback active: {status['reason']}")

st.divider()
st.caption(
    f"{CONFIG.app_name} — Built with Streamlit, Scikit-learn, Fairlearn, SHAP, Supabase PostgreSQL, ReportLab, and Google GenAI."
)
