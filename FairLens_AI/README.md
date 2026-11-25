# FairLens AI — Bias Detection in an AI Decision System

FairLens AI is a modular machine learning fairness auditing, explainability, bias mitigation, and continuous drift monitoring platform built with **Streamlit**, **Scikit-learn**, **Fairlearn**, **SHAP**, **Supabase PostgreSQL**, **ReportLab**, **Google Gemini (`google-genai`)**, and **Pytest**.

## Development Workflow (VS Code + Google AI Studio)

1. **Visual Studio Code**: Primary Python development environment for modules, Streamlit pages, ML pipelines, Supabase integration, and Pytest.
2. **Google AI Studio**: Provides the `GEMINI_API_KEY` consumed via `modules/gemini_client.py` and `modules/audit_narrator.py` using the official `google-genai` Python SDK.

## Setup Instructions

```bash
cd FairLens_AI
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Configure Environment Variables (`.env`)

- `GEMINI_API_KEY`: Obtain from Google AI Studio (optional; offline template fallback activates automatically if unset).
- `GEMINI_MODEL`: Defaults to `gemini-2.5-flash`.
- `SUPABASE_URL` & `SUPABASE_KEY`: Obtain from your Supabase project settings (anon/public key). Run `database/schema.sql` and `database/policies.sql` in the Supabase SQL Editor.

### Generate Datasets & Run Application

```bash
python dataset/generate_datasets.py
streamlit run app.py
```

### Run Test Suite

```bash
pytest -v
```
