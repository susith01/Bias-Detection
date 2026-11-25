"""Tests for Supabase operations, ReportLab PDF generation, and Streamlit 16-page navigation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from modules.bias_detection import detect_bias_across_attributes
from modules.report_generator import generate_pdf_report
from modules.risk_score import compute_bias_risk_score
from modules.supabase_client import SupabaseManager


def test_supabase_offline_and_mocked_operations(tmp_path):
    offline_db = SupabaseManager(url="", key="")
    assert offline_db.is_configured is False
    res = offline_db.save_analysis({"analysis_name": "Offline Test"})
    assert res["status"] == "offline_local"

    mock_client = MagicMock()
    mock_client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "123", "analysis_name": "Mocked"}
    ]
    db = SupabaseManager(url="https://example.supabase.co", key="anon-key")
    db._client = mock_client
    persisted = db.save_analysis({"analysis_name": "Mocked"})
    assert persisted["status"] == "persisted"
    assert persisted["data"]["id"] == "123"


def test_pdf_report_generation(tmp_path, trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    bias = detect_bias_across_attributes(
        bundle.y_test.to_numpy(), bundle.y_pred, bundle.test_sensitive_df
    )
    risk = compute_bias_risk_score(bias)
    out_pdf = tmp_path / "test_audit.pdf"
    generated = generate_pdf_report(
        report_title="Pytest Audit Report",
        dataset_info={"name": "Synthetic Loan", "source_type": "synthetic", "rows": 600, "cols": 12},
        model_info={"algorithm": bundle.model_name, **bundle.metrics},
        bias_results=bias,
        risk_summary=risk,
        gemini_summary="Executive test summary grounded in computed metrics.",
        output_path=out_pdf,
    )
    assert generated.exists()
    content = generated.read_bytes()
    assert content.startswith(b"%PDF")
    assert len(content) > 2000


def test_streamlit_pages_exist_and_no_unsafe_html():
    pages_dir = Path(__file__).resolve().parents[1] / "pages"
    expected = [
        "1_Dashboard.py",
        "2_Upload_Dataset.py",
        "3_Model_Training.py",
        "4_Bias_Detection.py",
        "5_Fairness_Analysis.py",
        "6_Bias_Mitigation.py",
        "7_Explainability.py",
        "8_Proxy_Leakage.py",
        "9_Root_Cause_Analysis.py",
        "10_Drift_Monitoring.py",
        "11_Individual_Fairness.py",
        "12_Intersectional_Analysis.py",
        "13_Live_Predictions.py",
        "14_Reports.py",
        "15_History.py",
        "16_About_Project.py",
    ]
    for filename in expected:
        page_path = pages_dir / filename
        assert page_path.exists(), f"Missing Streamlit page: {filename}"
        src = page_path.read_text(encoding="utf-8")
        assert "unsafe_allow_html" not in src
