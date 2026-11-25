"""Page 14: PDF Audit Report Generator (ReportLab + Gemini Executive Summary)."""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from modules.audit_narrator import generate_audit_narrative
from modules.report_generator import generate_pdf_report
from modules.ui_helpers import (
    get_db,
    refresh_bias_and_risk_for_bundle,
    render_sidebar_status,
    require_active_bundle,
    require_active_dataset,
)

st.set_page_config(page_title="Reports | FairLens AI", layout="wide")
render_sidebar_status()

st.title("14. PDF Fairness Audit Reports")
st.caption(
    "Compile dataset details, model metrics, fairness results, age-group & intersectional tables, "
    "proxy leakage, root cause, mitigation trade-offs, drift monitoring, and AI executive summaries into a ReportLab PDF."
)

df = require_active_dataset()
if df is None:
    st.stop()
bundle = require_active_bundle()
if bundle is None:
    st.stop()

bias_results = st.session_state.get("bias_results") or refresh_bias_and_risk_for_bundle(bundle)
risk_summary = st.session_state.get("risk_summary") or {}

with st.form("report_generation_form"):
    report_title = st.text_input(
        "Report Title",
        value=f"Fairness Audit Report — {st.session_state.get('active_dataset_name')}",
    )
    include_gemini = st.checkbox(
        "Include Gemini Executive Summary (falls back offline if key unavailable)",
        value=True,
    )
    gen_btn = st.form_submit_button("Generate PDF Report", type="primary")

if gen_btn:
    with st.spinner("Generating ReportLab PDF audit report..."):
        summary_text = None
        if include_gemini:
            nar = generate_audit_narrative(
                task_type="executive_summary",
                dataset_name=st.session_state.get("active_dataset_name") or "Dataset",
                model_name=bundle.model_name,
                model_metrics=bundle.metrics,
                bias_results=bias_results,
                risk_summary=risk_summary,
            )
            summary_text = nar["text"]

        from modules.intersectional_analysis import run_intersectional_analysis
        from modules.mitigation_recommender import recommend_mitigations
        from modules.proxy_analysis import compute_proxy_leakage
        from modules.root_cause import attribute_root_causes

        proxy_res = st.session_state.get("proxy_results")
        if not proxy_res:
            sens_candidates = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age" and c in df.columns]
            if sens_candidates:
                proxy_res = compute_proxy_leakage(df, sensitive_column=sens_candidates[0], target_column=bundle.target_column)
                st.session_state["proxy_results"] = proxy_res
        proxy_res = proxy_res or {}

        rc_res = st.session_state.get("root_cause_results")
        if not rc_res:
            sens_candidates = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age" and c in df.columns]
            if sens_candidates:
                rc_res = attribute_root_causes(
                    df=df,
                    target_column=bundle.target_column,
                    sensitive_column=sens_candidates[0],
                    y_true=bundle.y_test.to_numpy(),
                    y_pred=bundle.y_pred,
                    sensitive_test=bundle.test_sensitive_df[sens_candidates[0]],
                )
                st.session_state["root_cause_results"] = rc_res
        rc_res = rc_res or {}

        inter_attrs = [c for c in bundle.test_sensitive_df.columns if c.lower() != "age"][:2]
        inter_df = None
        if inter_attrs:
            inter_df = run_intersectional_analysis(
                y_true=bundle.y_test.to_numpy(),
                y_pred=bundle.y_pred,
                sensitive_df=bundle.test_sensitive_df,
                attributes=inter_attrs,
            ).get("intersectional_table")

        rec_data = recommend_mitigations(
            bias_results=bias_results,
            proxy_max_score=float(proxy_res.get("max_leakage_score", 0.0)),
            high_risk_proxies=proxy_res.get("high_risk_features", []),
        )

        mit_res = st.session_state.get("mitigation_results") or {}
        drift_df = st.session_state.get("drift_results")

        pdf_path = generate_pdf_report(
            report_title=report_title,
            dataset_info={
                "name": st.session_state.get("active_dataset_name"),
                "source_type": st.session_state.get("active_dataset_source"),
                "rows": len(df),
                "cols": len(df.columns),
            },
            model_info={
                "algorithm": bundle.model_name,
                **bundle.metrics,
            },
            bias_results=bias_results,
            risk_summary=risk_summary,
            intersectional_df=inter_df,
            proxy_df=proxy_res.get("proxy_table"),
            root_cause_df=rc_res.get("drivers_table"),
            mitigation_df=mit_res.get("comparison_table"),
            drift_df=drift_df,
            gemini_summary=summary_text,
            recommendations=rec_data.get("recommendations"),
        )
        st.session_state["last_report_path"] = str(pdf_path)

        db = get_db()
        db.upload_file("fairlens-reports", pdf_path, pdf_path.name)
        db.save_report_metadata(
            {
                "analysis_id": st.session_state.get("active_analysis_id"),
                "report_title": report_title,
                "file_name": pdf_path.name,
                "storage_path": f"fairlens-reports/{pdf_path.name}",
                "gemini_summary_included": bool(summary_text),
                "bias_risk_score": risk_summary.get("bias_risk_score"),
            }
        )
        st.success(f"Generated PDF report: {pdf_path.name}")

last_path = st.session_state.get("last_report_path")
if last_path and Path(last_path).exists():
    pdf_bytes = Path(last_path).read_bytes()
    st.download_button(
        label="Download Generated Audit PDF",
        data=pdf_bytes,
        file_name=Path(last_path).name,
        mime="application/pdf",
        type="primary",
    )
