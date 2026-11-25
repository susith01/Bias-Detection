"""ReportLab PDF generator producing comprehensive FairLens AI audit reports."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import REPORTS_DIR


def _build_selection_rate_chart(bias_results: Dict[str, Dict[str, Any]]) -> io.BytesIO:
    """Render a clean Matplotlib comparison chart of group selection rates."""
    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=150)
    labels: List[str] = []
    rates: List[float] = []
    bar_colors: List[str] = []
    palette = {"Gender": "#1D4ED8", "Age Group": "#0F172A", "Race": "#3B82F6"}

    for attr, res in bias_results.items():
        for grp in res.get("group_metrics", []):
            labels.append(f"{attr[:3]}: {grp['group']}")
            rates.append(float(grp["selection_rate"]) * 100.0)
            bar_colors.append(palette.get(attr, "#1E40AF"))

    if labels:
        ax.bar(range(len(labels)), rates, color=bar_colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Selection Rate (%)")
        ax.set_title("Group Selection Rates Across Protected Attributes", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def _styled_table(data: List[List[str]], col_widths: Optional[List[float]] = None) -> Table:
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def generate_pdf_report(
    report_title: str,
    dataset_info: Dict[str, Any],
    model_info: Dict[str, Any],
    bias_results: Dict[str, Dict[str, Any]],
    risk_summary: Dict[str, Any],
    intersectional_df: Optional[pd.DataFrame] = None,
    proxy_df: Optional[pd.DataFrame] = None,
    root_cause_df: Optional[pd.DataFrame] = None,
    mitigation_df: Optional[pd.DataFrame] = None,
    drift_df: Optional[pd.DataFrame] = None,
    gemini_summary: Optional[str] = None,
    recommendations: Optional[List[Dict[str, str]]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Create a structured multi-section PDF fairness audit report with ReportLab."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = output_path or (REPORTS_DIR / f"fairlens_audit_{timestamp}.pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FairLensTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0F172A"),
        fontSize=18,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "FairLensH2",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1D4ED8"),
        fontSize=12,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "FairLensBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
    )

    story: List[Any] = [
        Paragraph(f"FairLens AI — {report_title}", title_style),
        Paragraph(
            f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Bias Risk Score: {risk_summary.get('bias_risk_score', 'N/A')}/100 ({risk_summary.get('risk_level', 'N/A')})",
            body_style,
        ),
        Spacer(1, 0.15 * inch),
    ]

    # 1. Dataset & Model Overview
    story.append(Paragraph("1. Dataset & Model Evaluation", h2_style))
    meta_rows = [
        ["Field", "Value"],
        ["Dataset Name", str(dataset_info.get("name", "N/A"))],
        ["Source Type", str(dataset_info.get("source_type", "N/A")).upper()],
        ["Records / Columns", f"{dataset_info.get('rows', 0)} rows / {dataset_info.get('cols', 0)} cols"],
        ["Model Algorithm", str(model_info.get("algorithm", "N/A"))],
        ["Accuracy / F1 / ROC-AUC", (
            f"{model_info.get('accuracy', 0):.4f} / "
            f"{model_info.get('f1_score', 0):.4f} / "
            f"{model_info.get('roc_auc', 0):.4f}"
        )],
        ["Precision / Recall", f"{model_info.get('precision', 0):.4f} / {model_info.get('recall', 0):.4f}"],
    ]
    story.append(_styled_table(meta_rows, col_widths=[2.2 * inch, 4.5 * inch]))
    story.append(Spacer(1, 0.12 * inch))

    # 2. Gemini / Audit Summary
    if gemini_summary:
        story.append(Paragraph("2. Executive Audit Summary", h2_style))
        for para in gemini_summary.splitlines():
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
                story.append(Spacer(1, 0.04 * inch))

    # 3. Fairness Metrics & Chart
    story.append(Paragraph("3. Fairness Metrics (Gender, Age Group, Race)", h2_style))
    fair_rows = [["Attribute", "Status", "DP Diff", "DP Ratio", "EO Diff", "EqOdds Diff"]]
    for attr, res in bias_results.items():
        pm = res["prediction_only_metrics"]
        om = res.get("outcome_based_metrics") or {}
        fair_rows.append(
            [
                attr,
                res["overall_status"],
                f"{pm['demographic_parity_difference']:.4f}",
                f"{pm['demographic_parity_ratio']:.4f}",
                f"{om.get('equal_opportunity_difference', 0.0):.4f}",
                f"{om.get('equalized_odds_difference', 0.0):.4f}",
            ]
        )
    story.append(_styled_table(fair_rows, col_widths=[1.4 * inch, 0.9 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch]))
    story.append(Spacer(1, 0.1 * inch))

    chart_buf = _build_selection_rate_chart(bias_results)
    story.append(Image(chart_buf, width=6.5 * inch, height=2.85 * inch))

    # 4. Age Group Breakdown
    if "Age Group" in bias_results:
        story.append(Paragraph("4. Centralized Age Group Analysis", h2_style))
        age_rows = [["Age Band", "Count", "Selection Rate", "TPR", "FPR", "FNR"]]
        for g in bias_results["Age Group"]["group_metrics"]:
            age_rows.append(
                [
                    g["group"],
                    str(g["count"]),
                    f"{g['selection_rate']:.3f}",
                    f"{g['true_positive_rate']:.3f}" if g["true_positive_rate"] is not None else "N/A",
                    f"{g['false_positive_rate']:.3f}" if g["false_positive_rate"] is not None else "N/A",
                    f"{g['false_negative_rate']:.3f}" if g["false_negative_rate"] is not None else "N/A",
                ]
            )
        story.append(_styled_table(age_rows, col_widths=[1.2 * inch, 0.9 * inch, 1.2 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch]))

    # 5. Intersectional, Proxy, Root Cause, Mitigation, Drift
    if intersectional_df is not None and not intersectional_df.empty:
        story.append(Paragraph("5. Intersectional Subgroup Summary (Top 8)", h2_style))
        inter_rows = [["Subgroup", "Count", "Selection Rate", "FPR", "FNR"]]
        for _, r in intersectional_df.head(8).iterrows():
            inter_rows.append(
                [
                    str(r["group"])[:36],
                    str(r["count"]),
                    f"{float(r['selection_rate']):.3f}",
                    f"{float(r['false_positive_rate']):.3f}" if pd.notna(r.get("false_positive_rate")) else "N/A",
                    f"{float(r['false_negative_rate']):.3f}" if pd.notna(r.get("false_negative_rate")) else "N/A",
                ]
            )
        story.append(_styled_table(inter_rows))

    if proxy_df is not None and not proxy_df.empty:
        story.append(Paragraph("6. Proxy Leakage & Root Cause Attribution", h2_style))
        p_rows = [["Feature", "Type", "Mutual Info", "Leakage Score", "Tier"]]
        for _, r in proxy_df.head(6).iterrows():
            p_rows.append(
                [
                    str(r["feature"]),
                    str(r["feature_type"]),
                    f"{float(r['mutual_information']):.4f}",
                    f"{float(r['proxy_leakage_score']):.1f}",
                    str(r["risk_tier"]),
                ]
            )
        story.append(_styled_table(p_rows))
        story.append(Spacer(1, 0.08 * inch))

    if root_cause_df is not None and not root_cause_df.empty:
        rc_rows = [["Root Cause Driver", "Contribution %", "Evidence"]]
        for _, r in root_cause_df.iterrows():
            rc_rows.append([str(r["driver"]), f"{r['contribution_pct']}%", str(r["evidence"])[:72]])
        story.append(_styled_table(rc_rows, col_widths=[2.0 * inch, 1.1 * inch, 3.6 * inch]))

    if mitigation_df is not None and not mitigation_df.empty:
        story.append(Paragraph("7. Bias Mitigation Trade-Offs (Same Test Split)", h2_style))
        m_rows = [["Strategy", "Accuracy", "F1", "DP Diff", "EqOdds Diff"]]
        for _, r in mitigation_df.iterrows():
            m_rows.append(
                [
                    str(r["strategy"]),
                    f"{float(r['accuracy']):.3f}" if pd.notna(r["accuracy"]) else "Err",
                    f"{float(r['f1_score']):.3f}" if pd.notna(r["f1_score"]) else "Err",
                    f"{float(r['demographic_parity_diff']):.3f}" if pd.notna(r["demographic_parity_diff"]) else "Err",
                    f"{float(r['equalized_odds_diff']):.3f}" if pd.notna(r["equalized_odds_diff"]) else "Err",
                ]
            )
        story.append(_styled_table(m_rows))

    if drift_df is not None and not drift_df.empty:
        story.append(Paragraph("8. Drift Monitoring Summary", h2_style))
        d_rows = [["Batch", "N", "Selection Rate", "DP Diff", "PSI", "Alert"]]
        for _, r in drift_df.iterrows():
            d_rows.append(
                [
                    str(r["batch_label"]),
                    str(r["sample_size"]),
                    f"{float(r['selection_rate']):.3f}",
                    f"{float(r['dp_diff']):.3f}",
                    f"{float(r['feature_drift_psi']):.3f}",
                    "YES" if r["alert_triggered"] else "NO",
                ]
            )
        story.append(_styled_table(d_rows))

    # Recommendations & Limitations
    story.append(Paragraph("9. Recommendations & Methodological Limitations", h2_style))
    if recommendations:
        for rec in recommendations:
            story.append(
                Paragraph(
                    f"<b>[{rec['priority']}] {rec['strategy']}:</b> {rec['rationale']}",
                    body_style,
                )
            )
            story.append(Spacer(1, 0.04 * inch))

    story.append(
        Paragraph(
            "<b>Limitations:</b> Statistical fairness metrics depend on sample representativeness "
            "and historical label quality. Mitigating one metric (e.g., Demographic Parity) can trade "
            "off against Equalized Odds or Calibration when base rates differ. Synthetic monitoring "
            "batches simulate distribution shift and do not represent live regulated credit/hiring outcomes.",
            body_style,
        )
    )

    doc = SimpleDocTemplate(str(dest), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    doc.build(story)
    return dest
