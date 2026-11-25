"""Plotly and Matplotlib fairness visualizations grounded strictly in calculated metrics."""

from __future__ import annotations

from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import AGE_GROUP_LABELS

NAVY = "#0F172A"
ROYAL_BLUE = "#1D4ED8"
MID_BLUE = "#3B82F6"
LIGHT_BLUE = "#93C5FD"
SLATE = "#64748B"


def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color=NAVY),
        margin=dict(l=40, r=24, t=48, b=40),
    )
    return fig


def chart_group_selection_rates(
    group_metrics: List[Dict[str, Any]], attribute_name: str, order_age: bool = False
) -> go.Figure:
    """1 & 2. Gender / Age Group selection-rate comparison with 95% Wilson CIs."""
    df = pd.DataFrame(group_metrics)
    if order_age and not df.empty:
        rank = {lbl: idx for idx, lbl in enumerate(AGE_GROUP_LABELS + ["Missing", "Invalid"])}
        df["_ord"] = df["group"].map(lambda g: rank.get(g, 99))
        df = df.sort_values("_ord")
    err_plus = (df["selection_rate_ci_high"] - df["selection_rate"]).clip(lower=0.0)
    err_minus = (df["selection_rate"] - df["selection_rate_ci_low"]).clip(lower=0.0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["group"],
                y=df["selection_rate"],
                error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
                marker_color=ROYAL_BLUE,
                text=[f"{v:.1%}" for v in df["selection_rate"]],
                textposition="outside",
            )
        ]
    )
    fig.update_yaxes(title_text="Selection Rate", range=[0, 1.08], tickformat=".0%")
    fig.update_xaxes(title_text=attribute_name)
    return _apply_theme(fig, f"{attribute_name} Selection-Rate Comparison (95% Wilson CI)")


def chart_disparity_summary(bias_results: Dict[str, Dict[str, Any]], metric_key: str, title: str) -> go.Figure:
    """3, 4, 5. Demographic Parity, Equal Opportunity, or Equalized Odds across attributes."""
    rows = []
    for attr, res in bias_results.items():
        if metric_key in res["prediction_only_metrics"]:
            val = res["prediction_only_metrics"][metric_key]
        elif res["outcome_based_metrics"] and metric_key in res["outcome_based_metrics"]:
            val = res["outcome_based_metrics"][metric_key]
        else:
            continue
        rows.append({"Attribute": attr, "Disparity": float(val), "Status": res["overall_status"]})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Attribute",
        y="Disparity",
        color="Status",
        color_discrete_map={"PASS": "#16A34A", "WARN": "#D97706", "FAIL": "#DC2626"},
        text_auto=".3f",
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color=NAVY, annotation_text="Fail Threshold (0.10)")
    fig.update_yaxes(title_text="Disparity Value", rangemode="tozero")
    return _apply_theme(fig, title)


def chart_prediction_distribution(
    y_prob: np.ndarray, sensitive_series: pd.Series, attribute_name: str
) -> go.Figure:
    """6. Predicted probability distribution split by protected group."""
    df = pd.DataFrame(
        {
            "Predicted Probability": y_prob,
            attribute_name: sensitive_series.fillna("Missing").astype(str).to_numpy(),
        }
    )
    fig = px.histogram(
        df,
        x="Predicted Probability",
        color=attribute_name,
        barmode="overlay",
        nbins=25,
        opacity=0.65,
        color_discrete_sequence=[ROYAL_BLUE, "#0F172A", "#60A5FA", "#F59E0B", "#10B981"],
    )
    return _apply_theme(fig, f"Prediction Probability Distribution by {attribute_name}")


def chart_intersectional_fairness(intersectional_df: pd.DataFrame) -> go.Figure:
    """7. Intersectional subgroup selection rates."""
    df = intersectional_df.sort_values("selection_rate", ascending=True).tail(16)
    fig = px.bar(
        df,
        x="selection_rate",
        y="group",
        orientation="h",
        color="selection_rate",
        color_continuous_scale=["#DC2626", "#93C5FD", "#1D4ED8"],
        text=[f"{v:.1%} (n={c})" for v, c in zip(df["selection_rate"], df["count"])],
    )
    fig.update_xaxes(title_text="Selection Rate", range=[0, 1.08], tickformat=".0%")
    fig.update_yaxes(title_text="Intersectional Subgroup")
    return _apply_theme(fig, "Intersectional Fairness — Subgroup Selection Rates")


def chart_error_rates(
    group_metrics: List[Dict[str, Any]], rate_key: str, attribute_name: str, title: str
) -> go.Figure:
    """8 & 9. False Positive Rate (FPR) and False Negative Rate (FNR) by group."""
    df = pd.DataFrame(group_metrics)
    df = df[df[rate_key].notna()]
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["group"],
                y=df[rate_key],
                marker_color=NAVY if rate_key == "false_positive_rate" else MID_BLUE,
                text=[f"{v:.1%}" for v in df[rate_key]],
                textposition="outside",
            )
        ]
    )
    fig.update_yaxes(title_text=rate_key.replace("_", " ").title(), range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(title_text=attribute_name)
    return _apply_theme(fig, title)


def chart_model_fairness_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """10. Model performance vs fairness comparison across trained models."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Accuracy",
            x=comparison_df["Model"],
            y=comparison_df["Accuracy"],
            marker_color=ROYAL_BLUE,
        )
    )
    fig.add_trace(
        go.Bar(
            name="F1-Score",
            x=comparison_df["Model"],
            y=comparison_df["F1-Score"],
            marker_color=MID_BLUE,
        )
    )
    if "Max DP Diff" in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                name="Max DP Diff (Lower=Fairer)",
                x=comparison_df["Model"],
                y=comparison_df["Max DP Diff"],
                marker_color="#DC2626",
            )
        )
    if "Max EqOdds Diff" in comparison_df.columns:
        fig.add_trace(
            go.Bar(
                name="Max EqOdds Diff (Lower=Fairer)",
                x=comparison_df["Model"],
                y=comparison_df["Max EqOdds Diff"],
                marker_color="#D97706",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05])
    return _apply_theme(fig, "Model Performance & Fairness Comparison (Identical Test Split)")


def matplotlib_shap_bar(global_df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Matplotlib figure for SHAP global feature importance."""
    sub = global_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=140)
    ax.barh(sub["feature"], sub["mean_abs_shap"], color=ROYAL_BLUE)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)")
    ax.set_title("Global SHAP Feature Importance (Statistical Association)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def chart_selection_vs_confidence_scatter(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    sensitive_df: pd.DataFrame,
) -> go.Figure:
    """Interactive scatter chart comparing group selection rate vs mean model confidence score."""
    rows: List[Dict[str, Any]] = []
    probs = np.asarray(y_prob, dtype=float)
    preds = np.asarray(y_pred, dtype=int)
    confidence = np.maximum(probs, 1.0 - probs)

    for col in sensitive_df.columns:
        if col.lower() == "age":
            continue
        label = "Age Group" if col == "age_group" else col.replace("_", " ").title()
        series = sensitive_df[col].fillna("Missing").astype(str)
        for grp in series.unique():
            mask = (series == grp).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "Attribute": label,
                    "Group": str(grp),
                    "Count": int(mask.sum()),
                    "Selection Rate (%)": round(float(preds[mask].mean()) * 100.0, 2),
                    "Mean Model Confidence (%)": round(float(confidence[mask].mean()) * 100.0, 2),
                    "Mean P(Positive) (%)": round(float(probs[mask].mean()) * 100.0, 2),
                }
            )

    df_plot = pd.DataFrame(rows)
    fig = px.scatter(
        df_plot,
        x="Mean Model Confidence (%)",
        y="Selection Rate (%)",
        size="Count",
        color="Attribute",
        text="Group",
        hover_data=["Count", "Mean P(Positive) (%)"],
        color_discrete_map={
            "Gender": ROYAL_BLUE,
            "Age Group": NAVY,
            "Race": MID_BLUE,
        },
    )
    fig.update_traces(textposition="top center")
    fig.update_yaxes(range=[0, 105])
    fig.update_xaxes(range=[50, 100])
    return _apply_theme(fig, "Selection Rate vs. Mean Model Confidence Across Demographic Groups")


def chart_selection_vs_confidence_bar(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    sensitive_series: pd.Series,
    attribute_name: str,
) -> go.Figure:
    """Interactive grouped bar chart comparing selection rate and mean model confidence per group."""
    probs = np.asarray(y_prob, dtype=float)
    preds = np.asarray(y_pred, dtype=int)
    confidence = np.maximum(probs, 1.0 - probs)
    series = sensitive_series.fillna("Missing").astype(str)

    groups: List[str] = []
    sel_rates: List[float] = []
    conf_scores: List[float] = []
    for grp in sorted(series.unique()):
        mask = (series == grp).to_numpy()
        groups.append(str(grp))
        sel_rates.append(float(preds[mask].mean()) * 100.0)
        conf_scores.append(float(confidence[mask].mean()) * 100.0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Selection Rate (%)",
            x=groups,
            y=sel_rates,
            marker_color=ROYAL_BLUE,
            text=[f"{v:.1f}%" for v in sel_rates],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Mean Confidence (%)",
            x=groups,
            y=conf_scores,
            marker_color=NAVY,
            text=[f"{v:.1f}%" for v in conf_scores],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Percentage (%)", range=[0, 112])
    fig.update_xaxes(title_text=attribute_name)
    return _apply_theme(fig, f"{attribute_name}: Selection Rate vs. Model Confidence")

