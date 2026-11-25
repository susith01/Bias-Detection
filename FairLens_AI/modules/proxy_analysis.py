"""Proxy Leakage Scorer: detects non-sensitive features encoding protected attributes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.fillna("Missing"), y.fillna("Missing"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    n = table.values.sum()
    if n == 0:
        return 0.0
    row_sums = table.sum(axis=1).values[:, None]
    col_sums = table.sum(axis=0).values[None, :]
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table.values - expected) ** 2 / np.where(expected > 0, expected, 1.0))
    min_dim = min(table.shape) - 1
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_proxy_leakage(
    df: pd.DataFrame,
    sensitive_column: str,
    target_column: Optional[str] = None,  # type: ignore[name-defined]
    exclude_columns: Optional[List[str]] = None,  # type: ignore[name-defined]
) -> Dict[str, Any]:
    """
    Score how strongly non-sensitive features correlate with a protected attribute
    using Mutual Information (normalized) and Cramér's V / Eta-squared.
    """
    if sensitive_column not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_column}' missing from dataset.")

    excluded = set(exclude_columns or [])
    excluded.add(sensitive_column)
    if target_column:
        excluded.add(target_column)

    sens_raw = df[sensitive_column].fillna("Missing").astype(str)
    le_sens = LabelEncoder()
    sens_encoded = le_sens.fit_transform(sens_raw)

    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        if col in excluded or col.lower().endswith("_id") or col == "age_group":
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            vals = series.fillna(series.median()).to_numpy().reshape(-1, 1)
            mi = float(mutual_info_classif(vals, sens_encoded, random_state=42)[0])
            # Eta-squared (ANOVA R^2 against protected categories)
            overall_mean = float(np.mean(vals))
            ss_total = float(np.sum((vals.ravel() - overall_mean) ** 2))
            ss_between = 0.0
            for grp in np.unique(sens_encoded):
                grp_vals = vals.ravel()[sens_encoded == grp]
                if len(grp_vals) > 0:
                    ss_between += len(grp_vals) * (float(np.mean(grp_vals)) - overall_mean) ** 2
            assoc_score = float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0
            feat_type = "Numeric"
        else:
            cat_vals = series.fillna("Missing").astype(str)
            le_col = LabelEncoder()
            enc_col = le_col.fit_transform(cat_vals).reshape(-1, 1)
            mi = float(
                mutual_info_classif(enc_col, sens_encoded, discrete_features=True, random_state=42)[0]
            )
            assoc_score = _cramers_v(cat_vals, sens_raw)
            feat_type = "Categorical"

        # Composite proxy leakage score 0-100
        leakage_score = float(min(100.0, (0.55 * min(1.0, mi * 2.5) + 0.45 * assoc_score) * 100.0))
        risk_tier = (
            "HIGH" if leakage_score >= 35.0 else ("MODERATE" if leakage_score >= 15.0 else "LOW")
        )
        rows.append(
            {
                "feature": col,
                "feature_type": feat_type,
                "mutual_information": round(mi, 4),
                "association_strength": round(assoc_score, 4),
                "proxy_leakage_score": round(leakage_score, 2),
                "risk_tier": risk_tier,
            }
        )

    score_df = (
        pd.DataFrame(rows)
        .sort_values("proxy_leakage_score", ascending=False)
        .reset_index(drop=True)
    )
    high_risk_features = score_df[score_df["risk_tier"] == "HIGH"]["feature"].tolist()
    max_score = float(score_df["proxy_leakage_score"].max()) if not score_df.empty else 0.0

    return {
        "sensitive_attribute": sensitive_column,
        "proxy_table": score_df,
        "high_risk_features": high_risk_features,
        "max_leakage_score": round(max_score, 2),
    }
