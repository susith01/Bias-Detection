"""Data preprocessing, quality profiling, and centralized age grouping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import AGE_GROUP_LABELS


def categorize_age(age_value: Any) -> str:
    """
    Centralized age grouping function used across all fairness metrics and charts.

    Exact supported age bands:
      - <=25  -> "≤25"
      - 26-35 -> "26–35"
      - 36-45 -> "36–45"
      - 46-55 -> "46–55"
      - >=56  -> "56+"

    Missing values return "Missing"; non-numeric or unrealistic (<0 or >120) return "Invalid".
    Raw age columns must remain preserved separately.
    """
    if age_value is None or (isinstance(age_value, float) and np.isnan(age_value)):
        return "Missing"
    if isinstance(age_value, str):
        stripped = age_value.strip()
        if not stripped or stripped.lower() in {"na", "nan", "null", "none", "?"}:
            return "Missing"
    try:
        numeric_age = float(age_value)
    except (TypeError, ValueError):
        return "Invalid"

    if np.isnan(numeric_age) or np.isinf(numeric_age) or numeric_age < 0 or numeric_age > 120:
        return "Invalid"

    if numeric_age <= 25:
        return "≤25"
    if numeric_age <= 35:
        return "26–35"
    if numeric_age <= 45:
        return "36–45"
    if numeric_age <= 55:
        return "46–55"
    return "56+"


def apply_age_grouping(df: pd.DataFrame, age_col: str = "age") -> pd.DataFrame:
    """Add an `age_group` column while preserving the raw `age` column intact."""
    result = df.copy()
    if age_col in result.columns:
        result["age_group"] = result[age_col].apply(categorize_age)
    return result


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Profile dataset shape, missingness, duplicates, and column types."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    missing_per_col = {col: int(df[col].isna().sum()) for col in df.columns}
    duplicate_count = int(df.duplicated().sum())

    candidate_sensitive = [
        c
        for c in df.columns
        if any(tok in c.lower() for tok in ("gender", "sex", "race", "age", "ethnic"))
    ]
    candidate_targets = [
        c
        for c in df.columns
        if df[c].dropna().nunique() == 2
        and c not in candidate_sensitive
        and not c.lower().endswith("_id")
    ]

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_per_col,
        "total_missing": int(sum(missing_per_col.values())),
        "duplicate_count": duplicate_count,
        "candidate_sensitive": candidate_sensitive,
        "candidate_targets": candidate_targets,
    }


def detect_sensitive_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Identify standard Gender, Age, and Race columns if present."""
    cols_lower = {c.lower(): c for c in df.columns}
    gender_col = next(
        (cols_lower[k] for k in cols_lower if k in ("gender", "sex")), None
    )
    age_col = next((cols_lower[k] for k in cols_lower if k == "age"), None)
    race_col = next(
        (cols_lower[k] for k in cols_lower if k in ("race", "ethnicity")), None
    )
    return {"gender": gender_col, "age": age_col, "race": race_col}


def prepare_binary_target(series: pd.Series) -> Tuple[pd.Series, Dict[Any, int]]:
    """Convert binary target column to 0/1 integer series cleanly."""
    if pd.api.types.is_numeric_dtype(series):
        uniq = sorted(series.dropna().unique().tolist())
        if set(uniq).issubset({0, 1}):
            return series.astype(int), {0: 0, 1: 1}
    vals = series.dropna().unique().tolist()
    if len(vals) != 2:
        raise ValueError(f"Target column must have exactly 2 unique values; found {len(vals)}.")
    sorted_vals = sorted(vals, key=lambda x: str(x))
    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    return series.map(mapping).astype(int), mapping


def build_preprocessor(
    X: pd.DataFrame, exclude_columns: Optional[List[str]] = None
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """Build leakage-safe scikit-learn ColumnTransformer using only training feature schema."""
    excluded = set(exclude_columns or [])
    usable_cols = [
        c
        for c in X.columns
        if c not in excluded and not c.lower().endswith("_id") and c != "age_group"
    ]
    numeric_features = [c for c in usable_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in usable_cols if c not in numeric_features]

    transformers = []
    if numeric_features:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_features))

    if categorical_features:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, usable_cols, numeric_features, categorical_features


def ordered_age_groups(values: pd.Series) -> List[str]:
    """Return age group categories in canonical order."""
    present = set(values.dropna().astype(str).unique().tolist())
    ordered = [lbl for lbl in AGE_GROUP_LABELS if lbl in present]
    for extra in ("Missing", "Invalid"):
        if extra in present:
            ordered.append(extra)
    return ordered
