"""Fairness Drift Monitor across sequential evaluation or simulated incoming batches."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import CONFIG
from modules.fairness_metrics import compute_fairness_metrics


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI) for numeric feature/score drift."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=breaks)
    act_counts, _ = np.histogram(actual, bins=breaks)
    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-4, 1.0)
    act_pct = np.clip(act_counts / max(len(actual), 1), 1e-4, 1.0)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def evaluate_batch_drift(
    baseline_probs: np.ndarray,
    baseline_dp_diff: float,
    batch_probs: np.ndarray,
    batch_preds: np.ndarray,
    batch_sensitive: pd.Series,
    batch_label: str,
    batch_y_true: Optional[np.ndarray] = None,
    dp_alert_delta: float = 0.05,
    psi_alert_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Compare an incoming batch against baseline fairness and score distribution."""
    fair = compute_fairness_metrics(
        y_true=batch_y_true,
        y_pred=batch_preds,
        sensitive_features=batch_sensitive,
        compute_ci=False,
    )
    batch_dp = float(fair["prediction_only_metrics"]["demographic_parity_difference"])
    dp_shift = batch_dp - baseline_dp_diff
    psi = calculate_psi(baseline_probs, batch_probs)
    sel_rate = float(np.mean(batch_preds))

    alerts: List[str] = []
    if abs(dp_shift) >= dp_alert_delta or batch_dp > CONFIG.thresholds.dp_diff_warn:
        alerts.append(
            f"Fairness drift alert: DP Diff reached {batch_dp:.3f} (shift {dp_shift:+.3f})."
        )
    if psi >= psi_alert_threshold:
        alerts.append(f"Score distribution shift alert: PSI={psi:.3f} (>= {psi_alert_threshold}).")

    return {
        "batch_label": batch_label,
        "sample_size": int(len(batch_preds)),
        "selection_rate": round(sel_rate, 4),
        "dp_diff": round(batch_dp, 4),
        "dp_shift": round(dp_shift, 4),
        "feature_drift_psi": round(psi, 4),
        "equalized_odds_diff": (
            round(float(fair["outcome_based_metrics"]["equalized_odds_difference"]), 4)
            if fair["outcome_based_metrics"] is not None
            else None
        ),
        "alert_triggered": len(alerts) > 0,
        "alert_message": " | ".join(alerts) if alerts else "Within expected drift bounds.",
    }


def simulate_drift_batches(
    baseline_probs: np.ndarray,
    baseline_dp_diff: float,
    X_reference: pd.DataFrame,
    pipeline: Any,
    feature_columns: List[str],
    sensitive_column: str,
    n_batches: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create sequential simulated batches showing progressive covariate/fairness drift."""
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []
    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(X_reference[c])]

    for b in range(1, n_batches + 1):
        sampled = X_reference.sample(
            n=min(350, len(X_reference)), replace=True, random_state=seed + b
        ).copy()
        # Progressive synthetic shift in later batches for monitoring simulation
        drift_factor = (b - 1) * 0.14
        if drift_factor > 0 and numeric_cols:
            col = numeric_cols[0]
            std = float(sampled[col].std() or 1.0)
            sampled[col] = sampled[col] - drift_factor * std * (
                sampled[sensitive_column].astype(str) != sampled[sensitive_column].astype(str).mode()[0]
            ).astype(float)

        probs = (
            pipeline.predict_proba(sampled[feature_columns])[:, 1]
            if hasattr(pipeline, "predict_proba")
            else pipeline.predict(sampled[feature_columns]).astype(float)
        )
        preds = (probs >= 0.5).astype(int)
        # Apply slight stochastic noise on later batches
        if drift_factor > 0.25:
            flip_mask = rng.random(len(preds)) < 0.04
            preds = np.where(flip_mask, 1 - preds, preds)

        res = evaluate_batch_drift(
            baseline_probs=baseline_probs,
            baseline_dp_diff=baseline_dp_diff,
            batch_probs=probs,
            batch_preds=preds,
            batch_sensitive=sampled[sensitive_column],
            batch_label=f"Simulated Batch {b}",
        )
        records.append(res)

    return pd.DataFrame(records)
