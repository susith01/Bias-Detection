"""Individual Fairness: k-NN consistency and counterfactual sensitive flip testing."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from modules.model_training import TrainingBundle


def evaluate_individual_fairness(
    bundle: TrainingBundle,
    sensitive_column: str,
    k_neighbors: int = 5,
    max_samples: int = 300,
) -> Dict[str, Any]:
    """
    Measure two individual fairness properties:
      1. Lipschitz / k-NN Consistency: 1 - mean |pred_i - mean(pred_neighbors)|
         computed across non-sensitive merit features.
      2. Counterfactual Flip Rate: fraction of test records whose prediction changes
         when only the sensitive attribute is altered (holding all other features fixed).
    """
    X_sub = bundle.X_test.iloc[:max_samples].copy().reset_index(drop=True)
    y_prob_sub = bundle.y_prob[:max_samples]
    y_pred_sub = bundle.y_pred[:max_samples]

    non_sens_num = [
        c
        for c in bundle.feature_columns
        if c != sensitive_column and pd.api.types.is_numeric_dtype(X_sub[c])
    ]
    if non_sens_num:
        scaled = StandardScaler().fit_transform(
            X_sub[non_sens_num].fillna(X_sub[non_sens_num].median())
        )
    else:
        scaled = np.zeros((len(X_sub), 1))

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_sub)))
    nn.fit(scaled)
    _, indices = nn.kneighbors(scaled)

    diffs: List[float] = []
    for i, nbr_idx in enumerate(indices):
        neighbors = [idx for idx in nbr_idx if idx != i][:k_neighbors]
        if neighbors:
            nbr_mean = float(np.mean(y_prob_sub[neighbors]))
            diffs.append(abs(float(y_prob_sub[i]) - nbr_mean))
    consistency_score = float(1.0 - np.mean(diffs)) if diffs else 1.0

    # Counterfactual sensitive-attribute swap
    flip_count = 0
    cf_records: List[Dict[str, Any]] = []
    if sensitive_column in bundle.feature_columns:
        unique_groups = X_sub[sensitive_column].dropna().astype(str).unique().tolist()
        for i in range(len(X_sub)):
            orig_val = str(X_sub.loc[i, sensitive_column])
            orig_pred = int(y_pred_sub[i])
            orig_prob = float(y_prob_sub[i])
            flipped = False
            max_delta = 0.0
            flipped_to = orig_val
            for alt in unique_groups:
                if alt == orig_val:
                    continue
                cf_row = X_sub.iloc[[i]][bundle.feature_columns].copy()
                cf_row[sensitive_column] = alt
                cf_prob = (
                    float(bundle.pipeline.predict_proba(cf_row)[:, 1][0])
                    if hasattr(bundle.pipeline, "predict_proba")
                    else float(bundle.pipeline.predict(cf_row)[0])
                )
                cf_pred = int(cf_prob >= 0.5)
                delta = abs(cf_prob - orig_prob)
                if delta > max_delta:
                    max_delta = delta
                    flipped_to = alt
                if cf_pred != orig_pred:
                    flipped = True
            if flipped:
                flip_count += 1
            if i < 50:
                cf_records.append(
                    {
                        "record_index": i,
                        "original_group": orig_val,
                        "original_prob": round(orig_prob, 4),
                        "original_pred": orig_pred,
                        "most_sensitive_alt": flipped_to,
                        "max_prob_delta": round(max_delta, 4),
                        "decision_flipped": flipped,
                    }
                )
    else:
        # Sensitive column excluded from training features -> counterfactual flip is 0 by construction
        for i in range(min(25, len(X_sub))):
            cf_records.append(
                {
                    "record_index": i,
                    "original_group": str(bundle.test_sensitive_df.iloc[i].get(sensitive_column, "N/A")),
                    "original_prob": round(float(y_prob_sub[i]), 4),
                    "original_pred": int(y_pred_sub[i]),
                    "most_sensitive_alt": "Excluded from features",
                    "max_prob_delta": 0.0,
                    "decision_flipped": False,
                }
            )

    flip_rate = float(flip_count / max(len(X_sub), 1))
    return {
        "sensitive_attribute": sensitive_column,
        "knn_consistency_score": round(consistency_score, 4),
        "counterfactual_flip_rate": round(flip_rate, 4),
        "counterfactual_flip_count": flip_count,
        "evaluated_records": int(len(X_sub)),
        "counterfactual_samples": pd.DataFrame(cf_records),
    }
