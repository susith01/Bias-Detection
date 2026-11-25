"""Tests for preprocessing, age grouping, datasets, ML training, fairness, and mitigation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import AGE_GROUP_LABELS
from modules.bias_detection import detect_bias_across_attributes
from modules.bias_mitigation import run_mitigation_comparison
from modules.fairness_metrics import compute_fairness_metrics
from modules.preprocessing import apply_age_grouping, categorize_age, inspect_dataset


def test_categorize_age_exact_bands_and_edge_cases():
    assert categorize_age(18) == "≤25"
    assert categorize_age(25) == "≤25"
    assert categorize_age(26) == "26–35"
    assert categorize_age(35) == "26–35"
    assert categorize_age(36) == "36–45"
    assert categorize_age(45) == "36–45"
    assert categorize_age(46) == "46–55"
    assert categorize_age(55) == "46–55"
    assert categorize_age(56) == "56+"
    assert categorize_age(82) == "56+"
    assert categorize_age(None) == "Missing"
    assert categorize_age(np.nan) == "Missing"
    assert categorize_age("?") == "Missing"
    assert categorize_age(-5) == "Invalid"
    assert categorize_age(145) == "Invalid"
    assert categorize_age("abc") == "Invalid"


def test_apply_age_grouping_preserves_raw_age(sample_loan_df):
    grouped = apply_age_grouping(sample_loan_df.head(50), age_col="age")
    assert "age" in grouped.columns
    assert "age_group" in grouped.columns
    assert grouped["age"].equals(sample_loan_df.head(50)["age"])
    assert set(grouped["age_group"].unique()).issubset(set(AGE_GROUP_LABELS))


def test_dataset_upload_and_inspection(sample_loan_df):
    assert len(sample_loan_df) >= 2000
    profile = inspect_dataset(sample_loan_df)
    assert profile["row_count"] == len(sample_loan_df)
    assert profile["column_count"] == len(sample_loan_df.columns)
    assert "loan_approved" in profile["candidate_targets"]


def test_model_training_and_predictions_shared_split(trained_loan_bundles):
    assert set(trained_loan_bundles.keys()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
    lr = trained_loan_bundles["Logistic Regression"]
    rf = trained_loan_bundles["Random Forest"]
    assert lr.y_test.index.equals(rf.y_test.index)
    for bundle in trained_loan_bundles.values():
        assert 0.0 <= bundle.metrics["accuracy"] <= 1.0
        assert 0.0 <= bundle.metrics["roc_auc"] <= 1.0
        assert len(bundle.y_pred) == len(bundle.y_test)


def test_fairness_metrics_ci_and_unlabeled_separation(trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    labeled = compute_fairness_metrics(
        y_true=bundle.y_test.to_numpy(),
        y_pred=bundle.y_pred,
        sensitive_features=bundle.test_sensitive_df["gender"],
        compute_ci=True,
    )
    assert labeled["has_ground_truth"] is True
    assert labeled["outcome_based_metrics"] is not None
    assert "demographic_parity_difference" in labeled["confidence_intervals"]

    unlabeled = compute_fairness_metrics(
        y_true=None,
        y_pred=bundle.y_pred,
        sensitive_features=bundle.test_sensitive_df["gender"],
        compute_ci=True,
    )
    assert unlabeled["has_ground_truth"] is False
    assert unlabeled["outcome_based_metrics"] is None


def test_bias_detection_and_mitigation(trained_loan_bundles):
    bundle = trained_loan_bundles["Logistic Regression"]
    bias = detect_bias_across_attributes(
        y_true=bundle.y_test.to_numpy(),
        y_pred=bundle.y_pred,
        sensitive_df=bundle.test_sensitive_df,
    )
    assert "Gender" in bias
    assert "Age Group" in bias
    assert bias["Gender"]["overall_status"] in {"PASS", "WARN", "FAIL"}

    mit = run_mitigation_comparison(
        bundle=bundle,
        sensitive_column="gender",
        strategies=["Reweighing", "Threshold Adjustment", "ExponentiatedGradient", "ThresholdOptimizer"],
    )
    comp = mit["comparison_table"]
    assert len(comp) == 5
    assert "Baseline (Unmitigated)" in comp["strategy"].tolist()


def test_missing_target_rows_and_nan_safe_fairness(sample_loan_df):
    from modules.model_training import train_and_evaluate_models

    messy = sample_loan_df.iloc[:120].copy()
    messy.loc[messy.index[:5], "loan_approved"] = np.nan
    bundles = train_and_evaluate_models(
        df=messy,
        target_column="loan_approved",
        algorithms=["Logistic Regression"],
        save_artifacts=False,
    )
    b = bundles["Logistic Regression"]
    assert len(b.y_train) + len(b.y_test) == 115

    # Single-class edge group should not emit NaN rates
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    sens = pd.Series(["A", "A", "B", "B", "A", "B"])
    res = compute_fairness_metrics(y_true, y_pred, sens, compute_ci=False)
    for grp in res["group_metrics"]:
        assert not np.isnan(grp["false_positive_rate"])
        assert not np.isnan(grp["false_negative_rate"])

