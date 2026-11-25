"""Pytest configuration and shared fixtures for FairLens AI."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.generate_datasets import generate_synthetic_loan_dataset
from modules.model_training import train_and_evaluate_models


@pytest.fixture(scope="session")
def sample_loan_df(tmp_path_factory) -> pd.DataFrame:
    tmp_dir = tmp_path_factory.mktemp("data")
    csv_path = generate_synthetic_loan_dataset(
        n_records=2000, seed=42, output_path=tmp_dir / "loan_test.csv"
    )
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def trained_loan_bundles(sample_loan_df):
    return train_and_evaluate_models(
        df=sample_loan_df.iloc[:600].copy(),
        target_column="loan_approved",
        algorithms=["Logistic Regression", "Decision Tree", "Random Forest"],
        test_size=0.25,
        random_state=42,
        save_artifacts=False,
    )
