"""Dataset package for FairLens AI."""

from dataset.generate_datasets import (
    DATASET_CATALOG,
    ensure_all_datasets,
    fetch_or_create_uci_adult_dataset,
    generate_synthetic_loan_dataset,
    generate_synthetic_recruitment_dataset,
)

__all__ = [
    "DATASET_CATALOG",
    "ensure_all_datasets",
    "fetch_or_create_uci_adult_dataset",
    "generate_synthetic_loan_dataset",
    "generate_synthetic_recruitment_dataset",
]
