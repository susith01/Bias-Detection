"""Central configuration for FairLens AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class FairnessThresholds:
    """Configurable PASS / WARN / FAIL thresholds for fairness disparities."""

    dp_diff_pass: float = 0.05
    dp_diff_warn: float = 0.10
    dp_ratio_pass: float = 0.85
    dp_ratio_warn: float = 0.80  # 4/5ths rule boundary
    eo_diff_pass: float = 0.05
    eo_diff_warn: float = 0.10
    eodds_diff_pass: float = 0.05
    eodds_diff_warn: float = 0.10
    min_group_size: int = 30


AGE_GROUP_LABELS: List[str] = ["≤25", "26–35", "36–45", "46–55", "56+"]
AGE_SPECIAL_LABELS: List[str] = ["Missing", "Invalid"]

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"

for _folder in (DATASET_DIR, MODELS_DIR, UPLOADS_DIR, REPORTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Application runtime settings."""

    app_name: str = "FairLens AI"
    app_subtitle: str = "Bias Detection in an AI Decision System"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip())
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", "").strip())
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", "").strip())
    random_seed: int = int(os.getenv("DEFAULT_RANDOM_SEED", "42"))
    thresholds: FairnessThresholds = field(default_factory=FairnessThresholds)
    supported_models: Dict[str, str] = field(
        default_factory=lambda: {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "Random Forest": "random_forest",
        }
    )


CONFIG = AppConfig()
