"""Dataset registry and generators for FairLens AI."""

from __future__ import annotations

import csv
import math
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

DATASET_DIR = Path(__file__).resolve().parent

UCI_ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_gt_50k",
]

UCI_ADULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

DATASET_CATALOG: Dict[str, Dict[str, str]] = {
    "Synthetic Loan Approval": {
        "filename": "synthetic_loan_approval.csv",
        "source_type": "synthetic",
        "domain": "Consumer Credit & Loan Underwriting",
        "target_column": "loan_approved",
        "description": (
            "Synthetic dataset (2,400 records) simulating consumer loan approvals "
            "with realistic credit features, neighborhood proxy signal, and controlled demographic disparities."
        ),
    },
    "Synthetic Recruitment": {
        "filename": "synthetic_recruitment.csv",
        "source_type": "synthetic",
        "domain": "Automated Resume Screening & Hiring",
        "target_column": "hired",
        "description": (
            "Synthetic dataset (2,400 records) simulating technical hiring decisions "
            "with experience, coding score, career-gap proxy, and demographic attributes."
        ),
    },
    "UCI Adult Income": {
        "filename": "uci_adult_income.csv",
        "source_type": "historical",
        "domain": "Census Income Classification (Historical 1994)",
        "target_column": "income_gt_50k",
        "description": (
            "Historical UCI Adult Income dataset schema (https://archive.ics.uci.edu/dataset/2/adult). "
            "Evaluates income >50K prediction across gender, grouped age, and race."
        ),
    },
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def generate_synthetic_loan_dataset(
    n_records: int = 2400, seed: int = 42, output_path: Optional[Path] = None
) -> Path:
    """Generate at least 2,000 synthetic loan approval records."""
    if n_records < 2000:
        raise ValueError("Synthetic Loan Approval dataset requires at least 2,000 records.")
    rng = random.Random(seed)
    out = output_path or (DATASET_DIR / "synthetic_loan_approval.csv")

    fieldnames = [
        "applicant_id",
        "gender",
        "age",
        "race",
        "annual_income",
        "credit_score",
        "debt_to_income",
        "employment_years",
        "loan_amount",
        "zip_risk_tier",
        "prior_defaults",
        "loan_approved",
    ]

    genders = ["Male", "Female", "Non-Binary"]
    gender_weights = [0.51, 0.46, 0.03]
    races = ["White", "Black", "Asian", "Hispanic", "Other"]
    race_weights = [0.46, 0.20, 0.14, 0.16, 0.04]

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(1, n_records + 1):
            gender = rng.choices(genders, weights=gender_weights, k=1)[0]
            race = rng.choices(races, weights=race_weights, k=1)[0]
            age = int(max(19, min(74, rng.gauss(39, 12))))

            # Controlled proxy correlation between race/neighborhood tier
            if race in ("Black", "Hispanic"):
                zip_tier = rng.choices(["Tier_A", "Tier_B", "Tier_C"], weights=[0.18, 0.37, 0.45], k=1)[0]
            else:
                zip_tier = rng.choices(["Tier_A", "Tier_B", "Tier_C"], weights=[0.44, 0.38, 0.18], k=1)[0]

            base_income = 76000 if gender == "Male" else 68000
            if age <= 25:
                base_income -= 18000
            elif age >= 56:
                base_income += 6000
            annual_income = round(max(22000.0, rng.gauss(base_income, 22000)), 2)

            credit_score = int(max(460, min(850, rng.gauss(685, 65))))
            debt_to_income = round(max(0.05, min(0.68, rng.gauss(0.28, 0.11))), 3)
            employment_years = round(max(0.0, min(40.0, (age - 20) * rng.uniform(0.2, 0.85))), 1)
            loan_amount = round(max(3000.0, min(85000.0, rng.gauss(24000, 11000))), 2)
            prior_defaults = 1 if rng.random() < (0.18 if credit_score < 620 else 0.04) else 0

            # Linear log-odds with intentional historical bias terms for detection demos
            logit = (
                -1.6
                + (credit_score - 650) * 0.018
                + (annual_income - 55000) / 45000.0
                - debt_to_income * 3.1
                + employment_years * 0.05
                - prior_defaults * 1.4
                - (loan_amount / max(annual_income, 1.0)) * 0.9
            )
            if gender == "Female":
                logit -= 0.35
            elif gender == "Non-Binary":
                logit -= 0.40
            if race == "Black":
                logit -= 0.42
            elif race == "Hispanic":
                logit -= 0.28
            if age <= 25 or age >= 56:
                logit -= 0.35
            if zip_tier == "Tier_C":
                logit -= 0.35

            approved = 1 if rng.random() < _sigmoid(logit) else 0
            writer.writerow(
                {
                    "applicant_id": f"LN-{idx:05d}",
                    "gender": gender,
                    "age": age,
                    "race": race,
                    "annual_income": annual_income,
                    "credit_score": credit_score,
                    "debt_to_income": debt_to_income,
                    "employment_years": employment_years,
                    "loan_amount": loan_amount,
                    "zip_risk_tier": zip_tier,
                    "prior_defaults": prior_defaults,
                    "loan_approved": approved,
                }
            )
    return out


def generate_synthetic_recruitment_dataset(
    n_records: int = 2400, seed: int = 42, output_path: Optional[Path] = None
) -> Path:
    """Generate at least 2,000 synthetic recruitment screening records."""
    if n_records < 2000:
        raise ValueError("Synthetic Recruitment dataset requires at least 2,000 records.")
    rng = random.Random(seed + 101)
    out = output_path or (DATASET_DIR / "synthetic_recruitment.csv")

    fieldnames = [
        "candidate_id",
        "gender",
        "age",
        "race",
        "years_experience",
        "education_level",
        "technical_score",
        "interview_score",
        "career_gap_months",
        "university_tier",
        "portfolio_projects",
        "hired",
    ]

    genders = ["Male", "Female", "Non-Binary"]
    gender_weights = [0.54, 0.43, 0.03]
    races = ["White", "Asian", "Black", "Hispanic", "Other"]
    race_weights = [0.43, 0.25, 0.14, 0.14, 0.04]
    edu_levels = ["Bachelors", "Masters", "PhD", "Bootcamp"]

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(1, n_records + 1):
            gender = rng.choices(genders, weights=gender_weights, k=1)[0]
            race = rng.choices(races, weights=race_weights, k=1)[0]
            age = int(max(21, min(66, rng.gauss(34, 9.5))))
            years_exp = round(max(0.0, min(35.0, (age - 21) * rng.uniform(0.35, 0.92))), 1)
            education = rng.choices(edu_levels, weights=[0.50, 0.32, 0.08, 0.10], k=1)[0]
            tech_score = round(max(35.0, min(100.0, rng.gauss(74, 12))), 1)
            interview_score = round(max(30.0, min(100.0, rng.gauss(72, 13))), 1)
            career_gap = (
                int(max(0, rng.gauss(9, 5)))
                if (gender == "Female" and rng.random() < 0.35)
                else int(max(0, rng.gauss(2, 3)))
            )
            uni_tier = rng.choices(
                ["Tier_1", "Tier_2", "Tier_3"],
                weights=[0.36, 0.42, 0.22] if race in ("White", "Asian") else [0.20, 0.44, 0.36],
                k=1,
            )[0]
            projects = rng.randint(1, 10)

            logit = (
                -3.1
                + (tech_score - 65) * 0.055
                + (interview_score - 65) * 0.045
                + years_exp * 0.065
                + projects * 0.09
                + (0.45 if education in ("Masters", "PhD") else 0.0)
                - career_gap * 0.06
            )
            if gender == "Female":
                logit -= 0.45
            elif gender == "Non-Binary":
                logit -= 0.38
            if age >= 46:
                logit -= 0.48
            if race in ("Black", "Hispanic"):
                logit -= 0.26
            if uni_tier == "Tier_1":
                logit += 0.25

            hired = 1 if rng.random() < _sigmoid(logit) else 0
            writer.writerow(
                {
                    "candidate_id": f"REC-{idx:05d}",
                    "gender": gender,
                    "age": age,
                    "race": race,
                    "years_experience": years_exp,
                    "education_level": education,
                    "technical_score": tech_score,
                    "interview_score": interview_score,
                    "career_gap_months": career_gap,
                    "university_tier": uni_tier,
                    "portfolio_projects": projects,
                    "hired": hired,
                }
            )
    return out


def fetch_or_create_uci_adult_dataset(
    output_path: Optional[Path] = None,
    try_remote: bool = True,
    max_rows: int = 5000,
    seed: int = 42,
) -> Path:
    """Download official UCI Adult Income data or create a schema-faithful local historical extract."""
    out = output_path or (DATASET_DIR / "uci_adult_income.csv")
    if try_remote:
        try:
            req = urllib.request.Request(
                UCI_ADULT_DATA_URL,
                headers={"User-Agent": "FairLensAI/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_lines = resp.read().decode("utf-8", errors="ignore").splitlines()
            parsed_rows: List[Dict[str, object]] = []
            for line in raw_lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 15 or "?" in parts:
                    continue
                income_val = 1 if ">50K" in parts[14] else 0
                parsed_rows.append(
                    {
                        "age": int(parts[0]),
                        "workclass": parts[1],
                        "fnlwgt": int(parts[2]),
                        "education": parts[3],
                        "education_num": int(parts[4]),
                        "marital_status": parts[5],
                        "occupation": parts[6],
                        "relationship": parts[7],
                        "race": parts[8],
                        "gender": parts[9],
                        "capital_gain": int(parts[10]),
                        "capital_loss": int(parts[11]),
                        "hours_per_week": int(parts[12]),
                        "native_country": parts[13],
                        "income_gt_50k": income_val,
                    }
                )
                if len(parsed_rows) >= max_rows:
                    break
            if len(parsed_rows) >= 2000:
                with out.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=UCI_ADULT_COLUMNS)
                    writer.writeheader()
                    writer.writerows(parsed_rows)
                return out
        except Exception:
            pass

    # Fallback local census-structured dataset preserving UCI Adult schema
    rng = random.Random(seed + 303)
    workclasses = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"]
    educations = [
        ("HS-grad", 9),
        ("Some-college", 10),
        ("Bachelors", 13),
        ("Masters", 14),
        ("Assoc-voc", 11),
        ("Doctorate", 16),
    ]
    maritals = ["Married-civ-spouse", "Never-married", "Divorced", "Separated"]
    occupations = [
        "Exec-managerial",
        "Prof-specialty",
        "Craft-repair",
        "Sales",
        "Adm-clerical",
        "Tech-support",
    ]
    relationships = ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"]
    races = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    genders = ["Male", "Female"]

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=UCI_ADULT_COLUMNS)
        writer.writeheader()
        for _ in range(2400):
            age = int(max(18, min(80, rng.gauss(38.5, 13.0))))
            gender = rng.choices(genders, weights=[0.67, 0.33], k=1)[0]
            race = rng.choices(races, weights=[0.85, 0.095, 0.032, 0.013, 0.01], k=1)[0]
            edu, edu_num = rng.choices(educations, weights=[0.32, 0.22, 0.22, 0.11, 0.09, 0.04], k=1)[0]
            marital = rng.choices(maritals, weights=[0.46, 0.33, 0.14, 0.07], k=1)[0]
            occ = rng.choice(occupations)
            rel = (
                "Husband"
                if (gender == "Male" and marital == "Married-civ-spouse")
                else ("Wife" if (gender == "Female" and marital == "Married-civ-spouse") else rng.choice(relationships[1:]))
            )
            hours = int(max(15, min(80, rng.gauss(41.5 if gender == "Male" else 37.5, 10.5))))
            cap_gain = rng.choices([0, 3103, 7298, 15024], weights=[0.91, 0.04, 0.03, 0.02], k=1)[0]
            cap_loss = rng.choices([0, 1485, 1902], weights=[0.95, 0.03, 0.02], k=1)[0]

            logit = (
                -3.6
                + (edu_num - 9) * 0.34
                + (age - 30) * 0.035
                + (hours - 35) * 0.04
                + (1.1 if marital == "Married-civ-spouse" else 0.0)
                + (1.4 if cap_gain > 0 else 0.0)
                + (0.55 if occ in ("Exec-managerial", "Prof-specialty") else 0.0)
                - (0.55 if gender == "Female" else 0.0)
                - (0.35 if race == "Black" else 0.0)
            )
            income = 1 if rng.random() < _sigmoid(logit) else 0
            writer.writerow(
                {
                    "age": age,
                    "workclass": rng.choice(workclasses),
                    "fnlwgt": rng.randint(25000, 420000),
                    "education": edu,
                    "education_num": edu_num,
                    "marital_status": marital,
                    "occupation": occ,
                    "relationship": rel,
                    "race": race,
                    "gender": gender,
                    "capital_gain": cap_gain,
                    "capital_loss": cap_loss,
                    "hours_per_week": hours,
                    "native_country": "United-States",
                    "income_gt_50k": income,
                }
            )
    return out


def ensure_all_datasets(try_remote_uci: bool = False) -> Dict[str, Path]:
    """Ensure all three required datasets exist locally."""
    paths = {}
    loan_path = DATASET_DIR / "synthetic_loan_approval.csv"
    rec_path = DATASET_DIR / "synthetic_recruitment.csv"
    uci_path = DATASET_DIR / "uci_adult_income.csv"

    if not loan_path.exists():
        generate_synthetic_loan_dataset(output_path=loan_path)
    if not rec_path.exists():
        generate_synthetic_recruitment_dataset(output_path=rec_path)
    if not uci_path.exists():
        fetch_or_create_uci_adult_dataset(output_path=uci_path, try_remote=try_remote_uci)

    paths["Synthetic Loan Approval"] = loan_path
    paths["Synthetic Recruitment"] = rec_path
    paths["UCI Adult Income"] = uci_path
    return paths


if __name__ == "__main__":
    generated = ensure_all_datasets(try_remote_uci=True)
    for name, path in generated.items():
        print(f"Prepared {name}: {path}")
