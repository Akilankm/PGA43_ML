"""Generate the deterministic, offline Bank Marketing teaching dataset.

The schema follows the familiar UCI Bank Marketing variables, but the rows are
synthetic.  Keeping the generator in the repository makes the notebook fully
reproducible without requiring a network connection or redistributing the
original UCI data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 43
N_ROWS = 5_000


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def generate(n_rows: int = N_ROWS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a deterministic synthetic dataset with 5,000 base rows.

    The post-contact ``duration`` variable is deliberately predictive of the
    outcome so that the notebook can demonstrate why temporal availability and
    leakage matter.  A small amount of missingness and exact duplication is
    injected for the data-audit lesson.
    """

    rng = np.random.default_rng(random_state)

    jobs = np.array(
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ]
    )
    job_probability = np.array(
        [0.115, 0.215, 0.035, 0.027, 0.215, 0.050, 0.035, 0.095, 0.022, 0.205, 0.025, 0.001]
    )
    job = rng.choice(jobs, n_rows, p=job_probability / job_probability.sum())

    age = np.clip(np.round(rng.normal(40.8, 10.8, n_rows)), 18, 92).astype(int)
    age += np.where(job == "retired", rng.integers(12, 24, n_rows), 0)
    age -= np.where(job == "student", rng.integers(8, 16, n_rows), 0)
    age = np.clip(age, 18, 95)

    marital = rng.choice(["married", "single", "divorced"], n_rows, p=[0.60, 0.28, 0.12])
    marital = np.where(
        age < 27,
        rng.choice(["single", "married"], n_rows, p=[0.78, 0.22]),
        marital,
    )
    education = rng.choice(
        ["primary", "secondary", "tertiary", "unknown"],
        n_rows,
        p=[0.16, 0.52, 0.29, 0.03],
    )
    education = np.where(
        np.isin(job, ["management", "student"]),
        rng.choice(["secondary", "tertiary", "unknown"], n_rows, p=[0.30, 0.67, 0.03]),
        education,
    )

    default = rng.choice(["no", "yes"], n_rows, p=[0.983, 0.017])
    housing = rng.choice(["no", "yes"], n_rows, p=[0.45, 0.55])
    loan = rng.choice(["no", "yes"], n_rows, p=[0.84, 0.16])

    raw_balance = rng.lognormal(mean=7.0, sigma=1.15, size=n_rows) - 850
    raw_balance += np.where(np.isin(job, ["management", "retired"]), 900, 0)
    raw_balance -= np.where((default == "yes") | (loan == "yes"), 550, 0)
    balance = np.clip(np.round(raw_balance), -5_000, 100_000).astype(int)

    contact = rng.choice(["cellular", "telephone", "unknown"], n_rows, p=[0.65, 0.07, 0.28])
    month = rng.choice(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        n_rows,
        p=[0.031, 0.058, 0.011, 0.065, 0.305, 0.118, 0.153, 0.139, 0.013, 0.016, 0.088, 0.003],
    )
    day = rng.integers(1, 32, n_rows)
    campaign = np.clip(rng.geometric(0.36, n_rows), 1, 40)

    previous_contact = rng.random(n_rows) < 0.18
    previous = np.where(previous_contact, rng.poisson(2.0, n_rows) + 1, 0)
    pdays = np.where(
        previous_contact,
        np.clip(rng.normal(195, 95, n_rows).round().astype(int), 1, 871),
        -1,
    )
    poutcome = np.full(n_rows, "unknown", dtype=object)
    previous_indices = np.where(previous_contact)[0]
    poutcome[previous_indices] = rng.choice(
        ["failure", "other", "success"],
        len(previous_indices),
        p=[0.67, 0.20, 0.13],
    )

    # Known only after the current call ends: useful for leakage demonstration,
    # but excluded from the prospective pre-call model.
    duration = np.clip(rng.gamma(shape=2.0, scale=125.0, size=n_rows).round().astype(int), 5, 3_600)

    linear_predictor = (
        -3.25
        + 0.035 * (age - 40)
        + 0.00015 * np.clip(balance, -3_000, 25_000)
        - 0.80 * (housing == "yes")
        - 0.65 * (loan == "yes")
        - 0.85 * (default == "yes")
        - 0.25 * (campaign - 1)
        + 0.35 * np.log1p(previous)
        + 2.50 * (poutcome == "success")
        - 0.70 * (poutcome == "failure")
        + 0.70 * (contact == "cellular")
        - 0.65 * (contact == "unknown")
        + 0.90 * np.isin(month, ["mar", "sep", "oct", "dec"])
        - 0.60 * (month == "may")
        + 0.90 * (job == "student")
        + 0.70 * (job == "retired")
        + 0.35 * (education == "tertiary")
        + 0.0050 * np.clip(duration - 180, -180, 900)
    )
    probability = sigmoid(linear_predictor)
    target = np.where(rng.random(n_rows) < probability, "yes", "no")

    frame = pd.DataFrame(
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "y": target,
        }
    )

    # Missingness is small and realistic enough to demonstrate imputation.
    for column, fraction in {"balance": 0.004, "education": 0.003, "contact": 0.002}.items():
        count = max(1, int(round(n_rows * fraction)))
        missing_indices = rng.choice(n_rows, size=count, replace=False)
        frame.loc[missing_indices, column] = np.nan

    # Exact duplicates make the duplicate-audit and leakage-safe cleaning steps
    # visible without changing the underlying teaching signal.
    duplicates = frame.sample(24, random_state=random_state)
    return pd.concat([frame, duplicates], ignore_index=True)


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "bank_marketing_teaching.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(output, index=False)
    print(f"Wrote {len(data):,} rows to {output}")
    print(data["y"].value_counts(normalize=True).rename("share"))


if __name__ == "__main__":
    main()
