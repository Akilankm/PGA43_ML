from pathlib import Path
import numpy as np
import pandas as pd

SEED = 4317
N_ROWS = 2400


def generate_dataset(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(41, 11, n_rows).round(), 21, 70).astype(int)
    income = np.exp(rng.normal(np.log(62000), .48, n_rows)).clip(18000, 240000).round(2)
    employment = np.minimum(np.maximum(rng.gamma(2.5, 3.2, n_rows), 0), age - 18).round(1)
    credit = np.clip(rng.normal(670, 75, n_rows), 350, 850).round().astype(int)
    debt = np.clip(rng.beta(2.2, 4.5, n_rows), .02, .95).round(3)
    missed = np.clip(rng.poisson(.75, n_rows), 0, 8).astype(int)
    loan = np.exp(rng.normal(np.log(18000), .65, n_rows)).clip(1500, 90000).round(2)
    prior = rng.binomial(1, .105, n_rows).astype(int)
    home = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n_rows, p=[.39, .43, .15, .03])
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "education", "vehicle", "small_business"],
        n_rows, p=[.41, .15, .12, .10, .15, .07],
    )
    region = rng.choice(["north", "south", "east", "west"], n_rows)
    channel = rng.choice(["web", "branch", "partner", "mobile"], n_rows, p=[.39, .23, .17, .21])
    month = rng.integers(1, 13, n_rows)
    interest = np.clip(
        4 + .018 * (720 - credit) + 5.2 * debt + prior + .28 * missed + rng.normal(0, .9, n_rows),
        3.5, 28,
    ).round(2)
    income_to_loan = income / np.maximum(loan, 1)
    risk = (
        -4.55 + .010 * (650 - credit) + 2.25 * debt + .38 * missed + 1.25 * prior
        + .035 * (interest - 9) + .000012 * loan - .000004 * income - .045 * employment
        + 1.35 * (credit < 610) + 1.0 * (debt > .55)
        + 1.2 * ((credit < 650) & (debt > .42))
        + .85 * ((purpose == "small_business") & (employment < 3))
        + .55 * ((home == "RENT") & (income_to_loan < 2.4))
        + .45 * ((channel == "partner") & (credit < 670))
    )
    target = rng.binomial(1, 1 / (1 + np.exp(-risk))).astype(int)
    frame = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n_rows + 1)],
        "age": age, "annual_income": income, "employment_years": employment,
        "credit_score": credit, "debt_ratio": debt, "missed_payments_12m": missed,
        "loan_amount": loan, "interest_rate": interest, "prior_default": prior,
        "home_ownership": home, "purpose": purpose, "region": region,
        "channel": channel, "application_month": month, "default": target,
    })
    for column, fraction in {
        "annual_income": .025, "employment_years": .035,
        "debt_ratio": .018, "home_ownership": .012,
    }.items():
        frame.loc[rng.choice(frame.index, int(fraction * n_rows), replace=False), column] = np.nan
    return frame


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "loan_default_teaching.csv"
    generated = generate_dataset()
    generated.to_csv(output, index=False)
    reread = pd.read_csv(output)
    assert len(reread) == N_ROWS and reread["customer_id"].is_unique
    print(f"Generated and validated {len(reread):,} rows at {output}")
