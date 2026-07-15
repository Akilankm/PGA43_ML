from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

GENERATOR = '''from pathlib import Path
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
'''

CONFIG = {
    "decision_tree_classification_masterclass": {
        "title": "Decision Tree Classification Masterclass",
        "env": "decision-tree-masterclass",
        "artifact": "decision_tree_classifier.joblib",
        "coverage": [
            "Decision-tree vocabulary, recursive partitioning, leaf probabilities, entropy, Gini impurity, misclassification error, and weighted information gain",
            "Manual Gini and split-gain calculation before library modeling",
            "Schema, duplicate, missingness, range, cardinality, and class-balance audits",
            "Per-feature numerical and categorical univariate analysis",
            "Class-conditional box plots, category target-rate tables, and multivariate rank-correlation analysis",
            "Stratified train/validation/test isolation before learned preprocessing",
            "Median/mode imputation, one-hot encoding, majority baseline, and unpruned-tree baseline",
            "Depth capacity curves exposing underfitting and overfitting",
            "Three-fold cross-validated tuning of depth, leaf size, and class weighting",
            "Cost-complexity pruning with validation-selected ccp_alpha",
            "Fold stability, plotted tree structure, exported rules, and impurity/permutation importance",
            "ROC, precision-recall, asymmetric-cost threshold engineering, untouched-test evaluation, subgroup checks, serialization, and model card",
        ],
    },
    "random_forest_classification_masterclass": {
        "title": "Random Forest Classification Masterclass",
        "env": "random-forest-masterclass",
        "artifact": "random_forest_classifier.joblib",
        "coverage": [
            "Single-tree variance, bootstrap sampling, out-of-bag observations, random feature subspaces, aggregation, and the bias-variance-correlation trade-off",
            "Manual bootstrap and OOB demonstration before library modeling",
            "Schema, duplicate, missingness, range, cardinality, and class-balance audits",
            "Per-feature numerical and categorical univariate analysis",
            "Class-conditional box plots, category target-rate tables, and multivariate rank-correlation analysis",
            "Stratified train/validation/test isolation before learned preprocessing",
            "Majority, logistic-regression, single-tree, and random-forest baseline ladder",
            "OOB evaluation and convergence analysis over the number of trees",
            "Leaf-size and random-feature-subspace capacity analysis",
            "Three-fold cross-validated tuning and fold-stability reporting",
            "Tree-to-tree correlation and cumulative ensemble convergence diagnostics",
            "Impurity/permutation importance, partial dependence, ROC/PR analysis, asymmetric-cost threshold engineering, untouched-test comparison, subgroup checks, serialization, and model card",
        ],
    },
}


def enrich(name: str, cfg: dict[str, object]) -> None:
    package = ROOT / f"{name}_conda_executed" / name
    report = json.loads((package / "validation_report.json").read_text())
    bullets = "\n".join(f"- {item};" for item in cfg["coverage"][:-1])
    bullets += f"\n- {cfg['coverage'][-1]}."
    layout = f'''{name}/
├── {name}.ipynb
├── {name}.html
├── README.md
├── environment.yaml
├── requirements.txt
├── requirements-lock.txt
├── data/
│   ├── loan_default_teaching.csv
│   ├── DATA_SOURCE.md
│   └── data_dictionary.csv
├── artifacts/
│   └── {cfg['artifact']}
├── scripts/
│   ├── generate_teaching_data.py
│   ├── reexecute_notebook.py
│   ├── render_html.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
├── model_card.json
└── validation_report.json'''
    readme = f'''# {cfg['title']}

A fully executed, classroom-grade end-to-end classification package aligned with the repository's Linear Regression and Logistic Regression masterclass architecture.

## Start here

- `{name}.ipynb` — canonical executed notebook with tables, metrics, diagnostics, and embedded PNG figures.
- `{name}.html` — self-contained static rendering for GitHub, VS Code, and environments with inconsistent notebook rendering.

## Create the exact Conda environment

```bash
conda env create -f environment.yaml
conda activate {cfg['env']}
python -m ipykernel install --user --name {cfg['env']} --display-name "Python ({cfg['title']})"
```

Use `conda env create -f environment.yaml`; do not use `conda create --file environment.yaml`.

## Re-execute, render, and validate

```bash
python scripts/generate_teaching_data.py
python scripts/reexecute_notebook.py
python scripts/render_html.py
python scripts/verify_package.py
```

The committed notebook and HTML are already executed. Regeneration is deterministic and requires no network access.

## Package layout

```text
{layout}
```

## Analytical coverage

{bullets}

## Reproducibility gate

- all {report['executed_code_cells']}/{report['code_cells']} code cells executed;
- zero error outputs;
- {report['embedded_png_figures']} embedded PNG figures;
- {report['dataset_rows']:,} bundled deterministic rows;
- fixed random seed `4317`;
- single-process tuning and evaluation for stable classroom reproduction.

## Dataset and interpretation boundary

The loan-default dataset is synthetic and intentionally contains nonlinear thresholds, interactions, mixed data types, controlled missingness, imbalance, and asymmetric decision costs. It exists only to teach modeling mechanics. It is not evidence about customers and must never be used for lending, causal, fairness, or policy conclusions.
'''
    (package / "README.md").write_text(readme)
    (package / "requirements.txt").write_text((package / "requirements-lock.txt").read_text())
    (package / "scripts" / "generate_teaching_data.py").write_text(GENERATOR)
    (package / "instructor_notes.md").write_text(f'''# Instructor Notes — {cfg['title']}

## Teaching sequence

1. Start from the decision objective and asymmetric false-negative cost.
2. Derive the algorithm mechanics before using scikit-learn.
3. Audit every feature and establish the synthetic-data boundary.
4. Separate train, validation, and test before preprocessing or model selection.
5. Use the baseline ladder to establish incremental value.
6. Diagnose capacity and variance instead of treating tuning as a black box.
7. Distinguish ranking quality, probability quality, and threshold policy.
8. Interpret importance as model reliance, never causality.
9. Reserve the test set for one final evaluation.
10. End with serialization, limitations, and monitoring questions.

## Classroom checkpoints

- Calculate one impurity or bootstrap example by hand.
- Explain why validation and test have different roles.
- Compare ROC-AUC with average precision under imbalance.
- Show how the threshold changes when the cost ratio changes.
- Treat subgroup tables as diagnostics, not fairness certification.

## Common misconceptions

- A visible split is not a causal mechanism.
- High training accuracy is not evidence of generalization.
- Feature importance does not identify a treatment effect.
- OOB performance does not replace a final untouched test.
- A threshold of 0.5 is not automatically operationally correct.
''')
    (package / "student_exercises.md").write_text(f'''# Student Exercises — {cfg['title']}

## Foundations

1. Recompute the manual impurity or bootstrap example without code.
2. Explain every term in the split or ensemble objective.
3. Identify learned transformations and leakage risks.

## Exploratory analysis

4. Add ECDF plots for all numerical variables.
5. Add confidence intervals to categorical target-rate tables.
6. Investigate missingness as a predictor without target leakage.
7. Create a two-feature interaction heatmap and explain its limitations.

## Modeling

8. Remove `interest_rate` and quantify the performance change.
9. Compare median imputation with explicit missing indicators.
10. Replace one-hot encoding with another train-fitted method and justify it.
11. Change class weights and explain precision-recall movement.
12. Implement nested cross-validation and compare estimates.

## Decision policy

13. Recompute thresholds for false-negative costs of 2, 4, 8, and 12.
14. Add expected cost per 1,000 applications.
15. Compare threshold stability across folds.

## Interpretation and robustness

16. Compare impurity and permutation importance under correlated predictors.
17. Repeat across five random seeds and summarize variance.
18. Produce a worst-error table with feature-level explanations.
19. Write a monitoring specification for schema, drift, calibration, and subgroup behavior.
20. Rewrite the model card for another educational use while preserving prohibited uses.
''')
    (package / "references.md").write_text('''# References

## Primary sources

- Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. (1984). *Classification and Regression Trees*.
- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24, 123–140.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.

## Textbooks

- Hastie, T., Tibshirani, R., and Friedman, J. *The Elements of Statistical Learning*.
- James, G., Witten, D., Hastie, T., Tibshirani, R., and Taylor, J. *An Introduction to Statistical Learning*.

## Implementation references

- scikit-learn user guide: decision trees, ensemble methods, evaluation, inspection, preprocessing, and pipelines.
- Jupyter nbformat and nbconvert documentation.
''')


def update_root_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    if "### 4. Decision Tree Classification Masterclass" in text:
        return
    addition = '''### 4. Decision Tree Classification Masterclass

Path:

`decision_tree_classification_masterclass_conda_executed/decision_tree_classification_masterclass/`

Covers manual Gini gain, leakage-safe mixed-feature preprocessing, baselines, capacity diagnostics, cross-validated tuning, cost-complexity pruning, tree visualization, rule tracing, importance analysis, threshold engineering, untouched-test evaluation, subgroup diagnostics, serialization, and model-card practice.

[Open the Decision Tree Classification masterclass README](decision_tree_classification_masterclass_conda_executed/decision_tree_classification_masterclass/README.md)

### 5. Random Forest Classification Masterclass

Path:

`random_forest_classification_masterclass_conda_executed/random_forest_classification_masterclass/`

Covers bootstrap aggregation, random feature subspaces, OOB evaluation, convergence and diversity diagnostics, baseline comparisons, cross-validated tuning, importance and partial dependence, probability and threshold engineering, untouched-test comparison, subgroup diagnostics, serialization, and production limitations.

[Open the Random Forest Classification masterclass README](random_forest_classification_masterclass_conda_executed/random_forest_classification_masterclass/README.md)

'''
    path.write_text(text.replace("## Reproducibility", addition + "## Reproducibility"))


for package_name, configuration in CONFIG.items():
    enrich(package_name, configuration)
update_root_readme()
print("Enriched both packages and root README")
