# Random Forest Classification Masterclass

A fully executed, classroom-grade end-to-end classification package aligned with the repository's Linear Regression and Logistic Regression masterclass architecture.

## Start here

- `random_forest_classification_masterclass.ipynb` — canonical executed notebook with tables, metrics, diagnostics, and embedded PNG figures.
- `random_forest_classification_masterclass.html` — self-contained static rendering for GitHub, VS Code, and environments with inconsistent notebook rendering.

## Create the exact Conda environment

```bash
conda env create -f environment.yaml
conda activate random-forest-masterclass
python -m ipykernel install --user --name random-forest-masterclass --display-name "Python (Random Forest Classification Masterclass)"
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
random_forest_classification_masterclass/
├── random_forest_classification_masterclass.ipynb
├── random_forest_classification_masterclass.html
├── README.md
├── environment.yaml
├── requirements.txt
├── requirements-lock.txt
├── data/
│   ├── loan_default_teaching.csv
│   ├── DATA_SOURCE.md
│   └── data_dictionary.csv
├── artifacts/
│   └── random_forest_classifier.joblib
├── scripts/
│   ├── generate_teaching_data.py
│   ├── reexecute_notebook.py
│   ├── render_html.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
├── model_card.json
└── validation_report.json
```

## Analytical coverage

- Single-tree variance, bootstrap sampling, out-of-bag observations, random feature subspaces, aggregation, and the bias-variance-correlation trade-off;
- Manual bootstrap and OOB demonstration before library modeling;
- Schema, duplicate, missingness, range, cardinality, and class-balance audits;
- Per-feature numerical and categorical univariate analysis;
- Class-conditional box plots, category target-rate tables, and multivariate rank-correlation analysis;
- Stratified train/validation/test isolation before learned preprocessing;
- Majority, logistic-regression, single-tree, and random-forest baseline ladder;
- OOB evaluation and convergence analysis over the number of trees;
- Leaf-size and random-feature-subspace capacity analysis;
- Three-fold cross-validated tuning and fold-stability reporting;
- Tree-to-tree correlation and cumulative ensemble convergence diagnostics;
- Impurity/permutation importance, partial dependence, ROC/PR analysis, asymmetric-cost threshold engineering, untouched-test comparison, subgroup checks, serialization, and model card.

## Reproducibility gate

- all 12/12 code cells executed;
- zero error outputs;
- 11 embedded PNG figures;
- 2,400 bundled deterministic rows;
- fixed random seed `4317`;
- single-process tuning and evaluation for stable classroom reproduction.

## Dataset and interpretation boundary

The loan-default dataset is synthetic and intentionally contains nonlinear thresholds, interactions, mixed data types, controlled missingness, imbalance, and asymmetric decision costs. It exists only to teach modeling mechanics. It is not evidence about customers and must never be used for lending, causal, fairness, or policy conclusions.
