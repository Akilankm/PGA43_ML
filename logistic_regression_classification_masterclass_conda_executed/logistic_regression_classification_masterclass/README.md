# Logistic Regression Classification Masterclass

A ready-to-open, classroom-grade logistic-regression package structured to mirror the Abalone linear-regression project.

## Start here

- `logistic_regression_classification_masterclass.ipynb` — instructor notebook with execution counts and validated result checkpoints.
- `logistic_regression_classification_masterclass_clean.ipynb` — student notebook with execution counts and outputs cleared.
- `logistic_regression_classification_masterclass.html` — static companion and package navigation page.

The notebooks are committed directly to GitHub. No ZIP extraction or notebook-generation step is required before opening them.

## Create the Conda environment

```bash
conda env create -f environment.yaml
conda activate logistic-regression-masterclass
python -m ipykernel install --user \
  --name logistic-regression-masterclass \
  --display-name "Python (Logistic Regression Masterclass)"
```

Open the project folder in VS Code, select **Python (Logistic Regression Masterclass)**, and open either notebook.

## Data behavior

The notebook looks for `data/bank_marketing_teaching.csv`. On the first run, if the CSV is absent, it executes `scripts/generate_teaching_data.py` and creates the deterministic 5,024-row teaching dataset locally. This keeps the repository lightweight while remaining fully offline and reproducible.

## Re-execute and validate

```bash
python scripts/reexecute_notebook.py
python scripts/verify_package.py
```

## Package layout

```text
logistic_regression_classification_masterclass/
├── logistic_regression_classification_masterclass.ipynb
├── logistic_regression_classification_masterclass_clean.ipynb
├── logistic_regression_classification_masterclass.html
├── environment.yaml
├── requirements-lock.txt
├── requirements.txt
├── data/
│   ├── DATA_SOURCE.md
│   └── data_dictionary.csv
├── scripts/
│   ├── generate_teaching_data.py
│   ├── reexecute_notebook.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
└── validation_report.json
```

## Analytical coverage

- probability, odds, log-odds, sigmoid and Bernoulli likelihood;
- data-quality auditing and explicit post-contact leakage analysis;
- numerical and categorical EDA;
- point-biserial association and target-rate analysis;
- stratified train/validation/test isolation;
- `Pipeline` and `ColumnTransformer` preprocessing;
- dummy baseline and logistic regression;
- ROC, precision-recall, confusion matrix and calibration;
- L1/L2 regularization and stratified cross-validation;
- class weighting and SMOTE;
- validation-only cost-sensitive threshold selection;
- Statsmodels GLM inference and adjusted odds ratios;
- residual, leverage and influence diagnostics;
- untouched-test evaluation, serialization and model-card guidance.

## Dataset statement

The generated dataset is a deterministic UCI Bank Marketing schema-compatible teaching dataset. It is synthetic and is not represented as the original UCI observations. See `data/DATA_SOURCE.md` for provenance and the official-data workflow.
