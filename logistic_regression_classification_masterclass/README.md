# Logistic Regression Classification Masterclass

A classroom-grade binary-classification codebase covering logistic regression with both **Statsmodels** and **scikit-learn**.

## Materialize the complete package

The repository contains the complete deterministic source and a one-command materializer. After creating the environment, run:

```bash
python scripts/materialize_package.py
```

This command generates and validates all expanded deliverables directly inside this folder:

- `logistic_regression_classification_masterclass.ipynb` — fully executed notebook with embedded tables and plots;
- `logistic_regression_classification_masterclass_clean.ipynb` — clean student notebook;
- `logistic_regression_classification_masterclass.html` — static rendered notebook;
- `data/bank_marketing_teaching.csv` — deterministic offline teaching dataset;
- `artifacts/final_logistic_pipeline.joblib` — fitted end-to-end prediction pipeline;
- `artifacts/model_metrics.json` and `artifacts/selected_threshold.json`;
- `validation_report.json` and `CHECKSUMS.sha256`.

No ZIP archive is stored in the repository.

## Environment setup

```bash
conda env create -f environment.yaml
conda activate logistic-regression-masterclass
python -m ipykernel install --user \
  --name logistic-regression-masterclass \
  --display-name "Python (Logistic Regression Masterclass)"
```

Then open the folder in VS Code or JupyterLab and select the registered kernel.

## Re-execute and validate

```bash
python scripts/reexecute_notebook.py
python scripts/verify_package.py
```

## Package structure

```text
logistic_regression_classification_masterclass/
├── README.md
├── environment.yaml
├── requirements-lock.txt
├── data/
│   ├── data_dictionary.csv
│   └── DATA_SOURCE.md
├── scripts/
│   ├── materialize_package.py
│   ├── build_notebook.py
│   ├── generate_teaching_data.py
│   ├── download_official_data.py
│   ├── reexecute_notebook.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
└── generated after materialization:
    ├── logistic_regression_classification_masterclass.ipynb
    ├── logistic_regression_classification_masterclass_clean.ipynb
    ├── logistic_regression_classification_masterclass.html
    ├── CHECKSUMS.sha256
    ├── validation_report.json
    ├── data/bank_marketing_teaching.csv
    └── artifacts/
        ├── final_logistic_pipeline.joblib
        ├── selected_threshold.json
        └── model_metrics.json
```

## Analytical coverage

The notebook includes:

- probability, odds, log-odds, sigmoid, Bernoulli likelihood, cross-entropy, and coefficient interpretation;
- schema validation, duplicates, target imbalance, data-quality checks, and leakage analysis;
- numerical and categorical EDA, point-biserial correlation, chi-square tests, and Cramér's V;
- stratified train/validation/test isolation before learned transformations;
- `Pipeline`, `ColumnTransformer`, imputation, scaling, one-hot encoding, and unknown-category handling;
- dummy baseline, operational logistic regression, deliberate leakage comparison, L1/L2 regularization, class weighting, and SMOTE inside the modeling pipeline;
- ROC-AUC, average precision, confusion matrices, precision, recall, specificity, F1, MCC, log-loss, Brier score, and calibration curves;
- validation-only business-threshold optimization with explicit false-positive and false-negative costs;
- Statsmodels GLM inference, adjusted odds ratios, confidence intervals, AIC, deviance, and McFadden pseudo-R²;
- VIF, Pearson residuals, leverage, standardized residuals, Cook's distance, and binned calibration diagnostics;
- coefficient interpretation, subgroup error analysis, model-card limitations, serialized model artifacts, and a validated prediction function.

## Dataset statement

The generated `bank_marketing_teaching.csv` is a deterministic synthetic teaching dataset that follows the UCI Bank Marketing schema and preserves realistic mixed feature types, imbalance, nonlinear probability behavior, and a deliberate post-contact leakage variable. It is included for offline reproducibility and is **not claimed to contain the original UCI records**.

To use the official UCI data:

```bash
python scripts/download_official_data.py
```

Then update `DATA_PATH` in the notebook to `data/bank-full.csv` and rerun. The official dataset is described by Moro, Cortez, and Rita and distributed through the UCI Machine Learning Repository.

## Validated result snapshot

On the deterministic teaching data, the validated local execution produced approximately:

- ROC-AUC: `0.785`
- Average precision: `0.344`
- Brier score: `0.063`
- selected validation business threshold: `0.229`

These are teaching-run results, not claims about production performance.
