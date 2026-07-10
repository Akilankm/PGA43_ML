# PGA43_ML

Curated, reproducible statistical machine-learning teaching resources.

This repository contains only validated, classroom-ready masterclass packages. Each package includes an executed notebook with embedded outputs, a clean student notebook, reproducible environment instructions, supporting documentation, and validation metadata.

## Curated learning paths

### 1. Abalone Linear Regression Masterclass

Path:

`abalone_linear_regression_masterclass_conda_executed/abalone_linear_regression_masterclass/`

Covers exploratory analysis, OLS with Statsmodels, scikit-learn regression, assumptions, diagnostics, influence, multicollinearity, regularisation, interpretation, and reproducible execution.

[Open the Abalone masterclass README](abalone_linear_regression_masterclass_conda_executed/abalone_linear_regression_masterclass/README.md)

### 2. Logistic Regression Classification Masterclass

Path:

`logistic_regression_classification_masterclass_conda_executed/logistic_regression_classification_masterclass/`

Covers probability and log-odds, leakage-aware EDA, preprocessing pipelines, Statsmodels GLM, scikit-learn logistic regression, imbalance handling, calibration, threshold engineering, diagnostics, test evaluation, and model-card practice.

[Open the Logistic Regression masterclass README](logistic_regression_classification_masterclass_conda_executed/logistic_regression_classification_masterclass/README.md)

## Reproducibility

Each masterclass provides its own `environment.yaml`. Create an environment with:

```bash
conda env create -f environment.yaml
```

Then open the executed notebook in VS Code or Jupyter using the environment's registered kernel.

## Curation policy

Legacy theory dumps, duplicated notebooks, generated build-only files, and unvalidated examples are intentionally excluded. The repository prioritises a small number of coherent, runnable, review-ready learning paths.
