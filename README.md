# PGA43_ML

Curated, reproducible statistical machine-learning teaching resources.

This repository contains only validated, classroom-ready masterclass packages. Each package includes one canonical end-to-end notebook with embedded outputs, a rendered HTML companion, reproducible environment instructions, supporting documentation, and validation metadata.

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

### 3. Decision Tree from Scratch — Play Tennis Masterclass

Path:

`decision tree from scratch/`

Covers manual entropy, information gain, Gini impurity, every root and recursive split calculation, categorical and numerical splitting concepts, a from-scratch ID3 implementation, rendered tree visualization, prediction tracing, stopping criteria, pruning, and critical generalization failure modes.

[Open the Decision Tree from Scratch README](decision%20tree%20from%20scratch/README.md)

### 4. Decision Tree Classification Masterclass

Path:

`decision_tree_classification_masterclass_conda_executed/decision_tree_classification_masterclass/`

Covers manual Gini gain, leakage-safe mixed-feature preprocessing, baselines, capacity diagnostics, cross-validated tuning, cost-complexity pruning, tree visualization, rule tracing, importance analysis, threshold engineering, untouched-test evaluation, subgroup diagnostics, serialization, and model-card practice.

[Open the Decision Tree Classification masterclass README](decision_tree_classification_masterclass_conda_executed/decision_tree_classification_masterclass/README.md)

### 5. Random Forest Classification Masterclass

Path:

`random_forest_classification_masterclass_conda_executed/random_forest_classification_masterclass/`

Covers bootstrap aggregation, random feature subspaces, OOB evaluation, convergence and diversity diagnostics, baseline comparisons, cross-validated tuning, importance and partial dependence, probability and threshold engineering, untouched-test comparison, subgroup diagnostics, serialization, and production limitations.

[Open the Random Forest Classification masterclass README](random_forest_classification_masterclass_conda_executed/random_forest_classification_masterclass/README.md)

### 6. Gradient Boosting Masterclass

Path:

`gradient_boosting_masterclass_conda_executed/gradient_boosting_masterclass/`

Covers additive stage-wise modelling, functional gradient descent, squared-error and binary log-loss derivations, manual boosting rounds, from-scratch regression and classification implementations, scikit-learn Gradient Boosting models, regularisation, early stopping, cross-validation, residual and calibration diagnostics, threshold-cost engineering, permutation importance, partial dependence, failure modes, and production monitoring.

[Open the Gradient Boosting masterclass README](gradient_boosting_masterclass_conda_executed/gradient_boosting_masterclass/README.md)

## Reproducibility

Each masterclass provides its own `environment.yaml`. Create an environment with:

```bash
conda env create -f environment.yaml
```

Then run the package's validation command or open the executed notebook in VS Code or Jupyter using the environment's registered kernel. The committed notebooks already contain their end-to-end outputs; the HTML files are the static rendered references.

## Curation policy

Legacy theory dumps, duplicated notebooks, outputless showcase notebooks, generated build-only files, and unvalidated examples are intentionally excluded. The repository prioritises a small number of coherent, runnable, review-ready learning paths.
