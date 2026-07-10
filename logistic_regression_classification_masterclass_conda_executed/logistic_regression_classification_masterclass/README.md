# Logistic Regression Classification Masterclass

This is the complete binary-classification companion to the Abalone linear-regression teaching project. It is designed for students to read cell by cell in VS Code or Jupyter and for instructors to open immediately with outputs already embedded.

## Start here

- `logistic_regression_classification_masterclass.ipynb` — executed instructor notebook with 66 cells, tables, metrics, and embedded plots.
- `logistic_regression_classification_masterclass_clean.ipynb` — student/re-execution copy with code preserved and outputs cleared.
- `logistic_regression_classification_masterclass.html` — self-contained static companion with code, tables, and embedded PNG plots.

The executed notebook is committed directly to GitHub. It does not require a ZIP extraction or a notebook-generation step before opening.

## Create the Conda environment

Use `conda env create`, not `conda create --file`:

```bash
conda env create -f environment.yaml
conda activate logistic-regression-masterclass
python -m ipykernel install --user \
  --name logistic-regression-masterclass \
  --display-name "Python (Logistic Regression Masterclass)"
```

Open this directory in VS Code, select the `Python (Logistic Regression Masterclass)` kernel, and open the executed notebook. The data generator is local and deterministic; no network download is needed.

## Re-execute and validate

```bash
python scripts/reexecute_notebook.py
python scripts/render_html.py
python scripts/verify_package.py
```

`reexecute_notebook.py` starts from the clean notebook and writes a fresh executed notebook. `render_html.py` creates the static companion. `verify_package.py` checks notebook validity, execution counts, output/error state, required package files, and rich-output coverage.

## Package layout

```text
logistic_regression_classification_masterclass/
├── logistic_regression_classification_masterclass.ipynb
├── logistic_regression_classification_masterclass_clean.ipynb
├── logistic_regression_classification_masterclass.html
├── environment.yaml
├── requirements.txt
├── requirements-lock.txt
├── data/
│   ├── DATA_SOURCE.md
│   └── data_dictionary.csv
├── scripts/
│   ├── generate_teaching_data.py
│   ├── reexecute_notebook.py
│   ├── render_html.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
└── validation_report.json
```

The CSV is generated on first execution by `scripts/generate_teaching_data.py`. It is intentionally not represented as the original UCI observations; see `data/DATA_SOURCE.md`.

## Analytical coverage

- probability, odds, log-odds, sigmoid, likelihood, and binary cross-entropy;
- reproducible data loading, schema audit, duplicates, missingness, ranges, and class imbalance;
- temporal leakage analysis for post-contact `duration`;
- numerical and categorical univariate EDA;
- bivariate outcome distributions, point-biserial associations, target-rate tables, and confidence intervals;
- multivariate correlation and age/balance target-rate heatmaps;
- stratified train/validation/test isolation;
- leakage-safe `Pipeline` and `ColumnTransformer` preprocessing;
- majority baseline and scikit-learn logistic regression;
- coefficient and odds-multiplier interpretation;
- ROC, precision-recall, confusion matrix, calibration, log loss, Brier score, and MCC;
- stratified cross-validation, L1/L2 regularisation, and hyperparameter search;
- class weighting and SMOTE comparison with calibration caveats;
- validation-only cost-sensitive threshold engineering;
- Statsmodels GLM inference, adjusted odds ratios, confidence intervals, and p-values;
- Pearson residuals, leverage, Cook's distance, DFBETAs, empirical-logit/LOWESS checks, Box-Tidwell-style terms, and VIF;
- untouched-test evaluation, subgroup checks, serialization, feature contract, threshold policy, and model card.

## Important interpretation boundary

This is a teaching artifact. It demonstrates a complete workflow and reliable engineering practices, but the synthetic rows are not evidence about real customers. Coefficients are associational, the threshold is a policy choice, and all results must be revalidated under real data, campaign drift, capacity limits, and governance requirements.
