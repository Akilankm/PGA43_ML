# Abalone Linear Regression Masterclass

A fully executed, classroom-grade notebook for teaching linear regression end to end with both **Statsmodels** and **scikit-learn**.

## Start here

The package contains two notebook variants:

- `abalone_linear_regression_masterclass.ipynb` — **executed notebook with all tables and plots embedded**. Open this first when teaching or reviewing.
- `abalone_linear_regression_masterclass_clean.ipynb` — identical source notebook with outputs cleared, intended for a controlled student rerun.
- `abalone_linear_regression_masterclass.html` — static rendered companion for environments where Markdown or MathJax rendering in VS Code is inconsistent.

## Create the exact Conda environment

Run these commands from the extracted project directory:

```bash
conda env create -f environment.yaml
conda activate abalone-linear-regression
python -m ipykernel install --user \
  --name abalone-linear-regression \
  --display-name "Python (Abalone Linear Regression)"
```

Open the folder in VS Code:

```bash
code .
```

Then open the notebook and choose:

**Select Kernel → Python (Abalone Linear Regression)**

The Conda file pins the same Python and package versions used to generate the embedded outputs. Thread counts and random seeds are constrained to reduce platform-dependent numerical drift.

## Recommended VS Code setup

Install the official **Python** and **Jupyter** extensions. Open the `.ipynb` file with the **Jupyter Notebook Editor**, not as raw JSON or a plain text document. Trust the workspace and notebook when prompted.

The notebook uses standard Markdown, fenced code blocks, `$...$` inline mathematics, and `$$...$$` display mathematics. Raw HTML was avoided. Figures are embedded as PNG outputs for reliable display in VS Code, JupyterLab, and classic Notebook.

When VS Code still fails to display a Markdown cell correctly, use the included HTML file as the exact rendered reference and update the VS Code Jupyter extension before changing notebook content.

## Re-execute and validate

The distributed teaching notebook is already executed. To regenerate the outputs from the clean notebook:

```bash
python scripts/reexecute_notebook.py
```

This creates:

```text
abalone_linear_regression_masterclass_rerun.ipynb
```

The script fails when:

- the runtime package versions differ from `environment.yaml`;
- a code cell raises an exception;
- any code cell is skipped;
- fewer than 30 graph outputs are embedded.

Validate the supplied package without changing it:

```bash
python scripts/verify_package.py
```

## Package contents

```text
abalone_linear_regression_masterclass/
├── abalone_linear_regression_masterclass.ipynb
├── abalone_linear_regression_masterclass_clean.ipynb
├── abalone_linear_regression_masterclass.html
├── environment.yaml
├── requirements-lock.txt
├── data/
│   └── abalone.csv
├── scripts/
│   ├── reexecute_notebook.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
└── validation_report.json
```

## Analytical coverage

The notebook includes:

- data loading, schema validation, duplicate and domain-quality audits;
- per-feature univariate analysis with descriptive statistics, histogram/KDE, box plot, and ECDF;
- bivariate Pearson/Spearman analysis, OLS and LOWESS trends, categorical-group inference, ANOVA, Kruskal–Wallis, and effect size;
- multivariate correlation analysis, scatter matrix, PCA scores, explained variance, and loadings;
- train/test isolation before learned transformations;
- categorical encoding, imputation, scaling, Yeo–Johnson transformation, domain feature engineering, and controlled polynomial expansion;
- Statsmodels OLS estimation, coefficient inference, confidence intervals, test prediction, and prediction intervals;
- RESET, Durbin–Watson, Breusch–Pagan, White, Jarque–Bera, omnibus normality, VIF, leverage, externally studentized residuals, and Cook's distance;
- HC3 robust inference, influence sensitivity analysis, reduced-collinearity specification, and log-target retransformation with Duan smearing;
- scikit-learn `Pipeline` and `ColumnTransformer` workflows;
- raw linear regression, engineered regression, polynomial regression, RidgeCV, and Huber regression;
- deterministic five-fold cross-validation, an explicit fold-by-fold learning curve, final test comparison, subgroup analysis, and target-band error analysis;
- reusable validated prediction function, model card, remedy framework, limitations, and student exercises.

## Reproducibility guarantees and limits

The data split, random sampling, PCA, cross-validation, and learning-curve subsets use fixed seeds. The exact runtime versions are printed and checked in the first executable cell.

The supplied executed notebook was validated with:

- all code cells executed;
- no error outputs;
- 35 embedded PNG graph outputs;
- the bundled 4,177-row dataset;
- package versions matching `environment.yaml` exactly.

Floating-point results can still differ at extremely small precision across operating systems or BLAS implementations. Interpret “same results” as identical model conclusions and metrics within normal floating-point tolerance, not byte-identical binary output across every machine.

## Dataset attribution

Nash, W., Sellers, T., Talbot, S., Cawthorn, A., and Ford, W. (1994). **Abalone**. UCI Machine Learning Repository. DOI: `10.24432/C55C7W`.

The bundled CSV is included for offline classroom execution.
