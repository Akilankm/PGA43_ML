# Linear and Multiple Regression End-to-End Notebooks

This folder is a classroom-ready resource pack for teaching **Simple Linear Regression** and **Multiple Linear Regression** using Python.

It contains executed notebooks with results, synthetic datasets, and a manual that explains the full modelling workflow from first principles to library implementation.

## Folder structure

```text
Linear-Multiple-Regression-End-to-End-Notebooks/
├── README.md
├── requirements.txt
├── data/
│   ├── simple_linear_student_scores.csv
│   └── multiple_regression_marketing_sales.csv
├── manual/
│   └── LINEAR_MULTIPLE_REGRESSION_MANUAL.md
└── notebooks/
    ├── 00_environment_setup_and_dataset_overview.ipynb
    ├── 01_simple_linear_regression_end_to_end.ipynb
    ├── 02_multiple_linear_regression_end_to_end.ipynb
    └── 03_diagnostics_regularization_and_interpretability.ipynb
```

## Learning path

| Order | Notebook | Purpose |
|---:|---|---|
| 0 | `00_environment_setup_and_dataset_overview.ipynb` | Verify libraries, load datasets, understand the resource pack |
| 1 | `01_simple_linear_regression_end_to_end.ipynb` | Simple regression from formula, manual implementation, scikit-learn, statsmodels, metrics |
| 2 | `02_multiple_linear_regression_end_to_end.ipynb` | Multiple regression with numeric and categorical features, one-hot encoding, pipelines, OLS summary |
| 3 | `03_diagnostics_regularization_and_interpretability.ipynb` | Residual checks, Breusch-Pagan, Jarque-Bera, VIF, Ridge, Lasso, cross-validation |

## Concepts covered

- Regression objective
- Dependent and independent variables
- Intercept and slope
- Ordinary Least Squares
- Manual formula for simple regression
- Matrix / normal equation implementation
- Train-test split
- MAE, RMSE, R²
- Cross-validation
- Multiple regression with categorical variables
- One-hot encoding
- scikit-learn pipelines
- statsmodels OLS summary
- p-values and confidence intervals
- Residual diagnostics
- Homoscedasticity / heteroscedasticity
- Normality of residuals
- Multicollinearity and VIF
- Ridge and Lasso regression
- Coefficient interpretation
- Model comparison and final decision framework

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook notebooks/
```

## Teaching recommendation

Use the existing PDFs in `Simple-Multiple-Regression-in-Python/` for theory-first explanation, then use this folder for hands-on execution and analysis.

Suggested class flow:

1. Explain the formula and intuition.
2. Run simple regression manually and with scikit-learn.
3. Explain residuals and metrics.
4. Move to multiple regression.
5. Compare scikit-learn vs statsmodels.
6. Teach assumptions and diagnostics.
7. Introduce Ridge/Lasso as practical extensions.
8. Ask students to modify features and compare metrics.

## Datasets

Both datasets are synthetic and deterministic. They are safe for classroom use and do not contain personal/private data.

| Dataset | Rows | Target | Use case |
|---|---:|---|---|
| `simple_linear_student_scores.csv` | 60 | `exam_score` | Simple regression |
| `multiple_regression_marketing_sales.csv` | 120 | `sales_k_units` | Multiple regression + diagnostics |

## Expected student outcomes

After completing this pack, students should be able to:

- Build regression models end to end.
- Explain slope, intercept, residuals, and model metrics.
- Use both scikit-learn and statsmodels appropriately.
- Interpret coefficients without overclaiming causality.
- Diagnose common regression problems.
- Compare OLS, Ridge, and Lasso models.
