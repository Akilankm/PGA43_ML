# Linear and Multiple Regression Manual

This manual supports the notebooks in this folder. It explains the concepts, formulas, implementation choices, and interpretation discipline required for end-to-end regression analysis.

---

## 1. What is regression?

Regression is a supervised learning method used when the target variable is numeric.

Examples:

| Problem | Target |
|---|---|
| Predict exam score | Numeric score |
| Predict monthly sales | Units sold |
| Predict house price | Price |
| Predict delivery time | Minutes |

Regression estimates the relationship between input variables and a numeric target.

---

## 2. Simple linear regression

Simple linear regression uses one input variable.

```text
y = beta_0 + beta_1 * x + error
```

| Symbol | Meaning |
|---|---|
| y | target / dependent variable |
| x | input / independent variable |
| beta_0 | intercept |
| beta_1 | slope |
| error | residual / unexplained component |

### Interpretation

If the fitted equation is:

```text
exam_score = 37.5 + 5.7 * study_hours
```

Then:

- `37.5` is the predicted score when study hours are zero.
- `5.7` means one additional study hour is associated with an average increase of 5.7 score points.

The wording **associated with** is important. Regression alone does not prove causality.

---

## 3. Ordinary Least Squares

OLS finds the line that minimizes the sum of squared residuals.

Residual:

```text
residual_i = actual_i - predicted_i
```

OLS objective:

```text
minimize sum((actual_i - predicted_i)^2)
```

Why squared errors?

1. Negative and positive errors do not cancel.
2. Larger errors are penalized more.
3. It gives a mathematically convenient closed-form solution.

---

## 4. Simple regression coefficient formulas

Slope:

```text
beta_1 = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
```

Intercept:

```text
beta_0 = mean_y - beta_1 * mean_x
```

This is implemented manually in the simple regression notebook before using scikit-learn.

---

## 5. Multiple linear regression

Multiple regression uses more than one feature.

```text
y = beta_0 + beta_1*x_1 + beta_2*x_2 + ... + beta_p*x_p + error
```

Example:

```text
sales = beta_0 + beta_1*tv + beta_2*radio + beta_3*social + beta_4*price + error
```

### Coefficient interpretation

A coefficient is interpreted as:

> Expected change in the target for a one-unit increase in that feature, holding other variables constant.

This **holding other variables constant** condition is critical.

---

## 6. Matrix form of OLS

The compact matrix form:

```text
beta_hat = inverse(X_transpose * X) * X_transpose * y
```

Where:

| Symbol | Meaning |
|---|---|
| X | design matrix |
| y | target vector |
| beta_hat | estimated coefficient vector |

In implementation, use `np.linalg.pinv` for numerical stability instead of directly calculating an inverse.

---

## 7. Train-test split

A regression model must be evaluated on unseen data.

Typical workflow:

1. Split data into train and test sets.
2. Fit the model only on train data.
3. Predict on test data.
4. Calculate metrics using test predictions.

This gives a better estimate of generalization performance.

---

## 8. Regression metrics

### MAE

```text
MAE = mean(abs(actual - predicted))
```

Mean absolute error is easy to explain because it is in the same unit as the target.

### RMSE

```text
RMSE = sqrt(mean((actual - predicted)^2))
```

RMSE penalizes large errors more strongly than MAE.

### R2

```text
R2 = 1 - SSE / SST
```

R2 measures the proportion of target variation explained by the model.

Do not use only R2. A high R2 can still hide poor assumptions, leakage, overfitting, or unstable coefficients.

---

## 9. scikit-learn vs statsmodels

| Library | Best for |
|---|---|
| scikit-learn | Predictive modelling, pipelines, cross-validation, production-style workflows |
| statsmodels | Statistical inference, p-values, confidence intervals, detailed OLS summaries |

Use both in teaching:

- Use scikit-learn for ML workflow.
- Use statsmodels for regression interpretation.

---

## 10. Categorical variables

Linear regression needs numeric inputs. Categorical variables must be encoded.

Common approach: one-hot encoding.

Example:

```text
season = Q1, Q2, Q3, Q4
```

After one-hot encoding with one dropped baseline:

```text
season_Q2
season_Q3
season_Q4
```

The dropped category becomes the baseline.

---

## 11. Regression assumptions

Regression is not only about fitting a line. The interpretation depends on assumptions.

| Assumption | Meaning |
|---|---|
| Linearity | Relationship between features and target is approximately linear |
| Independent errors | Residuals should not be systematically related |
| Homoscedasticity | Residual variance should be roughly constant |
| Normal residuals | Useful for valid inference and confidence intervals |
| No serious multicollinearity | Features should not be extremely redundant |
| No major outliers/leverage points | Model should not be dominated by a few points |

---

## 12. Residual analysis

Residuals are the difference between actual and predicted values.

```text
residual = actual - predicted
```

Good residual behavior:

- centered around zero
- no strong pattern
- roughly constant variance
- no extreme outliers

Bad residual behavior:

- funnel shape
- curve pattern
- clusters
- systematic positive/negative bias

---

## 13. Heteroscedasticity

Heteroscedasticity means residual variance is not constant.

Why it matters:

- Coefficients may still be unbiased.
- Standard errors and p-values may become unreliable.
- Prediction intervals may be misleading.

The diagnostic notebook uses the Breusch-Pagan test.

---

## 14. Normality of residuals

Normal residuals are especially important for statistical inference.

The diagnostic notebook uses the Jarque-Bera test.

In real data, mild non-normality is often acceptable for prediction, especially with enough data. For inference-heavy use cases, inspect this carefully.

---

## 15. Multicollinearity and VIF

Multicollinearity means input variables are highly correlated with each other.

Problems caused:

- coefficients become unstable
- p-values become unreliable
- signs may become unintuitive
- small data changes can change interpretation

Variance Inflation Factor:

| VIF | Interpretation |
|---:|---|
| ~1 | no issue |
| 1-5 | usually acceptable |
| >5 | inspect carefully |
| >10 | serious concern |

---

## 16. Ridge regression

Ridge regression adds L2 regularization:

```text
Loss = RSS + alpha * sum(beta_j^2)
```

Effect:

- shrinks coefficients
- reduces variance
- handles multicollinearity better
- usually keeps all variables

Use Ridge when many predictors are correlated.

---

## 17. Lasso regression

Lasso regression adds L1 regularization:

```text
Loss = RSS + alpha * sum(abs(beta_j))
```

Effect:

- shrinks coefficients
- can set some coefficients exactly to zero
- performs feature selection

Use Lasso when sparse feature selection is useful.

---

## 18. End-to-end regression workflow

Use this sequence:

1. Define the question.
2. Load data.
3. Inspect target and features.
4. Handle missing values.
5. Encode categorical variables.
6. Split train/test.
7. Fit baseline linear regression.
8. Evaluate MAE, RMSE, R2.
9. Interpret coefficients.
10. Check residuals.
11. Check VIF.
12. Compare Ridge/Lasso if needed.
13. Validate with cross-validation.
14. Document final model and limitations.

---

## 19. Common student mistakes

| Mistake | Correction |
|---|---|
| Saying regression proves causality | Say association unless experiment/causal design exists |
| Using only R2 | Always include MAE/RMSE and diagnostics |
| Ignoring categorical encoding | Use one-hot encoding |
| Interpreting coefficients without holding other variables constant | Use conditional interpretation |
| Skipping train-test split | Always validate on unseen data |
| Ignoring multicollinearity | Calculate VIF |
| Trusting p-values blindly | Check assumptions and data quality |
| Comparing unscaled Ridge/Lasso coefficients | Scale numeric features before regularization |

---

## 20. Classroom exercises

### Exercise 1

In the simple regression notebook:

- replace `study_hours` with `attendance_pct`
- compare MAE, RMSE, and R2
- decide which single feature is better

### Exercise 2

In the multiple regression notebook:

- remove `competitor_spend_k`
- retrain the model
- compare coefficient changes and test metrics

### Exercise 3

In the diagnostics notebook:

- increase Ridge `alpha`
- observe what happens to coefficients and CV metrics

### Exercise 4

Use Lasso with different alpha values and identify which variables are removed.

### Exercise 5

Ask students to write a final modelling conclusion:

1. Which model is selected?
2. What are the strongest drivers?
3. What are the diagnostics saying?
4. What are the limitations?

---

## 21. Final model selection principle

A good regression model is not just the model with the highest R2.

A good model should be:

- predictive
- stable
- interpretable
- diagnostically acceptable
- aligned with domain logic
- honest about uncertainty
