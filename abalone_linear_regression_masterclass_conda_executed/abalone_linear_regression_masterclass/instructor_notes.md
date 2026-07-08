# Instructor Notes

## Delivery recommendation

Use the executed notebook for instruction so students can see the expected output before running cells. Use the clean notebook for laboratories, assessments, and controlled reruns.

Before class:

1. create the Conda environment from `environment.yaml`;
2. install the named Jupyter kernel;
3. run `python scripts/verify_package.py`;
4. open the executed notebook in VS Code's Jupyter Notebook Editor;
5. keep the HTML companion available as a rendering fallback.

## Suggested teaching sequence

1. **Problem framing:** distinguish statistical inference, prediction, and causality.
2. **Data audit:** schema, missingness, duplicates, implausible values, and why a diagnostic flag is not an automatic deletion rule.
3. **Univariate EDA:** center, spread, skewness, kurtosis, histogram/KDE, box plot, ECDF, and target-count behavior.
4. **Bivariate EDA:** Pearson versus Spearman, marginal association, OLS line versus LOWESS, group comparisons, ANOVA, Kruskal–Wallis, and effect size.
5. **Multivariate EDA:** correlation structure, scatter matrix, PCA scores, loadings, and the distinction between dimensional structure and predictive evidence.
6. **Leakage-safe splitting:** isolate the test set before fitting imputers, encoders, scalers, power transforms, or feature-selection decisions.
7. **Statsmodels OLS:** specification, reference coding, coefficients, standard errors, confidence intervals, p-values, adjusted R², and out-of-sample prediction.
8. **Assumptions:** separate visual evidence, formal tests, practical consequences, and remedies.
9. **Robustness:** HC3 standard errors, influence sensitivity, reduced specifications, and log-target retransformation.
10. **Scikit-learn engineering:** `Pipeline`, `ColumnTransformer`, one-hot encoding, scaling, Yeo–Johnson transformation, polynomial features, Ridge, and Huber regression.
11. **Validation:** cross-validation, explicit learning curves, final holdout evaluation, subgroup error, target-band error, and uncertainty.
12. **Communication:** model card, limitations, causal restraint, and deployment monitoring.

## Core conceptual checkpoints

Students should be able to explain all of the following without referring to library syntax:

- Why OLS is linear in parameters even when transformed or polynomial predictors are used.
- Why a strong marginal correlation can become weak or reverse after controlling for correlated predictors.
- Why multicollinearity primarily destabilizes coefficient interpretation rather than automatically destroying prediction.
- Why heteroscedasticity can invalidate conventional standard errors while point predictions remain usable.
- Why residual normality is not required for OLS coefficient unbiasedness, but matters for small-sample exact inference.
- Why a low p-value is not evidence of causality, practical importance, or predictive usefulness.
- Why scaling is essential for penalized models but not required for unpenalized OLS predictions.
- Why target transformation changes the estimand and requires retransformation-bias treatment.
- Why an influential observation can be valid and informative rather than erroneous.
- Why the final test set must not be used repeatedly for model selection.

## Discussion prompts

- Why can `shucked_weight` receive a negative conditional coefficient while being positively correlated with rings marginally?
- Why is `sex=I` biologically entangled with maturity and size rather than functioning as a simple demographic label?
- Which assumption failures affect coefficients, conventional standard errors, prediction intervals, or all three?
- Why does the controlled polynomial model improve holdout RMSE while some engineered or robust alternatives do not?
- Why is the empirical 95% prediction-interval coverage below 95%, and what remedies could improve calibration?
- Why is the high-ring group substantially harder to predict?
- When should the analyst prefer HC3, weighted least squares, a transformed target, a count model, or a nonlinear model?

## Important interpretation cautions

- Do not describe every point outside an IQR fence as a bad record.
- Do not use RESET, Breusch–Pagan, White, or Jarque–Bera as mechanical pass/fail gates without inspecting effect magnitude and plots.
- Do not delete Cook's-distance cases merely to improve assumptions.
- Do not interpret one-hot coefficients without stating the reference group.
- Do not compare raw coefficients measured in different units as though their magnitudes were directly comparable.
- Do not use the test set to decide feature engineering and then report the same test score as unbiased final performance.
- Do not claim age causation from observational shell measurements.

## Assessment rubric

| Dimension | Weight |
|---|---:|
| Correctness and reproducibility | 25% |
| EDA depth and graphical interpretation | 20% |
| Assumption diagnostics and remedy reasoning | 20% |
| Leakage prevention and model validation | 15% |
| Uncertainty and subgroup error analysis | 10% |
| Communication, limitations, and causal restraint | 10% |
