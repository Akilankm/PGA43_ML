# Student Exercises — Decision Tree Classification Masterclass

## Foundations

1. Recompute the manual impurity or bootstrap example without code.
2. Explain every term in the split or ensemble objective.
3. Identify learned transformations and leakage risks.

## Exploratory analysis

4. Add ECDF plots for all numerical variables.
5. Add confidence intervals to categorical target-rate tables.
6. Investigate missingness as a predictor without target leakage.
7. Create a two-feature interaction heatmap and explain its limitations.

## Modeling

8. Remove `interest_rate` and quantify the performance change.
9. Compare median imputation with explicit missing indicators.
10. Replace one-hot encoding with another train-fitted method and justify it.
11. Change class weights and explain precision-recall movement.
12. Implement nested cross-validation and compare estimates.

## Decision policy

13. Recompute thresholds for false-negative costs of 2, 4, 8, and 12.
14. Add expected cost per 1,000 applications.
15. Compare threshold stability across folds.

## Interpretation and robustness

16. Compare impurity and permutation importance under correlated predictors.
17. Repeat across five random seeds and summarize variance.
18. Produce a worst-error table with feature-level explanations.
19. Write a monitoring specification for schema, drift, calibration, and subgroup behavior.
20. Rewrite the model card for another educational use while preserving prohibited uses.
