# Student exercises

1. Re-run the notebook with false-negative cost 10 and false-positive cost 2. Explain why the selected threshold moves and which operating metrics change.
2. Fit a deliberately leaky model that includes `duration`. Compare validation ranking and test metrics, then explain why the result is unusable for pre-call targeting.
3. Remove `poutcome` and compare ROC-AUC, average precision, calibration, and subgroup recall.
4. Replace the linear `age` effect with bins, a spline, or a polynomial. Compare validation log loss and inspect the empirical-logit plot.
5. Add an interaction between `housing` and `balance_1000` to the Statsmodels formula. Interpret the interaction in context.
6. Compare L1 and L2 regularisation by counting coefficients near zero and discussing the interpretability trade-off.
7. Replace the one-hot SMOTE demonstration with a mixed-type `SMOTENC` design. Document the preprocessing order and limitations.
8. Calculate bootstrap confidence intervals for test ROC-AUC and average precision without repeatedly tuning the threshold.
9. Build subgroup tables for `job`, `education`, and `marital`, then flag groups with fewer than 50 test observations.
10. Create a capacity-constrained policy that contacts only the top 10% of customers by predicted probability.
11. Write a prediction validation function that rejects missing columns, impossible ages, invalid categories, and an accidentally supplied `duration` column.
12. Replace the synthetic CSV with an approved official dataset, document the provenance, and compare which conclusions remain stable.
