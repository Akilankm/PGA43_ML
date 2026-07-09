# Student exercises

1. Replace the cost matrix with false-positive cost 2 and false-negative cost 10. Recompute the validation-selected threshold and explain the direction of change.
2. Remove `poutcome` and quantify the impact on ROC-AUC, average precision, and calibration.
3. Add an interaction between `housing` and `balance` in the Statsmodels specification. Interpret it carefully.
4. Fit a spline or binned representation of `age` and compare it with the linear-logit assumption.
5. Compare `class_weight='balanced'` with SMOTE using the same cross-validation folds.
6. Calculate bootstrap confidence intervals for test ROC-AUC and average precision.
7. Build subgroup recall tables for `job`, `education`, and `marital` status. Identify unstable small groups.
8. Replace the synthetic file with official UCI `bank-full.csv`, rerun every cell, and document which conclusions remain stable.
9. Create a capacity-constrained policy that contacts only the top 10% of customers by predicted probability.
10. Write a deployment validation function that rejects missing columns, invalid categories, and impossible numerical ranges.
