# Diagnostics and evaluation

For classification report confusion matrix, precision, recall, F1, balanced accuracy, ROC-AUC where meaningful, PR-AUC for rare positives, and probability quality through log loss or Brier score. Threshold selection must use validation data, not the final test set.

Diagnostics:
- Plot validation score versus number of trees to confirm convergence.
- Compare train and validation performance to detect excessive depth or leakage.
- Examine out-of-bag score as a low-cost internal estimate, while retaining external validation for grouped or temporal data.
- Inspect permutation importance and partial dependence cautiously; correlated predictors can share or mask importance.
- Check calibration curves because averaged tree probabilities may be poorly calibrated.
- Analyze errors by subgroup, class, time, geography, and data-quality segment.
- Monitor prediction drift, feature drift, missingness, latency, and forest size in production.

Failure modes include correlated trees, noisy high-cardinality features, extrapolation failure, biased impurity importance, unstable minority-class recall, leakage hidden by random splits, and oversized forests with little marginal benefit.
