# Hyperparameter tuning, comparisons, and production

Important parameters:
- `n_estimators`: increase until validation or OOB performance stabilizes.
- `max_features`: lower values decorrelate trees but may raise bias.
- `max_depth`, `min_samples_split`, `min_samples_leaf`: regularize individual trees.
- `bootstrap`: enables classical bagging and OOB estimates.
- `class_weight`: changes split incentives under imbalance.
- `max_samples`: trades tree diversity and training cost.
- `criterion`: Gini or entropy/log-loss variants.

Use randomized search or successive halving before a narrow grid. Tune the full preprocessing pipeline and use repeated, grouped, or time-aware validation where appropriate.

Comparisons:
- Single tree: more interpretable, much higher variance.
- Gradient boosting: often stronger tabular accuracy, more sequential tuning sensitivity.
- Extra Trees: more random split thresholds, often faster and lower variance.
- Logistic regression: better for linear effects, extrapolation, sparse data, and coefficient interpretation.
- SVM/KNN: geometry-sensitive and scaling-dependent; forests are usually easier on heterogeneous tabular data.

Production considerations include model serialization compatibility, batch versus online latency, deterministic seeds, CPU parallelism, memory, schema validation, probability calibration, fairness checks, drift monitoring, and retraining policy.
