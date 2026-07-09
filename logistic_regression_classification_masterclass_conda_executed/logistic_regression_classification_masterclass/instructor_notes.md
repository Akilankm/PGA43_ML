# Instructor notes

## Recommended teaching sequence

1. Start with odds and log-odds before introducing the sigmoid.
2. Ask learners why linear regression can produce impossible probabilities.
3. Use `duration` to distinguish a statistically predictive variable from a deployable variable.
4. Compare accuracy with balanced accuracy and average precision before discussing imbalance remedies.
5. Interpret coefficients in three layers: log-odds, odds ratio, and context-dependent probability change.
6. Separate model discrimination from calibration and decision thresholding.
7. Treat the test set as a one-time audit, not a tuning resource.

## Discussion prompts

- Why does threshold `0.5` produce near-zero recall on an imbalanced problem?
- Why can class weighting improve recall while degrading probability calibration?
- Why is a statistically significant variable not necessarily operationally important?
- Why does an odds ratio not imply causality?
- How would contact capacity constraints change threshold selection?

## Suggested classroom timing

- Foundations and leakage: 60 minutes
- EDA and association measures: 60 minutes
- Pipelines and model fitting: 75 minutes
- Evaluation and thresholds: 75 minutes
- Inference and diagnostics: 75 minutes
- Deployment, model card, and exercises: 45 minutes
