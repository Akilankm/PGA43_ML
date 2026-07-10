# Instructor notes

## Suggested teaching sequence

1. Begin with the probability → odds → log-odds transformation and use the sigmoid plot to explain why linear regression is not a probability model.
2. Ask learners to predict what will happen if `duration` is included. Then show why post-contact availability makes it leakage, even if it is highly predictive.
3. Make the class read the audit before opening the first model result. Discuss duplicates, missingness, rare categories, and the positive-class prevalence.
4. Compare the majority baseline and logistic model at threshold 0.50. Ask why a reasonable ROC-AUC can coexist with low positive recall.
5. Interpret one numeric coefficient in log-odds, odds-ratio, and probability language. Then interpret one one-hot category relative to its reference level.
6. Separate four questions: ranking, classification at a threshold, probability calibration, and business cost.
7. Show that class weighting and SMOTE can increase recall while changing calibration and precision.
8. Use the Statsmodels table to discuss adjusted effects, standard errors, confidence intervals, and the difference between association and causation.
9. Treat diagnostics as questions for model improvement. A high Cook's distance is a reason to investigate, not an automatic reason to delete a row.
10. End with the untouched test set, threshold policy, feature contract, and model card. Ask what must be monitored after deployment.

## Suggested timing

| Block | Time |
|---|---:|
| Foundations and mathematical intuition | 45 minutes |
| Audit, target, and leakage | 45 minutes |
| Univariate, bivariate, and multivariate EDA | 75 minutes |
| Splits, preprocessing, baseline, and first model | 60 minutes |
| Metrics, curves, tuning, and imbalance | 90 minutes |
| Threshold policy and calibration | 45 minutes |
| Statsmodels inference and diagnostics | 90 minutes |
| Final evaluation, deployment, and exercises | 45 minutes |

## Discussion prompts

- Why is a 0.5 threshold not a universal law?
- Why can average precision be more informative than accuracy for a rare positive class?
- Why does a one-unit odds multiplier depend on how a numeric variable was scaled?
- Why should a test set not be used to choose a threshold?
- Why can a model have good discrimination and poor calibration?
- What operational process would make `duration` available at decision time, and would that change the problem definition?
- Which subgroup metrics should be reported when a category has only a few observations?
- What would you monitor if the campaign mix changed from one month to the next?

## Common learner mistakes

- Calling `conda create --file environment.yaml`; the correct command is `conda env create -f environment.yaml`.
- Fitting an encoder or scaler on the full dataset before splitting.
- Reporting accuracy alone on an imbalanced target.
- Treating a p-value as proof of practical importance or causality.
- Tuning a threshold on the test set.
- Describing `class_weight="balanced"` or SMOTE as a calibration remedy.
- Interpreting a coefficient as a fixed probability-point change.
- Including `duration` in a prospective pre-call targeting model.
