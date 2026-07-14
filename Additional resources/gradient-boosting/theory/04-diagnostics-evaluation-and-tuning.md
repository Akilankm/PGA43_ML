# Diagnostics, Evaluation, and Hyperparameter Tuning

## Regression metrics

- MAE is robust and directly interpretable.
- RMSE emphasizes large errors.
- R-squared expresses relative variance explanation but is not a deployment cost.
- Median absolute error is robust under heavy tails.
- Quantile loss supports asymmetric uncertainty.

## Classification metrics

Use log loss, ROC-AUC, precision-recall AUC, balanced accuracy, macro F1, Brier score, and calibration diagnostics according to the decision objective.

## Residual diagnostics

Inspect residuals against fitted values, important features, target range, subgroup, and time. Structured residual patterns may indicate missing features, insufficient interactions, leakage, label problems, or distribution shift.

## Staged diagnostics

Plot training and validation loss after every boosting stage. Select the stage count near minimum validation loss, not minimum training loss.

## Core hyperparameters

- `n_estimators`: capacity and inference cost.
- `learning_rate`: correction size.
- `max_depth`: interaction order.
- `min_samples_leaf`: leaf stability.
- `subsample`: stochastic regularization.
- `max_features`: feature subsampling.
- `loss`: objective and robustness.
- Early-stopping controls: validation fraction, patience, and tolerance.

## Search strategy

Start with shallow trees and a moderate learning rate. Search learning rate and estimator count jointly. Use randomized search for broad exploration, then narrow around stable regions.

## Feature importance

Impurity importance is biased toward continuous and high-cardinality predictors. Prefer held-out permutation importance. Use partial dependence or accumulated local effects for response shape. Treat all importance as predictive reliance, not causality.

## Probability calibration

Boosted classifiers may be overconfident. Compare reliability curves and Brier score. Calibrate using held-out or cross-validated predictions.