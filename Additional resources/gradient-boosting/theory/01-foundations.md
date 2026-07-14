# Foundations

Gradient boosting builds an additive ensemble of weak learners, usually shallow decision trees. Each new learner is trained to reduce the current ensemble's errors.

## Why it exists

A shallow tree is stable but often underfits. A deep tree can fit complex patterns but may have high variance. Gradient boosting combines many constrained trees, each contributing a small correction.

## Core mechanics

1. Start with a simple constant prediction.
2. Compute the loss gradient or residuals.
3. Fit a weak learner to the correction target.
4. Scale the learner by the learning rate.
5. Add it to the ensemble.
6. Repeat while validation performance improves.

## Prerequisites

Students should understand supervised learning, train-validation-test splits, regression and classification losses, decision trees, derivatives, bias-variance trade-offs, cross-validation, and leakage.

## Assumptions

Gradient boosting assumes representative training data, consistent feature definitions at inference, manageable label noise, deployment-aligned validation, and a metric aligned with the business objective. It does not require linearity, normal residuals, feature scaling, or independent predictors.

## When to use it

Use it for tabular data with nonlinearities and interactions, especially when predictive quality matters and careful tuning is possible.

## When not to use it

Reconsider it when a linear model already meets requirements, coefficient-level interpretation is mandatory, labels are extremely noisy, online learning is required, extrapolation is central, or latency and memory budgets are extremely tight.

## Advantages

- Strong tabular performance
- Automatic nonlinearities and interactions
- Flexible losses
- Little need for scaling
- Multiple regularization controls
- Useful staged diagnostics

## Limitations

- Sequential training
- Sensitive hyperparameter interactions
- Potential label-noise overfitting
- Weak extrapolation
- Misleading impurity importance
- Possible probability miscalibration
- Larger inference cost as the ensemble grows
