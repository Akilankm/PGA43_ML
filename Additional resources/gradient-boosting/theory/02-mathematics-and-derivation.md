# Mathematics and Derivation

## Additive model

Gradient boosting constructs an additive predictor:

`F_M(x) = F_0(x) + sum_{m=1..M} learning_rate * h_m(x)`.

Each weak learner is chosen to reduce empirical risk over the training observations.

## Functional gradient descent

At stage `m`, compute pseudo-residuals as the negative derivative of the loss with respect to the current prediction. Fit a weak learner to the input-pseudo-residual pairs and add its shrunken prediction to the ensemble.

## Squared-error derivation

For half squared error, the derivative with respect to the current prediction is `prediction - target`. The negative gradient is therefore `target - prediction`, which is the ordinary residual. Residual fitting is a special case of gradient descent in function space.

The constant minimizing squared error is the sample mean, so regression begins with the training-target mean.

## Logistic-loss intuition

For binary classification, the negative gradient behaves like the difference between the observed class and the current predicted probability. Trees correct probability errors in log-odds space.

## Leaf values

A tree partitions feature space into terminal regions. Each terminal value is selected to minimize the chosen loss within that region.

## Regularization

- Learning rate controls shrinkage.
- Estimator count controls correction stages.
- Tree depth controls interaction complexity.
- Minimum leaf size controls local stability.
- Row subsampling creates stochastic gradient boosting.
- Feature subsampling reduces reliance on a few predictors.
- Early stopping limits effective complexity.

Learning rate and estimator count must be tuned jointly.

## Bias-variance interpretation

Shallow trees begin with high bias. Sequential corrections reduce bias. Too many stages, excessive depth, or noisy labels increase variance. Shrinkage, subsampling, shallow learners, and early stopping control the transition.