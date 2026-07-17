# Gradient Boosting Technical Notes

## 1. Additive stage-wise model

Gradient boosting constructs

\[
F_M(x)=F_0(x)+\sum_{m=1}^{M}\nu\,\rho_m h_m(x),
\]

where `h_m` is a weak learner, `rho_m` is a stage/leaf correction, and `nu` is the learning rate. Unlike bagging, learners are not independent: stage `m` is trained against the current model's errors.

## 2. Functional gradient descent

For empirical risk

\[
\mathcal{R}(F)=\sum_i L(y_i,F(x_i)),
\]

the pseudo-residual at stage `m` is

\[
r_{im}=-\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}.
\]

A tree is fitted to `r_im`, converting an abstract functional gradient into a learnable function over feature space.

## 3. Squared-error regression

For `L(y,F)=0.5(y-F)^2`, the negative gradient is `y-F`: the ordinary residual. The initial constant minimizing squared error is the sample mean.

## 4. Binary log-loss

For raw score `F`, probability `p=sigmoid(F)`, and binary target `y`, the negative gradient is `y-p`. A Newton leaf correction uses `sum(y-p)/sum(p(1-p))`, providing curvature-aware steps.

## 5. Regularization controls

- Lower `learning_rate` requires more stages but usually smooths optimization.
- `max_depth` controls interaction order; depth 1 is additive in one-way effects.
- `min_samples_leaf` prevents fragile terminal corrections.
- `subsample < 1` creates stochastic gradient boosting and can reduce variance.
- Early stopping chooses stage count using validation loss.

## 6. Evaluation contract

Hyperparameters and thresholds are selected on training/validation data. The test set remains untouched until the final evaluation. Classification quality must include discrimination, probability quality, threshold metrics, and decision cost; regression quality must include point metrics and residual structure.

## 7. Limitations

Classical gradient boosting is sequential, sensitive to noisy labels and hyperparameters, and may extrapolate poorly outside the training support. Impurity importance can be biased; permutation importance is validation-dependent and not causal. Monitoring must cover input drift, target drift, calibration drift, performance, latency, and data-contract violations.
