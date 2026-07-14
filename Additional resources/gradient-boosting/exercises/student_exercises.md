# Student Exercises and Interview Questions

## Guided exercises

1. Derive the negative gradient for squared error.
2. Explain why the initial prediction is the target mean.
3. Execute three boosting stages manually on five rows.
4. Compare learning rates while adjusting estimator count.
5. Plot training and validation loss by stage.
6. Add target outliers and compare squared, Huber, and absolute-error losses.
7. Compare tree depths and explain interaction order.
8. Add missing values and place imputation correctly inside cross-validation.
9. Compare impurity and permutation importance.
10. Evaluate errors by target quartile and subgroup.
11. Replace random splitting with time-based validation.
12. Benchmark latency as the ensemble grows.

## Challenge exercises

- Implement early stopping.
- Add stochastic row subsampling.
- Implement binary logistic-loss boosting.
- Add quantile loss.
- Reproduce the notebook with a real-world dataset.
- Write unit tests for determinism, shape validation, and staged predictions.

## Interview questions

1. Why is boosting gradient descent in function space?
2. What is a pseudo-residual?
3. How do learning rate and estimator count interact?
4. Why are shallow trees used?
5. How does boosting differ from bagging?
6. Why can boosting overfit label noise?
7. Is scaling required?
8. How should temporal data be validated?
9. Why is impurity importance misleading?
10. How are probabilities calibrated?
11. Why do tree ensembles extrapolate poorly?
12. What should production monitoring include?