# Student Exercises

1. Derive the negative gradient for absolute-error loss and explain why it differs from squared error.
2. Reproduce the first three squared-error boosting stages by hand on the toy dataset.
3. Set `learning_rate=1.0`; compare validation behavior with `learning_rate=0.03`.
4. Quantify how tree depth changes interaction capacity and overfitting.
5. Compare deterministic and stochastic boosting using identical total stage counts.
6. Implement early stopping using a validation set for the scratch regressor.
7. Replace the classification threshold with one minimizing a false-negative-heavy cost function.
8. Compare impurity and permutation importance; explain disagreements.
9. Create a synthetic covariate-shift scenario and evaluate degradation.
10. Write a deployment model card covering objective, data contract, metrics, failure modes, monitoring, and rollback.
