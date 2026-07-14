# Comparisons, Failure Modes, and Production

## Comparisons
Logistic regression offers natural probabilities and coefficients; SVM emphasizes margin. KNN is local and prediction-heavy; SVM learns a global boundary. Trees require little scaling; SVM requires careful geometry. Gradient boosting often dominates heterogeneous tabular data; SVM excels on smaller high-dimensional data. Neural networks are better for massive unstructured representation learning.

## Failure modes
Scaling before splitting, tuning on test data, omitting scaling, defaulting to RBF without a linear baseline, narrow grids, ignoring imbalance, treating scores as probabilities, ignoring support-vector count, random temporal splits, and applying nonlinear SVM at impractical scale.

## Production
Serialize preprocessing and model together, validate schema, benchmark latency, monitor feature and score drift and calibration, version data/code/packages, define fallback and retraining policies, and test compatibility after upgrades.

## Prefer another model
Use logistic regression for transparent probabilities, boosting for heterogeneous tabular data, random forests for lower-tuning nonlinear baselines, and neural networks for massive unstructured data.
