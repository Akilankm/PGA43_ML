# Comparisons, Failure Modes, and Production

## Comparisons

### Single decision tree
Gradient boosting is usually more accurate and stable, but slower and less interpretable.

### Random forest
Random forests build trees independently and mainly reduce variance. Gradient boosting builds trees sequentially and reduces residual bias. Random forests are easier to parallelize and usually require less tuning.

### AdaBoost
AdaBoost changes observation emphasis. Gradient boosting optimizes an explicit differentiable loss through pseudo-residuals.

### XGBoost, LightGBM, and CatBoost
These are optimized boosting systems with regularized objectives, histogram or specialized split finding, and advanced missing-value or categorical handling. Classic gradient boosting remains ideal for learning and smaller baselines.

### Linear models
Linear models extrapolate and provide coefficients. Gradient boosting captures nonlinearities and interactions without manual basis expansion.

## Failure modes

- preprocessing leakage;
- tuning on the test set;
- overly deep weak learners;
- excessive stages on noisy labels;
- high learning rate;
- random splitting for temporal data;
- blind trust in impurity importance;
- assuming probabilities are calibrated;
- expecting extrapolation;
- ignoring latency growth;
- relying only on aggregate metrics.

## Production checklist

- serialize preprocessing and model together;
- validate input schema;
- log model, data, seed, and versions;
- benchmark p95 latency and memory;
- monitor feature drift, prediction drift, error, and calibration;
- define fallback behavior;
- establish retraining triggers;
- test serialization compatibility;
- maintain champion-challenger evaluation;
- document thresholds and business costs.

## Prefer another model when

Choose linear models for sparse high-dimensional data or coefficient-level interpretation, random forests for a lower-tuning variance-reduction baseline, histogram boosting for larger datasets, and monotonic models when constraints must be guaranteed.