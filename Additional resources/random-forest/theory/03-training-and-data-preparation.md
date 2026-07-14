# Mechanics, data preparation, and training

1. Define the prediction unit, target, observation time, and deployment horizon.
2. Split data before learning preprocessing. Use stratification for classification and grouped or temporal splits when observations are dependent.
3. Impute missing values inside the training pipeline. Encode categorical features consistently; one-hot encoding is safest for nominal variables in standard scikit-learn forests.
4. Scaling is usually unnecessary because tree splits depend on ordering, not Euclidean distance.
5. For each tree, bootstrap rows, recursively grow splits, and randomly restrict candidate features at each node.
6. Aggregate votes or probabilities across trees.
7. Tune with cross-validation using a metric aligned with business cost.
8. Evaluate once on an untouched test set.

Key leakage risks include target-derived predictors, post-outcome variables, preprocessing before splitting, duplicate entities across folds, and tuning on the final test set.

Class imbalance is not solved automatically. Use stratified validation, class-weighting, balanced metrics, threshold analysis, or resampling performed only within training folds.
