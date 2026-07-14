# Training, Data Preparation, and Mechanics

## Leakage-safe workflow

1. Define the target and deployment unit.
2. Create an untouched test split before exploration or tuning.
3. Fit imputers, encoders, and feature selectors only on training folds.
4. Use cross-validation for hyperparameter search.
5. Refit the selected pipeline on the full training set.
6. Evaluate once on the untouched test set.
7. Record data, code, seed, and library versions.

## Scaling

Tree splits depend on ordering rather than Euclidean distance, so standardization is normally unnecessary. Scaling may still be needed for upstream transformations or mixed-model pipelines.

## Missing values

Classic scikit-learn gradient boosting requires explicit imputation. Put imputation inside a pipeline. Histogram-based boosting systems may support native missing-value routing.

## Categorical variables

Use one-hot encoding for low-cardinality nominal variables. For high-cardinality features, use leakage-safe target-aware encoding or a library with native categorical handling.

## Outliers

Squared-error boosting can chase extreme residuals. Investigate data quality and compare Huber or absolute-error objectives when tails are heavy.

## Class imbalance

For classification, evaluate balanced accuracy, macro F1, precision-recall behavior, class-specific errors, and threshold-dependent costs.

## Step-by-step regression mechanics

1. Initialize predictions to the training-target mean.
2. Compute residuals.
3. Fit a shallow regression tree to residuals.
4. Predict corrections.
5. Multiply corrections by the learning rate.
6. Add them to the current ensemble.
7. Repeat.
8. Monitor validation loss at every stage.

## Computational behavior

Training is sequential across trees. Prediction cost grows approximately with the number of trees multiplied by tree depth.