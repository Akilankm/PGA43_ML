# Modeling workflow

## Data preparation

1. Define the target and unit of analysis.
2. Remove leakage variables and entity duplicates across splits.
3. Stratify classification splits when appropriate.
4. Impute missing values inside the pipeline.
5. encode categorical variables using a representation compatible with the selected metric.
6. Scale numeric features inside the validation pipeline.
7. Inspect class balance and irrelevant dimensions.
8. Tune only on training folds.
9. Evaluate once on an untouched test set.

Distance aggregates coordinate differences. Without scaling, a feature measured in thousands can dominate a feature measured between zero and one. `StandardScaler` should therefore be placed before KNN in a scikit-learn `Pipeline`, so fold-specific means and standard deviations are learned only from each training partition.

## Hyperparameters

- `n_neighbors`: locality versus smoothing.
- `weights`: uniform or distance-weighted aggregation.
- `metric`: geometry of similarity.
- `p`: Minkowski power.
- `algorithm`: brute force, KD tree, ball tree, or automatic selection.
- `leaf_size`: tree construction and query efficiency.

The scoring function used during tuning must represent the real decision cost.

## Evaluation

For classification, examine accuracy, balanced accuracy, per-class precision and recall, macro or weighted F1, confusion matrices, ROC-AUC, and PR-AUC where imbalance matters. Assess Brier score or reliability curves when probability quality matters.

For regression, examine MAE, RMSE, R-squared, residual structure, and performance across target ranges.

## Validation discipline

Cross-validation selects preprocessing and hyperparameters. The test set is not a tuning surface. Report cross-validation variation as well as mean performance, then perform one final test evaluation after the modeling decisions are frozen.