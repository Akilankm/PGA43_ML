# Data Preparation and Mechanics

## Leakage-safe workflow
1. Define visualization, compression, denoising, or predictive use.
2. Create an untouched test split for predictive workflows.
3. Fit imputation and scaling only in training folds.
4. Fit PCA in the same pipeline.
5. Select component count with training-only criteria.
6. Evaluate downstream performance and held-out reconstruction.
7. Persist means, scales, components, and versions.

## Scaling
Covariance PCA is dominated by large-variance variables. Standardize when units differ or scale should not determine importance. Do not scale automatically when physical variance magnitude is meaningful.

## Missing and categorical data
Classical PCA needs complete numeric data. Impute inside the pipeline. One-hot data can be used, but Multiple Correspondence Analysis may better suit purely categorical data.

## Outliers
Outliers can rotate components. Compare results with and without suspicious observations or use robust PCA.

## Mechanics
Center, optionally scale, compute SVD, order singular directions, retain k components, transform to scores, reconstruct if needed, and evaluate retained variance plus downstream utility.

## Choosing k
Use scree elbows, cumulative variance, parallel analysis, reconstruction error, or downstream cross-validation. A 95% threshold is only a heuristic.

## Complexity
Full SVD may be expensive. Randomized SVD is useful for a small number of components; IncrementalPCA supports out-of-core data.
