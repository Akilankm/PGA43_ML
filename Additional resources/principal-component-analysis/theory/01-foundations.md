# Foundations

PCA is an unsupervised linear transformation that replaces correlated variables with orthogonal principal components. PC1 captures maximum variance; each later component captures maximum remaining variance subject to orthogonality.

## Why it exists
Datasets often contain redundant, noisy, correlated measurements. PCA gives a lower-dimensional representation preserving as much linear variance as possible.

## Terms
Scores are observations in component coordinates. Loadings are feature coefficients. Explained variance is component variance. Reconstruction maps scores back to original space. Whitening rescales scores to unit variance.

## Assumptions and prerequisites
Students need vectors, matrices, covariance, eigenvalues, eigenvectors, SVD, and train-validation-test splitting. PCA is most useful when linear structure and Euclidean geometry are meaningful and variance is a defensible proxy for information. Normality is not required for computation.

## Use PCA when
Correlated numeric predictors, visualization, compression, denoising, multicollinearity mitigation, or latent linear structure are present.

## Avoid PCA when
Direct feature meaning must remain, low-variance variables may be predictive, structure is strongly nonlinear, units are incompatible, outliers dominate, or sparse nonnegative semantics must be preserved.

## Advantages
Deterministic linear compression, orthogonal features, efficient SVD, reduced multicollinearity, and exact reconstruction mathematics.

## Limitations
Scaling and outlier sensitivity, unsupervised objective, sign ambiguity, linearity, and weaker interpretability.
