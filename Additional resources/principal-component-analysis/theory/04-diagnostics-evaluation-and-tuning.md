# Diagnostics, Evaluation, and Tuning

Inspect explained-variance ratios, cumulative variance, scree plots, score plots, loadings, biplots, reconstruction error, outliers, and component stability.

When PCA feeds a supervised model, tune `n_components` through cross-validation on the complete pipeline and compare with a no-PCA baseline.

Large absolute loadings identify features defining a component. Interpret patterns, not isolated coefficients. Sign is arbitrary.

Inspect overall and feature-wise held-out reconstruction error. High feature error indicates poor representation by the retained subspace.

## Hyperparameters
- `n_components`: integer, variance fraction, or supported MLE;
- `svd_solver`: full, randomized, covariance-based, or auto;
- `whiten`: unit component variance;
- randomized solver iteration controls;
- preprocessing choices.

Whitening can help scale-sensitive downstream models but discards original variance magnitude and may amplify weak-component noise.

Bootstrap rows and compare component subspaces. Individual loadings can be unstable when eigenvalues are close. High explained variance does not guarantee predictive value because PCA ignores the target.
