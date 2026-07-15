# Comparisons, Failure Modes, and Production

## Comparisons
Feature selection retains original variables; PCA creates composites. Truncated SVD usually does not center and is suitable for sparse text. Kernel PCA captures nonlinear structure but scales and inverts poorly. ICA seeks independent sources. NMF yields nonnegative additive parts. t-SNE and UMAP optimize nonlinear embedding objectives mainly for visualization. Autoencoders support nonlinear compression but require more data and tuning.

## Failure modes
Fitting before splitting, unjustified scaling choices, universal variance thresholds, treating variance as predictive importance, ignoring outliers, treating signs as fixed, claiming causal factors, omitting no-PCA baselines, and using dense PCA on massive sparse matrices.

## Production
Persist preprocessing and PCA together. Validate schema and feature order. Record means, scales, components, and versions. Monitor feature drift, score drift, reconstruction error, latency, and transformation consistency.

## Prefer another method
Use feature selection for interpretability, Truncated SVD for sparse text, NMF for nonnegative parts, Kernel PCA for small nonlinear datasets, supervised reduction when labels should guide compression, and autoencoders for large nonlinear representation learning.
