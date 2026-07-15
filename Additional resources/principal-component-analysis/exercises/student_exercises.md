# Student Exercises and Interview Questions

## Guided exercises

1. Center a small matrix manually.
2. Compute a two-feature covariance matrix.
3. Derive the first eigenvector and projected scores.
4. Show that distinct principal-component scores are uncorrelated.
5. Reconstruct observations using one component.
6. Compare covariance PCA before and after standardization.
7. Match the scratch implementation with scikit-learn.
8. Plot cumulative explained variance.
9. Select component count using reconstruction error.
10. Compare downstream classification with and without PCA.
11. Inspect feature-wise reconstruction error.
12. Test sensitivity to an extreme outlier.
13. Compare PCA and Truncated SVD on sparse data.
14. Assess component stability under bootstrap resampling.

## Challenge exercises

Implement covariance-eigendecomposition PCA, whitening, randomized SVD, incremental updates, parallel analysis, a biplot, and comparison with a linear autoencoder.

## Interview questions

1. Why must PCA center data?
2. When should data be standardized?
3. Derive PCA as a constrained optimization problem.
4. Why are eigenvalues component variances?
5. How does SVD produce PCA?
6. What is the relation between retained variance and reconstruction error?
7. Why can PCA reduce multicollinearity?
8. Why can PCA hurt predictive performance?
9. What does whitening do?
10. Why are component signs arbitrary?
11. PCA versus Truncated SVD?
12. How would you monitor PCA in production?
