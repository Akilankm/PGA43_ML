# Mathematics and Derivation

For centered data $X_c$, covariance is $S=X_c^T X_c/(n-1)$. For unit direction $w$, projected variance is $w^T S w$. PCA solves $\max_w w^T S w$ subject to $w^T w=1$.

The Lagrangian derivative gives $Sw=\lambda w$, so principal directions are covariance eigenvectors and eigenvalues are component variances.

For top directions $W_k$, scores are $Z=X_cW_k$ and reconstruction is $\hat X=ZW_k^T+\mu$. PCA also minimizes $\|X-\hat X\|_F^2$ among rank-$k$ linear approximations.

If $X_c=U\Sigma V^T$, columns of $V$ are principal directions and $\lambda_j=\sigma_j^2/(n-1)$. SVD is usually numerically preferable to explicitly forming covariance.

Explained variance ratio is $\lambda_j/\sum_l\lambda_l$. Discarded singular values determine reconstruction error. Component signs are arbitrary because $v$ and $-v$ define the same axis.
