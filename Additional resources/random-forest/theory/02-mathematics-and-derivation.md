# Mathematics and derivation

For classification, a node with class proportions p_k has Gini impurity G=1-sum_k p_k^2. A candidate split partitions node S into S_L and S_R and minimizes |S_L|/|S| G(S_L)+|S_R|/|S| G(S_R), equivalently maximizing impurity decrease.

For tree b, draw a bootstrap sample D_b of n observations with replacement. At every node, sample m_try predictors from p predictors and choose the best split only among them. The forest classifier is f_hat(x)=mode{T_b(x): b=1,...,B}. For regression it is f_hat(x)=B^{-1} sum_b T_b(x).

If individual trees have variance sigma^2 and pairwise correlation rho, the variance of their average is approximately rho sigma^2 + (1-rho)sigma^2/B. Increasing B reduces only the second term; random feature selection targets rho.

A training observation is excluded from a bootstrap sample with probability (1-1/n)^n, approaching exp(-1), approximately 0.368. These out-of-bag observations provide internal validation predictions without a separate fold for each tree.

Impurity-based feature importance sums weighted impurity decreases but can favor continuous or high-cardinality features. Permutation importance measures predictive degradation after shuffling a feature and should be evaluated on held-out data.
