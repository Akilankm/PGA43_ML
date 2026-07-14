# Mathematics and mechanics

Given training pairs `(x_i, y_i)`, let `N_k(x)` denote the indices of the `k` training points closest to query `x`.

## Classification

For each class, add the weights of neighboring observations belonging to that class. Predict the class with the largest total weight. Uniform voting assigns weight 1. Distance voting commonly assigns a weight proportional to the reciprocal of distance, with a small numerical safeguard.

## Regression

Predict the weighted neighborhood average: add each neighboring target multiplied by its weight, then divide by the sum of weights. Uniform weighting produces the ordinary neighborhood mean.

## Minkowski distance

For power `p`, calculate the absolute coordinate differences, raise each to `p`, sum them, and take the `p`th root.

- `p=1` gives Manhattan distance.
- `p=2` gives Euclidean distance.
- Larger values increasingly emphasize the largest coordinate difference.

Cosine, Hamming, and Mahalanobis distances may suit directional, binary, or covariance-aware similarity. The metric defines what similarity means for the problem.

## Bias and variance

- `k=1` is highly flexible and sensitive to noise.
- Increasing `k` lowers variance but increases smoothing bias.
- Very large `k` approaches the global class frequency or target mean.

Select `k` with cross-validation rather than training or test accuracy.

## Probability interpretation

KNN probabilities are normalized local vote weights. They are local frequency estimates, not automatically calibrated probabilities. Small neighborhoods create coarse probability values, and class imbalance can distort local proportions.