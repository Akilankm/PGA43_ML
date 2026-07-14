# Foundations

A Support Vector Machine selects a hyperplane `f(x)=w^T x+b` with maximum geometric margin. The closest observations are support vectors.

## Why it exists
Many hyperplanes may separate training data. Maximum-margin selection improves robustness and often generalizes well.

## Prerequisites
Vectors, dot products, norms, derivatives, constrained optimization, binary metrics, splitting, and scaling.

## Use it when
Data is small or medium, dimensionality is high, sparse features are common, or a suitable kernel can model nonlinear structure.

## Avoid it when
Data has millions of rows, incremental learning is mandatory, calibrated probabilities are central, preprocessing cannot be controlled, or nonlinear inference with many support vectors is too slow.

## Advantages
Convex standard objective, strong high-dimensional performance, kernel flexibility, and support-vector sparsity.

## Limitations
Scaling sensitivity, expensive nonlinear training, hyperparameter sensitivity, non-native probabilities, and difficult nonlinear interpretation.
