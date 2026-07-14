# Foundations: what, why, prerequisites, and intuition

## What KNN is

K-nearest neighbors predicts a query from the labels or targets of the most similar retained training observations. It is non-parametric, instance-based, lazy, and local: fitting primarily stores examples, while most computation occurs during prediction.

For classification, neighbors vote for a class. For regression, their targets are averaged. Both can use uniform or distance-based weights.

## Why it exists

Many real problems exhibit local regularity: observations that are close in a meaningful feature space tend to have similar outcomes. KNN models that principle without imposing a global linear equation or recursive tree partition.

It is valuable as an interpretable baseline, a flexible estimator for irregular boundaries, and a teaching model for representation, distance, and the bias-variance trade-off.

## Prerequisites

Students should understand vectors, train/validation/test splits, classification metrics, NumPy, and basic probability.

Before using KNN, confirm that:

1. rows are independent observations;
2. features are available at prediction time;
3. missing values are handled;
4. categorical features are encoded consistently;
5. scaling is fitted only on training folds;
6. distance corresponds to domain similarity.

## Step-by-step mechanics

For each query:

1. calculate its distance to every retained training example;
2. identify the `k` smallest distances;
3. retrieve corresponding labels or targets;
4. calculate optional distance weights;
5. aggregate the neighbors;
6. apply deterministic tie-breaking;
7. return a prediction and, where required, local class proportions.

The simplicity is deceptive: feature representation and distance definition are the central modeling choices.