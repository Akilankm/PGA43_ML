# Foundations: what, why, and prerequisites

## What it is
A Random Forest is a collection of decision trees trained on different bootstrap samples. At each split, each tree considers only a random subset of features. Classification uses majority vote; regression uses averaging.

## Why it exists
A deep decision tree has low bias but high variance: small data changes can produce a very different tree. Bagging reduces variance by averaging unstable estimators. Random feature selection further reduces correlation among trees, making averaging more effective.

## Prerequisites
Students should understand supervised learning, train/validation/test separation, categorical versus continuous variables, decision-tree splits, probability, expectation, variance, and classification metrics.

## Assumptions
Random Forest makes few distributional assumptions. It assumes training examples are representative, labels are meaningful, predictors available at inference match training semantics, and observations are sufficiently independent for the validation design. It does not eliminate leakage, sampling bias, concept drift, or causal confounding.

## When to use
Use it for strong tabular baselines, nonlinear interactions, mixed-scale numeric predictors, moderate missingness after explicit handling, and problems where predictive accuracy matters more than a compact parametric equation.

## When not to use
Avoid it when strict extrapolation is required, latency or memory budgets are extremely small, very high-dimensional sparse linear structure dominates, calibrated probabilities are mandatory without calibration, or a transparent monotonic/causal model is required.
