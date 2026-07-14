# Diagnostics, limitations, comparisons, and production

## Diagnostics

Inspect cross-validation performance across neighborhood sizes, train-versus-validation gaps, confusion matrices, per-class recall, prediction confidence, and distances to selected neighbors. Large neighbor distances indicate weak local support: the algorithm still returns a class, but the query may lie outside dense training regions.

Test sensitivity to scaling, metrics, feature subsets, weighting, and random split seeds. Measure prediction latency and memory as the retained dataset grows.

## When to use KNN

KNN is suitable when datasets are small or medium, local similarity is meaningful, dimensionality is moderate, nonlinear boundaries are plausible, prediction latency is acceptable, and example-based explanations are valuable.

## When not to use it

Reconsider KNN when prediction must be extremely fast at large scale, memory is restricted, dimensionality is high, features mix incompatible semantics, label noise is substantial, reliable extrapolation is required, or deployment demands a compact parametric model.

## Failure modes

- Scaling before splitting, causing leakage.
- Omitting scaling.
- Selecting hyperparameters from test results.
- Using accuracy alone for imbalanced classes.
- Treating local vote proportions as calibrated probabilities.
- Allowing duplicate entities across splits.
- Ignoring tie-breaking behavior.
- Retaining irrelevant high-dimensional features.
- Ignoring prediction-time cost.

## Curse of dimensionality

As dimensionality rises, points become sparse and nearest and farthest distances can become similar. Neighborhood contrast weakens. Mitigate this using feature selection, domain-informed embeddings, PCA where appropriate, metric learning, and removal of noisy dimensions.

## Comparisons

- Logistic regression is compact and fast but globally linear without feature engineering.
- Decision trees model interactions without scaling but can be unstable.
- SVMs can perform strongly in high-dimensional spaces but are less example-based.
- Nearest-centroid methods are compact but miss multimodal class structure.
- Random forests are robust nonlinear baselines but less locally transparent.

## Production concerns

Version preprocessing, feature definitions, retained reference data, class order, tie rules, hyperparameters, and validation results. Monitor input drift, neighbor-distance drift, class-frequency drift, latency, and memory. Because training examples are retained, privacy, deletion, and retention controls are especially important.