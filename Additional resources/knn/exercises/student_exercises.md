# Student exercises and interview review

## Applied exercises

1. Hand-calculate Manhattan and Euclidean distances for three points.
2. Implement deterministic tie-breaking without `Counter.most_common`.
3. Extend the scratch classifier with a `kneighbors` method.
4. Compare uniform and distance weighting on a noisy moons dataset.
5. Plot validation score for `k=1..40` and explain the bias-variance pattern.
6. Demonstrate preprocessing leakage, then correct it with a pipeline.
7. Create an imbalanced dataset and compare accuracy, balanced accuracy, F1, and PR-AUC.
8. Add PCA inside the pipeline and explain why performance changes.
9. Benchmark brute-force prediction latency as training size grows.
10. Write a model card describing intended use, limitations, privacy, and monitoring.

## Interview questions

1. Why is KNN called a lazy learner?
2. What does `k` control?
3. Why must scaling be fitted inside the pipeline?
4. How are class probabilities computed?
5. Why does KNN struggle in high dimensions?
6. Compare Euclidean and Manhattan distance.
7. How would you handle class imbalance?
8. Why can a KD tree become ineffective?
9. What are fit-time, prediction-time, and memory costs?
10. How would you detect an out-of-distribution query?
11. When is distance weighting beneficial?
12. Why does KNN extrapolate poorly?
13. How would you represent mixed numeric and categorical features?
14. What privacy issue is distinctive for instance-based learning?
15. Compare KNN with nearest centroid, logistic regression, trees, and SVMs.

## Mastery criterion

A student has mastered the module when they can implement classification from scratch, justify the distance metric, construct a leakage-safe tuning pipeline, interpret neighbor-distance diagnostics, and reject KNN for an unsuitable production scenario.