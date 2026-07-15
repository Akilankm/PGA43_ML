# Foundations

AdaBoost converts many weak classifiers into a strong classifier. It trains learners sequentially, increasing the importance of misclassified observations so later learners focus on difficult regions.

Use it for small-to-medium tabular classification problems with weak base learners such as decision stumps. Avoid it when labels are very noisy, outliers dominate, datasets are extremely large, or calibrated probabilities are central.

Advantages include conceptual clarity, low implementation complexity, strong performance with shallow trees, and useful margin behavior. Limitations include sensitivity to label noise and outliers, sequential training, and dependence on weak-learner quality.

Prerequisites: binary classification, weighted averages, logarithms, exponential functions, decision trees, train-validation-test splitting, and classification metrics.
