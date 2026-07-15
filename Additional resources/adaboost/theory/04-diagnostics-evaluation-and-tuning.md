# Diagnostics, Evaluation, and Hyperparameter Tuning

Evaluate accuracy, balanced accuracy, macro F1, per-class precision and recall, confusion matrix, ROC-AUC, and precision-recall AUC.

Diagnostics should include staged training and validation error, weak-learner error trajectory, learner-contribution trajectory, concentration of observation importance, margin distributions, subgroup errors, and calibration when probabilities are used.

Core hyperparameters are estimator count, learning rate, base-estimator depth, and minimum leaf size. Tune estimator count and learning rate jointly. Begin with decision stumps, then compare depth two or three. Use stratified cross-validation and preserve an untouched test set. Validation-based early stopping can prevent unnecessary rounds and inference cost.
