# Diagnostics, Evaluation, and Hyperparameter Tuning

Use accuracy only for balanced costs. Otherwise inspect balanced accuracy, macro F1, per-class precision and recall, confusion matrix, ROC-AUC, PR-AUC, Brier score, and calibration.

## Margin diagnostics
Inspect decision scores by class. Many scores near zero indicate ambiguity. Confident errors suggest label problems, drift, or an unsuitable kernel.

## Support-vector diagnostics
A high support-vector fraction can indicate overlap, noise, weak features, small `C`, or an over-flexible kernel; it also raises inference cost.

## Learning curves
Both train and validation low means underfitting. A persistent gap means variance or insufficient data.

## Hyperparameters
- `C`: small gives stronger regularization and wider margin; large penalizes violations more.
- `gamma`: small gives smooth RBF boundaries; large gives local boundaries.
- `kernel`: linear is a strong sparse baseline; RBF captures nonlinear structure.
- `class_weight`: changes class-specific penalties.

Search `C` and `gamma` on logarithmic scales with stratified cross-validation. Tune preprocessing and model together. SVM scores are not probabilities; use cross-validated calibration when required.
