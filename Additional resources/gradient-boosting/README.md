# Gradient Boosting Masterclass

This module teaches gradient boosting from first principles through a validated, executed notebook and production-oriented implementation.

## Learning objectives

By the end, students will be able to:

- explain additive stage-wise modeling and functional gradient descent;
- derive squared-error boosting updates;
- distinguish learning rate, number of estimators, tree depth, and subsampling effects;
- prepare data without leakage;
- implement a gradient boosting regressor from scratch;
- train and tune scikit-learn GradientBoosting models;
- diagnose underfitting, overfitting, calibration, residual structure, and feature reliance;
- decide when gradient boosting is appropriate and when another model is safer.

## Module map

1. `theory/01-foundations.md`
2. `theory/02-mathematics-and-derivation.md`
3. `theory/03-training-data-and-mechanics.md`
4. `theory/04-diagnostics-evaluation-and-tuning.md`
5. `theory/05-comparisons-failure-modes-and-production.md`
6. `src/gradient_boosting_from_scratch.py`
7. `src/library_pipeline.py`
8. `notebooks/gradient-boosting-masterclass.ipynb`
9. `exercises/student_exercises.md`
10. `references.md`

## Reproducibility

Use the supplied `environment.yaml`, then execute the notebook top-to-bottom. All stochastic operations use fixed random seed `42`. Run `python validate_module.py` after execution.
