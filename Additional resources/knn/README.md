# K-Nearest Neighbors (KNN) Masterclass

K-nearest neighbors is a non-parametric, instance-based learning method for classification and regression. This module develops KNN from first principles, implements it with NumPy, validates it against scikit-learn, and teaches the practical reasoning needed to use it responsibly.

## Learning outcomes

By the end, a student can:

- explain lazy learning, neighborhoods, distance metrics, and voting;
- derive the classification and regression estimators;
- identify why scaling and dimensionality matter;
- implement KNN without a machine-learning estimator;
- build leakage-safe scikit-learn pipelines;
- tune `n_neighbors`, distance, weighting, and Minkowski power;
- diagnose underfitting, overfitting, imbalance, and weak local support;
- compare KNN with linear models, trees, SVMs, and nearest-centroid methods;
- decide when KNN is appropriate and when it is not.

## Learning sequence

1. `theory/01-foundations.md`
2. `theory/02-mathematics-and-mechanics.md`
3. `theory/03-modeling-workflow.md`
4. `theory/04-diagnostics-limitations-and-production.md`
5. `src/knn_from_scratch.py`
6. `src/knn_regressor_from_scratch.py`
7. `notebooks/knn-masterclass.ipynb`
8. `exercises/student_exercises.md`
9. `references.md`

## Reproducibility

```bash
conda env create -f environment.yaml
conda activate knn-masterclass
python validate_module.py
```

The committed notebook contains rendered outputs, fixed random seeds, and an untouched final test evaluation.