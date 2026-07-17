# Gradient Boosting Masterclass — From Functional Gradients to Production Diagnostics

This package is the canonical Gradient Boosting learning path in `PGA43_ML`. It mirrors the repository's linear, logistic, decision-tree, and random-forest masterclasses: one executed notebook is the primary source of understanding, the data are committed locally, and a rendered HTML companion is included.

## What the notebook teaches

- supervised-learning placement: additive ensembles, bagging versus boosting, and weak learners;
- functional gradient descent and stage-wise additive modeling;
- squared-error derivation where the negative gradient is the residual;
- binary log-loss derivation where the pseudo-residual is `y - p`;
- manual stage calculations and prediction decomposition;
- from-scratch regression and binary-classification implementations;
- scikit-learn `GradientBoostingRegressor` and `GradientBoostingClassifier`;
- leakage-safe train/validation/test protocol;
- learning-rate/estimator trade-offs, depth, leaf size, subsampling, and early stopping;
- regression residual diagnostics, classification discrimination, calibration, threshold policy, and decision cost;
- permutation importance, partial dependence, model limitations, drift, monitoring, and deployment gates.

## Structure

```text
gradient_boosting_masterclass/
├── README.md
├── environment.yaml
├── data/
│   ├── diabetes_regression.csv
│   └── breast_cancer_classification.csv
├── notebooks/
│   └── gradient_boosting_masterclass.ipynb
├── rendered/
│   └── gradient_boosting_masterclass.html
├── src/
│   ├── gradient_boosting_from_scratch.py
│   └── library_pipeline.py
├── artifacts/
│   ├── metrics_summary.csv
│   └── validation_report.json
├── exercises/student_exercises.md
├── theory/technical_notes.md
└── scripts/verify_package.py
```

## Reproduce

```bash
conda env create -f environment.yaml
conda activate pga43-gradient-boosting
jupyter lab
```

Open `notebooks/gradient_boosting_masterclass.ipynb` and run all cells from the package root. The committed notebook is already executed and contains all outputs.

Validate the package:

```bash
python scripts/verify_package.py
```

## Dataset policy

The committed CSV files are deterministic snapshots of scikit-learn's diabetes regression and breast-cancer classification teaching datasets. They are used strictly for education and benchmarking, not clinical decision-making.
