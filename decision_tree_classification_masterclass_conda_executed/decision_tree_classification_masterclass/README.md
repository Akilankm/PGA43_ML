# Decision Tree Classification Masterclass

A fully executed, classroom-grade end-to-end classification package aligned with the repository's Linear Regression and Logistic Regression masterclass architecture.

## Start here

- `decision_tree_classification_masterclass.ipynb` — canonical executed notebook with tables, metrics, diagnostics, and embedded PNG figures.
- `decision_tree_classification_masterclass.html` — self-contained static rendering for GitHub, VS Code, and environments with inconsistent notebook rendering.

## Create the exact Conda environment

```bash
conda env create -f environment.yaml
conda activate decision-tree-masterclass
python -m ipykernel install --user --name decision-tree-masterclass --display-name "Python (Decision Tree Classification Masterclass)"
```

Use `conda env create -f environment.yaml`; do not use `conda create --file environment.yaml`.

## Re-execute, render, and validate

```bash
python scripts/generate_teaching_data.py
python scripts/reexecute_notebook.py
python scripts/render_html.py
python scripts/verify_package.py
```

The committed notebook and HTML are already executed. Regeneration is deterministic and requires no network access.

## Package layout

```text
decision_tree_classification_masterclass/
├── decision_tree_classification_masterclass.ipynb
├── decision_tree_classification_masterclass.html
├── README.md
├── environment.yaml
├── requirements.txt
├── requirements-lock.txt
├── data/
│   ├── loan_default_teaching.csv
│   ├── DATA_SOURCE.md
│   └── data_dictionary.csv
├── artifacts/
│   └── decision_tree_classifier.joblib
├── scripts/
│   ├── generate_teaching_data.py
│   ├── reexecute_notebook.py
│   ├── render_html.py
│   └── verify_package.py
├── instructor_notes.md
├── student_exercises.md
├── references.md
├── model_card.json
└── validation_report.json
```

## Analytical coverage

- Decision-tree vocabulary, recursive partitioning, leaf probabilities, entropy, Gini impurity, misclassification error, and weighted information gain;
- Manual Gini and split-gain calculation before library modeling;
- Schema, duplicate, missingness, range, cardinality, and class-balance audits;
- Per-feature numerical and categorical univariate analysis;
- Class-conditional box plots, category target-rate tables, and multivariate rank-correlation analysis;
- Stratified train/validation/test isolation before learned preprocessing;
- Median/mode imputation, one-hot encoding, majority baseline, and unpruned-tree baseline;
- Depth capacity curves exposing underfitting and overfitting;
- Three-fold cross-validated tuning of depth, leaf size, and class weighting;
- Cost-complexity pruning with validation-selected ccp_alpha;
- Fold stability, plotted tree structure, exported rules, and impurity/permutation importance;
- ROC, precision-recall, asymmetric-cost threshold engineering, untouched-test evaluation, subgroup checks, serialization, and model card.

## Reproducibility gate

- all 11/11 code cells executed;
- zero error outputs;
- 10 embedded PNG figures;
- 2,400 bundled deterministic rows;
- fixed random seed `4317`;
- single-process tuning and evaluation for stable classroom reproduction.

## Dataset and interpretation boundary

The loan-default dataset is synthetic and intentionally contains nonlinear thresholds, interactions, mixed data types, controlled missingness, imbalance, and asymmetric decision costs. It exists only to teach modeling mechanics. It is not evidence about customers and must never be used for lending, causal, fairness, or policy conclusions.
