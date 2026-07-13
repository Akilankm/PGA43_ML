# Decision Tree from Scratch — Play Tennis Masterclass

A calculation-first, fully executed teaching resource for constructing a categorical decision tree manually with the canonical 14-row Play Tennis dataset.

## Package contents

- `decision_tree_from_scratch_play_tennis.ipynb` — canonical executed notebook with embedded outputs, tables, LaTeX formulas, visual tree, and complete from-scratch ID3 implementation.
- `decision_tree_from_scratch_play_tennis.html` — static rendered companion for browser-based review.
- `data/play_tennis.csv` — source dataset used by every calculation.
- `environment.yaml` — reproducible Conda environment.
- `validation.json` — execution and structural validation metadata.

## Covered in depth

The notebook covers tree terminology, recursive partitioning, greedy split selection, entropy, Gini impurity, misclassification error, weighted child impurity, information gain, every candidate root split, the complete Sunny and Rain recursive calculations, final rule extraction, a reusable categorical ID3 implementation, deterministic tie-breaking, unseen-category fallback, inference tracing, numerical threshold splitting, pre-pruning, post-pruning, bias–variance trade-offs, leakage, high-cardinality overfitting, missing values, and generalization limitations.

## Create the environment

```bash
conda env create -f environment.yaml
conda activate decision-tree-from-scratch
```

## Run the notebook

```bash
jupyter lab decision_tree_from_scratch_play_tennis.ipynb
```

The committed notebook is already executed and contains rendered outputs. To reproduce them from this directory:

```bash
jupyter nbconvert \
  --to notebook \
  --execute decision_tree_from_scratch_play_tennis.ipynb \
  --output decision_tree_from_scratch_play_tennis.ipynb \
  --ExecutePreprocessor.cwd=. \
  --ExecutePreprocessor.timeout=600
```

The implementation deliberately does not use `sklearn.tree`. NumPy and pandas perform the calculations; Matplotlib and Seaborn provide the visualizations.
