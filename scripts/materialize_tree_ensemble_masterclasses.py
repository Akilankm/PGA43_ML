"""Materialize curated Decision Tree and Random Forest masterclass packages.

This script intentionally follows the repository's existing masterclass pattern:
- one canonical notebook per algorithm;
- deterministic local teaching data;
- Conda environment file;
- re-execution, HTML rendering and validation scripts;
- instructor notes, student exercises and references.

Run from the repository root:

    python scripts/materialize_tree_ensemble_masterclasses.py

Then validate each package:

    cd decision_tree_classification_masterclass_conda_executed/decision_tree_classification_masterclass
    python scripts/reexecute_notebook.py
    python scripts/render_html.py
    python scripts/verify_package.py

    cd ../../random_forest_classification_masterclass_conda_executed/random_forest_classification_masterclass
    python scripts/reexecute_notebook.py
    python scripts/render_html.py
    python scripts/verify_package.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 43

COMMON_ENV = """channels:\n  - conda-forge\ndependencies:\n  - python=3.11\n  - numpy=1.26.4\n  - pandas=2.2.2\n  - scikit-learn=1.5.1\n  - matplotlib=3.9.1\n  - scipy=1.14.0\n  - jupyter=1.0.0\n  - nbclient=0.10.0\n  - nbconvert=7.16.4\n  - ipykernel=6.29.5\n"""

REQUIREMENTS = """numpy==1.26.4\npandas==2.2.2\nscikit-learn==1.5.1\nmatplotlib==3.9.1\nscipy==1.14.0\nnbclient==0.10.0\nnbconvert==7.16.4\nipykernel==6.29.5\n"""

DATA_SOURCE = """# Data source and generation policy\n\n`credit_risk_teaching.csv` is generated deterministically by the canonical notebook. It is synthetic teaching data, not real credit, banking, lending or customer data. The target contains nonlinear threshold effects and interactions so decision trees and random forests have meaningful structure to learn.\n\nUse this dataset only for classroom instruction, algorithm diagnostics and reproducibility practice.\n"""

REFERENCES = """# References\n\n- Breiman, L. (1984). *Classification and Regression Trees*.\n- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5-32.\n- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1, 81-106.\n- Hastie, Tibshirani, and Friedman. *The Elements of Statistical Learning*.\n- scikit-learn User Guide: Decision Trees, Ensembles, Model Evaluation and Inspection.\n"""

EXERCISES = """# Student exercises\n\n1. Recalculate the Gini gain for another numeric feature and compare it with `debt_to_income`.\n2. Change the validation threshold from 0.50 to a cost-sensitive policy and report precision/recall movement.\n3. Run the notebook with shallower and deeper models and explain underfitting/overfitting evidence.\n4. Compare impurity importance with permutation importance.\n5. Write a model card explaining why the synthetic artifact must not be used for real lending decisions.\n"""

DATA_GENERATION_CODE = r'''
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 43
DATA_PATH = Path("data/credit_risk_teaching.csv")

def generate_credit_risk_data(path: Path, n: int = 1400, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(39, 12, n).round(), 18, 75).astype(int)
    annual_income = np.exp(rng.normal(10.75, 0.55, n)).round(0)
    loan_amount = np.exp(rng.normal(9.7, 0.65, n)).round(0)
    credit_score = np.clip(rng.normal(665, 80, n).round(), 350, 850).astype(int)
    employment_years = np.clip((age - 22) * rng.uniform(0.05, 0.65, n) + rng.normal(0, 2, n), 0, 40).round(1)
    delinquencies_2yr = rng.poisson(np.clip((700 - credit_score) / 180, 0.05, 2.5)).astype(int)
    debt_to_income = np.clip((loan_amount / np.maximum(annual_income, 1)) + rng.normal(0.18, 0.11, n), 0.02, 1.3)
    savings_ratio = np.clip(rng.beta(2.2, 7.5, n) + (credit_score - 650) / 1400, 0, 0.9)
    home_ownership = rng.choice(["rent", "mortgage", "own", "other"], p=[0.39, 0.37, 0.19, 0.05], size=n)
    purpose = rng.choice(["debt_consolidation", "home_improvement", "medical", "education", "small_business", "vehicle"], p=[0.34,0.18,0.14,0.13,0.11,0.10], size=n)
    region = rng.choice(["north", "south", "east", "west"], p=[0.24,0.28,0.22,0.26], size=n)
    prior_default = rng.binomial(1, np.clip(0.06 + delinquencies_2yr*0.08 + (credit_score < 600)*0.10, 0.01, 0.55))
    logit = (-3.35 + 2.25 * (debt_to_income > 0.48) + 1.45 * (credit_score < 610)
             + 0.95 * ((loan_amount > 23000) & (annual_income < 52000))
             + 0.85 * prior_default + 0.55 * (delinquencies_2yr >= 2)
             + 0.45 * np.isin(purpose, ["small_business", "medical"])
             + 0.35 * (home_ownership == "rent") - 0.65 * (savings_ratio > 0.28)
             - 0.45 * (employment_years > 8))
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({
        "age": age, "annual_income": annual_income, "loan_amount": loan_amount,
        "credit_score": credit_score, "employment_years": employment_years,
        "debt_to_income": debt_to_income.round(3), "savings_ratio": savings_ratio.round(3),
        "delinquencies_2yr": delinquencies_2yr, "prior_default": prior_default,
        "home_ownership": home_ownership, "purpose": purpose, "region": region,
        "default": default,
    })
    for col, frac in {"annual_income": 0.018, "employment_years": 0.025, "home_ownership": 0.012}.items():
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

df = generate_credit_risk_data(DATA_PATH)
print(df.shape)
df.head()
'''

EDA_CODE = r'''
display(df.dtypes.to_frame("dtype"))
display(df["default"].value_counts(normalize=True).rename("target_rate").to_frame())
display((df.isna().mean() * 100).round(2).rename("missing_pct").to_frame())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), ["credit_score", "debt_to_income", "loan_amount", "annual_income"]):
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Univariate distribution: {col}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["credit_score", "debt_to_income", "savings_ratio"]):
    df.boxplot(column=col, by="default", ax=ax)
    ax.set_title(f"{col} by default")
plt.suptitle("")
plt.tight_layout()
plt.show()

num = df.select_dtypes(include="number").columns.tolist()
cor = df[num].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cor, aspect="auto")
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=45, ha="right")
ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
fig.colorbar(im, ax=ax)
ax.set_title("Multivariate correlation map")
plt.tight_layout()
plt.show()

for col in ["home_ownership", "purpose", "region"]:
    display(df.groupby(col)["default"].agg(["mean", "count"]).sort_values("mean", ascending=False))
'''

SPLIT_CODE = r'''
X = df.drop(columns="default")
y = df["default"]
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE)

num_features = X.select_dtypes(include="number").columns.tolist()
cat_features = X.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_features),
])

def evaluate(name, model, X_ref, y_ref):
    pred = model.predict(X_ref)
    proba = model.predict_proba(X_ref)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_ref, pred),
        "balanced_accuracy": balanced_accuracy_score(y_ref, pred),
        "precision": precision_score(y_ref, pred, zero_division=0),
        "recall": recall_score(y_ref, pred, zero_division=0),
        "f1": f1_score(y_ref, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_ref, proba),
        "average_precision": average_precision_score(y_ref, proba),
        "log_loss": log_loss(y_ref, proba),
        "brier": brier_score_loss(y_ref, proba),
    }
print("train/valid/test:", X_train.shape, X_valid.shape, X_test.shape)
'''

VIS_CODE = r'''
results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
display(results_df)
best_model = models[results_df.iloc[0]["model"]]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ConfusionMatrixDisplay.from_estimator(best_model, X_valid, y_valid, ax=axes[0])
RocCurveDisplay.from_estimator(best_model, X_valid, y_valid, ax=axes[1])
PrecisionRecallDisplay.from_estimator(best_model, X_valid, y_valid, ax=axes[2])
plt.tight_layout(); plt.show()
CalibrationDisplay.from_estimator(best_model, X_valid, y_valid, n_bins=8)
plt.title("Validation calibration")
plt.show()
print(classification_report(y_valid, best_model.predict(X_valid), digits=3))
'''

FINAL_CODE = r'''
final_model = best_model
final = pd.DataFrame([evaluate("selected_model_test", final_model, X_test, y_test)])
display(final)
ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test)
plt.title("Untouched test confusion matrix")
plt.show()
perm = permutation_importance(final_model, X_valid, y_valid, scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE)
imp = pd.DataFrame({"feature": X_valid.columns, "permutation_importance_auc": perm.importances_mean}).sort_values("permutation_importance_auc", ascending=False)
display(imp)

def predict_default_risk(records: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    required = list(X_train.columns)
    missing = sorted(set(required) - set(records.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    proba = final_model.predict_proba(records[required])[:, 1]
    return records.assign(default_probability=proba, predicted_default=(proba >= threshold).astype(int))

display(predict_default_risk(X_test.head(8))[["default_probability", "predicted_default"] + list(X_test.columns[:5])])
'''


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook(kind: str) -> nbf.NotebookNode:
    is_rf = kind == "random_forest"
    title = "Random Forest Classification Masterclass" if is_rf else "Decision Tree Classification Masterclass"
    cells = [
        md(f"""# {title}\n\nA complete end-to-end tree-based classification workflow following the Linear and Logistic Regression masterclass structure: deterministic data, EDA, leakage-safe split, baselines, diagnostics, tuning, interpretation, final test evaluation and model card boundaries."""),
        md("""## Learning objectives\n\n- audit tabular classification data before modeling;\n- isolate train/validation/test data before learned transformations;\n- understand impurity, splitting, variance and overfitting;\n- tune tree-based hyperparameters using validation data;\n- evaluate with ROC, PR, confusion matrix, calibration and final test metrics;\n- communicate interpretation limits clearly."""),
        code(DATA_GENERATION_CODE),
        md("## Data audit and EDA"),
        code(EDA_CODE),
        md("## Leakage-safe split and preprocessing contract"),
        code(SPLIT_CODE),
    ]
    if is_rf:
        cells += [
            md("""## Random forest mechanics\n\nA forest averages many high-variance trees trained on bootstrap samples and random feature subsets. This usually reduces variance compared with a single tree."""),
            code(r'''
rng = np.random.default_rng(RANDOM_STATE)
for i in range(3):
    sample_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    print(f"Bootstrap {i + 1}: unique_rows={len(np.unique(sample_idx))}, oob_rate={1 - len(np.unique(sample_idx)) / len(X_train):.2%}")
models = {
    "logistic_baseline": Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "single_tree": Pipeline([("preprocess", preprocess), ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE))]),
    "random_forest": Pipeline([("preprocess", preprocess), ("model", RandomForestClassifier(n_estimators=120, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", oob_score=True, n_jobs=1, random_state=RANDOM_STATE))]),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(evaluate(name, model, X_valid, y_valid))
print("OOB score:", models["random_forest"].named_steps["model"].oob_score_)
'''),
            md("## Validation comparison"), code(VIS_CODE),
            md("## Tuning tree count and leaf constraints"),
            code(r'''
param_grid = {"model__n_estimators": [100, 150], "model__min_samples_leaf": [8, 16]}
search = GridSearchCV(models["random_forest"], param_grid, scoring="roc_auc", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE), n_jobs=1)
search.fit(X_train_full, y_train_full)
print("Best CV ROC-AUC:", search.best_score_)
print("Best params:", search.best_params_)
models["random_forest_tuned"] = search.best_estimator_
best_model = search.best_estimator_
'''),
        ]
    else:
        cells += [
            md("""## Decision-tree mechanics from scratch\n\nThe hand calculation scores candidate thresholds by weighted Gini impurity reduction before using scikit-learn for the production implementation."""),
            code(r'''
def gini(y_values):
    p = np.bincount(np.asarray(y_values), minlength=2) / len(y_values)
    return 1 - np.sum(p ** 2)
root_gini = gini(df["default"])
manual = []
for t in df["debt_to_income"].quantile([0.25, 0.5, 0.75]).round(3):
    left = df[df["debt_to_income"] <= t]["default"]
    right = df[df["debt_to_income"] > t]["default"]
    weighted = len(left) / len(df) * gini(left) + len(right) / len(df) * gini(right)
    manual.append({"feature": "debt_to_income", "threshold": t, "weighted_gini": weighted, "gini_gain": root_gini - weighted})
display(pd.DataFrame(manual).sort_values("gini_gain", ascending=False))
'''),
            md("## Baselines and decision-tree models"),
            code(r'''
models = {
    "logistic_baseline": Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "decision_tree_depth_4": Pipeline([("preprocess", preprocess), ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE))]),
    "decision_tree_unconstrained": Pipeline([("preprocess", preprocess), ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    results.append(evaluate(name, model, X_valid, y_valid))
'''),
            md("## Validation comparison"), code(VIS_CODE),
            md("## Tree visualization and rule extraction"),
            code(r'''
shallow = models["decision_tree_depth_4"]
feature_names = shallow.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(20, 10))
plot_tree(shallow.named_steps["model"], feature_names=feature_names, class_names=["non_default", "default"], max_depth=3, fontsize=8)
plt.title("Regularized decision tree, top levels")
plt.show()
print(export_text(shallow.named_steps["model"], feature_names=list(feature_names), max_depth=3))
'''),
            md("## Cost-complexity and depth tuning"),
            code(r'''
base = Pipeline([("preprocess", preprocess), ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))])
param_grid = {"model__max_depth": [3, 4, 6, None], "model__min_samples_leaf": [20, 40], "model__ccp_alpha": [0.0, 0.001, 0.003]}
search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), n_jobs=1)
search.fit(X_train_full, y_train_full)
print("Best CV ROC-AUC:", search.best_score_)
print("Best params:", search.best_params_)
models["decision_tree_tuned"] = search.best_estimator_
best_model = search.best_estimator_
'''),
        ]
    cells += [md("## Final untouched-test evaluation and interpretation safeguards"), code(FINAL_CODE), md("""## Key takeaways\n\nTree-based models capture nonlinear interactions but can overfit. Validation design, pruning/regularization, calibration checks and interpretation boundaries matter more than visual appeal. The final model is a reproducible teaching artifact, not a real credit-risk system.""")]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": f"Python ({title})", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
    return nb


def write_supporting_scripts(pkg: Path, notebook_name: str, package_name: str, min_figures: int = 4) -> None:
    scripts = pkg / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "reexecute_notebook.py").write_text(f'''from pathlib import Path\nimport nbformat\nfrom nbclient import NotebookClient\nROOT = Path(__file__).resolve().parents[1]\nNB = ROOT / "{notebook_name}"\nnb = nbformat.read(NB, as_version=4)\nNotebookClient(nb, timeout=900, kernel_name="python3", resources={{"metadata": {{"path": str(ROOT)}}}}).execute()\nnbformat.write(nb, NB)\nprint(f"re-executed {{NB}}")\n''', encoding="utf-8")
    (scripts / "render_html.py").write_text(f'''from pathlib import Path\nimport nbformat\nfrom nbconvert import HTMLExporter\nROOT = Path(__file__).resolve().parents[1]\nNB = ROOT / "{notebook_name}"\nHTML = ROOT / "{notebook_name.replace('.ipynb', '.html')}"\nbody, _ = HTMLExporter().from_notebook_node(nbformat.read(NB, as_version=4))\nHTML.write_text(body, encoding="utf-8")\nprint(f"rendered {{HTML}}")\n''', encoding="utf-8")
    (scripts / "verify_package.py").write_text(f'''from pathlib import Path\nimport json\nimport nbformat\nROOT = Path(__file__).resolve().parents[1]\nNB = ROOT / "{notebook_name}"\nHTML = ROOT / "{notebook_name.replace('.ipynb', '.html')}"\nrequired = [ROOT / "README.md", ROOT / "environment.yaml", ROOT / "requirements-lock.txt", ROOT / "data" / "DATA_SOURCE.md", NB, HTML]\nmissing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]\nnb = nbformat.read(NB, as_version=4)\ncode_cells = [c for c in nb.cells if c.cell_type == "code"]\nunexecuted = [c for c in code_cells if c.get("execution_count") is None]\nerrors = [o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]\nfigures = sum("image/png" in o.get("data", {{}}) for c in code_cells for o in c.get("outputs", []))\nreport = {{"package": "{package_name}", "code_cells": len(code_cells), "unexecuted_code_cells": len(unexecuted), "error_outputs": len(errors), "embedded_png_figures": figures, "missing_required_files": missing, "passes": not missing and not unexecuted and not errors and figures >= {min_figures}}}\n(ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")\nprint(json.dumps(report, indent=2))\nraise SystemExit(0 if report["passes"] else 1)\n''', encoding="utf-8")


def materialize_package(folder: str, kind: str, title: str, env_name: str, coverage: str, notes: str) -> None:
    pkg = ROOT / folder
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "data").mkdir(exist_ok=True)
    notebook_name = f"{kind}_classification_masterclass.ipynb"
    nbf.write(build_notebook("random_forest" if kind == "random_forest" else "decision_tree"), pkg / notebook_name)
    (pkg / "README.md").write_text(f"""# {title}\n\nA complete classroom-grade masterclass package generated by `scripts/materialize_tree_ensemble_masterclasses.py`.\n\n## Start here\n\n- `{notebook_name}` — canonical notebook.\n- `{notebook_name.replace('.ipynb', '.html')}` — rendered companion after running `python scripts/render_html.py`.\n\n## Create the exact Conda environment\n\n```bash\nconda env create -f environment.yaml\nconda activate {env_name}\npython -m ipykernel install --user --name {env_name} --display-name \"Python ({title})\"\n```\n\n## Re-execute, render and validate\n\n```bash\npython scripts/reexecute_notebook.py\npython scripts/render_html.py\npython scripts/verify_package.py\n```\n\n## Analytical coverage\n\n{coverage}\n\n## Reproducibility boundary\n\nThe dataset is synthetic and deterministic. The artifact is for teaching; it must not be used for real lending, banking, pricing or eligibility decisions.\n""", encoding="utf-8")
    (pkg / "environment.yaml").write_text(f"name: {env_name}\n" + COMMON_ENV, encoding="utf-8")
    (pkg / "requirements-lock.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (pkg / "data" / "DATA_SOURCE.md").write_text(DATA_SOURCE, encoding="utf-8")
    (pkg / "references.md").write_text(REFERENCES, encoding="utf-8")
    (pkg / "student_exercises.md").write_text(EXERCISES, encoding="utf-8")
    (pkg / "instructor_notes.md").write_text(notes, encoding="utf-8")
    write_supporting_scripts(pkg, notebook_name, Path(folder).name)


def main() -> None:
    materialize_package(
        "decision_tree_classification_masterclass_conda_executed/decision_tree_classification_masterclass",
        "decision_tree",
        "Decision Tree Classification Masterclass",
        "decision-tree-classification-masterclass",
        """- synthetic-data generation and schema audit;\n- univariate, bivariate and multivariate EDA;\n- manual Gini and split-gain calculation;\n- leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`;\n- logistic baseline, shallow tree, unconstrained tree and tuned tree comparison;\n- tree visualization, cost-complexity/depth tuning, final test evaluation and model-card boundary.""",
        """# Instructor notes - Decision Tree\n\nTeach this module after logistic regression. Emphasize impurity, recursive partitioning, overfitting, pruning and interpretation limits.""",
    )
    materialize_package(
        "random_forest_classification_masterclass_conda_executed/random_forest_classification_masterclass",
        "random_forest",
        "Random Forest Classification Masterclass",
        "random-forest-classification-masterclass",
        """- synthetic-data generation and schema audit;\n- bootstrap and OOB mechanics;\n- leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`;\n- logistic baseline, single-tree baseline and random-forest comparison;\n- hyperparameter tuning, permutation importance, final test evaluation and model-card boundary.""",
        """# Instructor notes - Random Forest\n\nTeach this after the decision-tree masterclass. Focus on variance reduction, bootstrap sampling, feature subspacing, OOB validation and the transparency trade-off.""",
    )
    print("Materialized decision-tree and random-forest masterclass packages.")


if __name__ == "__main__":
    main()
