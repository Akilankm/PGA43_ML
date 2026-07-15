from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SEED = 4317
PACKAGES = {
    "decision_tree": ROOT / "decision_tree_classification_masterclass_conda_executed" / "decision_tree_classification_masterclass",
    "random_forest": ROOT / "random_forest_classification_masterclass_conda_executed" / "random_forest_classification_masterclass",
}


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


COMMON_IMPORTS = r'''
from pathlib import Path
import json, math, platform, warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

RANDOM_STATE = 4317
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
plt.rcParams.update({"figure.figsize": (10, 5), "axes.grid": True, "figure.dpi": 110})
print({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__})
'''

COMMON_DATA = r'''
DATA_PATH = Path("data/loan_default_teaching.csv")
if not DATA_PATH.exists():
    subprocess_result = __import__("subprocess").run([sys.executable, "scripts/generate_teaching_data.py"], check=True)
df = pd.read_csv(DATA_PATH)
TARGET = "default"
ID_COL = "customer_id"
print("shape:", df.shape)
display(df.head())
'''

COMMON_AUDIT = r'''
audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_n": df.isna().sum(),
    "missing_pct": (100 * df.isna().mean()).round(2),
    "unique": df.nunique(dropna=False),
})
audit["role"] = np.where(audit.index == TARGET, "target", np.where(audit.index == ID_COL, "identifier", "predictor"))
display(audit)
print("duplicate rows:", int(df.duplicated().sum()))
print("duplicate ids:", int(df[ID_COL].duplicated().sum()))
print("target classes:", sorted(df[TARGET].unique().tolist()))
print("positive rate:", round(df[TARGET].mean(), 4))
assert set(df[TARGET].unique()) == {0, 1}
assert df[ID_COL].is_unique
'''

COMMON_EDA_NUM = r'''
model_df = df.drop(columns=ID_COL)
num_cols = [c for c in model_df.select_dtypes(include="number").columns if c != TARGET]
cat_cols = model_df.select_dtypes(exclude="number").columns.tolist()
summary = model_df[num_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
summary["skew"] = model_df[num_cols].skew()
display(summary.round(3))

for col in num_cols:
    values = model_df[col].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(values, bins=30, edgecolor="black")
    axes[0].axvline(values.median(), linestyle="--", label="median")
    axes[0].set_title(f"{col}: distribution"); axes[0].legend()
    model_df.boxplot(column=col, by=TARGET, ax=axes[1])
    axes[1].set_title(f"{col} by target"); axes[1].set_xlabel("default")
    sorted_v = np.sort(values)
    axes[2].plot(sorted_v, np.arange(1, len(sorted_v)+1)/len(sorted_v))
    axes[2].set_title(f"{col}: empirical CDF"); axes[2].set_ylabel("cumulative probability")
    plt.suptitle(""); plt.tight_layout(); plt.show()
'''

COMMON_EDA_CAT = r'''
for col in cat_cols:
    table = model_df.groupby(col, dropna=False)[TARGET].agg(default_rate="mean", n="size").sort_values("default_rate", ascending=False)
    display(table.style.format({"default_rate": "{:.2%}"}))
    ax = table["default_rate"].plot(kind="bar", ylim=(0, max(.2, table["default_rate"].max()*1.2)), title=f"Target rate by {col}")
    ax.axhline(model_df[TARGET].mean(), linestyle="--", label="overall rate")
    ax.set_ylabel("default rate"); ax.legend(); plt.tight_layout(); plt.show()
'''

COMMON_MULTI = r'''
corr = model_df[num_cols + [TARGET]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
fig.colorbar(im, ax=ax, label="Spearman correlation")
ax.set_title("Rank-correlation structure: association, not causation")
plt.tight_layout(); plt.show()

credit_bins = pd.qcut(model_df["credit_score"], 6, duplicates="drop")
debt_bins = pd.qcut(model_df["debt_ratio"], 6, duplicates="drop")
heat = model_df.assign(credit_bin=credit_bins, debt_bin=debt_bins).pivot_table(index="debt_bin", columns="credit_bin", values=TARGET, aggfunc="mean", observed=False)
fig, ax = plt.subplots(figsize=(11, 6)); im = ax.imshow(heat, aspect="auto", cmap="magma")
ax.set_xticks(range(heat.shape[1])); ax.set_xticklabels([str(x) for x in heat.columns], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(heat.shape[0])); ax.set_yticklabels([str(x) for x in heat.index], fontsize=8)
fig.colorbar(im, ax=ax, label="observed default rate")
ax.set_title("Interaction map: credit score × debt ratio")
plt.tight_layout(); plt.show()
'''

COMMON_SPLIT = r'''
X = model_df.drop(columns=TARGET)
y = model_df[TARGET].astype(int)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=.25, stratify=y_train_full, random_state=RANDOM_STATE)
num_features = X.select_dtypes(include="number").columns.tolist()
cat_features = X.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_features),
], verbose_feature_names_out=False)
print({"train": X_train.shape, "validation": X_valid.shape, "test": X_test.shape})
print("positive rates:", {"train": y_train.mean(), "validation": y_valid.mean(), "test": y_test.mean()})
'''

COMMON_METRICS = r'''
def metric_row(name, y_true, proba, threshold=.5):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "model": name, "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1-1e-6)),
        "brier": brier_score_loss(y_true, proba),
    }

def evaluate_model(name, model, X_ref=X_valid, y_ref=y_valid, threshold=.5):
    return metric_row(name, y_ref, model.predict_proba(X_ref)[:, 1], threshold)

def plot_classifier_diagnostics(model, X_ref, y_ref, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_ref, y_ref, ax=axes[0])
    RocCurveDisplay.from_estimator(model, X_ref, y_ref, ax=axes[1])
    PrecisionRecallDisplay.from_estimator(model, X_ref, y_ref, ax=axes[2])
    axes[0].set_title("Confusion matrix"); axes[1].set_title("ROC curve"); axes[2].set_title("Precision-recall curve")
    fig.suptitle(title); plt.tight_layout(); plt.show()
'''

COMMON_CALIBRATION = r'''
raw_proba = selected_model.predict_proba(X_valid)[:, 1]
calibrated_model = CalibratedClassifierCV(clone(selected_model), method="sigmoid", cv=2)
calibrated_model.fit(X_train, y_train)
cal_proba = calibrated_model.predict_proba(X_valid)[:, 1]
cal_table = pd.DataFrame([
    metric_row("raw", y_valid, raw_proba),
    metric_row("sigmoid_calibrated", y_valid, cal_proba),
])
display(cal_table[["model","roc_auc","average_precision","log_loss","brier"]])
fig, ax = plt.subplots(figsize=(7, 6))
CalibrationDisplay.from_predictions(y_valid, raw_proba, n_bins=8, name="raw", ax=ax)
CalibrationDisplay.from_predictions(y_valid, cal_proba, n_bins=8, name="calibrated", ax=ax)
ax.set_title("Reliability diagram: ranking and probability quality differ")
plt.tight_layout(); plt.show()
if cal_table.loc[1, "brier"] <= cal_table.loc[0, "brier"]:
    selected_probability_model = calibrated_model
else:
    selected_probability_model = selected_model
'''

COMMON_THRESHOLD = r'''
validation_proba = selected_probability_model.predict_proba(X_valid)[:, 1]
thresholds = np.linspace(.05, .80, 76)
rows = []
for t in thresholds:
    pred = (validation_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, pred).ravel()
    rows.append({"threshold": t, "precision": precision_score(y_valid,pred,zero_division=0), "recall": recall_score(y_valid,pred,zero_division=0), "f1": f1_score(y_valid,pred,zero_division=0), "cost": 5*fn + fp})
threshold_table = pd.DataFrame(rows)
chosen_threshold = float(threshold_table.loc[threshold_table["cost"].idxmin(), "threshold"])
print("validation-selected cost-sensitive threshold:", chosen_threshold)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
threshold_table.plot(x="threshold", y=["precision","recall","f1"], ax=axes[0])
threshold_table.plot(x="threshold", y="cost", ax=axes[1], legend=False)
axes[0].axvline(chosen_threshold, linestyle="--"); axes[1].axvline(chosen_threshold, linestyle="--")
axes[0].set_title("Threshold changes classification trade-offs"); axes[1].set_title("Operational cost: 5×FN + 1×FP")
plt.tight_layout(); plt.show()
'''

COMMON_SUBGROUP = r'''
valid_scored = X_valid.copy()
valid_scored["actual"] = y_valid.values
valid_scored["probability"] = selected_probability_model.predict_proba(X_valid)[:,1]
valid_scored["prediction"] = (valid_scored["probability"] >= chosen_threshold).astype(int)
for group_col in ["region", "channel", "home_ownership"]:
    rows=[]
    for group, part in valid_scored.groupby(group_col, dropna=False):
        if part["actual"].nunique() < 2: continue
        rows.append({"group": group, "n": len(part), "base_rate": part.actual.mean(), "roc_auc": roc_auc_score(part.actual, part.probability), "recall": recall_score(part.actual, part.prediction, zero_division=0), "precision": precision_score(part.actual, part.prediction, zero_division=0)})
    table = pd.DataFrame(rows).sort_values("roc_auc")
    print(group_col); display(table)
    table.set_index("group")[["roc_auc","recall","precision"]].plot(kind="bar", ylim=(0,1), title=f"Diagnostic variation by {group_col}")
    plt.tight_layout(); plt.show()
'''

COMMON_ROBUSTNESS = r'''
base = selected_probability_model.predict_proba(X_valid)[:,1]
perturbed = X_valid.copy()
perturbed["annual_income"] = perturbed["annual_income"] * 1.05
perturbed["loan_amount"] = perturbed["loan_amount"] * 1.05
perturbed["debt_ratio"] = np.clip(perturbed["debt_ratio"] + .02, 0, 1)
shifted = selected_probability_model.predict_proba(perturbed)[:,1]
delta = shifted - base
print(pd.Series(delta).describe(percentiles=[.01,.05,.5,.95,.99]))
plt.hist(delta, bins=30, edgecolor="black")
plt.axvline(0, linestyle="--"); plt.title("Prediction sensitivity under plausible input perturbation")
plt.xlabel("probability change"); plt.tight_layout(); plt.show()
'''

COMMON_FINAL = r'''
test_proba = selected_probability_model.predict_proba(X_test)[:,1]
test_metrics = metric_row("final_test", y_test, test_proba, chosen_threshold)
display(pd.DataFrame([test_metrics]))
plot_classifier_diagnostics(selected_probability_model, X_test, y_test, "Final untouched-test diagnostics")
print(classification_report(y_test, (test_proba >= chosen_threshold).astype(int), digits=3))

rng = np.random.default_rng(RANDOM_STATE)
boot=[]
for _ in range(300):
    idx = rng.integers(0, len(y_test), len(y_test))
    yt = y_test.to_numpy()[idx]; pt = test_proba[idx]
    if np.unique(yt).size < 2: continue
    boot.append((roc_auc_score(yt, pt), average_precision_score(yt, pt), brier_score_loss(yt, pt)))
boot = np.asarray(boot)
ci = pd.DataFrame({"metric":["roc_auc","average_precision","brier"], "estimate":[test_metrics["roc_auc"],test_metrics["average_precision"],test_metrics["brier"]], "ci_low":np.quantile(boot,.025,axis=0), "ci_high":np.quantile(boot,.975,axis=0)})
display(ci)
'''

COMMON_DEPLOY = r'''
ARTIFACT_DIR = Path("artifacts"); ARTIFACT_DIR.mkdir(exist_ok=True)
joblib.dump({"model": selected_probability_model, "threshold": chosen_threshold, "feature_columns": list(X.columns)}, ARTIFACT_DIR / ARTIFACT_NAME)

def predict_records(records: pd.DataFrame) -> pd.DataFrame:
    required = list(X.columns)
    missing = sorted(set(required) - set(records.columns))
    extra = sorted(set(records.columns) - set(required))
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if extra: print("Ignoring extra columns:", extra)
    clean = records[required].copy()
    proba = selected_probability_model.predict_proba(clean)[:,1]
    return clean.assign(default_probability=proba, predicted_default=(proba >= chosen_threshold).astype(int))

display(predict_records(X_test.head(6))[["default_probability","predicted_default"] + list(X.columns[:5])])
model_card = {
    "algorithm": ALGORITHM_NAME, "intended_use": "education only", "prohibited_use": "real lending or eligibility decisions",
    "data": "deterministic synthetic teaching data", "threshold": chosen_threshold, "test_metrics": test_metrics,
    "limitations": ["synthetic data", "noncausal interpretation", "performance may drift", "subgroup diagnostics are not fairness certification"],
}
Path("model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
print(json.dumps(model_card, indent=2))
'''


def common_prefix(title: str, algorithm_map: str):
    return [
        md(f"""# {title}\n\n## Notebook contract\nThis notebook is the **primary learning source**. Read it sequentially: every concept is derived, visualized, implemented, diagnosed, and interpreted. The dataset is deterministic synthetic teaching data; no result is evidence about real people."""),
        md(f"""## Learning architecture\n\n{algorithm_map}\n\nThe workflow separates **ranking**, **probability estimation**, and **decision policy**. A strong classifier may still be poorly calibrated, and a threshold is an operational choice—not a mathematical constant."""),
        md("## 1. Runtime and reproducibility"), code(COMMON_IMPORTS),
        md("## 2. Dataset provenance and first inspection"), code(COMMON_DATA),
        md("## 3. Schema, missingness, duplicates, identifiers, and target audit"), code(COMMON_AUDIT),
        md("## 4. Metric strategy under class imbalance\n\nAccuracy is reported but not trusted alone. ROC-AUC measures ranking, average precision emphasizes the minority class, Brier/log loss assess probability quality, and precision/recall/F1/MCC depend on a threshold."),
        md("## 5. Numerical EDA: distribution, outliers, class separation, ECDF"), code(COMMON_EDA_NUM),
        md("## 6. Categorical EDA: support and conditional target rates"), code(COMMON_EDA_CAT),
        md("## 7. Multivariate structure and interaction evidence"), code(COMMON_MULTI),
        md("## 8. Leakage-safe train/validation/test contract"), code(COMMON_SPLIT),
        md("## 9. Reusable evaluation functions"), code(COMMON_METRICS),
    ]


def decision_tree_notebook():
    cells = common_prefix("Decision Tree Classification — Exhaustive Visual Masterclass", "**Path:** impurity → candidate splits → recursive partitions → stopping/pruning → probability leaves → threshold policy.")
    cells += [
        md(r"""## 10. CART mechanics from first principles
For node class proportions $p_k$, Gini impurity is $1-\sum_kp_k^2$ and entropy is $-\sum_kp_k\log_2p_k$. A split is valuable only when the weighted child impurity is lower than the parent impurity. Misclassification error is less sensitive and is usually not the split criterion."""),
        code(r'''
p = np.linspace(.001,.999,400)
gini_curve = 2*p*(1-p)
entropy_curve = -(p*np.log2(p)+(1-p)*np.log2(1-p))
misclass_curve = np.minimum(p,1-p)
plt.plot(p,gini_curve,label="Gini"); plt.plot(p,entropy_curve,label="Entropy"); plt.plot(p,misclass_curve,label="Misclassification error")
plt.axvline(.5,linestyle="--"); plt.xlabel("positive-class proportion"); plt.ylabel("impurity"); plt.title("Impurity is maximal at maximum class uncertainty"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 11. Manual numerical split search and inference"),
        code(r'''
def gini(yv):
    counts=np.bincount(np.asarray(yv,dtype=int),minlength=2); p=counts/counts.sum(); return 1-np.sum(p*p)
def split_scan(feature):
    tmp=pd.DataFrame({"x":X_train[feature],"y":y_train}).dropna().sort_values("x")
    candidates=np.unique(np.quantile(tmp.x,np.linspace(.05,.95,80)))
    parent=gini(tmp.y); rows=[]
    for t in candidates:
        left=tmp[tmp.x<=t].y; right=tmp[tmp.x>t].y
        if min(len(left),len(right))<20: continue
        child=len(left)/len(tmp)*gini(left)+len(right)/len(tmp)*gini(right)
        rows.append({"threshold":t,"gain":parent-child,"left_n":len(left),"right_n":len(right),"left_rate":left.mean(),"right_rate":right.mean()})
    return pd.DataFrame(rows).sort_values("gain",ascending=False)
scan=split_scan("debt_ratio"); display(scan.head(10))
best=scan.iloc[0]
plt.plot(scan.threshold,scan.gain); plt.axvline(best.threshold,linestyle="--",label=f"best={best.threshold:.3f}")
plt.title("Every candidate threshold competes on weighted impurity reduction"); plt.xlabel("candidate threshold"); plt.ylabel("Gini gain"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 12. Manual categorical split logic\n\nCART partitions categories into two groups. One-hot encoding turns this into transparent yes/no membership tests. Rare categories require caution because high apparent gain may be sampling noise."),
        code(r'''
rates=X_train.assign(target=y_train.values).groupby("purpose").target.agg(["mean","count"]).sort_values("mean")
display(rates)
rates["mean"].plot(kind="bar",title="Categorical split intuition: ordered target rates")
plt.axhline(y_train.mean(),linestyle="--"); plt.ylabel("target rate"); plt.tight_layout(); plt.show()
'''),
        md("## 13. A decision stump from scratch"),
        code(r'''
class NumericDecisionStump:
    def fit(self, x, y):
        table=pd.DataFrame({"x":x,"y":y}).dropna(); self.fill_=table.x.median(); table.x=table.x.fillna(self.fill_)
        candidates=np.unique(np.quantile(table.x,np.linspace(.05,.95,100))); parent=gini(table.y); best=None
        for t in candidates:
            left=table[table.x<=t].y; right=table[table.x>t].y
            if min(len(left),len(right))<15: continue
            gain=parent-(len(left)*gini(left)+len(right)*gini(right))/len(table)
            if best is None or gain>best[0]: best=(gain,t,left.mean(),right.mean())
        self.gain_,self.threshold_,self.left_probability_,self.right_probability_=best; return self
    def predict_proba(self,x):
        x=pd.Series(x).fillna(self.fill_).to_numpy(); p=np.where(x<=self.threshold_,self.left_probability_,self.right_probability_); return np.c_[1-p,p]
stump=NumericDecisionStump().fit(X_train.credit_score,y_train)
print(vars(stump)); display(pd.DataFrame([metric_row("scratch_stump",y_valid,stump.predict_proba(X_valid.credit_score)[:,1])]))
'''),
        md("## 14. Baseline ladder: majority, logistic, shallow tree, unconstrained tree"),
        code(r'''
models={
 "logistic":Pipeline([("preprocess",preprocess),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))]),
 "tree_depth4":Pipeline([("preprocess",preprocess),("model",DecisionTreeClassifier(max_depth=4,min_samples_leaf=25,class_weight="balanced",random_state=RANDOM_STATE))]),
 "tree_unconstrained":Pipeline([("preprocess",preprocess),("model",DecisionTreeClassifier(class_weight="balanced",random_state=RANDOM_STATE))]),
}
rows=[]
majority=np.repeat(y_train.mode().iloc[0],len(y_valid)); majority_proba=np.repeat(y_train.mean(),len(y_valid)); rows.append(metric_row("majority",y_valid,majority_proba))
for name,model in models.items(): model.fit(X_train,y_train); rows.append(evaluate_model(name,model))
comparison=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); display(comparison)
comparison.set_index("model")[["roc_auc","average_precision","balanced_accuracy","f1"]].plot(kind="bar",ylim=(0,1),title="Baseline ladder")
plt.tight_layout(); plt.show()
'''),
        md("## 15. Capacity diagnostics: depth and minimum leaf size"),
        code(r'''
rows=[]
for depth in [1,2,3,4,5,6,8,12,None]:
 for leaf in [1,10,25,50]:
  m=Pipeline([("preprocess",preprocess),("model",DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight="balanced",random_state=RANDOM_STATE))])
  m.fit(X_train,y_train)
  rows.append({"depth":99 if depth is None else depth,"leaf":leaf,"train_auc":roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),"valid_auc":roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1])})
capacity=pd.DataFrame(rows)
for leaf,part in capacity.groupby("leaf"):
 plt.plot(part.depth,part.valid_auc,marker="o",label=f"leaf={leaf}")
plt.xlabel("max depth (99 = unlimited)"); plt.ylabel("validation ROC-AUC"); plt.title("Capacity surface: depth × minimum leaf size"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 16. Cost-complexity pruning path"),
        code(r'''
prep=clone(preprocess); Xt=prep.fit_transform(X_train); Xv=prep.transform(X_valid)
base_tree=DecisionTreeClassifier(class_weight="balanced",random_state=RANDOM_STATE).fit(Xt,y_train)
path=base_tree.cost_complexity_pruning_path(Xt,y_train)
alphas=np.unique(np.quantile(path.ccp_alphas[:-1],np.linspace(0,1,30)))
rows=[]
for a in alphas:
 t=DecisionTreeClassifier(ccp_alpha=float(a),class_weight="balanced",random_state=RANDOM_STATE).fit(Xt,y_train)
 rows.append({"alpha":a,"nodes":t.tree_.node_count,"depth":t.get_depth(),"train_auc":roc_auc_score(y_train,t.predict_proba(Xt)[:,1]),"valid_auc":roc_auc_score(y_valid,t.predict_proba(Xv)[:,1])})
prune=pd.DataFrame(rows); display(prune.sort_values("valid_auc",ascending=False).head())
fig,axes=plt.subplots(1,2,figsize=(14,4)); prune.plot(x="alpha",y=["train_auc","valid_auc"],ax=axes[0]); prune.plot(x="alpha",y=["nodes","depth"],secondary_y="depth",ax=axes[1]); axes[0].set_title("Pruning trades fit for generalization"); axes[1].set_title("Structural simplification"); plt.tight_layout(); plt.show()
'''),
        md("## 17. Cross-validated hyperparameter search"),
        code(r'''
base=Pipeline([("preprocess",preprocess),("model",DecisionTreeClassifier(class_weight="balanced",random_state=RANDOM_STATE))])
search=RandomizedSearchCV(base,{"model__criterion":["gini","entropy","log_loss"],"model__max_depth":[3,4,5,6,8,12,None],"model__min_samples_split":[2,10,25,50],"model__min_samples_leaf":[5,10,20,35,50],"model__max_features":[None,"sqrt",.7],"model__ccp_alpha":[0,.0005,.001,.002,.004]},n_iter=25,scoring="roc_auc",cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE),random_state=RANDOM_STATE,n_jobs=1,return_train_score=True)
search.fit(X_train_full,y_train_full); selected_model=search.best_estimator_
print("best CV AUC:",search.best_score_); print(search.best_params_)
cv_results=pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(15); display(cv_results[["mean_train_score","mean_test_score","std_test_score","params"]])
'''),
        md("## 18. Repeated-CV stability and learning curve"),
        code(r'''
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=RANDOM_STATE)
scores=cross_validate(selected_model,X_train_full,y_train_full,cv=cv,scoring={"auc":"roc_auc","ap":"average_precision","bal":"balanced_accuracy"},n_jobs=1)
stability=pd.DataFrame({k.replace("test_",""):v for k,v in scores.items() if k.startswith("test_")}); display(stability.describe().T)
stability.boxplot(); plt.title("Repeated-CV performance distribution"); plt.ylabel("score"); plt.tight_layout(); plt.show()
sizes,tr,va=learning_curve(selected_model,X_train_full,y_train_full,train_sizes=np.linspace(.3,1,3),cv=StratifiedKFold(2,shuffle=True,random_state=RANDOM_STATE),scoring="roc_auc",n_jobs=1)
plt.plot(sizes,tr.mean(1),marker="o",label="train"); plt.plot(sizes,va.mean(1),marker="o",label="CV"); plt.fill_between(sizes,va.mean(1)-va.std(1),va.mean(1)+va.std(1),alpha=.2); plt.xlabel("training rows"); plt.ylabel("ROC-AUC"); plt.title("Learning curve: data limitation vs model limitation"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 19. Tree structure, rules, local paths, and leaf audit"),
        code(r'''
selected_model.fit(X_train_full,y_train_full)
feature_names=selected_model.named_steps["preprocess"].get_feature_names_out(); tree=selected_model.named_steps["model"]
plt.figure(figsize=(22,11)); plot_tree(tree,feature_names=feature_names,class_names=["no default","default"],max_depth=3,filled=True,rounded=True,fontsize=8); plt.title("Top levels of selected tree"); plt.show()
print(export_text(tree,feature_names=list(feature_names),max_depth=4))
Xt_full=selected_model.named_steps["preprocess"].transform(X_train_full); Xv=selected_model.named_steps["preprocess"].transform(X_valid)
path=tree.decision_path(Xv[:3]); leaves=tree.apply(Xv[:3])
for i in range(3): print(f"record {i}: leaf={leaves[i]}, path nodes={path.indices[path.indptr[i]:path.indptr[i+1]].tolist()}")
leaf_id=tree.apply(Xt_full); leaf_table=pd.DataFrame({"leaf":leaf_id,"y":y_train_full.values}).groupby("leaf").y.agg(["size","mean"]).sort_values("size",ascending=False); display(leaf_table.head(15))
leaf_table["size"].hist(bins=25); plt.title("Leaf support distribution: tiny leaves signal variance"); plt.xlabel("samples per leaf"); plt.tight_layout(); plt.show()
'''),
        md("## 20. Global importance: impurity vs permutation"),
        code(r'''
mdi=pd.Series(tree.feature_importances_,index=feature_names,name="MDI").sort_values(ascending=False)
perm=permutation_importance(selected_model,X_valid,y_valid,scoring="roc_auc",n_repeats=8,random_state=RANDOM_STATE,n_jobs=1)
perm_s=pd.Series(perm.importances_mean,index=X_valid.columns,name="permutation_auc_drop").sort_values(ascending=False)
fig,axes=plt.subplots(1,2,figsize=(15,6)); mdi.head(15).sort_values().plot(kind="barh",ax=axes[0],title="Impurity importance"); perm_s.head(15).sort_values().plot(kind="barh",ax=axes[1],title="Permutation importance"); plt.tight_layout(); plt.show(); display(perm_s.to_frame())
'''),
        md("## 21. Partial dependence: conditional model response, not causality"),
        code(r'''
PartialDependenceDisplay.from_estimator(selected_model,X_valid,["credit_score","debt_ratio","loan_amount"],kind="both",subsample=200,random_state=RANDOM_STATE,grid_resolution=25)
plt.suptitle("PDP + ICE: average response can hide heterogeneous paths"); plt.tight_layout(); plt.show()
'''),
        md("## 22. Validation diagnostics before probability calibration"),
        code(r'''
display(pd.DataFrame([evaluate_model("selected_tree",selected_model)])); plot_classifier_diagnostics(selected_model,X_valid,y_valid,"Selected decision tree")
'''),
        md("## 23. Calibration"), code(COMMON_CALIBRATION),
        md("## 24. Validation-only threshold engineering"), code(COMMON_THRESHOLD),
        md("## 25. Subgroup diagnostics—not fairness certification"), code(COMMON_SUBGROUP),
        md("## 26. Robustness to plausible perturbations"), code(COMMON_ROBUSTNESS),
        md("## 27. Final untouched-test evaluation with bootstrap uncertainty"), code(COMMON_FINAL),
        md("## 28. Serialization, prediction contract, and model card"), code('ARTIFACT_NAME="decision_tree_classifier.joblib"\nALGORITHM_NAME="Calibrated pruned DecisionTreeClassifier"\n'+COMMON_DEPLOY),
        md("""## 29. Failure modes and remedies
- **High variance:** constrain depth/leaf size or prune; confirm with repeated CV.
- **Fragmented categories:** inspect support before trusting category splits.
- **Unstable probabilities:** calibrate using training folds only.
- **Misleading importance:** compare MDI with permutation and inspect correlated predictors.
- **Policy leakage:** choose thresholds only on validation, never on test.
- **Distribution shift:** monitor schema, missingness, base rate, calibration, and subgroup behavior."""),
        md("""## 30. Mastery exercises
1. Derive Gini and entropy for three different class proportions.
2. Hand-calculate weighted impurity for a candidate split.
3. Explain why an unconstrained tree can have perfect training accuracy and poor validation AUC.
4. Re-run pruning with a one-standard-error selection rule.
5. Compare Gini, entropy, and log-loss criteria.
6. Explain why one-hot encoded categories become binary membership rules.
7. Trace one prediction from root to leaf.
8. Compare MDI and permutation importance under correlated features.
9. Design a different false-negative/false-positive cost function.
10. Write monitoring thresholds for drift, calibration, and missingness."""),
    ]
    return notebook(cells, "Decision Tree Classification — Exhaustive Visual Masterclass")


def random_forest_notebook():
    cells = common_prefix("Random Forest Classification — Exhaustive Visual Masterclass", "**Path:** bootstrap samples + randomized feature subsets → diverse trees → probability averaging → OOB evidence → calibration and threshold policy.")
    cells += [
        md(r"""## 10. Why forests work
Bagging reduces variance by averaging unstable learners. If individual trees have variance $\sigma^2$ and average pairwise correlation $\rho$, ensemble variance is approximately $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. More trees reduce the second term; random feature subspaces aim to reduce correlation."""),
        md("## 11. Bootstrap sampling and the 63.2% rule"),
        code(r'''
rng=np.random.default_rng(RANDOM_STATE); n=len(X_train); unique_rates=[]
for b in range(400): unique_rates.append(len(np.unique(rng.integers(0,n,n)))/n)
print(pd.Series(unique_rates).describe())
plt.hist(unique_rates,bins=25,edgecolor="black"); plt.axvline(1-np.exp(-1),linestyle="--",label="1 - e^-1 ≈ 0.632"); plt.xlabel("unique fraction in bootstrap sample"); plt.title("Bootstrap samples leave out about 36.8% of observations"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 12. Correlation controls the benefit of averaging"),
        code(r'''
B=np.arange(1,301); sigma2=1
for rho in [0,.05,.15,.35,.7]: plt.plot(B,rho*sigma2+(1-rho)*sigma2/B,label=f"rho={rho}")
plt.xlabel("number of trees"); plt.ylabel("relative ensemble variance"); plt.title("More trees cannot eliminate correlated error"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 13. Mini-bagging from scratch"),
        code(r'''
prep_demo=clone(preprocess); Xt=prep_demo.fit_transform(X_train); Xv=prep_demo.transform(X_valid)
rng=np.random.default_rng(RANDOM_STATE); tree_prob=[]
for b in range(25):
 idx=rng.integers(0,len(y_train),len(y_train)); t=DecisionTreeClassifier(max_depth=5,min_samples_leaf=10,max_features="sqrt",random_state=RANDOM_STATE+b).fit(Xt[idx],y_train.to_numpy()[idx]); tree_prob.append(t.predict_proba(Xv)[:,1])
stack=np.vstack(tree_prob); cumulative=np.cumsum(stack,axis=0)/np.arange(1,len(stack)+1)[:,None]
auc=[roc_auc_score(y_valid,p) for p in cumulative]
plt.plot(range(1,len(auc)+1),auc,marker="o"); plt.xlabel("trees averaged"); plt.ylabel("validation ROC-AUC"); plt.title("From individual high-variance trees to a stable average"); plt.tight_layout(); plt.show()
'''),
        md("## 14. Baseline ladder: logistic, single tree, bagging, forest, extra trees"),
        code(r'''
models={
 "logistic":Pipeline([("preprocess",preprocess),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))]),
 "single_tree":Pipeline([("preprocess",preprocess),("model",DecisionTreeClassifier(max_depth=6,min_samples_leaf=15,class_weight="balanced",random_state=RANDOM_STATE))]),
 "bagging":Pipeline([("preprocess",preprocess),("model",BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8,min_samples_leaf=8,class_weight="balanced",random_state=RANDOM_STATE),n_estimators=30,random_state=RANDOM_STATE,n_jobs=1))]),
 "random_forest":Pipeline([("preprocess",preprocess),("model",RandomForestClassifier(n_estimators=50,min_samples_leaf=6,max_features="sqrt",class_weight="balanced_subsample",oob_score=True,random_state=RANDOM_STATE,n_jobs=1))]),
 "extra_trees":Pipeline([("preprocess",preprocess),("model",ExtraTreesClassifier(n_estimators=50,min_samples_leaf=6,max_features="sqrt",class_weight="balanced",random_state=RANDOM_STATE,n_jobs=1))]),
}
rows=[]
for name,m in models.items(): m.fit(X_train,y_train); rows.append(evaluate_model(name,m))
comparison=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); display(comparison)
comparison.set_index("model")[["roc_auc","average_precision","balanced_accuracy","f1"]].plot(kind="bar",ylim=(0,1),title="Baseline ladder")
plt.tight_layout(); plt.show()
print("forest OOB score:",models["random_forest"].named_steps["model"].oob_score_)
'''),
        md("## 15. OOB and tree-count convergence"),
        code(r'''
rows=[]
for ntrees in [10,30,60,100]:
 m=Pipeline([("preprocess",preprocess),("model",RandomForestClassifier(n_estimators=ntrees,min_samples_leaf=6,max_features="sqrt",class_weight="balanced_subsample",oob_score=True,random_state=RANDOM_STATE,n_jobs=1))]); m.fit(X_train,y_train)
 rows.append({"trees":ntrees,"oob_accuracy":m.named_steps["model"].oob_score_,"valid_auc":roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1]),"valid_ap":average_precision_score(y_valid,m.predict_proba(X_valid)[:,1])})
convergence=pd.DataFrame(rows); display(convergence)
convergence.plot(x="trees",y=["oob_accuracy","valid_auc","valid_ap"],marker="o",title="OOB and validation convergence"); plt.tight_layout(); plt.show()
'''),
        md("## 16. Random feature subspaces and tree correlation"),
        code(r'''
rows=[]
for mf in [1,.25,"sqrt",.75,1.0]:
 m=Pipeline([("preprocess",preprocess),("model",RandomForestClassifier(n_estimators=40,max_features=mf,min_samples_leaf=6,class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=1))]); m.fit(X_train,y_train)
 rf=m.named_steps["model"]; Xv_t=m.named_steps["preprocess"].transform(X_valid); probs=np.vstack([t.predict_proba(Xv_t)[:,1] for t in rf.estimators_[:20]])
 corr=np.corrcoef(probs); mean_corr=corr[np.triu_indices_from(corr,1)].mean()
 rows.append({"max_features":str(mf),"mean_tree_correlation":mean_corr,"validation_auc":roc_auc_score(y_valid,probs.mean(0))})
subspace=pd.DataFrame(rows); display(subspace)
fig,ax1=plt.subplots(figsize=(9,4)); ax1.plot(subspace.max_features,subspace.validation_auc,marker="o",label="AUC"); ax2=ax1.twinx(); ax2.plot(subspace.max_features,subspace.mean_tree_correlation,marker="s",linestyle="--",label="correlation"); ax1.set_ylabel("validation AUC"); ax2.set_ylabel("mean tree correlation"); ax1.set_title("Random subspaces trade tree strength for diversity"); plt.tight_layout(); plt.show()
'''),
        md("## 17. Capacity controls: minimum leaf size and row subsampling"),
        code(r'''
rows=[]
for leaf in [2,8,20]:
 for max_samples in [None,.75]:
  m=Pipeline([("preprocess",preprocess),("model",RandomForestClassifier(n_estimators=40,min_samples_leaf=leaf,max_features="sqrt",max_samples=max_samples,class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=1))]); m.fit(X_train,y_train)
  rows.append({"leaf":leaf,"max_samples":str(max_samples),"train_auc":roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),"valid_auc":roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1])})
capacity=pd.DataFrame(rows); display(capacity.sort_values("valid_auc",ascending=False).head(10))
for label,part in capacity.groupby("max_samples"): plt.plot(part.leaf,part.valid_auc,marker="o",label=f"max_samples={label}")
plt.xscale("log"); plt.xlabel("min samples leaf"); plt.ylabel("validation ROC-AUC"); plt.title("Forest regularization surface"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 18. Cross-validated hyperparameter search"),
        code(r'''
base=Pipeline([("preprocess",preprocess),("model",RandomForestClassifier(class_weight="balanced_subsample",oob_score=True,random_state=RANDOM_STATE,n_jobs=1))])
search=RandomizedSearchCV(base,{"model__n_estimators":[40,60,80],"model__max_depth":[None,6,10,16],"model__min_samples_leaf":[2,4,8,16,25],"model__max_features":["sqrt","log2",.35,.6],"model__max_samples":[None,.7,.9],"model__criterion":["gini","entropy","log_loss"]},n_iter=6,scoring="roc_auc",cv=StratifiedKFold(3,shuffle=True,random_state=RANDOM_STATE),random_state=RANDOM_STATE,n_jobs=1,return_train_score=True)
search.fit(X_train_full,y_train_full); selected_model=search.best_estimator_
print("best CV AUC:",search.best_score_); print(search.best_params_); print("selected OOB:",selected_model.named_steps["model"].oob_score_)
display(pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(12)[["mean_train_score","mean_test_score","std_test_score","params"]])
'''),
        md("## 19. Cross-validation stability and learning curve"),
        code(r'''
cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE)
scores=cross_validate(selected_model,X_train_full,y_train_full,cv=cv,scoring={"auc":"roc_auc","ap":"average_precision","bal":"balanced_accuracy"},n_jobs=1)
stability=pd.DataFrame({k.replace("test_",""):v for k,v in scores.items() if k.startswith("test_")}); display(stability.describe().T)
stability.boxplot(); plt.title("Fold-to-fold stability"); plt.tight_layout(); plt.show()
sizes,tr,va=learning_curve(selected_model,X_train_full,y_train_full,train_sizes=np.linspace(.3,1,3),cv=StratifiedKFold(2,shuffle=True,random_state=RANDOM_STATE),scoring="roc_auc",n_jobs=1)
plt.plot(sizes,tr.mean(1),marker="o",label="train"); plt.plot(sizes,va.mean(1),marker="o",label="CV"); plt.fill_between(sizes,va.mean(1)-va.std(1),va.mean(1)+va.std(1),alpha=.2); plt.xlabel("training rows"); plt.ylabel("ROC-AUC"); plt.title("Learning curve"); plt.legend(); plt.tight_layout(); plt.show()
'''),
        md("## 20. Tree diversity, vote dispersion, and local uncertainty"),
        code(r'''
selected_model.fit(X_train_full,y_train_full); rf=selected_model.named_steps["model"]; Xv_t=selected_model.named_steps["preprocess"].transform(X_valid)
tree_prob=np.vstack([t.predict_proba(Xv_t)[:,1] for t in rf.estimators_]); vote_sd=tree_prob.std(0); mean_prob=tree_prob.mean(0)
fig,axes=plt.subplots(1,2,figsize=(14,4)); axes[0].hist(vote_sd,bins=30,edgecolor="black"); axes[0].set_title("Tree-vote dispersion"); axes[1].scatter(mean_prob,vote_sd,s=10,alpha=.5); axes[1].set_xlabel("ensemble probability"); axes[1].set_ylabel("tree-vote SD"); axes[1].set_title("Disagreement is often largest near ambiguous regions"); plt.tight_layout(); plt.show()
for idx in np.argsort(vote_sd)[-3:]:
 plt.hist(tree_prob[:,idx],bins=15,range=(0,1),edgecolor="black"); plt.axvline(mean_prob[idx],linestyle="--"); plt.title(f"Per-tree probability distribution for validation row {idx}"); plt.xlabel("tree probability"); plt.tight_layout(); plt.show()
'''),
        md("## 21. Importance: impurity vs permutation"),
        code(r'''
feature_names=selected_model.named_steps["preprocess"].get_feature_names_out(); mdi=pd.Series(rf.feature_importances_,index=feature_names).sort_values(ascending=False)
perm=permutation_importance(selected_model,X_valid,y_valid,scoring="roc_auc",n_repeats=8,random_state=RANDOM_STATE,n_jobs=1); perm_s=pd.Series(perm.importances_mean,index=X_valid.columns).sort_values(ascending=False)
fig,axes=plt.subplots(1,2,figsize=(15,6)); mdi.head(15).sort_values().plot(kind="barh",ax=axes[0],title="Impurity importance"); perm_s.head(15).sort_values().plot(kind="barh",ax=axes[1],title="Permutation importance"); plt.tight_layout(); plt.show(); display(perm_s.to_frame("AUC drop"))
'''),
        md("## 22. Partial dependence: global response summaries"),
        code(r'''
PartialDependenceDisplay.from_estimator(selected_model,X_valid,["credit_score","debt_ratio","loan_amount"],kind="both",subsample=200,random_state=RANDOM_STATE,grid_resolution=25)
plt.suptitle("PDP + ICE: forest response is nonlinear and heterogeneous"); plt.tight_layout(); plt.show()
'''),
        md("## 23. Validation diagnostics"), code(r'''display(pd.DataFrame([evaluate_model("selected_forest",selected_model)])); plot_classifier_diagnostics(selected_model,X_valid,y_valid,"Selected random forest")'''),
        md("## 24. Calibration"), code(COMMON_CALIBRATION),
        md("## 25. Validation-only threshold engineering"), code(COMMON_THRESHOLD),
        md("## 26. Subgroup diagnostics—not fairness certification"), code(COMMON_SUBGROUP),
        md("## 27. Robustness to plausible perturbations"), code(COMMON_ROBUSTNESS),
        md("## 28. Final untouched-test evaluation with bootstrap uncertainty"), code(COMMON_FINAL),
        md("## 29. Serialization, prediction contract, and model card"), code('ARTIFACT_NAME="random_forest_classifier.joblib"\nALGORITHM_NAME="Calibrated RandomForestClassifier"\n'+COMMON_DEPLOY),
        md("""## 30. Failure modes and remedies
- **Correlated trees:** reduce `max_features`, diversify samples, or revisit dominant predictors.
- **Overconfident probabilities:** use out-of-fold calibration and monitor reliability.
- **Too few trees:** inspect metric and OOB convergence.
- **Opaque local behavior:** inspect per-tree votes, PDP/ICE, and representative cases.
- **Biased importance:** compare impurity and permutation methods.
- **Compute inflation:** tune tree count only until convergence; parallelize carefully in production.
- **Shift:** monitor schema, OOB/test gap, calibration, base rate, and subgroup performance."""),
        md("""## 31. Mastery exercises
1. Derive the 63.2% bootstrap uniqueness approximation.
2. Explain the variance-correlation formula in your own words.
3. Implement majority-vote bagging from scratch.
4. Compare bagging, random forest, and extra trees.
5. Measure how `max_features` changes tree correlation.
6. Explain why OOB is useful but does not replace a final test set.
7. Compare impurity and permutation importance.
8. Analyze three records with high tree-vote dispersion.
9. Design a threshold policy for a new cost ratio.
10. Write a monitoring plan for drift, calibration, latency, and model size."""),
    ]
    return notebook(cells, "Random Forest Classification — Exhaustive Visual Masterclass")


def notebook(cells, title):
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata={"kernelspec":{"display_name":f"Python ({title})","language":"python","name":"python3"},"language_info":{"name":"python","pygments_lexer":"ipython3"}}
    return nb


def write_runner(pkg: Path, nb_name: str, min_code: int, min_figures: int, min_chapters: int):
    scripts=pkg/"scripts"; scripts.mkdir(exist_ok=True)
    (scripts/"reexecute_notebook.py").write_text(f'''from pathlib import Path\nimport nbformat\nfrom nbclient import NotebookClient\nROOT=Path(__file__).resolve().parents[1]; NB=ROOT/"{nb_name}"\nnb=nbformat.read(NB,as_version=4); NotebookClient(nb,timeout=1800,kernel_name="python3",resources={{"metadata":{{"path":str(ROOT)}}}}).execute(); nbformat.write(nb,NB); print("executed",NB)\n''')
    (scripts/"render_html.py").write_text(f'''from pathlib import Path\nimport nbformat\nfrom nbconvert import HTMLExporter\nROOT=Path(__file__).resolve().parents[1]; NB=ROOT/"{nb_name}"; HTML=ROOT/"{nb_name.replace('.ipynb','.html')}"\nbody,_=HTMLExporter().from_notebook_node(nbformat.read(NB,as_version=4)); HTML.write_text(body,encoding="utf-8"); print("rendered",HTML)\n''')
    (scripts/"verify_package.py").write_text(f'''from pathlib import Path\nimport json,nbformat\nROOT=Path(__file__).resolve().parents[1]; NB=ROOT/"{nb_name}"; HTML=ROOT/"{nb_name.replace('.ipynb','.html')}"\nnb=nbformat.read(NB,as_version=4); code=[c for c in nb.cells if c.cell_type=="code"]; md=[c for c in nb.cells if c.cell_type=="markdown"]; errors=[o for c in code for o in c.get("outputs",[]) if o.get("output_type")=="error"]; unexecuted=[c for c in code if c.get("execution_count") is None]; figs=sum("image/png" in o.get("data",{{}}) for c in code for o in c.get("outputs",[])); chapters=sum(c.source.lstrip().startswith("## ") for c in md); required=[ROOT/"README.md",ROOT/"environment.yaml",ROOT/"requirements-lock.txt",ROOT/"data"/"loan_default_teaching.csv",NB,HTML,ROOT/"model_card.json"]; missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]; report={{"total_cells":len(nb.cells),"markdown_cells":len(md),"code_cells":len(code),"chapters":chapters,"unexecuted_code_cells":len(unexecuted),"error_outputs":len(errors),"embedded_png_figures":figs,"missing_required_files":missing,"minimum_code_cells":{min_code},"minimum_figures":{min_figures},"minimum_chapters":{min_chapters}}}; report["passes"]=not missing and not unexecuted and not errors and len(code)>={min_code} and figs>={min_figures} and chapters>={min_chapters}; (ROOT/"validation_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); raise SystemExit(0 if report["passes"] else 1)\n''')


def write_docs(pkg: Path, title: str, nb_name: str, report: dict):
    (pkg/"README.md").write_text(f"""# {title}\n\nThe executed notebook is the primary source of understanding. It contains derivations, visual intuition, exhaustive EDA, stable modeling techniques, diagnostics, interpretation, uncertainty, failure modes, and deployment contracts.\n\n## Start here\n- `{nb_name}` — executed canonical notebook with all outputs embedded.\n- `{nb_name.replace('.ipynb','.html')}` — self-contained rendered companion.\n\n## Reproduce\n```bash\nconda env create -f environment.yaml\npython scripts/reexecute_notebook.py\npython scripts/render_html.py\npython scripts/verify_package.py\n```\n\n## Validation\n- {report['total_cells']} total cells\n- {report['code_cells']} executed code cells\n- {report['chapters']} instructional chapters\n- {report['embedded_png_figures']} embedded figures\n- zero execution errors\n\nThe data are deterministic and synthetic. This is an educational artifact, not a real lending model.\n""")
    (pkg/"instructor_notes.md").write_text("""# Instructor notes\n\nTeach in notebook order. Require students to explain the inference after each visual, not merely run cells. Emphasize split isolation, validation-only selection, probability calibration, threshold policy, uncertainty, noncausal interpretation, and monitoring.\n""")
    (pkg/"student_exercises.md").write_text("""# Student exercises\n\nThe notebook concludes with ten mastery exercises. Students should submit calculations, plots, metric interpretation, error analysis, and a deployment/monitoring plan.\n""")


def build_package(kind: str):
    pkg=PACKAGES[kind]; pkg.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,"scripts/generate_teaching_data.py"],cwd=pkg,check=True)
    if kind=="decision_tree": nb=decision_tree_notebook(); nb_name="decision_tree_classification_masterclass.ipynb"; title="Decision Tree Classification — Exhaustive Visual Masterclass"; mins=(25,35,28)
    else: nb=random_forest_notebook(); nb_name="random_forest_classification_masterclass.ipynb"; title="Random Forest Classification — Exhaustive Visual Masterclass"; mins=(25,35,28)
    nbf.write(nb,pkg/nb_name); write_runner(pkg,nb_name,*mins)
    subprocess.run([sys.executable,"scripts/reexecute_notebook.py"],cwd=pkg,check=True)
    subprocess.run([sys.executable,"scripts/render_html.py"],cwd=pkg,check=True)
    subprocess.run([sys.executable,"scripts/verify_package.py"],cwd=pkg,check=True)
    report=json.loads((pkg/"validation_report.json").read_text()); write_docs(pkg,title,nb_name,report)
    subprocess.run([sys.executable,"scripts/verify_package.py"],cwd=pkg,check=True)


def main():
    for kind in PACKAGES: build_package(kind)
    print("Upgraded both tree masterclasses successfully.")


if __name__ == "__main__":
    main()
