from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

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


COMMON_SETUP = r'''
# Expert enrichment runtime: intentionally self-contained so this chapter can be studied independently.
from pathlib import Path
import math, warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 4317
rng_expert = np.random.default_rng(RANDOM_STATE)
DATA_PATH = Path("data/loan_default_teaching.csv")
expert_df = pd.read_csv(DATA_PATH)
TARGET = "default"
ID_COL = "customer_id"
expert_X = expert_df.drop(columns=[TARGET, ID_COL])
expert_y = expert_df[TARGET].astype(int)
X_dev, X_test_expert, y_dev, y_test_expert = train_test_split(
    expert_X, expert_y, test_size=0.20, stratify=expert_y, random_state=RANDOM_STATE
)
X_train_expert, X_valid_expert, y_train_expert, y_valid_expert = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=RANDOM_STATE
)
num_expert = expert_X.select_dtypes(include="number").columns.tolist()
cat_expert = expert_X.select_dtypes(exclude="number").columns.tolist()
expert_preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_expert),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_expert),
], verbose_feature_names_out=False)

expert_logit = Pipeline([
    ("preprocess", expert_preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
])
expert_tree = Pipeline([
    ("preprocess", expert_preprocess),
    ("model", DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=25, min_samples_split=50,
        class_weight="balanced", random_state=RANDOM_STATE,
    )),
])
expert_forest = Pipeline([
    ("preprocess", expert_preprocess),
    ("model", RandomForestClassifier(
        n_estimators=320, min_samples_leaf=6, max_features="sqrt",
        class_weight="balanced_subsample", oob_score=True,
        n_jobs=1, random_state=RANDOM_STATE,
    )),
])
for model in (expert_logit, expert_tree, expert_forest):
    model.fit(X_train_expert, y_train_expert)


def metric_row(name, model, X, y, threshold=0.5):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z*z/n
    centre = (phat + z*z/(2*n)) / denom
    half = z * np.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / denom
    return centre-half, centre+half


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def paired_bootstrap_auc(y, p_a, p_b, n_boot=500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    return np.asarray(diffs)

print("Expert enrichment split:", X_train_expert.shape, X_valid_expert.shape, X_test_expert.shape)
display(pd.DataFrame([
    metric_row("logistic", expert_logit, X_valid_expert, y_valid_expert),
    metric_row("decision_tree", expert_tree, X_valid_expert, y_valid_expert),
    metric_row("random_forest", expert_forest, X_valid_expert, y_valid_expert),
]).set_index("model").round(4))
'''


COMMON_STATISTICAL = r'''
# Statistical evidence map: association strength, uncertainty and multiplicity control.
stat_rows = []
for col in num_expert:
    a = expert_df.loc[expert_df[TARGET] == 0, col].dropna()
    b = expert_df.loc[expert_df[TARGET] == 1, col].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 2*u/(len(a)*len(b)) - 1
    point_r, point_p = stats.pointbiserialr(expert_df[TARGET], expert_df[col].fillna(expert_df[col].median()))
    stat_rows.append({"feature": col, "type": "numeric", "test": "Mann-Whitney U",
                      "p_value": p, "effect_size": rank_biserial,
                      "secondary_effect": point_r, "secondary_p": point_p})
for col in cat_expert:
    table = pd.crosstab(expert_df[col].fillna("<MISSING>"), expert_df[TARGET])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.values.sum()
    phi2 = chi2/n
    r, k = table.shape
    cramer_v = np.sqrt(phi2 / max(1, min(k-1, r-1)))
    stat_rows.append({"feature": col, "type": "categorical", "test": "Chi-square",
                      "p_value": p, "effect_size": cramer_v,
                      "secondary_effect": np.nan, "secondary_p": np.nan})
stat_table = pd.DataFrame(stat_rows)
stat_table["q_value_bh"] = bh_fdr(stat_table["p_value"])
stat_table["evidence_strength"] = -np.log10(stat_table["q_value_bh"].clip(lower=1e-12))
display(stat_table.sort_values(["q_value_bh", "effect_size"], ascending=[True, False]).round(5))

plt.figure(figsize=(11, 6))
ordered = stat_table.sort_values("evidence_strength")
plt.barh(ordered["feature"], ordered["evidence_strength"])
plt.axvline(-np.log10(0.05), linestyle="--", label="FDR q=0.05")
plt.xlabel("-log10(Benjamini-Hochberg q-value)")
plt.title("Feature-level statistical evidence after multiplicity control")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(11, 6))
ordered = stat_table.reindex(stat_table["effect_size"].abs().sort_values().index)
plt.barh(ordered["feature"], ordered["effect_size"])
plt.axvline(0, linewidth=1)
plt.xlabel("Effect size: rank-biserial (numeric) or Cramér's V (categorical)")
plt.title("Association magnitude is different from statistical significance")
plt.tight_layout(); plt.show()

strong = stat_table.sort_values("evidence_strength", ascending=False).iloc[0]
print(
    f"Inference: {strong.feature} has the strongest multiplicity-adjusted evidence in this synthetic sample. "
    "This is associational evidence, not a causal claim; tree models may still prioritize interaction-capable features differently."
)
'''


COMMON_RATE_INTERVALS = r'''
# Target-rate uncertainty: Wilson intervals prevent over-reading noisy bins or small categories.
rate_feature = "debt_ratio"
work = expert_df[[rate_feature, TARGET]].dropna().copy()
work["bin"] = pd.qcut(work[rate_feature], q=10, duplicates="drop")
rows = []
for label, group in work.groupby("bin", observed=True):
    lo, hi = wilson_interval(group[TARGET].sum(), len(group))
    rows.append({"bin": str(label), "mid": group[rate_feature].mean(), "rate": group[TARGET].mean(),
                 "low": lo, "high": hi, "n": len(group)})
rate_ci = pd.DataFrame(rows)
display(rate_ci.round(4))
plt.figure(figsize=(11, 5))
plt.errorbar(rate_ci["mid"], rate_ci["rate"],
             yerr=[rate_ci["rate"]-rate_ci["low"], rate_ci["high"]-rate_ci["rate"]],
             marker="o", capsize=4)
plt.xlabel(rate_feature); plt.ylabel("Observed target rate with 95% Wilson interval")
plt.title("Binned risk pattern with sampling uncertainty")
plt.tight_layout(); plt.show()

cat_feature = "purpose"
cat_rows = []
for label, group in expert_df.groupby(cat_feature, dropna=False):
    lo, hi = wilson_interval(group[TARGET].sum(), len(group))
    cat_rows.append({"category": str(label), "rate": group[TARGET].mean(), "low": lo, "high": hi, "n": len(group)})
cat_ci = pd.DataFrame(cat_rows).sort_values("rate")
plt.figure(figsize=(10, 5))
plt.errorbar(cat_ci["rate"], cat_ci["category"],
             xerr=[cat_ci["rate"]-cat_ci["low"], cat_ci["high"]-cat_ci["rate"]],
             fmt="o", capsize=4)
plt.xlabel("Observed target rate with 95% Wilson interval")
plt.title("Category risk estimates: magnitude, support and uncertainty")
plt.tight_layout(); plt.show()
print("Inference: intervals widen when support is limited. A high raw rate without adequate support is weak evidence and should not drive a split or policy alone.")
'''


COMMON_DECISION_ANALYSIS = r'''
# Decision-curve and threshold analysis: ranking, probability quality and operational policy are distinct.
p_valid = selected_expert_model.predict_proba(X_valid_expert)[:, 1]
thresholds = np.linspace(0.05, 0.80, 76)
rows = []
prevalence = y_valid_expert.mean()
for t in thresholds:
    pred = p_valid >= t
    tp = np.sum((pred == 1) & (y_valid_expert.values == 1))
    fp = np.sum((pred == 1) & (y_valid_expert.values == 0))
    fn = np.sum((pred == 0) & (y_valid_expert.values == 1))
    tn = np.sum((pred == 0) & (y_valid_expert.values == 0))
    n = len(y_valid_expert)
    net_benefit = tp/n - fp/n * (t/(1-t))
    treat_all = prevalence - (1-prevalence) * (t/(1-t))
    expected_cost = 5*fn + fp
    rows.append({"threshold": t, "net_benefit_model": net_benefit,
                 "net_benefit_all": treat_all, "net_benefit_none": 0,
                 "expected_cost": expected_cost, "recall": tp/max(1,tp+fn),
                 "precision": tp/max(1,tp+fp)})
policy = pd.DataFrame(rows)
best_t = policy.loc[policy["expected_cost"].idxmin(), "threshold"]
display(policy.loc[policy["expected_cost"].nsmallest(10).index].round(4))

plt.figure(figsize=(10, 5))
plt.plot(policy["threshold"], policy["net_benefit_model"], label="model")
plt.plot(policy["threshold"], policy["net_benefit_all"], label="treat all")
plt.plot(policy["threshold"], policy["net_benefit_none"], label="treat none")
plt.xlabel("Decision threshold"); plt.ylabel("Net benefit")
plt.title("Decision-curve analysis")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(policy["threshold"], policy["expected_cost"], label="5×FN + 1×FP")
plt.axvline(best_t, linestyle="--", label=f"selected threshold={best_t:.2f}")
plt.xlabel("Decision threshold"); plt.ylabel("Validation decision cost")
plt.title("Threshold is a policy choice, not an intrinsic model property")
plt.legend(); plt.tight_layout(); plt.show()
print(f"Inference: under the illustrated 5:1 cost ratio, the validation-selected threshold is {best_t:.2f}; a different business cost ratio would justify a different threshold.")
'''


COMMON_CALIBRATION = r'''
# Calibration with bootstrap uncertainty bands and expected calibration error (ECE).
def calibration_summary(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({"bin": b, "count": mask.sum(), "pred": p[mask].mean(), "obs": np.asarray(y)[mask].mean()})
    out = pd.DataFrame(rows)
    ece = np.sum(out["count"]/len(y) * np.abs(out["pred"]-out["obs"]))
    return out, ece

cal, ece = calibration_summary(y_valid_expert.values, p_valid, 10)
boot_curves = []
for _ in range(300):
    idx = rng_expert.integers(0, len(y_valid_expert), len(y_valid_expert))
    c, _ = calibration_summary(y_valid_expert.values[idx], p_valid[idx], 10)
    if len(c) == len(cal):
        boot_curves.append(c["obs"].values)
boot_curves = np.asarray(boot_curves)
cal["low"] = np.quantile(boot_curves, 0.025, axis=0)
cal["high"] = np.quantile(boot_curves, 0.975, axis=0)
display(cal.round(4))
plt.figure(figsize=(7, 6))
plt.plot([0,1], [0,1], linestyle="--", label="perfect calibration")
plt.errorbar(cal["pred"], cal["obs"], yerr=[cal["obs"]-cal["low"], cal["high"]-cal["obs"]], marker="o", capsize=4, label="model")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
plt.title(f"Reliability diagram with bootstrap bands (ECE={ece:.3f})")
plt.legend(); plt.tight_layout(); plt.show()
print("Inference: discrimination can be strong while probability calibration remains imperfect. Use calibration diagnostics before treating scores as absolute risk estimates.")
'''


COMMON_MODEL_COMPARISON = r'''
# Paired statistical comparison: same observations, paired bootstrap AUC and exact McNemar test.
p_logit = expert_logit.predict_proba(X_test_expert)[:, 1]
p_selected = selected_expert_model.predict_proba(X_test_expert)[:, 1]
diffs = paired_bootstrap_auc(y_test_expert.values, p_selected, p_logit, n_boot=700)
ci = np.quantile(diffs, [0.025, 0.975])
plt.figure(figsize=(9, 5))
plt.hist(diffs, bins=35)
plt.axvline(0, linestyle="--")
plt.axvline(ci[0], linestyle=":"); plt.axvline(ci[1], linestyle=":")
plt.xlabel("Paired bootstrap ROC-AUC difference: selected model − logistic")
plt.title("Sampling uncertainty in model comparison")
plt.tight_layout(); plt.show()

pred_a = p_selected >= best_t
pred_b = p_logit >= best_t
correct_a = pred_a == y_test_expert.values
correct_b = pred_b == y_test_expert.values
b = int(np.sum(correct_a & ~correct_b))
c = int(np.sum(~correct_a & correct_b))
mc_p = stats.binomtest(min(b,c), n=b+c, p=0.5, alternative="two-sided").pvalue if (b+c) else 1.0
comparison = pd.DataFrame({
    "statistic": ["mean paired AUC difference", "95% CI low", "95% CI high", "P(diff>0)", "McNemar exact p"],
    "value": [diffs.mean(), ci[0], ci[1], np.mean(diffs>0), mc_p],
})
display(comparison.round(5))
print(
    "Inference: the paired bootstrap quantifies ranking uncertainty, while McNemar tests disagreement in thresholded errors. "
    "Neither test proves universal superiority beyond this data-generating process."
)
'''


COMMON_STRESS = r'''
# Robustness and distribution-shift stress tests.
def stress_copy(X, scenario):
    out = X.copy()
    scenario_seed = {
        "5% extra numeric missingness": 101,
        "10% numeric noise": 202,
        "unseen categories": 303,
        "high-risk covariate shift": 404,
    }.get(scenario, 0)
    rng = np.random.default_rng(RANDOM_STATE + scenario_seed)
    if scenario == "5% extra numeric missingness":
        for col in num_expert:
            idx = rng.choice(out.index, max(1, int(0.05*len(out))), replace=False)
            out.loc[idx, col] = np.nan
    elif scenario == "10% numeric noise":
        for col in num_expert:
            sd = pd.to_numeric(out[col], errors="coerce").std()
            out[col] = pd.to_numeric(out[col], errors="coerce") + rng.normal(0, 0.10*sd, len(out))
    elif scenario == "unseen categories":
        for col in cat_expert:
            out.loc[out.sample(frac=0.08, random_state=RANDOM_STATE).index, col] = "<UNSEEN>"
    elif scenario == "high-risk covariate shift":
        if "debt_ratio" in out: out["debt_ratio"] = np.clip(out["debt_ratio"]*1.25, 0, 1)
        if "credit_score" in out: out["credit_score"] = out["credit_score"] - 35
    return out

stress_rows = []
for scenario in ["clean", "5% extra numeric missingness", "10% numeric noise", "unseen categories", "high-risk covariate shift"]:
    Xs = X_test_expert if scenario == "clean" else stress_copy(X_test_expert, scenario)
    p = selected_expert_model.predict_proba(Xs)[:, 1]
    stress_rows.append({"scenario": scenario, "roc_auc": roc_auc_score(y_test_expert, p),
                        "average_precision": average_precision_score(y_test_expert, p),
                        "brier": brier_score_loss(y_test_expert, p), "mean_score": p.mean()})
stress = pd.DataFrame(stress_rows)
display(stress.round(4))
plt.figure(figsize=(11, 5))
x = np.arange(len(stress))
plt.plot(x, stress["roc_auc"], marker="o", label="ROC-AUC")
plt.plot(x, stress["average_precision"], marker="o", label="Average precision")
plt.plot(x, stress["brier"], marker="o", label="Brier score")
plt.xticks(x, stress["scenario"], rotation=25, ha="right")
plt.title("Robustness profile under controlled input stress")
plt.legend(); plt.tight_layout(); plt.show()
print("Inference: stress tests are not forecasts of production drift; they reveal directional sensitivities that monitoring and retraining policy must cover.")
'''


TREE_SURFACE = r'''
# Visual geometry of recursive partitioning in a two-feature slice.
features_2d = ["credit_score", "debt_ratio"]
slice_df = expert_df[features_2d + [TARGET]].dropna()
X2 = slice_df[features_2d]
y2 = slice_df[TARGET]
tree_2d = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE).fit(X2, y2)
x0 = np.linspace(X2.iloc[:,0].quantile(.01), X2.iloc[:,0].quantile(.99), 220)
x1 = np.linspace(X2.iloc[:,1].quantile(.01), X2.iloc[:,1].quantile(.99), 220)
xx, yy = np.meshgrid(x0, x1)
grid = pd.DataFrame({features_2d[0]: xx.ravel(), features_2d[1]: yy.ravel()})
zz = tree_2d.predict_proba(grid)[:,1].reshape(xx.shape)
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, zz, levels=np.linspace(0,1,11), alpha=.75)
plt.scatter(X2.iloc[:,0], X2.iloc[:,1], c=y2, s=9, alpha=.35)
plt.colorbar(label="Predicted target probability")
plt.xlabel(features_2d[0]); plt.ylabel(features_2d[1])
plt.title("Axis-aligned decision regions created by recursive splits")
plt.tight_layout(); plt.show()

plt.figure(figsize=(15, 7))
plot_tree(tree_2d, feature_names=features_2d, class_names=["0","1"], filled=True, rounded=True, fontsize=8)
plt.title("The tree representation of the same geometric partition")
plt.tight_layout(); plt.show()
print("Inference: the rectangular regions explain both the interpretability and the limitation of standard trees: oblique relationships require many staircase-like splits.")
'''


TREE_SPLIT_STABILITY = r'''
# Bootstrap split stability: how often does the root feature and threshold repeat?
Xt = expert_tree.named_steps["preprocess"].transform(X_train_expert)
feature_names = expert_tree.named_steps["preprocess"].get_feature_names_out()
root_rows = []
for b in range(120):
    idx = rng_expert.integers(0, len(Xt), len(Xt))
    m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE+b)
    m.fit(Xt[idx], y_train_expert.values[idx])
    root_idx = m.tree_.feature[0]
    root_rows.append({"feature": feature_names[root_idx], "threshold": m.tree_.threshold[0]})
root_stability = pd.DataFrame(root_rows)
freq = root_stability["feature"].value_counts().head(12).sort_values()
display(freq.rename("bootstrap_root_count").to_frame())
plt.figure(figsize=(10, 5))
plt.barh(freq.index, freq.values)
plt.xlabel("Bootstrap selections as root split")
plt.title("Root-split stability across 120 bootstrap samples")
plt.tight_layout(); plt.show()

leading = freq.index[-1]
plt.figure(figsize=(9, 5))
plt.hist(root_stability.loc[root_stability["feature"] == leading, "threshold"], bins=20)
plt.xlabel("Transformed threshold"); plt.ylabel("Bootstrap frequency")
plt.title(f"Threshold uncertainty when root feature is {leading}")
plt.tight_layout(); plt.show()
print("Inference: a visually simple tree can hide substantial structural uncertainty. Repeated split selection is stronger evidence than a single fitted tree.")
'''


TREE_PRUNING = r'''
# Cost-complexity pruning with the one-standard-error rule.
Xt = expert_tree.named_steps["preprocess"].fit_transform(X_train_expert)
path = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE).cost_complexity_pruning_path(Xt, y_train_expert)
alphas = np.unique(path.ccp_alphas)
if len(alphas) > 40:
    alphas = np.quantile(alphas, np.linspace(0, 1, 40))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
for alpha in alphas:
    model = DecisionTreeClassifier(ccp_alpha=float(alpha), class_weight="balanced", min_samples_leaf=10, random_state=RANDOM_STATE)
    scores = cross_val_score(model, Xt, y_train_expert, scoring="roc_auc", cv=cv, n_jobs=1)
    model.fit(Xt, y_train_expert)
    rows.append({"alpha": alpha, "mean_auc": scores.mean(), "se_auc": scores.std(ddof=1)/np.sqrt(len(scores)),
                 "nodes": model.tree_.node_count, "depth": model.tree_.max_depth})
prune = pd.DataFrame(rows).sort_values("alpha")
best_idx = prune["mean_auc"].idxmax()
cutoff = prune.loc[best_idx, "mean_auc"] - prune.loc[best_idx, "se_auc"]
one_se = prune[prune["mean_auc"] >= cutoff].sort_values("nodes").iloc[0]
display(prune.sort_values("mean_auc", ascending=False).head(10).round(5))
plt.figure(figsize=(10, 5))
plt.errorbar(prune["alpha"], prune["mean_auc"], yerr=prune["se_auc"], marker="o", markersize=3)
plt.axhline(cutoff, linestyle="--", label="one-SE cutoff")
plt.axvline(one_se["alpha"], linestyle=":", label=f"simplest eligible α={one_se['alpha']:.4g}")
plt.xscale("symlog", linthresh=1e-5)
plt.xlabel("ccp_alpha"); plt.ylabel("Cross-validated ROC-AUC")
plt.title("Pruning selection with uncertainty, not point-estimate chasing")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(prune["alpha"], prune["nodes"], marker="o", label="nodes")
plt.plot(prune["alpha"], prune["depth"], marker="o", label="depth")
plt.xscale("symlog", linthresh=1e-5)
plt.xlabel("ccp_alpha"); plt.title("Complexity path under pruning")
plt.legend(); plt.tight_layout(); plt.show()
print("Inference: the one-SE rule deliberately trades negligible validation performance for a simpler and usually more stable tree.")
'''


TREE_LEAF_AUDIT = r'''
# Leaf-level reliability: support, observed rate, raw probability and Laplace smoothing.
Xt_leaf = expert_tree.named_steps["preprocess"].transform(X_valid_expert)
leaf_ids = expert_tree.named_steps["model"].apply(Xt_leaf)
p_tree = expert_tree.predict_proba(X_valid_expert)[:,1]
leaf_table = pd.DataFrame({"leaf": leaf_ids, "y": y_valid_expert.values, "raw_p": p_tree}).groupby("leaf").agg(
    n=("y","size"), events=("y","sum"), observed_rate=("y","mean"), raw_probability=("raw_p","mean")
).reset_index()
intervals = leaf_table.apply(lambda r: wilson_interval(r.events, r.n), axis=1)
leaf_table["low"] = [x[0] for x in intervals]
leaf_table["high"] = [x[1] for x in intervals]
leaf_table["laplace_probability"] = (leaf_table["events"] + 1) / (leaf_table["n"] + 2)
display(leaf_table.sort_values("n").round(4))

plt.figure(figsize=(11, 6))
plot_df = leaf_table.sort_values("observed_rate")
plt.errorbar(plot_df["observed_rate"], np.arange(len(plot_df)),
             xerr=[plot_df["observed_rate"]-plot_df["low"], plot_df["high"]-plot_df["observed_rate"]], fmt="o", capsize=3)
plt.scatter(plot_df["raw_probability"], np.arange(len(plot_df)), marker="x", label="model leaf probability")
plt.yticks(np.arange(len(plot_df)), plot_df["leaf"])
plt.xlabel("Probability / observed rate"); plt.ylabel("Leaf id")
plt.title("Leaf reliability with Wilson uncertainty intervals")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(leaf_table["raw_probability"], leaf_table["laplace_probability"], s=np.sqrt(leaf_table["n"])*18)
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("Raw leaf probability"); plt.ylabel("Laplace-smoothed probability")
plt.title("Small leaves receive the largest probability shrinkage")
plt.tight_layout(); plt.show()
print("Inference: leaf probabilities are sample proportions. Their uncertainty and support should be audited before using them as risk estimates.")
'''


TREE_LOCAL = r'''
# Local decision-path explanation and feasible one-feature counterfactual search.
row_index = int(np.argmax(expert_tree.predict_proba(X_valid_expert)[:,1]))
row = X_valid_expert.iloc[[row_index]]
transformed = expert_tree.named_steps["preprocess"].transform(row)
model = expert_tree.named_steps["model"]
feature_names = expert_tree.named_steps["preprocess"].get_feature_names_out()
node_indicator = model.decision_path(transformed)
leaf_id = model.apply(transformed)[0]
path_nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
records = []
parent_p = y_train_expert.mean()
for node in path_nodes:
    value = model.tree_.value[node][0]
    node_p = value[1] / value.sum()
    feat_idx = model.tree_.feature[node]
    if feat_idx >= 0:
        threshold = model.tree_.threshold[node]
        actual = transformed[0, feat_idx]
        direction = "≤" if actual <= threshold else ">"
        rule = f"{feature_names[feat_idx]} {direction} {threshold:.3f}"
    else:
        rule = "leaf"
    records.append({"node": node, "rule": rule, "probability": node_p, "delta": node_p-parent_p})
    parent_p = node_p
path_table = pd.DataFrame(records)
display(path_table.round(4))
plt.figure(figsize=(11, 5))
plt.bar(np.arange(len(path_table)), path_table["delta"])
plt.axhline(0, linewidth=1)
plt.xticks(np.arange(len(path_table)), path_table["rule"], rotation=35, ha="right")
plt.ylabel("Change in node probability")
plt.title("Local path waterfall: how each split changes the conditional risk")
plt.tight_layout(); plt.show()

base_p = expert_tree.predict_proba(row)[0,1]
candidates = []
for col in num_expert:
    for q in np.linspace(.05,.95,19):
        candidate = row.copy()
        candidate[col] = expert_df[col].quantile(q)
        p = expert_tree.predict_proba(candidate)[0,1]
        scale = expert_df[col].std() or 1
        distance = abs(float(candidate[col].iloc[0]) - float(row[col].iloc[0]))/scale
        candidates.append({"feature": col, "new_value": float(candidate[col].iloc[0]), "probability": p, "standardized_change": distance})
cf = pd.DataFrame(candidates)
cf["crosses_0_5"] = (cf["probability"] >= .5) != (base_p >= .5)
cf_best = cf[cf["crosses_0_5"]].sort_values("standardized_change").head(12)
display(cf_best.round(4))
plt.figure(figsize=(10, 5))
show = cf.sort_values("probability").groupby("feature").head(1).sort_values("probability")
plt.barh(show["feature"], show["probability"])
plt.axvline(base_p, linestyle="--", label=f"original={base_p:.3f}")
plt.axvline(.5, linestyle=":", label="classification threshold")
plt.xlabel("Lowest probability reached by one-feature quantile search")
plt.title("Counterfactual sensitivity is diagnostic, not prescriptive recourse")
plt.legend(); plt.tight_layout(); plt.show()
print("Inference: local paths are exact for the fitted tree. Counterfactual searches show model sensitivity, but they do not establish feasibility, ethics or causal effect.")
'''


FOREST_MARGIN = r'''
# Forest margins and per-tree uncertainty.
forest_model = expert_forest.named_steps["model"]
Xt_valid = expert_forest.named_steps["preprocess"].transform(X_valid_expert)
tree_prob = np.vstack([t.predict_proba(Xt_valid)[:,1] for t in forest_model.estimators_])
mean_prob = tree_prob.mean(axis=0)
std_prob = tree_prob.std(axis=0)
margin = np.where(y_valid_expert.values == 1, 2*mean_prob-1, 1-2*mean_prob)
plt.figure(figsize=(10, 5))
plt.hist(margin[y_valid_expert.values==0], bins=35, alpha=.65, label="true class 0")
plt.hist(margin[y_valid_expert.values==1], bins=35, alpha=.65, label="true class 1")
plt.axvline(0, linestyle="--")
plt.xlabel("Correct-class vote margin")
plt.title("Margin distribution: confidence and error severity")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(mean_prob, std_prob, c=(margin<0), alpha=.65)
plt.xlabel("Mean forest probability"); plt.ylabel("Across-tree probability standard deviation")
plt.title("Ensemble disagreement is highest near ambiguous regions")
plt.tight_layout(); plt.show()

uncertain = pd.DataFrame({"mean_probability": mean_prob, "tree_sd": std_prob, "margin": margin, "target": y_valid_expert.values}, index=X_valid_expert.index)
display(uncertain.sort_values("tree_sd", ascending=False).head(15).round(4))
print("Inference: tree disagreement is a model-based uncertainty diagnostic. It is not a calibrated posterior interval and can remain low under shared model bias.")
'''


FOREST_PROXIMITY = r'''
# Random-forest proximity: observations are similar when they land in the same leaves.
sample_n = min(180, len(X_valid_expert))
sample_idx = rng_expert.choice(np.arange(len(X_valid_expert)), sample_n, replace=False)
X_sample_t = Xt_valid[sample_idx]
y_sample = y_valid_expert.values[sample_idx]
leaves = np.vstack([t.apply(X_sample_t) for t in forest_model.estimators_]).T
proximity = np.mean(leaves[:,None,:] == leaves[None,:,:], axis=2)
plt.figure(figsize=(8, 7))
plt.imshow(proximity, aspect="auto")
plt.colorbar(label="Fraction of trees sharing a leaf")
plt.title("Forest proximity matrix on a validation sample")
plt.tight_layout(); plt.show()

embedding = MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_STATE, normalized_stress="auto", n_init=2, max_iter=250).fit_transform(1-proximity)
plt.figure(figsize=(9, 6))
plt.scatter(embedding[:,0], embedding[:,1], c=y_sample, s=35, alpha=.75)
plt.xlabel("MDS dimension 1"); plt.ylabel("MDS dimension 2")
plt.title("Two-dimensional map induced by forest leaf co-occurrence")
plt.tight_layout(); plt.show()
print("Inference: proximity reveals the representation learned by the ensemble. Cluster separation supports discriminative structure but does not imply naturally occurring causal groups.")
'''


FOREST_IMPORTANCE_STABILITY = r'''
# Feature-importance stability across repeated resamples.
importance_runs = []
for repeat in range(12):
    idx = rng_expert.integers(0, len(X_train_expert), len(X_train_expert))
    m = clone(expert_forest)
    m.set_params(model__n_estimators=140, model__random_state=RANDOM_STATE+repeat, model__oob_score=False)
    m.fit(X_train_expert.iloc[idx], y_train_expert.iloc[idx])
    pi = permutation_importance(m, X_valid_expert, y_valid_expert, scoring="roc_auc", n_repeats=3, random_state=RANDOM_STATE+repeat, n_jobs=1)
    importance_runs.append(pi.importances_mean)
imp = pd.DataFrame(importance_runs, columns=X_valid_expert.columns)
summary = pd.DataFrame({"mean": imp.mean(), "sd": imp.std(), "positive_fraction": (imp>0).mean()}).sort_values("mean", ascending=False)
display(summary.round(5))

plot = summary.head(14).sort_values("mean")
plt.figure(figsize=(10, 6))
plt.errorbar(plot["mean"], plot.index, xerr=plot["sd"], fmt="o", capsize=3)
plt.axvline(0, linestyle="--")
plt.xlabel("Permutation importance in ROC-AUC ± resample SD")
plt.title("Importance magnitude and stability across bootstrap refits")
plt.tight_layout(); plt.show()

ranks = imp.rank(axis=1, ascending=False)
rank_summary = pd.DataFrame({"median_rank": ranks.median(), "rank_iqr": ranks.quantile(.75)-ranks.quantile(.25)}).sort_values("median_rank")
plt.figure(figsize=(10, 6))
top = rank_summary.head(14).sort_values("median_rank", ascending=False)
plt.barh(top.index, top["median_rank"], xerr=top["rank_iqr"])
plt.xlabel("Median importance rank ± rank IQR")
plt.title("Rank stability is often more informative than one fitted importance value")
plt.tight_layout(); plt.show()
print("Inference: features with high mean importance but unstable rank should be described as model-dependent, not universally dominant.")
'''


FOREST_ALE = r'''
# First-order accumulated local effects (ALE), implemented directly for numeric features.
def ale_1d(model, X, feature, bins=12):
    x = X[feature].astype(float)
    edges = np.unique(np.quantile(x.dropna(), np.linspace(0,1,bins+1)))
    ids = np.clip(np.digitize(x.fillna(x.median()), edges)-1, 0, len(edges)-2)
    effects, counts = [], []
    for b in range(len(edges)-1):
        mask = ids == b
        if not np.any(mask):
            effects.append(0.0); counts.append(0); continue
        low = X.loc[mask].copy(); high = X.loc[mask].copy()
        low[feature] = edges[b]; high[feature] = edges[b+1]
        diff = model.predict_proba(high)[:,1] - model.predict_proba(low)[:,1]
        effects.append(diff.mean()); counts.append(mask.sum())
    accumulated = np.cumsum(effects)
    weights = np.asarray(counts)/max(1,np.sum(counts))
    accumulated = accumulated - np.sum(accumulated*weights)
    mids = (edges[:-1]+edges[1:])/2
    return pd.DataFrame({"mid": mids, "ale": accumulated, "count": counts})

for feature in ["credit_score", "debt_ratio", "interest_rate"]:
    ale = ale_1d(expert_forest, X_valid_expert, feature)
    display(ale.round(5))
    plt.figure(figsize=(9, 4.5))
    plt.plot(ale["mid"], ale["ale"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel(feature); plt.ylabel("Centered accumulated local effect")
    plt.title(f"ALE: local average effect of {feature} on forest probability")
    plt.tight_layout(); plt.show()
print("Inference: ALE reduces extrapolation across correlated feature regions compared with naive marginal plots, but it remains descriptive of the fitted model rather than causal.")
'''


FOREST_SURROGATE = r'''
# Global surrogate tree: compress forest behavior and measure fidelity explicitly.
forest_train_p = expert_forest.predict_proba(X_train_expert)[:,1]
forest_valid_p = expert_forest.predict_proba(X_valid_expert)[:,1]
Xt_train = expert_forest.named_steps["preprocess"].transform(X_train_expert)
Xt_valid2 = expert_forest.named_steps["preprocess"].transform(X_valid_expert)
feature_names = expert_forest.named_steps["preprocess"].get_feature_names_out()
surrogate = DecisionTreeClassifier(max_depth=4, min_samples_leaf=35, random_state=RANDOM_STATE)
surrogate.fit(Xt_train, (forest_train_p >= .5).astype(int))
surrogate_class = surrogate.predict(Xt_valid2)
fidelity = np.mean(surrogate_class == (forest_valid_p >= .5))
mae = np.mean(np.abs(surrogate.predict_proba(Xt_valid2)[:,1] - forest_valid_p))
print({"surrogate_classification_fidelity": fidelity, "surrogate_probability_MAE": mae})
plt.figure(figsize=(18, 8))
plot_tree(surrogate, feature_names=feature_names, class_names=["forest low","forest high"], filled=True, rounded=True, max_depth=4, fontsize=7)
plt.title("Shallow surrogate tree approximating the random forest")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(forest_valid_p, surrogate.predict_proba(Xt_valid2)[:,1], alpha=.55)
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("Random forest probability"); plt.ylabel("Surrogate probability")
plt.title("Surrogate fidelity must be measured, not assumed")
plt.tight_layout(); plt.show()
print("Inference: the surrogate is an approximation of the forest, not the forest itself. Explanations are trustworthy only to the extent quantified by fidelity.")
'''


FOREST_CONFORMAL = r'''
# Split-conformal prediction sets for finite-sample uncertainty communication.
# Calibration nonconformity is 1 - probability assigned to the true class.
p_cal = expert_forest.predict_proba(X_valid_expert)
true_prob = p_cal[np.arange(len(y_valid_expert)), y_valid_expert.values]
nonconformity = 1 - true_prob
alpha = 0.10
q_level = np.ceil((len(nonconformity)+1)*(1-alpha))/len(nonconformity)
q_hat = np.quantile(nonconformity, min(q_level,1), method="higher")
p_test_matrix = expert_forest.predict_proba(X_test_expert)
pred_sets = p_test_matrix >= (1-q_hat)
coverage = np.mean(pred_sets[np.arange(len(y_test_expert)), y_test_expert.values])
set_size = pred_sets.sum(axis=1)
print({"target_coverage": 1-alpha, "empirical_coverage": coverage, "mean_set_size": set_size.mean(), "q_hat": q_hat})

plt.figure(figsize=(8, 5))
values, counts = np.unique(set_size, return_counts=True)
plt.bar(values.astype(str), counts)
plt.xlabel("Prediction-set size"); plt.ylabel("Test observations")
plt.title("Conformal prediction-set efficiency")
plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 5))
plt.hist(p_test_matrix.max(axis=1)[set_size==1], bins=25, alpha=.7, label="singleton set")
if np.any(set_size>1):
    plt.hist(p_test_matrix.max(axis=1)[set_size>1], bins=25, alpha=.7, label="ambiguous set")
plt.xlabel("Maximum class probability")
plt.title("Prediction sets communicate uncertainty beyond a hard class label")
plt.legend(); plt.tight_layout(); plt.show()
print("Inference: split conformal coverage is marginal under exchangeability. It does not guarantee equal coverage for every subgroup or under distribution shift.")
'''


FINAL_SYNTHESIS = r'''
# Evidence-to-conclusion synthesis: every strong claim is paired with evidence and a boundary.
synthesis = pd.DataFrame([
    ["Association tests", "Features differ across outcome groups", "Mann-Whitney/Chi-square + FDR + effect size", "Association is not causation"],
    ["Cross-validation", "Model generalizes within sampled conditions", "Repeated/fold performance and uncertainty", "Dependent on split design and stationarity"],
    ["Calibration", "Probabilities approximate observed frequencies", "Reliability diagram, ECE, Brier score", "May deteriorate after drift"],
    ["Explainability", "The fitted model relies on specific features and interactions", "Path/ALE/permutation/surrogate diagnostics", "Model behavior is not a causal mechanism"],
    ["Robustness", "Performance has measured sensitivity to perturbations", "Controlled stress scenarios", "Not a substitute for real drift monitoring"],
    ["Decision policy", "A threshold is justified for a stated cost ratio", "Decision curves and validation-only selection", "Costs and capacity can change"],
    ["Uncertainty", "Reported metrics and predictions have sampling/model uncertainty", "Bootstrap intervals, leaf intervals, tree disagreement or conformal sets", "No guarantee under non-exchangeable deployment"],
], columns=["evidence_domain", "justified_conclusion", "supporting_method", "required_caveat"])
display(synthesis)
print("Expert standard: a model conclusion is complete only when it states the evidence, uncertainty, operational meaning and limitation.")
'''


def expert_cells(kind: str):
    selected = "expert_tree" if kind == "decision_tree" else "expert_forest"
    title = "Decision Tree" if kind == "decision_tree" else "Random Forest"
    cells = [
        md(f"""# Expert Enrichment — {title}: Statistical Evidence, Explainability and Decision Science

This extension turns the notebook from a strong implementation guide into an expert training resource. Every chapter follows the discipline **visual evidence → statistical evidence → model inference → operational justification → limitation**."""),
        md("""## 1. Independent expert runtime and benchmark ladder

The enrichment is intentionally self-contained. It rebuilds a leakage-safe split and comparable Logistic Regression, Decision Tree and Random Forest pipelines so every advanced inference can be reproduced."""),
        code(COMMON_SETUP + f"\nselected_expert_model = {selected}\nprint('Selected expert model:', type(selected_expert_model.named_steps['model']).__name__)"),
        md("""## 2. Statistical feature-evidence map

P-values alone are insufficient. We combine non-parametric or categorical tests, multiplicity correction and effect sizes."""),
        code(COMMON_STATISTICAL),
        md("""## 3. Risk estimates with confidence intervals

Observed target rates are estimates. Wilson intervals make uncertainty visible and prevent students from over-interpreting small groups."""),
        code(COMMON_RATE_INTERVALS),
    ]
    if kind == "decision_tree":
        cells += [
            md("""## 4. Visual geometry of recursive partitioning

The two-feature surface connects the mathematical split rule, rectangular decision regions and rendered tree."""), code(TREE_SURFACE),
            md("""## 5. Bootstrap split stability

A single tree structure can look definitive even when small resamples select different root features or thresholds."""), code(TREE_SPLIT_STABILITY),
            md("""## 6. Cost-complexity pruning with the one-standard-error rule

The one-SE rule selects the simplest model statistically indistinguishable from the best point estimate."""), code(TREE_PRUNING),
            md("""## 7. Leaf reliability and probability smoothing

Leaves are local samples, not laws. We audit support, event rate, Wilson intervals and Laplace shrinkage."""), code(TREE_LEAF_AUDIT),
            md("""## 8. Exact local path explanation and counterfactual sensitivity

A Decision Tree permits an exact root-to-leaf explanation. Counterfactual search is added as a sensitivity diagnostic with explicit non-causal boundaries."""), code(TREE_LOCAL),
        ]
    else:
        cells += [
            md("""## 4. Margin distributions and ensemble disagreement

Forest confidence is better understood through correct-class margins and across-tree dispersion than through one hard prediction."""), code(FOREST_MARGIN),
            md("""## 5. Forest proximity and learned representation

Leaf co-occurrence creates a model-induced similarity measure that can be inspected as a heatmap and low-dimensional embedding."""), code(FOREST_PROXIMITY),
            md("""## 6. Importance stability across refits

A single importance ranking can be accidental. Repeated bootstrap refits reveal magnitude, variance and rank stability."""), code(FOREST_IMPORTANCE_STABILITY),
            md("""## 7. Accumulated local effects

ALE provides a correlation-aware alternative to naive marginal feature-response plots."""), code(FOREST_ALE),
            md("""## 8. Surrogate-tree explanation with measured fidelity

A simple surrogate is useful only when its approximation quality is reported."""), code(FOREST_SURROGATE),
            md("""## 9. Split-conformal prediction sets

Prediction sets communicate uncertainty and empirical coverage beyond a forced hard label."""), code(FOREST_CONFORMAL),
        ]
    cells += [
        md("""## 10. Decision-curve analysis and cost-sensitive threshold justification

A model score becomes a decision only after costs, benefits and capacity are specified."""), code(COMMON_DECISION_ANALYSIS),
        md("""## 11. Calibration with bootstrap uncertainty

Reliability diagrams are extended with bootstrap intervals and expected calibration error."""), code(COMMON_CALIBRATION),
        md("""## 12. Paired statistical model comparison

Because models predict the same observations, uncertainty analysis must preserve pairing."""), code(COMMON_MODEL_COMPARISON),
        md("""## 13. Controlled robustness and shift diagnostics

Missingness, noise, unseen categories and covariate shift expose failure surfaces before deployment."""), code(COMMON_STRESS),
        md("""## 14. Evidence-to-conclusion synthesis

The final table teaches students how to convert analysis into a defensible expert conclusion rather than a metric dump."""), code(FINAL_SYNTHESIS),
        md("""## Expert mastery checkpoint

A top-level practitioner should now be able to:

1. separate statistical association, predictive value, probability quality and operational utility;
2. quantify uncertainty in metrics, leaves, feature importance and local predictions;
3. distinguish exact explanations, approximations and non-causal sensitivity analyses;
4. justify regularization and threshold choices with evidence rather than convention;
5. state clearly what the model supports, what it does not support, and what must be monitored after deployment."""),
    ]
    return cells


def update_supporting_docs(pkg: Path, kind: str, report: dict):
    title = "Decision Tree" if kind == "decision_tree" else "Random Forest"
    readme = pkg / "README.md"
    original = readme.read_text(encoding="utf-8")
    section = f"""

## Expert enrichment

The notebook now includes a dedicated expert sequence covering statistical hypothesis tests, false-discovery-rate control, effect sizes, uncertainty intervals, richer visual inference, model-comparison tests, calibration uncertainty, decision-curve analysis, robustness testing and advanced explainability.

Final validated scale:

- {report['total_cells']} total cells
- {report['code_cells']} executed code cells
- {report['chapters']} instructional chapters
- {report['embedded_png_figures']} embedded PNG figures
- zero execution errors

The notebook remains the primary source of understanding. Every major result is paired with an inference, justification and limitation.
"""
    if "## Expert enrichment" in original:
        original = original.split("## Expert enrichment")[0].rstrip()
    readme.write_text(original.rstrip() + section, encoding="utf-8")

    notes = pkg / "instructor_notes.md"
    notes.write_text(notes.read_text(encoding="utf-8") + f"""

## Expert enrichment teaching sequence

- Require students to distinguish significance from effect size and prediction.
- Ask for one visual inference and one statistical inference per chapter.
- Require every explanation to name whether it is exact, approximate, global or local.
- Discuss uncertainty before deployment thresholds.
- Use the final evidence-to-conclusion matrix as the oral-examination rubric.
""", encoding="utf-8")

    exercises = pkg / "student_exercises.md"
    exercises.write_text(exercises.read_text(encoding="utf-8") + f"""

## Expert-level extensions

1. Replace the illustrated false-negative cost ratio and recompute the optimal threshold and decision curve.
2. Repeat the paired bootstrap comparison with average precision and explain why conclusions may differ from ROC-AUC.
3. Audit feature-importance stability under a new random seed and quantify rank agreement with Spearman correlation.
4. Design a subgroup calibration audit with minimum-support rules and bootstrap intervals.
5. Create a synthetic drift scenario that preserves prevalence but changes conditional feature distributions.
6. Write an executive conclusion that includes evidence, uncertainty, decision implication and limitation.
""", encoding="utf-8")


def enrich(kind: str):
    pkg = PACKAGES[kind]
    nb_name = f"{kind}_classification_masterclass.ipynb"
    nb_path = pkg / nb_name
    html_path = pkg / nb_name.replace(".ipynb", ".html")
    notebook = nbf.read(nb_path, as_version=4)
    marker = "# Expert Enrichment —"
    notebook.cells = [cell for cell in notebook.cells if marker not in cell.get("source", "")]
    notebook.cells.extend(expert_cells(kind))
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(pkg)}})
    client.execute()
    nbf.write(notebook, nb_path)
    body, _ = HTMLExporter().from_notebook_node(notebook)
    html_path.write_text(body, encoding="utf-8")

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    figures = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.get("outputs", []))
    chapters = sum(c.cell_type == "markdown" and c.source.lstrip().startswith("##") for c in notebook.cells)
    minimum_figures = 52 if kind == "decision_tree" else 58
    report = {
        "total_cells": len(notebook.cells),
        "markdown_cells": sum(c.cell_type == "markdown" for c in notebook.cells),
        "code_cells": len(code_cells),
        "chapters": chapters,
        "unexecuted_code_cells": sum(c.get("execution_count") is None for c in code_cells),
        "error_outputs": len(errors),
        "embedded_png_figures": figures,
        "minimum_code_cells": 38,
        "minimum_figures": minimum_figures,
        "minimum_chapters": 42,
    }
    report["passes"] = (
        report["unexecuted_code_cells"] == 0 and report["error_outputs"] == 0
        and report["code_cells"] >= report["minimum_code_cells"]
        and report["embedded_png_figures"] >= report["minimum_figures"]
        and report["chapters"] >= report["minimum_chapters"]
    )
    (pkg / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_supporting_docs(pkg, kind, report)
    print(kind, json.dumps(report, indent=2))
    if not report["passes"]:
        raise RuntimeError(f"{kind} expert validation failed")


def main():
    for kind in PACKAGES:
        enrich(kind)
    print("Expert enrichment completed and validated for both tree masterclasses.")


if __name__ == "__main__":
    main()
