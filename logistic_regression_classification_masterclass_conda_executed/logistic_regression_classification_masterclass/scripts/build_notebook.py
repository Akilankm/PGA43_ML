"""Build the clean teaching notebook used by the logistic-regression package.

The notebook is assembled from small, explicit cells so that learners can
follow the reasoning in order and instructors can re-execute or adapt a single
section without reverse-engineering a monolithic script.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"


def markdown(notebook: nbf.NotebookNode, source: str) -> None:
    notebook.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(notebook: nbf.NotebookNode, source: str) -> None:
    notebook.cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python (Logistic Regression Masterclass)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    markdown(
        notebook,
        """
        # Logistic Regression Classification Masterclass

        **A complete, teaching-grade workflow for binary classification**

        This notebook develops logistic regression from first principles and
        then turns the mathematics into a leakage-aware, reproducible modelling
        workflow. The running example is a deterministic Bank Marketing-style
        dataset. The schema is inspired by the UCI Bank Marketing data, but the
        rows are synthetic and generated locally so every learner can reproduce
        the same analysis without a network connection.

        **Central question:** given information available *before* a marketing
        contact, how can we estimate subscription probability, evaluate the
        model honestly, and choose a decision threshold that reflects business
        costs?
        """,
    )

    markdown(
        notebook,
        """
        ## Learning outcomes

        By the end of this notebook, you should be able to:

        1. explain probability, odds, log-odds, the sigmoid, likelihood, and
           binary cross-entropy;
        2. audit a binary dataset and identify post-outcome leakage;
        3. perform univariate, bivariate, and multivariate classification EDA;
        4. build leakage-safe preprocessing with `Pipeline` and
           `ColumnTransformer`;
        5. fit and interpret logistic regression with scikit-learn and
           Statsmodels;
        6. distinguish discrimination, calibration, classification metrics, and
           decision-threshold performance;
        7. compare regularisation, class weighting, and SMOTE;
        8. diagnose linearity in the logit, collinearity, influence, and
           separation risks; and
        9. refit once on development data, evaluate on an untouched test set,
           serialize the model, and document its limitations.
        """,
    )

    markdown(
        notebook,
        """
        ## Notebook roadmap

        **Foundations → audit → EDA → split → preprocessing → baseline →
        logistic model → evaluation → tuning → imbalance → threshold policy →
        inference → diagnostics → final test audit → model card.**

        The order is intentional. Decisions that can leak information from the
        validation or test sets are postponed until after the data split.
        """,
    )

    markdown(
        notebook,
        r"""
        ## 1. Why logistic regression?

        Linear regression predicts an unrestricted real number. A probability
        must stay in the interval \([0,1]\). Logistic regression solves this by
        modelling the **log-odds** as a linear function of the predictors:

        \[
        \log\left(\frac{p}{1-p}\right)=\beta_0 + \beta_1x_1+\cdots+\beta_kx_k.
        \]

        The inverse transformation is the sigmoid:

        \[
        p = \sigma(z)=\frac{1}{1+e^{-z}}.
        \]

        This gives a probability for ranking and calibration, while a separate
        threshold converts probability into an operational class decision.
        """,
    )

    code(
        notebook,
        """
        from pathlib import Path
        import json
        import os
        import runpy
        import warnings

        # Keep Matplotlib's cache writable in managed or read-only environments.
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/logistic-regression-mpl")

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from IPython.display import display
        from scipy.stats import chi2_contingency, pointbiserialr
        from sklearn.base import clone
        from sklearn.calibration import calibration_curve
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            log_loss,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
            precision_recall_curve,
            ConfusionMatrixDisplay,
        )
        from sklearn.model_selection import (
            GridSearchCV,
            StratifiedKFold,
            cross_validate,
            train_test_split,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from statsmodels.nonparametric.smoothers_lowess import lowess
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        warnings.filterwarnings("ignore")
        SEED = 43
        np.random.seed(SEED)
        sns.set_theme(style="whitegrid", context="notebook")
        COLORS = {"No": "#456990", "Yes": "#F45B69", "accent": "#2A9D8F", "dark": "#263238"}
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 160,
                "axes.titleweight": "bold",
                "axes.titlepad": 12,
                "figure.constrained_layout.use": True,
            }
        )

        def find_project_root() -> Path:
            candidates = [Path.cwd(), *Path.cwd().parents]
            for candidate in candidates:
                if (candidate / "scripts" / "generate_teaching_data.py").exists():
                    return candidate
            return Path.cwd()

        PROJECT_ROOT = find_project_root()
        DATA_PATH = PROJECT_ROOT / "data" / "bank_marketing_teaching.csv"
        if not DATA_PATH.exists():
            generator = PROJECT_ROOT / "scripts" / "generate_teaching_data.py"
            if not generator.exists():
                raise FileNotFoundError(
                    f"Could not find {generator}. Open the notebook from the package directory."
                )
            runpy.run_path(str(generator), run_name="__main__")

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Data path: {DATA_PATH}")
        print(f"Random seed: {SEED}")
        """,
    )

    code(
        notebook,
        """
        versions = pd.Series(
            {
                "python": __import__("sys").version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": __import__("sklearn").__version__,
                "statsmodels": sm.__version__,
                "imbalanced_learn": __import__("imblearn").__version__,
                "seaborn": sns.__version__,
            },
            name="version",
        )
        display(versions.to_frame())
        """,
    )

    markdown(
        notebook,
        r"""
        ### Mathematical intuition: one linear score, three useful scales

        The same model score can be read in three ways:

        | Scale | Interpretation |
        |---|---|
        | Probability \(p\) | estimated chance of `Yes` |
        | Odds \(p/(1-p)\) | how much more likely `Yes` is than `No` |
        | Log-odds | an additive linear scale for modelling |

        A coefficient of \(0.69\) adds about \(0.69\) to log-odds and multiplies
        odds by \(e^{0.69}\approx 2\). It does **not** add 69 percentage points
        to probability.
        """,
    )

    code(
        notebook,
        """
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

        z_values = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
        intuition = pd.DataFrame(
            {
                "linear_score_z": z_values,
                "probability": sigmoid(z_values),
                "odds": sigmoid(z_values) / (1 - sigmoid(z_values)),
                "log_odds": np.log(sigmoid(z_values) / (1 - sigmoid(z_values))),
            }
        )
        display(intuition.round(4))

        z_curve = np.linspace(-7, 7, 400)
        p_curve = sigmoid(z_curve)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(z_curve, p_curve, color=COLORS["accent"], linewidth=3)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.scatter([0], [0.5], color=COLORS["Yes"], zorder=3, s=70)
        ax.set(title="The sigmoid maps any real-valued score to a probability", xlabel="Linear score / log-odds (z)", ylabel="P(Y=1 | X)")
        ax.set_ylim(-0.02, 1.02)
        plt.show()
        """,
    )

    markdown(
        notebook,
        r"""
        ### Likelihood and binary cross-entropy

        For one observation with label \(y\in\{0,1\}\) and predicted
        probability \(p\), the Bernoulli likelihood contribution is

        \[
        p^y(1-p)^{1-y}.
        \]

        Maximising the product of these contributions is equivalent to minimising
        the negative log-likelihood, commonly reported as **log loss** or binary
        cross-entropy:

        \[
        -\frac{1}{n}\sum_{i=1}^n\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right].
        \]

        A confident wrong prediction is penalised much more heavily than an
        uncertain prediction. That is why probability quality matters even when
        the final business decision is binary.
        """,
    )

    markdown(
        notebook,
        """
        ## 2. Load the data and state its provenance

        The package creates `data/bank_marketing_teaching.csv` deterministically
        from `scripts/generate_teaching_data.py` when the file is absent. It has
        the familiar Bank Marketing column names, but it is synthetic. This is a
        deliberate reproducibility choice: learners can run the notebook offline
        and the repository does not pretend that generated rows are the original
        UCI observations.
        """,
    )

    code(
        notebook,
        """
        df_raw = pd.read_csv(DATA_PATH)
        df_raw.columns = df_raw.columns.str.strip()
        print(f"Rows: {len(df_raw):,} | Columns: {df_raw.shape[1]}")
        display(df_raw.head(8))
        """,
    )

    code(
        notebook,
        """
        dictionary_path = PROJECT_ROOT / "data" / "data_dictionary.csv"
        data_dictionary = pd.read_csv(dictionary_path)
        display(data_dictionary)
        """,
    )

    markdown(
        notebook,
        """
        ## 3. Data-quality audit and target definition

        Before modelling, we answer four questions:

        1. What is the grain of one row?
        2. Which columns are numeric, categorical, identifiers, or targets?
        3. Are there missing values, duplicates, impossible values, or rare levels?
        4. Is the target sufficiently represented in both classes?

        A model trained before this audit can be numerically impressive and still
        be conceptually invalid.
        """,
    )

    code(
        notebook,
        """
        df = df_raw.copy()
        df["target"] = df["y"].map({"no": 0, "yes": 1}).astype("int8")
        df["target_label"] = df["target"].map({0: "No", 1: "Yes"})

        audit = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_count": df.isna().sum(),
                "missing_pct": 100 * df.isna().mean(),
                "unique_values": df.nunique(dropna=False),
            }
        ).sort_values(["missing_count", "unique_values"], ascending=[False, True])
        display(audit)
        """,
    )

    code(
        notebook,
        """
        duplicate_count = int(df.duplicated().sum())
        duplicate_rows = df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(10)
        quality_summary = pd.DataFrame(
            {
                "check": ["exact duplicate rows", "rows", "columns", "total missing cells"],
                "value": [duplicate_count, len(df), df.shape[1], int(df.isna().sum().sum())],
            }
        )
        display(quality_summary)
        display(duplicate_rows)
        """,
    )

    code(
        notebook,
        """
        class_distribution = (
            df["target_label"]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="count")
        )
        class_distribution["share"] = class_distribution["count"] / class_distribution["count"].sum()
        display(class_distribution.style.format({"share": "{:.2%}"}))

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=class_distribution, x="class", y="count", hue="class", palette=COLORS, legend=False, ax=ax)
        ax.set(title="The target is imbalanced: a majority-class baseline is necessary", xlabel="Subscription outcome", ylabel="Rows")
        for container in ax.containers:
            ax.bar_label(container, fmt="%d")
        plt.show()
        """,
    )

    code(
        notebook,
        """
        range_checks = pd.DataFrame(
            {
                "rule": [
                    "age is between 18 and 100",
                    "day is between 1 and 31",
                    "campaign is at least 1",
                    "duration is positive",
                    "pdays is -1 or positive",
                    "target has exactly two classes",
                ],
                "violations": [
                    int((~df["age"].between(18, 100)).sum()),
                    int((~df["day"].between(1, 31)).sum()),
                    int((df["campaign"] < 1).sum()),
                    int((df["duration"] <= 0).sum()),
                    int((~((df["pdays"] == -1) | (df["pdays"] > 0))).sum()),
                    int(df["target"].nunique() != 2),
                ],
            }
        )
        display(range_checks)
        assert range_checks["violations"].sum() == 0
        """,
    )

    markdown(
        notebook,
        """
        ## 4. Leakage analysis: predictive is not the same as deployable

        `duration` is the length of the current call. It can be highly predictive
        of whether the customer eventually subscribed, but it is observed only
        after the call has happened. A model intended to prioritise *who to call*
        cannot use it.

        This is a temporal availability decision, not a statistical significance
        decision. We will inspect `duration`, show its predictive relationship,
        and exclude it from the prospective model.
        """,
    )

    code(
        notebook,
        """
        leakage_audit = pd.DataFrame(
            {
                "feature": ["duration", "poutcome", "previous", "pdays"],
                "available_before_current_contact": ["No", "Usually yes", "Yes", "Yes"],
                "role_in_this_notebook": ["Exclude", "Use with interpretation", "Use", "Use"],
                "reason": [
                    "Recorded after the current contact ends",
                    "History from a previous campaign, not the current call",
                    "Historical contact count",
                    "Days since a previous contact; -1 means not previously contacted",
                ],
            }
        )
        display(leakage_audit)
        display(
            df.groupby("target_label", observed=True)["duration"]
            .agg(["count", "median", "mean", "std"])
            .round(2)
        )
        """,
    )

    code(
        notebook,
        """
        target_col = "target"
        raw_target_col = "y"
        leakage_columns = ["duration"]
        model_features = [
            column for column in df.columns
            if column not in {target_col, "target_label", raw_target_col, *leakage_columns}
        ]
        numeric_features = [column for column in model_features if pd.api.types.is_numeric_dtype(df[column])]
        categorical_features = [column for column in model_features if column not in numeric_features]
        print(f"Model features ({len(model_features)}): {model_features}")
        print(f"Numeric: {numeric_features}")
        print(f"Categorical: {categorical_features}")
        """,
    )

    markdown(
        notebook,
        """
        ## 5. Exploratory data analysis

        EDA is not a decorative preface. It helps us understand scale, skewness,
        missingness, rare categories, possible non-linearity, class separation,
        and interactions that affect both modelling and interpretation.

        We separate:

        - **univariate:** one variable at a time;
        - **bivariate:** one predictor against the target; and
        - **multivariate:** relationships among several predictors and the target.
        """,
    )

    code(
        notebook,
        """
        eda_numeric = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        numeric_summary = df[eda_numeric].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
        numeric_summary["missing"] = df[eda_numeric].isna().sum()
        display(numeric_summary.round(2))

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for ax, feature in zip(axes.flat, eda_numeric):
            sns.histplot(data=df, x=feature, hue="target_label", bins=32, stat="density", common_norm=False, alpha=0.45, palette=COLORS, ax=ax)
            ax.set_title(f"{feature}: distribution by outcome")
        for ax in axes.flat[len(eda_numeric):]:
            ax.remove()
        plt.show()
        """,
    )

    code(
        notebook,
        """
        plot_df = df.copy()
        for feature in categorical_features:
            plot_df[feature] = plot_df[feature].astype("object").where(plot_df[feature].notna(), "Missing")

        categorical_eda = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
        fig, axes = plt.subplots(3, 3, figsize=(16, 15))
        for ax, feature in zip(axes.flat, categorical_eda):
            order = plot_df[feature].value_counts().index
            sns.countplot(data=plot_df, y=feature, hue="target_label", order=order, palette=COLORS, ax=ax)
            ax.set_title(f"{feature}: category counts by outcome")
            ax.set_xlabel("Rows")
            ax.set_ylabel("")
        plt.show()
        """,
    )

    code(
        notebook,
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, feature in zip(axes.flat, ["age", "balance", "campaign", "pdays", "previous", "duration"]):
            sns.boxenplot(data=df, x="target_label", y=feature, order=["No", "Yes"], palette=COLORS, hue="target_label", legend=False, ax=ax)
            ax.set_title(f"{feature}: distribution shape and tails")
            ax.set_xlabel("Outcome")
        plt.show()
        """,
    )

    markdown(
        notebook,
        """
        ### Bivariate analysis: numerical predictors versus the target

        Visual separation is useful, but an association measure should accompany
        it. The point-biserial correlation is the Pearson correlation between a
        binary target and a continuous predictor. It describes linear association,
        not causality and not the full multivariable effect.
        """,
    )

    code(
        notebook,
        """
        point_biserial_rows = []
        for feature in eda_numeric:
            clean = df[[target_col, feature]].dropna()
            correlation, p_value = pointbiserialr(clean[target_col], clean[feature])
            point_biserial_rows.append(
                {"feature": feature, "point_biserial_r": correlation, "p_value": p_value, "n": len(clean)}
            )
        point_biserial_table = pd.DataFrame(point_biserial_rows).sort_values("point_biserial_r", key=lambda s: s.abs(), ascending=False)
        display(point_biserial_table.style.format({"point_biserial_r": "{:.3f}", "p_value": "{:.3g}"}))
        """,
    )

    code(
        notebook,
        """
        def target_rate_table(data: pd.DataFrame, feature: str) -> pd.DataFrame:
            grouped = data[[feature, target_col]].copy()
            grouped[feature] = grouped[feature].astype("object").where(grouped[feature].notna(), "Missing")
            result = grouped.groupby(feature, observed=True)[target_col].agg(["sum", "count"])
            result["rate"] = result["sum"] / result["count"]
            z = 1.96
            denominator = 1 + z**2 / result["count"]
            centre = (result["rate"] + z**2 / (2 * result["count"])) / denominator
            half_width = z * np.sqrt(
                (result["rate"] * (1 - result["rate"]) / result["count"])
                + z**2 / (4 * result["count"]**2)
            ) / denominator
            result["ci_low"] = (centre - half_width).clip(0, 1)
            result["ci_high"] = (centre + half_width).clip(0, 1)
            return result.reset_index().rename(columns={feature: "category"}).sort_values("rate", ascending=False)

        for feature in ["job", "education", "contact", "poutcome"]:
            table = target_rate_table(df, feature)
            print(f"{feature} target rates")
            display(table.style.format({"rate": "{:.2%}", "ci_low": "{:.2%}", "ci_high": "{:.2%}"}))
        """,
    )

    code(
        notebook,
        """
        rate_features = ["job", "education", "contact", "poutcome", "month"]
        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        for ax, feature in zip(axes.flat, rate_features):
            rates = target_rate_table(df, feature).sort_values("rate")
            ax.barh(rates["category"].astype(str), rates["rate"], color=COLORS["accent"], alpha=0.85)
            ax.axvline(df[target_col].mean(), color=COLORS["Yes"], linestyle="--", label="overall rate")
            ax.set_title(f"{feature}: observed target rate")
            ax.set_xlabel("P(Yes)")
            ax.legend(loc="lower right")
        axes.flat[-1].remove()
        plt.show()
        """,
    )

    markdown(
        notebook,
        """
        ### Multivariate analysis: association is conditional

        A feature can look useful by itself and become weak after adjustment for
        other variables. Conversely, an interaction can be hidden in marginal
        plots. The following views are descriptive diagnostics; they do not
        replace a pre-specified model or causal design.
        """,
    )

    code(
        notebook,
        """
        correlation_features = ["age", "balance", "day", "campaign", "pdays", "previous", "duration", target_col]
        correlation_matrix = df[correlation_features].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
        ax.set_title("Numeric correlation matrix: useful for screening, not causality")
        plt.show()
        """,
    )

    code(
        notebook,
        """
        binned = df[["age", "balance", target_col]].copy()
        binned["age_band"] = pd.cut(binned["age"], bins=[17, 25, 35, 45, 55, 65, 100], right=False)
        binned["balance_band"] = pd.qcut(binned["balance"], q=6, duplicates="drop")
        age_balance_rates = (
            binned.groupby(["age_band", "balance_band"], observed=True)[target_col]
            .agg(rate="mean", n="size")
            .reset_index()
        )
        pivot_rates = age_balance_rates.pivot(index="age_band", columns="balance_band", values="rate")
        display(pivot_rates.style.format("{:.2%}"))

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(pivot_rates, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
        ax.set_title("Target rate across age and balance bands")
        ax.set_xlabel("Balance quantile band")
        ax.set_ylabel("Age band")
        plt.show()
        """,
    )

    markdown(
        notebook,
        """
        ## 6. Split first, preprocess second

        The test set is a one-time final audit. We create three partitions:

        - **train:** fit preprocessing and model parameters;
        - **validation:** tune choices such as regularisation and threshold; and
        - **test:** evaluate once after all decisions are frozen.

        Exact duplicates are removed before the split. Stratification preserves the
        rare positive class in each partition. Every learned transformation is
        placed inside a pipeline so that it is fitted only on training data.
        """,
    )

    code(
        notebook,
        """
        df_clean = df.drop_duplicates().reset_index(drop=True)
        X = df_clean[model_features].copy()
        y = df_clean[target_col].copy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.40, stratify=y, random_state=SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
        )
        split_table = pd.DataFrame(
            {
                "partition": ["train", "validation", "test"],
                "rows": [len(X_train), len(X_val), len(X_test)],
                "positive_count": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
                "positive_rate": [y_train.mean(), y_val.mean(), y_test.mean()],
            }
        )
        display(split_table.style.format({"positive_rate": "{:.2%}"}))

        assert set(X_train.index).isdisjoint(X_val.index)
        assert set(X_train.index).isdisjoint(X_test.index)
        assert set(X_val.index).isdisjoint(X_test.index)
        print("Index-overlap checks passed.")
        """,
    )

    code(
        notebook,
        """
        numeric_features_model = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
        categorical_features_model = [column for column in X.columns if column not in numeric_features_model]

        def make_ohe():
            # `sparse_output` is the current parameter; the fallback keeps the
            # teaching notebook usable with older scikit-learn releases.
            try:
                return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
            except TypeError:
                return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

        def make_preprocessor():
            return ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        numeric_features_model,
                    ),
                    (
                        "categorical",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("one_hot", make_ohe()),
                            ]
                        ),
                        categorical_features_model,
                    ),
                ],
                remainder="drop",
            )

        print(f"Numeric model columns: {numeric_features_model}")
        print(f"Categorical model columns: {categorical_features_model}")
        """,
    )

    markdown(
        notebook,
        """
        ## 7. Metrics and the majority-class baseline

        On an imbalanced dataset, accuracy can be misleading. We track:

        - ROC-AUC: ranking quality across all thresholds;
        - average precision: positive-class ranking quality, useful under
          imbalance;
        - log loss and Brier score: probability quality;
        - precision, recall, specificity, F1, balanced accuracy, and MCC at a
          chosen threshold.

        No single metric is universally best. The metric must reflect the use
        case and the cost of false positives and false negatives.
        """,
    )

    code(
        notebook,
        """
        def evaluate_predictions(y_true, probabilities, threshold=0.50):
            predictions = (np.asarray(probabilities) >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
            return {
                "threshold": threshold,
                "roc_auc": roc_auc_score(y_true, probabilities),
                "average_precision": average_precision_score(y_true, probabilities),
                "log_loss": log_loss(y_true, probabilities),
                "brier": brier_score_loss(y_true, probabilities),
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "specificity": specificity,
                "f1": f1_score(y_true, predictions, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
                "mcc": matthews_corrcoef(y_true, predictions),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }

        dummy_model = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        )
        dummy_model.fit(X_train, y_train)
        dummy_val_probability = dummy_model.predict_proba(X_val)[:, 1]
        dummy_metrics = evaluate_predictions(y_val, dummy_val_probability)
        display(pd.DataFrame([dummy_metrics], index=["majority-class baseline"]).round(4))
        """,
    )

    markdown(
        notebook,
        """
        ## 8. Fit a leakage-safe scikit-learn logistic model

        The pipeline performs imputation, scaling, and one-hot encoding inside the
        training workflow. This avoids the common mistake of fitting a scaler or
        category encoder on all rows before the split.

        `C` is the inverse regularisation strength in scikit-learn: smaller `C`
        means stronger regularisation. We begin with a transparent L2 model and
        tune the choices later using cross-validation.
        """,
    )

    code(
        notebook,
        """
        logistic_model = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="liblinear",
                        max_iter=4_000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        logistic_model.fit(X_train, y_train)
        logistic_val_probability = logistic_model.predict_proba(X_val)[:, 1]
        baseline_logistic_metrics = evaluate_predictions(y_val, logistic_val_probability)
        display(pd.DataFrame([baseline_logistic_metrics], index=["logistic regression"]).round(4))
        """,
    )

    code(
        notebook,
        """
        fitted_preprocessor = logistic_model.named_steps["preprocess"]
        transformed_feature_names = fitted_preprocessor.get_feature_names_out()
        coefficients = logistic_model.named_steps["model"].coef_[0]
        coefficient_table = pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "coefficient": coefficients,
                "odds_multiplier": np.exp(coefficients),
                "absolute_coefficient": np.abs(coefficients),
            }
        ).sort_values("absolute_coefficient", ascending=False)
        display(coefficient_table.head(20).style.format({"coefficient": "{:.3f}", "odds_multiplier": "{:.3f}"}))

        top_coefficients = coefficient_table.head(16).sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=top_coefficients, x="coefficient", y="feature", hue="feature", palette="vlag", legend=False, ax=ax)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Largest logistic coefficients: direction and magnitude on log-odds scale")
        ax.set_xlabel("Coefficient (change in log-odds per one transformed unit)")
        ax.set_ylabel("")
        plt.show()
        """,
    )

    markdown(
        notebook,
        r"""
        ### Interpreting a coefficient correctly

        With all other encoded variables held fixed:

        - \(\beta_j\) is the additive change in log-odds for a one-unit change in
          the transformed feature;
        - \(e^{\beta_j}\) is the multiplicative change in odds; and
        - the probability change depends on the starting probability, so it is
          not constant across customers.

        For a one-hot category, the coefficient compares that category with the
        dropped reference category. Interpretation is associational, not causal.
        """,
    )

    markdown(
        notebook,
        """
        ## 9. Validation-set evaluation: ranking, classification, and calibration

        ROC and precision-recall curves show behaviour over many thresholds. The
        confusion matrix shows one operating point. The calibration curve asks a
        different question: among observations assigned probability around 0.2,
        do roughly 20% actually belong to the positive class?
        """,
    )

    code(
        notebook,
        """
        fpr, tpr, _ = roc_curve(y_val, logistic_val_probability)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, logistic_val_probability)
        calibration_observed, calibration_predicted = calibration_curve(
            y_val, logistic_val_probability, n_bins=10, strategy="quantile"
        )

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        axes[0].plot(fpr, tpr, color=COLORS["Yes"], linewidth=2.5, label=f"AUC={roc_auc_score(y_val, logistic_val_probability):.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="grey")
        axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
        axes[0].legend()

        axes[1].plot(recall_curve, precision_curve, color=COLORS["accent"], linewidth=2.5, label=f"AP={average_precision_score(y_val, logistic_val_probability):.3f}")
        axes[1].axhline(y_val.mean(), linestyle="--", color="grey", label="positive prevalence")
        axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
        axes[1].legend()

        axes[2].plot(calibration_predicted, calibration_observed, "o-", color=COLORS["accent"])
        axes[2].plot([0, 1], [0, 1], "--", color="grey")
        axes[2].set(title="Calibration", xlabel="Mean predicted probability", ylabel="Observed frequency")

        ConfusionMatrixDisplay.from_predictions(
            y_val,
            (logistic_val_probability >= 0.5).astype(int),
            display_labels=["No", "Yes"],
            cmap="Blues",
            colorbar=False,
            ax=axes[3],
        )
        axes[3].set_title("Confusion matrix at threshold 0.50")
        plt.show()
        """,
    )

    markdown(
        notebook,
        """
        ## 10. Regularisation and cross-validation

        We use stratified folds so each fold retains the rare-class proportion.
        Cross-validation belongs inside the training data. The validation set
        remains available for the later threshold policy, and the test set remains
        untouched.
        """,
    )

    code(
        notebook,
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_rows = []
        for penalty in ["l1", "l2"]:
            for regularization_C in [0.03, 0.10, 0.30, 1.00, 3.00]:
                candidate = Pipeline(
                    [
                        ("preprocess", make_preprocessor()),
                        (
                            "model",
                            LogisticRegression(
                                penalty=penalty,
                                C=regularization_C,
                                solver="liblinear",
                                max_iter=4_000,
                                random_state=SEED,
                            ),
                        ),
                    ]
                )
                scores = cross_validate(
                    candidate,
                    X_train,
                    y_train,
                    scoring={"roc_auc": "roc_auc", "average_precision": "average_precision", "log_loss": "neg_log_loss"},
                    cv=cv,
                    n_jobs=1,
                )
                cv_rows.append(
                    {
                        "penalty": penalty,
                        "C": regularization_C,
                        "roc_auc_mean": scores["test_roc_auc"].mean(),
                        "average_precision_mean": scores["test_average_precision"].mean(),
                        "log_loss_mean": -scores["test_log_loss"].mean(),
                    }
                )
        cv_summary = pd.DataFrame(cv_rows).sort_values("average_precision_mean", ascending=False)
        display(cv_summary.style.format({column: "{:.4f}" for column in cv_summary.select_dtypes("number").columns}))
        """,
    )

    code(
        notebook,
        """
        tuning_grid = GridSearchCV(
            estimator=Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear",
                            max_iter=4_000,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__penalty": ["l1", "l2"],
                "model__C": [0.03, 0.10, 0.30, 1.00, 3.00],
                "model__class_weight": [None, "balanced"],
            },
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc", "log_loss": "neg_log_loss"},
            refit="average_precision",
            cv=cv,
            n_jobs=1,
            return_train_score=True,
        )
        tuning_grid.fit(X_train, y_train)
        tuning_results = pd.DataFrame(tuning_grid.cv_results_).sort_values("rank_test_average_precision")
        display(
            tuning_results[
                [
                    "rank_test_average_precision",
                    "param_model__penalty",
                    "param_model__C",
                    "param_model__class_weight",
                    "mean_test_average_precision",
                    "mean_test_roc_auc",
                    "mean_test_log_loss",
                ]
            ].head(12).style.format(
                {
                    "mean_test_average_precision": "{:.4f}",
                    "mean_test_roc_auc": "{:.4f}",
                    "mean_test_log_loss": "{:.4f}",
                }
            )
        )
        selected_model = tuning_grid.best_estimator_
        selected_params = tuning_grid.best_params_
        selected_val_probability = selected_model.predict_proba(X_val)[:, 1]
        print(f"Selected by validation average precision: {selected_params}")
        """,
    )

    markdown(
        notebook,
        """
        ## 11. Imbalance remedies: class weighting and SMOTE

        Two common responses to a rare positive class are:

        - **class weighting:** make positive errors more expensive during fitting;
        - **SMOTE:** create synthetic minority examples inside each training fold.

        Both can change recall and ranking behaviour. They do not automatically
        improve calibration. The SMOTE demonstration below runs after the
        preprocessing pipeline and is useful pedagogically, but mixed categorical
        data requires care: one-hot interpolation is not the same as semantic
        category generation. In a production mixed-type workflow, evaluate
        alternatives such as SMOTENC or carefully designed sampling policies.
        """,
    )

    code(
        notebook,
        """
        weighted_model = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=4_000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        weighted_model.fit(X_train, y_train)
        weighted_val_probability = weighted_model.predict_proba(X_val)[:, 1]

        smote_model = ImbPipeline(
            [
                ("preprocess", make_preprocessor()),
                ("smote", SMOTE(random_state=SEED)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=0.30,
                        solver="liblinear",
                        max_iter=4_000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        smote_model.fit(X_train, y_train)
        smote_val_probability = smote_model.predict_proba(X_val)[:, 1]

        imbalance_comparison = pd.DataFrame(
            [
                evaluate_predictions(y_val, logistic_val_probability),
                evaluate_predictions(y_val, selected_val_probability),
                evaluate_predictions(y_val, weighted_val_probability),
                evaluate_predictions(y_val, smote_val_probability),
            ],
            index=["baseline logistic", "tuned selected", "class_weight=balanced", "SMOTE"],
        )
        display(imbalance_comparison[["roc_auc", "average_precision", "log_loss", "brier", "precision", "recall", "f1"]].round(4))
        """,
    )

    code(
        notebook,
        """
        calibration_models = {
            "baseline": logistic_val_probability,
            "selected": selected_val_probability,
            "balanced": weighted_val_probability,
            "SMOTE": smote_val_probability,
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        for name, probabilities in calibration_models.items():
            observed, predicted = calibration_curve(y_val, probabilities, n_bins=10, strategy="quantile")
            ax.plot(predicted, observed, marker="o", linewidth=2, label=name)
        ax.set(title="Imbalance remedies can change calibration", xlabel="Mean predicted probability", ylabel="Observed frequency")
        ax.legend()
        plt.show()
        """,
    )

    markdown(
        notebook,
        r"""
        ## 12. Business threshold engineering

        A probability model does not decide how many customers to contact. The
        threshold encodes a policy. We tune it on validation data only.

        For illustration, suppose a missed subscriber (false negative) costs 6
        units and an unnecessary contact (false positive) costs 1 unit:

        \[
        \text{total cost}=1\times FP + 6\times FN.
        \]

        Changing these costs, campaign capacity, or customer experience priorities
        should change the threshold. The test set is not used to choose it.
        """,
    )

    code(
        notebook,
        """
        false_positive_cost = 1.0
        false_negative_cost = 6.0
        threshold_rows = []
        for threshold in np.linspace(0.01, 0.80, 160):
            predictions = (selected_val_probability >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "total_cost": false_positive_cost * fp + false_negative_cost * fn,
                    "cost_per_row": (false_positive_cost * fp + false_negative_cost * fn) / len(y_val),
                    "precision": precision_score(y_val, predictions, zero_division=0),
                    "recall": recall_score(y_val, predictions, zero_division=0),
                    "contact_rate": predictions.mean(),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )
        threshold_table = pd.DataFrame(threshold_rows)
        selected_threshold = float(threshold_table.loc[threshold_table["total_cost"].idxmin(), "threshold"])
        display(threshold_table.nsmallest(10, "total_cost").style.format({column: "{:.4f}" for column in ["threshold", "cost_per_row", "precision", "recall", "contact_rate"]}))
        print(f"Selected validation threshold: {selected_threshold:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        axes[0].plot(threshold_table["threshold"], threshold_table["total_cost"], color=COLORS["Yes"], linewidth=2.5)
        axes[0].axvline(selected_threshold, linestyle="--", color=COLORS["dark"], label=f"selected={selected_threshold:.3f}")
        axes[0].set(title="Validation cost by threshold", xlabel="Decision threshold", ylabel="Total cost")
        axes[0].legend()
        axes[1].plot(threshold_table["threshold"], threshold_table["precision"], label="precision", linewidth=2)
        axes[1].plot(threshold_table["threshold"], threshold_table["recall"], label="recall", linewidth=2)
        axes[1].plot(threshold_table["threshold"], threshold_table["contact_rate"], label="contact rate", linewidth=2)
        axes[1].axvline(selected_threshold, linestyle="--", color=COLORS["dark"])
        axes[1].set(title="Threshold changes the operating point", xlabel="Decision threshold", ylabel="Rate")
        axes[1].legend()
        plt.show()
        """,
    )

    markdown(
        notebook,
        """
        ## 13. Statsmodels GLM: inference and adjusted odds ratios

        scikit-learn is excellent for predictive pipelines. Statsmodels exposes a
        statistical summary: coefficients, standard errors, Wald tests, confidence
        intervals, deviance, and information criteria.

        The formula model below is fit on the training partition only. Numeric
        features are expressed in teaching-friendly units (`age_10` and
        `balance_1000`) so the odds ratios are easier to discuss. Missing values
        are imputed using training-only summaries, and categorical missingness is
        represented explicitly as `Missing`. `pdays_known` is deliberately not
        included in this inferential formula because it is deterministically
        encoded by the `poutcome` levels in this teaching generator; including
        both would create exact multicollinearity.
        """,
    )

    code(
        notebook,
        """
        stats_train = df_clean.loc[X_train.index].copy()
        for feature in numeric_features_model:
            stats_train[feature] = stats_train[feature].fillna(X_train[feature].median())
        for feature in categorical_features_model:
            stats_train[feature] = stats_train[feature].astype("object").where(stats_train[feature].notna(), "Missing")

        stats_train["age_10"] = stats_train["age"] / 10.0
        stats_train["balance_1000"] = stats_train["balance"] / 1_000.0
        stats_train["campaign_log"] = np.log1p(stats_train["campaign"])
        stats_train["previous_log"] = np.log1p(stats_train["previous"])
        stats_train["pdays_known"] = (stats_train["pdays"] != -1).astype(int)

        stats_formula = (
            "target ~ age_10 + balance_1000 + day + campaign_log + previous_log "
            "+ C(job) + C(marital) + C(education) + C(default) + C(housing) + C(loan) "
            "+ C(contact) + C(month) + C(poutcome)"
        )
        stats_glm = smf.glm(stats_formula, data=stats_train, family=sm.families.Binomial()).fit()
        print(stats_glm.summary())
        """,
    )

    code(
        notebook,
        """
        confidence_intervals = stats_glm.conf_int()
        odds_ratio_table = pd.DataFrame(
            {
                "term": stats_glm.params.index,
                "coefficient": stats_glm.params.values,
                "odds_ratio": np.exp(stats_glm.params.values),
                "ci_low": np.exp(confidence_intervals[0].values),
                "ci_high": np.exp(confidence_intervals[1].values),
                "p_value": stats_glm.pvalues.values,
            }
        ).sort_values("p_value")
        display(
            odds_ratio_table.head(25).style.format(
                {
                    "coefficient": "{:.4f}",
                    "odds_ratio": "{:.3f}",
                    "ci_low": "{:.3f}",
                    "ci_high": "{:.3f}",
                    "p_value": "{:.3g}",
                }
            )
        )

        odds_to_plot = odds_ratio_table.query("term != 'Intercept'").head(15).sort_values("odds_ratio")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.errorbar(
            odds_to_plot["odds_ratio"],
            odds_to_plot["term"],
            xerr=[odds_to_plot["odds_ratio"] - odds_to_plot["ci_low"], odds_to_plot["ci_high"] - odds_to_plot["odds_ratio"]],
            fmt="o",
            color=COLORS["accent"],
            ecolor=COLORS["dark"],
            capsize=3,
        )
        ax.axvline(1, linestyle="--", color="grey", label="no change in odds")
        ax.set_xscale("log")
        ax.set(title="Adjusted odds ratios with 95% confidence intervals", xlabel="Odds ratio (log scale)", ylabel="Term")
        ax.legend()
        plt.show()
        """,
    )

    markdown(
        notebook,
        """
        ## 14. Diagnostics and model adequacy

        Logistic regression does not require normally distributed predictors or
        constant-variance residuals in the linear-regression sense. It does rely
        on important structural assumptions and practical conditions:

        - observations are independent enough for the intended inference;
        - the logit is approximately linear in continuous predictors;
        - categories and predictors are not so sparse or collinear that estimates
          become unstable; and
        - a few highly influential observations do not determine the conclusion.

        These are diagnostic signals, not automatic proof that a model is valid or
        invalid.
        """,
    )

    code(
        notebook,
        """
        train_probability_stats = stats_glm.predict(stats_train)
        influence = stats_glm.get_influence(observed=False).summary_frame()
        pearson_residuals = stats_glm.resid_pearson

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        axes[0].scatter(train_probability_stats, pearson_residuals, s=14, alpha=0.35, color=COLORS["accent"])
        axes[0].axhline(0, linestyle="--", color="grey")
        axes[0].set(title="Pearson residuals vs fitted probability", xlabel="Fitted probability", ylabel="Pearson residual")
        axes[1].scatter(influence["hat_diag"], influence["standard_resid"], s=14, alpha=0.35, color=COLORS["Yes"])
        axes[1].axhline(3, linestyle=":", color="grey")
        axes[1].axhline(-3, linestyle=":", color="grey")
        axes[1].set(title="Standardised residuals vs leverage", xlabel="Leverage", ylabel="Standardised residual")
        axes[2].scatter(train_probability_stats, stats_train[target_col], s=12, alpha=0.20, color=COLORS["dark"], label="observations")
        fitted_values = train_probability_stats.to_numpy()
        order = np.argsort(fitted_values)
        axes[2].plot(fitted_values[order], fitted_values[order], color=COLORS["Yes"], linewidth=2, label="identity")
        axes[2].set(title="Observed labels versus fitted probability", xlabel="Fitted probability", ylabel="Observed target")
        axes[2].legend()
        plt.show()

        influence_thresholds = pd.DataFrame(
            {
                "diagnostic": ["high leverage guide", "large Cook's distance guide"],
                "threshold": [2 * len(stats_glm.params) / len(stats_train), 4 / len(stats_train)],
            }
        )
        display(influence_thresholds)
        influence_columns = ["cooks_d", "hat_diag", "standard_resid"]
        influence_columns += [column for column in ["dfbetas_Intercept", "dfb_Intercept"] if column in influence.columns]
        display(influence.nlargest(10, "cooks_d")[influence_columns].round(4))
        """,
    )

    code(
        notebook,
        """
        # Empirical-logit view for continuous predictors.
        def empirical_logit_table(data, feature, bins=10):
            temporary = data[[feature, target_col]].copy()
            temporary["bin"] = pd.qcut(temporary[feature], q=bins, duplicates="drop")
            grouped = temporary.groupby("bin", observed=True).agg(
                mean_feature=(feature, "mean"), successes=(target_col, "sum"), n=(target_col, "size")
            )
            # Add 0.5 pseudo-counts to avoid infinite empirical logits.
            grouped["rate_smoothed"] = (grouped["successes"] + 0.5) / (grouped["n"] + 1.0)
            grouped["empirical_logit"] = np.log(grouped["rate_smoothed"] / (1 - grouped["rate_smoothed"]))
            return grouped.reset_index(drop=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, feature in zip(axes, ["age", "balance"]):
            table = empirical_logit_table(stats_train, feature)
            ax.scatter(table["mean_feature"], table["empirical_logit"], color=COLORS["accent"], s=55, label="binned empirical logit")
            smooth = lowess(table["empirical_logit"], table["mean_feature"], frac=0.7, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color=COLORS["Yes"], linewidth=2, label="LOWESS trend")
            ax.set(title=f"Linearity-in-the-logit diagnostic: {feature}", xlabel=feature, ylabel="Empirical logit of target rate")
            ax.legend()
        plt.show()
        """,
    )

    code(
        notebook,
        """
        # Box-Tidwell-style terms test whether selected continuous effects need a
        # power transformation. A small p-value is a signal to investigate a
        # nonlinear representation, not a mandate to transform blindly.
        box_tidwell = stats_train.copy()
        box_tidwell["age_bt"] = box_tidwell["age_10"] * np.log(box_tidwell["age_10"].clip(lower=0.1))
        box_tidwell["balance_positive"] = (box_tidwell["balance_1000"] - box_tidwell["balance_1000"].min() + 0.1).clip(lower=0.1)
        box_tidwell["balance_bt"] = box_tidwell["balance_positive"] * np.log(box_tidwell["balance_positive"])
        box_tidwell_formula = (
            "target ~ age_10 + age_bt + balance_1000 + balance_bt + day + campaign_log + previous_log "
            "+ C(job) + C(marital) + C(education) + C(default) + C(housing) + C(loan) + C(contact) + C(month) + C(poutcome)"
        )
        box_tidwell_model = smf.glm(box_tidwell_formula, data=box_tidwell, family=sm.families.Binomial()).fit()
        box_tidwell_results = pd.DataFrame(
            {
                "term": ["age_bt", "balance_bt"],
                "coefficient": [box_tidwell_model.params["age_bt"], box_tidwell_model.params["balance_bt"]],
                "p_value": [box_tidwell_model.pvalues["age_bt"], box_tidwell_model.pvalues["balance_bt"]],
            }
        )
        display(box_tidwell_results.style.format({"coefficient": "{:.4f}", "p_value": "{:.4g}"}))
        """,
    )

    code(
        notebook,
        """
        vif_features = ["age_10", "balance_1000", "day", "campaign_log", "previous_log", "pdays_known"]
        vif_matrix = stats_train[vif_features].astype(float)
        vif_table = pd.DataFrame(
            {
                "feature": vif_features,
                "VIF": [variance_inflation_factor(vif_matrix.values, index) for index in range(vif_matrix.shape[1])],
            }
        ).sort_values("VIF", ascending=False)
        display(vif_table.style.format({"VIF": "{:.3f}"}))
        print("VIF is a collinearity screening measure; it is not a test of predictive usefulness.")
        """,
    )

    markdown(
        notebook,
        """
        ### Diagnostic interpretation and remedies

        | Signal | What it may mean | Possible response |
        |---|---|---|
        | Curved empirical-logit trend | linear logit is inadequate | splines, bins, transformations, interactions |
        | Large VIF or unstable signs | redundant predictors | combine, remove, regularise, or redesign |
        | High leverage / Cook's distance | observation strongly affects fit | verify data, sensitivity analysis, robust process |
        | Sparse categories / separation | estimates can diverge | pool levels, penalise, collect data, Firth-type methods |
        | Good AUC but poor calibration | ranking is useful, probabilities are not | recalibration and monitoring |

        Diagnostics guide the next modelling question. They do not justify
        deleting observations simply because they are inconvenient.
        """,
    )

    markdown(
        notebook,
        """
        ## 15. Final refit, untouched test evaluation, and subgroup checks

        All modelling choices are now frozen: the selected hyperparameters and the
        validation-derived threshold. We refit the selected pipeline on train plus
        validation, then evaluate once on test. This is the closest estimate in
        this notebook of performance on future data generated by the same process.
        """,
    )

    code(
        notebook,
        """
        X_development = pd.concat([X_train, X_val], axis=0)
        y_development = pd.concat([y_train, y_val], axis=0)
        final_model = clone(selected_model)
        final_model.fit(X_development, y_development)
        test_probability = final_model.predict_proba(X_test)[:, 1]

        final_metrics = pd.DataFrame(
            [
                evaluate_predictions(y_test, test_probability, threshold=0.50),
                evaluate_predictions(y_test, test_probability, threshold=selected_threshold),
            ],
            index=["test at threshold 0.50", "test at validation-selected threshold"],
        )
        display(final_metrics[["threshold", "roc_auc", "average_precision", "log_loss", "brier", "precision", "recall", "specificity", "f1", "balanced_accuracy", "tn", "fp", "fn", "tp"]].round(4))
        """,
    )

    code(
        notebook,
        """
        test_scored = X_test.copy()
        test_scored["actual_target"] = y_test
        test_scored["probability"] = test_probability
        test_scored["prediction"] = (test_probability >= selected_threshold).astype(int)
        subgroup_rows = []
        for feature in ["job", "education", "marital", "contact"]:
            for level, group in test_scored.groupby(feature, dropna=False, observed=True):
                if len(group) < 20 or group["actual_target"].nunique() < 2:
                    continue
                subgroup_rows.append(
                    {
                        "feature": feature,
                        "level": "Missing" if pd.isna(level) else level,
                        "n": len(group),
                        "actual_positive_rate": group["actual_target"].mean(),
                        "predicted_positive_rate": group["prediction"].mean(),
                        "recall": recall_score(group["actual_target"], group["prediction"], zero_division=0),
                        "precision": precision_score(group["actual_target"], group["prediction"], zero_division=0),
                    }
                )
        subgroup_table = pd.DataFrame(subgroup_rows)
        display(subgroup_table.sort_values(["feature", "recall"]).style.format({column: "{:.2%}" for column in ["actual_positive_rate", "predicted_positive_rate", "recall", "precision"]}))
        """,
    )

    markdown(
        notebook,
        """
        ## 16. Serialization, reproducibility, and model card

        A production handoff needs more than a fitted estimator. It needs the
        preprocessing graph, threshold policy, feature contract, evaluation
        metrics, data provenance, and known limitations.
        """,
    )

    code(
        notebook,
        """
        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        joblib.dump(final_model, artifacts_dir / "final_logistic_pipeline.joblib")
        (artifacts_dir / "selected_threshold.json").write_text(
            json.dumps(
                {
                    "threshold": selected_threshold,
                    "false_positive_cost": false_positive_cost,
                    "false_negative_cost": false_negative_cost,
                    "selection_partition": "validation",
                    "selection_metric": "minimum total cost",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        final_metrics.to_csv(artifacts_dir / "final_test_metrics.csv")
        (artifacts_dir / "feature_contract.json").write_text(
            json.dumps(
                {
                    "target": target_col,
                    "excluded_leakage_columns": leakage_columns,
                    "model_features": model_features,
                    "numeric_features": numeric_features_model,
                    "categorical_features": categorical_features_model,
                    "random_seed": SEED,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Saved final pipeline, threshold policy, feature contract, and test metrics to artifacts/.")
        """,
    )

    markdown(
        notebook,
        """
        ### Final model card

        **Intended use:** rank prospective customers before a marketing contact.

        **Target:** whether the customer subscribes (`yes`/`no`) in the synthetic
        Bank Marketing-style teaching data.

        **Excluded leakage:** `duration`, because it is only known after the
        current contact ends.

        **Primary outputs:** probability for ranking, plus a threshold chosen from
        validation costs. The threshold is a policy and must be revisited when
        contact capacity, costs, prevalence, or customer-experience constraints
        change.

        **Limitations:** synthetic data are not evidence about real customers;
        coefficients are associational; subgroup metrics with small samples are
        unstable; and good validation results do not guarantee performance under
        population or campaign drift.

        **Monitoring plan:** track prevalence, average precision, calibration,
        false-negative rate, subgroup recall, missingness, unseen categories,
        feature drift, contact capacity, and realised campaign value.
        """,
    )

    markdown(
        notebook,
        """
        ## Takeaways

        1. Logistic regression models a linear relationship on the log-odds scale,
           not directly on the probability scale.
        2. A statistically predictive variable can still be unusable because it is
           unavailable at decision time.
        3. EDA, preprocessing, tuning, and threshold selection must respect the
           train/validation/test boundary.
        4. AUC, average precision, calibration, confusion-matrix metrics, and
           business cost answer different questions.
        5. Class weighting and SMOTE alter the operating point; they are not magic
           fixes and may require recalibration.
        6. Statsmodels provides inference and diagnostics; scikit-learn provides a
           production-oriented pipeline. The two views complement each other.
        7. A final model is incomplete without a feature contract, threshold
           policy, reproducible environment, evaluation record, and model card.
        """,
    )

    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {len(notebook.cells)} cells to {OUTPUT}")


if __name__ == "__main__":
    main()
