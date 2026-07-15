from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def build_pipeline(random_state=42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
            n_estimators=100,
            learning_rate=0.5,
            random_state=random_state,
        )),
    ])


def parameter_grid():
    return {
        "model__n_estimators": [25, 50, 100, 200],
        "model__learning_rate": [0.05, 0.1, 0.5, 1.0],
        "model__estimator__max_depth": [1, 2, 3],
        "model__estimator__min_samples_leaf": [1, 5, 10],
    }
