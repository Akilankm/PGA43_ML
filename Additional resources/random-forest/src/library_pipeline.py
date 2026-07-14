"""Leakage-safe scikit-learn Random Forest training utilities."""
from __future__ import annotations
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

@dataclass(frozen=True)
class ForestSearchConfig:
    random_state: int = 42
    folds: int = 3
    iterations: int = 10
    scoring: str = "balanced_accuracy"


def build_search(config: ForestSearchConfig = ForestSearchConfig()) -> RandomizedSearchCV:
    estimator = RandomForestClassifier(n_jobs=-1, random_state=config.random_state)
    parameters = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(config.folds, shuffle=True, random_state=config.random_state)
    return RandomizedSearchCV(
        estimator,
        parameters,
        n_iter=config.iterations,
        scoring=config.scoring,
        cv=cv,
        n_jobs=-1,
        random_state=config.random_state,
        return_train_score=True,
    )
