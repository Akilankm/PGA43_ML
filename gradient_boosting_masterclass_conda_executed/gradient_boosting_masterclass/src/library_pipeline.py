from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def build_regressor(random_state: int = 42) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.85,
        validation_fraction=0.15,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=random_state,
    )


def build_classifier(random_state: int = 42) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.85,
        validation_fraction=0.15,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=random_state,
    )
