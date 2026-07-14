from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class GradientBoostingConfig:
    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 2
    min_samples_leaf: int = 5
    subsample: float = 0.8
    random_state: int = 42


def build_regression_pipeline(config: GradientBoostingConfig | None = None) -> Pipeline:
    """Create a leakage-safe scikit-learn pipeline."""
    cfg = config or GradientBoostingConfig()
    model = GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        subsample=cfg.subsample,
        random_state=cfg.random_state,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", model),
    ])


def parameter_grid() -> dict[str, list[Any]]:
    return {
        "model__n_estimators": [100, 200, 400],
        "model__learning_rate": [0.01, 0.03, 0.1],
        "model__max_depth": [1, 2, 3],
        "model__min_samples_leaf": [3, 5, 10],
        "model__subsample": [0.7, 0.9, 1.0],
    }
