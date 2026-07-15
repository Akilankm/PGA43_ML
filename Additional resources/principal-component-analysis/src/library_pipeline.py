from __future__ import annotations

from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PCAConfig:
    n_components: int | float | None = 0.95
    whiten: bool = False
    svd_solver: str = "full"
    random_state: int = 42


def build_pca_pipeline(config: PCAConfig | None = None) -> Pipeline:
    cfg = config or PCAConfig()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=cfg.n_components,
            whiten=cfg.whiten,
            svd_solver=cfg.svd_solver,
            random_state=cfg.random_state,
        )),
        ("model", LogisticRegression(max_iter=5000, random_state=cfg.random_state)),
    ])


def parameter_grid() -> dict[str, list]:
    return {
        "pca__n_components": [2, 5, 10, 15, 20, 0.90, 0.95, 0.99],
        "pca__whiten": [False, True],
        "model__C": [0.01, 0.1, 1.0, 10.0],
    }
