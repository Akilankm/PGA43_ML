from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class SVMConfig:
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    class_weight: str | dict | None = None
    probability: bool = False
    random_state: int = 42


def build_svm_pipeline(config: SVMConfig | None = None) -> Pipeline:
    cfg = config or SVMConfig()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel=cfg.kernel,
                    C=cfg.C,
                    gamma=cfg.gamma,
                    class_weight=cfg.class_weight,
                    probability=cfg.probability,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def parameter_grid() -> list[dict[str, list[Any]]]:
    return [
        {
            "model__kernel": ["linear"],
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": [0.001, 0.01, 0.1, 1.0, "scale"],
        },
    ]
