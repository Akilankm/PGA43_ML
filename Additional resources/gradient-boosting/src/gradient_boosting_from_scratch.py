from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.tree import DecisionTreeRegressor


@dataclass
class GradientBoostingRegressorScratch:
    """Educational squared-error gradient boosting regressor."""

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 2
    min_samples_leaf: int = 5
    random_state: int = 42
    init_: float | None = field(init=False, default=None)
    estimators_: list[DecisionTreeRegressor] = field(init=False, default_factory=list)
    train_loss_: list[float] = field(init=False, default_factory=list)

    def _validate_hyperparameters(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GradientBoostingRegressorScratch":
        self._validate_hyperparameters()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y have inconsistent lengths")
        if not np.isfinite(X_arr).all() or not np.isfinite(y_arr).all():
            raise ValueError("X and y must contain finite values")

        self.init_ = float(np.mean(y_arr))
        self.estimators_ = []
        self.train_loss_ = []
        prediction = np.full(y_arr.shape, self.init_, dtype=float)

        for stage in range(self.n_estimators):
            residual = y_arr - prediction
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + stage,
            )
            tree.fit(X_arr, residual)
            prediction += self.learning_rate * tree.predict(X_arr)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y_arr - prediction) ** 2)))
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.init_ is None:
            raise RuntimeError("fit must be called before predict")
        X_arr = np.asarray(X, dtype=float)
        prediction = np.full(X_arr.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(X_arr)
        return prediction

    def staged_predict(self, X: ArrayLike):
        if self.init_ is None:
            raise RuntimeError("fit must be called before staged_predict")
        X_arr = np.asarray(X, dtype=float)
        prediction = np.full(X_arr.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            prediction = prediction + self.learning_rate * tree.predict(X_arr)
            yield prediction.copy()
