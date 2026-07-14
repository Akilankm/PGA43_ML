from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class LinearSVMScratch:
    """Educational binary linear soft-margin SVM using batch subgradient descent."""

    learning_rate: float = 0.001
    lambda_param: float = 0.01
    n_epochs: int = 2000
    w_: NDArray[np.float64] | None = field(init=False, default=None)
    b_: float = field(init=False, default=0.0)
    objective_history_: list[float] = field(init=False, default_factory=list)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearSVMScratch":
        if self.learning_rate <= 0 or self.lambda_param <= 0 or self.n_epochs <= 0:
            raise ValueError("hyperparameters must be positive")

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).reshape(-1)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr):
            raise ValueError("X must be 2D and match y length")
        if not np.isfinite(X_arr).all():
            raise ValueError("X must contain finite values")
        if set(np.unique(y_arr).tolist()) not in ({-1, 1}, {0, 1}):
            raise ValueError("y must contain {-1,1} or {0,1}")

        y_signed = np.where(y_arr <= 0, -1.0, 1.0)
        self.w_ = np.zeros(X_arr.shape[1], dtype=float)
        self.b_ = 0.0
        self.objective_history_ = []

        for _ in range(self.n_epochs):
            margins = y_signed * (X_arr @ self.w_ + self.b_)
            violators = margins < 1
            grad_w = self.lambda_param * self.w_
            grad_b = 0.0
            if np.any(violators):
                grad_w -= np.mean(y_signed[violators, None] * X_arr[violators], axis=0)
                grad_b -= float(np.mean(y_signed[violators]))

            self.w_ -= self.learning_rate * grad_w
            self.b_ -= self.learning_rate * grad_b

            hinge = np.maximum(0.0, 1.0 - y_signed * (X_arr @ self.w_ + self.b_))
            objective = 0.5 * self.lambda_param * float(self.w_ @ self.w_) + float(hinge.mean())
            self.objective_history_.append(objective)
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.w_ is None:
            raise RuntimeError("fit must be called before prediction")
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        return np.where(self.decision_function(X) >= 0, 1, -1)
