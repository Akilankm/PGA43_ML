from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class PCAFromScratch:
    """Educational PCA implementation using singular value decomposition."""

    n_components: int | None = None
    mean_: NDArray[np.float64] | None = field(init=False, default=None)
    components_: NDArray[np.float64] | None = field(init=False, default=None)
    explained_variance_: NDArray[np.float64] | None = field(init=False, default=None)
    explained_variance_ratio_: NDArray[np.float64] | None = field(init=False, default=None)
    singular_values_: NDArray[np.float64] | None = field(init=False, default=None)
    n_features_in_: int | None = field(init=False, default=None)

    def fit(self, X: ArrayLike) -> "PCAFromScratch":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X_arr.shape[0] < 2:
            raise ValueError("PCA requires at least two observations")
        if not np.isfinite(X_arr).all():
            raise ValueError("X must contain finite values")

        n_samples, n_features = X_arr.shape
        maximum = min(n_samples, n_features)
        k = maximum if self.n_components is None else self.n_components
        if not isinstance(k, int) or not 1 <= k <= maximum:
            raise ValueError(f"n_components must be between 1 and {maximum}")

        self.mean_ = X_arr.mean(axis=0)
        self.n_features_in_ = n_features
        centered = X_arr - self.mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        all_variances = singular_values**2 / (n_samples - 1)

        self.components_ = vt[:k]
        self.singular_values_ = singular_values[:k]
        self.explained_variance_ = all_variances[:k]
        self.explained_variance_ratio_ = all_variances[:k] / all_variances.sum()
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        self._check_input(X_arr)
        return (X_arr - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim != 2 or Z_arr.shape[1] != self.components_.shape[0]:
            raise ValueError("Z has an incompatible shape")
        return Z_arr @ self.components_ + self.mean_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    def reconstruction_error(self, X: ArrayLike) -> float:
        X_arr = np.asarray(X, dtype=float)
        reconstructed = self.inverse_transform(self.transform(X_arr))
        return float(np.sqrt(np.mean((X_arr - reconstructed) ** 2)))

    def _check_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("fit must be called before transformation")

    def _check_input(self, X: NDArray[np.float64]) -> None:
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape")
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values")
