"""Educational NumPy implementation of KNN regression."""
from __future__ import annotations

import numpy as np


class KNNRegressorScratch:
    """Brute-force KNN regressor with uniform or distance weighting."""

    def __init__(self, n_neighbors: int = 5, p: float = 2.0, weights: str = "uniform", epsilon: float = 1e-12):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if p <= 0:
            raise ValueError("p must be positive")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.n_neighbors = int(n_neighbors)
        self.p = float(p)
        self.weights = weights
        self.epsilon = float(epsilon)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Expected X as 2D and y as matching 1D array")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed training rows")
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.sum(
            np.abs(X[:, None, :] - self.X_train_[None, :, :]) ** self.p,
            axis=2,
        ) ** (1.0 / self.p)
        indices = np.argpartition(
            distances, self.n_neighbors - 1, axis=1
        )[:, : self.n_neighbors]
        local_distances = np.take_along_axis(distances, indices, axis=1)
        local_targets = self.y_train_[indices]
        if self.weights == "uniform":
            return local_targets.mean(axis=1)
        weights = 1.0 / np.maximum(local_distances, self.epsilon)
        return np.sum(weights * local_targets, axis=1) / np.sum(weights, axis=1)
