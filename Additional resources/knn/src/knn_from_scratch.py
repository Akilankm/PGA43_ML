"""Educational NumPy implementation of KNN classification."""
from __future__ import annotations

import numpy as np


class KNNClassifierScratch:
    """Brute-force KNN classifier with Minkowski distance."""

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
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Expected X as 2D and y as matching 1D array")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed training rows")
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes_ = np.unique(y)
        return self

    def _distances(self, X):
        X = np.asarray(X, dtype=float)
        return np.sum(
            np.abs(X[:, None, :] - self.X_train_[None, :, :]) ** self.p,
            axis=2,
        ) ** (1.0 / self.p)

    def predict_proba(self, X):
        distances = self._distances(X)
        indices = np.argpartition(
            distances, self.n_neighbors - 1, axis=1
        )[:, : self.n_neighbors]
        local_distances = np.take_along_axis(distances, indices, axis=1)
        local_labels = self.y_train_[indices]

        if self.weights == "uniform":
            weights = np.ones_like(local_distances)
        else:
            weights = 1.0 / np.maximum(local_distances, self.epsilon)
            zero_mask = local_distances <= self.epsilon
            for row in range(len(weights)):
                if zero_mask[row].any():
                    weights[row] = zero_mask[row].astype(float)

        scores = np.column_stack(
            [np.sum(weights * (local_labels == cls), axis=1) for cls in self.classes_]
        )
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
