from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GradientBoostingRegressorScratch:
    """Educational squared-error gradient boosting regressor.

    The model is deliberately explicit: every stage fits a shallow regression
    tree to the current negative gradient, which is the residual y - F(x).
    """

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 2
    min_samples_leaf: int = 5
    subsample: float = 1.0
    random_state: int = 42

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")

        rng = np.random.default_rng(self.random_state)
        self.init_ = float(np.mean(y))
        self.estimators_ = []
        self.train_loss_ = []
        prediction = np.full(len(y), self.init_, dtype=float)
        sample_size = max(2, int(np.ceil(self.subsample * len(y))))

        for stage in range(self.n_estimators):
            negative_gradient = y - prediction
            if self.subsample < 1.0:
                index = rng.choice(len(y), size=sample_size, replace=False)
            else:
                index = np.arange(len(y))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + stage,
            )
            tree.fit(X[index], negative_gradient[index])
            prediction += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y - prediction) ** 2)))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        prediction = np.full(len(X), self.init_, dtype=float)
        for tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(X)
        return prediction

    def staged_predict(self, X) -> Iterator[np.ndarray]:
        X = np.asarray(X, dtype=float)
        prediction = np.full(len(X), self.init_, dtype=float)
        for tree in self.estimators_:
            prediction = prediction + self.learning_rate * tree.predict(X)
            yield prediction.copy()


@dataclass
class GradientBoostingBinaryClassifierScratch:
    """Educational binary log-loss gradient boosting classifier.

    Trees fit the pseudo-residual y - p. Each terminal leaf receives a Newton
    correction gamma = sum(y-p) / sum(p(1-p)), matching the core idea behind
    Friedman-style logistic gradient boosting.
    """

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 1
    min_samples_leaf: int = 8
    subsample: float = 1.0
    random_state: int = 42

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        unique = np.unique(y)
        if not np.array_equal(unique, np.array([0, 1])):
            raise ValueError("y must contain both binary classes encoded as 0 and 1")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")

        rng = np.random.default_rng(self.random_state)
        prevalence = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.init_ = float(np.log(prevalence / (1 - prevalence)))
        raw_score = np.full(len(y), self.init_, dtype=float)
        self.estimators_ = []
        self.leaf_values_ = []
        self.train_loss_ = []
        sample_size = max(2, int(np.ceil(self.subsample * len(y))))

        for stage in range(self.n_estimators):
            probability = _sigmoid(raw_score)
            pseudo_residual = y - probability
            if self.subsample < 1.0:
                index = rng.choice(len(y), size=sample_size, replace=False)
            else:
                index = np.arange(len(y))

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + stage,
            )
            tree.fit(X[index], pseudo_residual[index])
            leaf_id = tree.apply(X)
            values = {}
            for leaf in np.unique(leaf_id):
                mask = leaf_id == leaf
                numerator = np.sum(pseudo_residual[mask])
                denominator = np.sum(probability[mask] * (1 - probability[mask]))
                values[int(leaf)] = float(numerator / max(denominator, 1e-12))
            update = np.array([values[int(leaf)] for leaf in leaf_id])
            raw_score += self.learning_rate * update
            self.estimators_.append(tree)
            self.leaf_values_.append(values)
            p = np.clip(_sigmoid(raw_score), 1e-12, 1 - 1e-12)
            self.train_loss_.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        raw_score = np.full(len(X), self.init_, dtype=float)
        for tree, values in zip(self.estimators_, self.leaf_values_):
            leaf_id = tree.apply(X)
            update = np.array([values[int(leaf)] for leaf in leaf_id])
            raw_score += self.learning_rate * update
        return raw_score

    def predict_proba(self, X):
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def staged_predict_proba(self, X) -> Iterator[np.ndarray]:
        X = np.asarray(X, dtype=float)
        raw_score = np.full(len(X), self.init_, dtype=float)
        for tree, values in zip(self.estimators_, self.leaf_values_):
            leaf_id = tree.apply(X)
            update = np.array([values[int(leaf)] for leaf in leaf_id])
            raw_score += self.learning_rate * update
            p1 = _sigmoid(raw_score)
            yield np.column_stack([1 - p1, p1])
