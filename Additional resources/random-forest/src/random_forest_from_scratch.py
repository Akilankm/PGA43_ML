"""Educational Random Forest classifier implemented from first principles."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    value: int | None = None

class DecisionTreeScratch:
    def __init__(self, max_depth=6, min_samples_split=2, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - np.sum(p * p)

    def _best_split(self, X, y):
        n, d = X.shape
        m = self.max_features or d
        features = self.rng.choice(d, size=min(m, d), replace=False)
        best_feature, best_threshold, best_score = None, None, np.inf
        for feature in features:
            values = np.unique(X[:, feature])
            if len(values) < 2:
                continue
            for threshold in (values[:-1] + values[1:]) / 2:
                left = X[:, feature] <= threshold
                right = ~left
                if not left.any() or not right.any():
                    continue
                score = (left.sum() * self._gini(y[left]) + right.sum() * self._gini(y[right])) / n
                if score < best_score:
                    best_feature, best_threshold, best_score = feature, float(threshold), float(score)
        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=int(np.bincount(y).argmax()))
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=int(np.bincount(y).argmax()))
        left = X[:, feature] <= threshold
        return Node(feature, threshold, self._grow(X[left], y[left], depth + 1), self._grow(X[~left], y[~left], depth + 1))

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X, float), np.asarray(y, int), 0)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        child = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, child)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.asarray(X, float)])

class RandomForestClassifierScratch:
    def __init__(self, n_estimators=25, max_depth=6, min_samples_split=2, max_features="sqrt", bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.bootstrap_indices = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        n, d = X.shape
        feature_count = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else int(self.max_features)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        self.bootstrap_indices = []
        for index in range(self.n_estimators):
            sample = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree = DecisionTreeScratch(self.max_depth, self.min_samples_split, feature_count, self.random_state + index)
            tree.fit(X[sample], y[sample])
            self.trees.append(tree)
            self.bootstrap_indices.append(sample)
        return self

    def predict(self, X):
        votes = np.vstack([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda values: np.bincount(values).argmax(), 0, votes)

    def predict_proba(self, X):
        votes = np.vstack([tree.predict(X) for tree in self.trees])
        classes = np.unique(votes)
        return np.column_stack([(votes == class_id).mean(axis=0) for class_id in classes])
