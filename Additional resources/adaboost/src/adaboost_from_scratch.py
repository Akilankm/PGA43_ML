from dataclasses import dataclass, field
import numpy as np
from sklearn.tree import DecisionTreeClassifier

@dataclass
class AdaBoostClassifierScratch:
    n_estimators: int = 50
    learning_rate: float = 1.0
    max_depth: int = 1
    random_state: int = 42
    estimators_: list = field(init=False, default_factory=list)
    estimator_weights_: list = field(init=False, default_factory=list)
    estimator_errors_: list = field(init=False, default_factory=list)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        labels = np.unique(y)
        if X.ndim != 2 or len(X) != len(y) or len(labels) != 2:
            raise ValueError("Expected finite two-dimensional X and binary y")
        ys = np.where(y == labels[0], -1.0, 1.0)
        row_weights = np.full(len(y), 1.0 / len(y))
        self.classes_ = labels
        for stage in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state + stage)
            tree.fit(X, ys, sample_weight=row_weights)
            pred = tree.predict(X)
            err = float(np.clip(np.sum(row_weights[pred != ys]), 1e-12, 1 - 1e-12))
            if err >= 0.5:
                break
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)
            row_weights = row_weights * np.exp(-alpha * ys * pred)
            row_weights = row_weights / row_weights.sum()
            self.estimators_.append(tree)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(err)
        return self

    def decision_function(self, X):
        if not self.estimators_:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=float)
        return sum(a * model.predict(X) for a, model in zip(self.estimator_weights_, self.estimators_))

    def predict(self, X):
        signed = np.where(self.decision_function(X) >= 0, 1, -1)
        return np.where(signed == -1, self.classes_[0], self.classes_[1])
