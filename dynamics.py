"""A simple dynamics-aware reference classifier: nearest-neighbour prediction error.

For each series x and each embedding dimension d, build delay vectors

    V[t] = (x[t], x[t-1], ..., x[t-d+1])      for t = d-1 .. n-2
    T[t] = x[t+1]                              the value that follows

and predict each T[t] from the target of its nearest neighbour in V (excluding the
vector itself). The mean absolute error of that one-step predictor is low for a
deterministic process, where similar recent history means similar next value, and
high for a series whose order has been scrambled. The feature vector of a series is
[err_1, err_2, err_3] and a random forest is trained on those.

This is a SIMPLE REFERENCE, not a proposed best method. It needs a clean series: it
is fragile to observation noise, which blurs the nearest-neighbour structure it
relies on. It exists so the app can show how far a method that looks at the dynamics
sits above the shape-matching methods on the same problem (the "gap").

Only numpy and scikit-learn are needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

NAME = "Dynamics (NN)"
DIMENSIONS = (1, 2, 3)


def nn_prediction_error(x, d):
    """Mean |T[t] - T[nn(t)]| over the delay vectors of ``x`` at embedding dimension ``d``."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < d + 2:
        raise ValueError(f"a series of {n} samples is too short for embedding dimension {d}")
    # Row k of V is the vector ending at t = d - 1 + k, newest value first.
    t = np.arange(d - 1, n - 1)
    V = np.stack([x[t - j] for j in range(d)], axis=1)
    T = x[t + 1]
    dist = ((V[:, None, :] - V[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(dist, np.inf)          # a vector is not its own neighbour
    nearest = dist.argmin(axis=1)
    return float(np.abs(T - T[nearest]).mean())


def dynamics_features(X):
    """One row of [err_1, err_2, err_3] per series in ``X``."""
    return np.array([[nn_prediction_error(row, d) for d in DIMENSIONS] for row in X])


def evaluate_dynamics(X_train, X_test, y_train, y_test):
    """Test accuracy of a random forest on the nearest-neighbour error features."""
    classifier = RandomForestClassifier(200, random_state=0)
    classifier.fit(dynamics_features(X_train), y_train)
    return float(accuracy_score(y_test, classifier.predict(dynamics_features(X_test))))
