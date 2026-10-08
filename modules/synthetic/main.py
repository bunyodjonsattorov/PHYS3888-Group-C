"""The app's original built-in benchmark, as a module.

Two labelled tasks, 50 samples of length 100 each:

    Constant Parameter   AR(1) with coefficient 0.4 against 0.8
    Non-Stationary       random walk with drift +0.05 that flips to -0.05 halfway
                         through the record, against a steady +0.15

This is a port of the generator that used to live inside ``app.py``. The seeded
draws are reproduced exactly, so dataset 1 here is dataset 1 there: the recursion
is vectorised across the 50 samples, which changes the speed but not a single
value. The original built 100 datasets of each kind in one pass and the app's
slider defaulted to evaluating 10, so shipping the first ``N_DATASETS`` of each
keeps the default run identical while keeping the committed CSV small. Raise
``N_DATASETS`` if you want more.

Each dataset is 50 contiguous rows holding both labels, because ``app.py`` scores
each block of 50 rows and a single-class block is always 100%.

Run standalone to refresh the CSV::

    python3 modules/synthetic/main.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TITLE = "Synthetic AR and drift"
DESCRIPTION = (
    "The original built-in benchmark: AR(1) pairs, and random walks whose drift "
    "flips sign partway through."
)

N_DATASETS = 10
SAMPLES = 50
LENGTH = 100
SEED = 42

STATIONARY = "Constant Parameter"
NON_STATIONARY = "Non-Stationary"


def _stationary(rng: np.random.RandomState) -> list[tuple[np.ndarray, np.ndarray]]:
    """AR(1) pairs: coefficient 0.4 against 0.8."""
    out = []
    for _ in range(100):
        coefficients = np.where(np.arange(SAMPLES) < SAMPLES // 2, 0.4, 0.8)
        noise = rng.normal(0.0, 1.0, size=(SAMPLES, LENGTH))
        X = np.zeros((SAMPLES, LENGTH))
        for t in range(1, LENGTH):
            X[:, t] = coefficients * X[:, t - 1] + noise[:, t]
        y = np.where(np.arange(SAMPLES) < SAMPLES // 2, 0, 1)
        out.append((X, y))
    return out


def _non_stationary(rng: np.random.RandomState) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random walks: drift +0.05 flipping to -0.05 at the midpoint, against 0.15."""
    first_half = SAMPLES // 2
    steps = np.arange(LENGTH)
    drift = np.zeros((SAMPLES, LENGTH))
    drift[:first_half, 1:] = np.where(steps[1:] <= 50, 0.05, -0.05)
    drift[first_half:, 1:] = 0.15

    out = []
    for _ in range(100):
        noise = rng.normal(0.0, 1.0, size=(SAMPLES, LENGTH))
        X = np.zeros((SAMPLES, LENGTH))
        for t in range(1, LENGTH):
            X[:, t] = X[:, t - 1] + drift[:, t] + noise[:, t]
        y = np.where(np.arange(SAMPLES) < first_half, 0, 1)
        out.append((X, y))
    return out


def datasets(n: int = N_DATASETS):
    """Yield ``(condition, X, y)`` for the first ``n`` datasets of each condition."""
    rng = np.random.RandomState(SEED)
    for condition, built in ((STATIONARY, _stationary(rng)),
                             (NON_STATIONARY, _non_stationary(rng))):
        for X, y in built[:n]:
            yield condition, X, y


def generate_csv(path, n: int = N_DATASETS, length: int = LENGTH):
    """Write the module's CSV in the format the app parses. Returns the path."""
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Condition", "Label"] + [f"t{i}" for i in range(length)])
        for condition, X, y in datasets(n):
            for row, label in zip(X, y):
                writer.writerow([condition, int(label)]
                                + [f"{v:.6f}" for v in row[:length]])
    return path


def main() -> None:
    path = generate_csv(Path(__file__).with_name("data.csv"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
