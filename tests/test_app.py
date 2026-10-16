import numpy as np
import pandas as pd
import pytest

from dynamics import evaluate_dynamics
from scoring import (DYNAMICS, chance_verdict, gap_table, majority_class_accuracy,
                     standard_error, uncertainty_table)


def tiny_csv():
    rows = []
    for i in range(8):                                   # A: one block with both labels
        rows.append(["A", i % 2, *np.arange(5.0) + i])
    for i in range(6):                                   # B: one label only
        rows.append(["B", 1, *np.arange(5.0) + i])
    rows.append([None, 0, *np.zeros(5)])                 # no condition
    return pd.DataFrame(rows, columns=["Condition", "Label", *[f"t{i}" for i in range(5)]])


def test_parse_uploaded_csv(app):
    datasets, report = app["parse_uploaded_csv"](tiny_csv())
    assert list(datasets) == ["A", "B"]                  # order of first appearance
    X, y = datasets["A"][0]
    assert X.shape == (8, 5) and set(y) == {0, 1}
    assert datasets["B"] == []                           # one label, so not scorable
    assert report["single_label_chunks"] == 1
    assert report["missing_condition_rows"] == 1


def logistic_pair(n_series=20, length=100, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(n_series):
        x = np.empty(length + 50)
        x[0] = rng.uniform(0.2, 0.8)
        for t in range(length + 49):
            x[t + 1] = 3.95 * x[t] * (1 - x[t])
        s = x[50:]
        s = (s - s.mean()) / s.std()
        X.append(s if k % 2 else rng.permutation(s))      # odd: chaotic, even: scrambled
        y.append(k % 2)
    return np.array(X), np.array(y)


def test_evaluate_dynamics_is_a_proportion_and_sees_determinism():
    X, y = logistic_pair()
    acc = evaluate_dynamics(X[:14], X[14:], y[:14], y[14:])
    assert 0.0 <= acc <= 1.0
    assert acc > 0.8                                      # order matters here, shape does not


def results_frame():
    rows = []
    # Two datasets, one condition. Per dataset the best shape method is the max of
    # Rocket and DTW: 0.9 on dataset 1, 0.7 on dataset 2.
    for ds, (rocket, dtw, dyn, fsfa) in {1: (0.9, 0.6, 1.0, 0.5), 2: (0.5, 0.7, 0.9, 0.4)}.items():
        for name, acc in (("Rocket", rocket), ("DTW", dtw), (DYNAMICS, dyn), ("fSFA", fsfa)):
            rows.append({"Dataset": ds, "Condition": "C", "Algorithm": name,
                         "Accuracy": acc, "Chance": 0.5})
    return pd.DataFrame(rows)


def test_gap_uses_per_dataset_best_and_leaves_fsfa_out():
    gaps = gap_table(results_frame())
    assert len(gaps) == 1
    row = gaps.iloc[0]
    # per-dataset gaps: 1.0-0.9 = 0.1 and 0.9-0.7 = 0.2 (fSFA must not enter the max)
    assert row["Gap"] == pytest.approx(0.15)
    assert row["Gap SE"] == pytest.approx(np.std([0.1, 0.2], ddof=1) / np.sqrt(2))
    assert row["Best shape/feature"] == pytest.approx(0.8)
    assert row["fSFA"] == pytest.approx(0.45)


def test_gap_is_empty_without_the_reference():
    frame = results_frame()
    assert gap_table(frame[frame["Algorithm"] != DYNAMICS]).empty


def test_chance_and_uncertainty():
    assert majority_class_accuracy([0, 0, 0, 1]) == 0.75
    assert standard_error([0.5]) != standard_error([0.5])            # NaN with one value
    assert chance_verdict(0.9, 0.1, 0.5) == "above chance"
    assert chance_verdict(0.7, 0.1, 0.5) == "not clearly above chance"   # 0.7 - 0.2 = 0.5
    table = uncertainty_table(results_frame())
    assert set(table["Algorithm"]) == {"Rocket", "DTW", DYNAMICS, "fSFA"}
