"""Chance level, uncertainty and the gap, computed from the benchmark's results table.

None of this touches Streamlit or changes an accuracy: it only reads the long table
run_benchmark returns (one row per dataset, condition and algorithm) and derives
extra columns from it.

    chance   accuracy of always predicting the majority class of a dataset's test set
    SE       standard error of a mean across datasets, std(ddof=1) / sqrt(n)
    gap      dynamics-aware reference accuracy minus the best shape or feature method,
             worked out per dataset and then averaged, so it carries its own SE
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynamics import NAME as DYNAMICS

# What the gap is measured against. fSFA is reported next to the gap, not inside it.
SHAPE_METHODS = ("Rocket", "MiniRocket", "catch22", "DTW")
SEPARATE = "fSFA"

NOT_CLEARLY_ABOVE = "not clearly above chance"
ABOVE = "above chance"
NEEDS_REPEATS = "needs 2+ datasets"


def majority_class_accuracy(y_test):
    """Accuracy of predicting the most common class of ``y_test`` for every row."""
    counts = np.unique(np.asarray(y_test), return_counts=True)[1]
    return float(counts.max() / counts.sum())


def standard_error(values):
    """Standard error of the mean, or NaN when there are fewer than two values."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def chance_verdict(mean, se, chance):
    """'not clearly above chance' when mean - 2*SE <= chance, else 'above chance'."""
    if np.isnan(se):
        return NEEDS_REPEATS
    return NOT_CLEARLY_ABOVE if mean - 2 * se <= chance else ABOVE


def uncertainty_table(df_results):
    """Per condition and algorithm: SE of the accuracy, chance level and the verdict.

    The columns are keyed by Condition and Algorithm so they can be merged onto the
    app's existing summary without disturbing its numbers.
    """
    if "Chance" not in df_results.columns:
        raise ValueError("the results table has no Chance column")
    rows = []
    for (condition, algorithm), group in df_results.groupby(["Condition", "Algorithm"], sort=False):
        mean = float(group["Accuracy"].mean())
        se = standard_error(group["Accuracy"])
        chance = float(group["Chance"].mean())
        rows.append({"Condition": str(condition), "Algorithm": algorithm,
                     "Std. error": se, "Chance": chance,
                     "Vs chance": chance_verdict(mean, se, chance)})
    return pd.DataFrame(rows)


def gap_table(df_results):
    """One row per condition for which the reference and a shape method both ran.

    ``Gap`` is the mean over datasets of (reference accuracy minus the best of the
    shape and feature methods on that dataset), with its standard error. Returns an
    empty frame when the reference was not run, so a caller can simply show nothing.
    """
    columns = ["Condition", "Reference", "Best shape/feature", "Gap", "Gap SE", "fSFA"]
    if DYNAMICS not in set(df_results["Algorithm"]):
        return pd.DataFrame(columns=columns)
    wide = df_results.pivot_table(index=["Condition", "Dataset"], columns="Algorithm",
                                  values="Accuracy", aggfunc="first", sort=False)
    rows = []
    for condition in dict.fromkeys(df_results["Condition"]):
        sub = wide.loc[condition]
        shape = [name for name in SHAPE_METHODS if name in sub.columns]
        if not shape or DYNAMICS not in sub.columns:
            continue
        best = sub[shape].max(axis=1)
        gap = sub[DYNAMICS] - best
        rows.append({
            "Condition": str(condition),
            "Reference": float(sub[DYNAMICS].mean()),
            "Best shape/feature": float(best.mean()),
            "Gap": float(gap.mean()),
            "Gap SE": standard_error(gap),
            "fSFA": float(sub[SEPARATE].mean()) if SEPARATE in sub.columns else float("nan"),
        })
    return pd.DataFrame(rows, columns=columns)


def summary_rows(title, df_results):
    """The rows one module adds to the cross-module Summary: a row per condition.

    Columns: Module, Condition, the mean accuracy of every algorithm that ran, Chance,
    and the gap with its SE (blank when the reference was not run).
    """
    unc = uncertainty_table(df_results)
    gaps = gap_table(df_results).set_index("Condition") if DYNAMICS in set(df_results["Algorithm"]) else None
    rows = []
    for condition in dict.fromkeys(df_results["Condition"]):
        sub = df_results[df_results["Condition"] == condition]
        row = {"Module": title, "Condition": str(condition)}
        for algorithm in dict.fromkeys(sub["Algorithm"]):
            row[algorithm] = float(sub.loc[sub["Algorithm"] == algorithm, "Accuracy"].mean())
        row["Chance"] = float(unc.loc[unc["Condition"] == str(condition), "Chance"].iloc[0])
        row["Gap"] = row["Gap SE"] = float("nan")
        if gaps is not None and str(condition) in gaps.index:
            row["Gap"] = float(gaps.loc[str(condition), "Gap"])
            row["Gap SE"] = float(gaps.loc[str(condition), "Gap SE"])
        row["Datasets"] = int(sub["Dataset"].nunique())
        rows.append(row)
    return rows
