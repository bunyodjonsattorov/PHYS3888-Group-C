"""Time-series framework benchmarking app.

The app shows one tab per module, plus a Home tab that lists the modules and keeps
the manual CSV upload. A module is a folder under ``modules/`` holding ``main.py``
(which writes the module's CSV) and ``main.html`` (the page shown in its tab). See
the Modules section of README.md for how to add one.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Specialised time-series libraries. When one is not installed, the algorithms that
# need it are skipped and reported as such, never replaced by a stand-in.
try:
    from sktime.transformations.panel.rocket import MiniRocket, Rocket
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False

try:
    import pycatch22
    CATCH22_AVAILABLE = True
except ImportError:
    CATCH22_AVAILABLE = False

try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False

try:
    from fsfa import fSFA   # from the file fsfa.py
    FSFA_AVAILABLE = True
except ImportError:
    FSFA_AVAILABLE = False

from dynamics import NAME as DYNAMICS, evaluate_dynamics   # numpy + scikit-learn only
from ui import csv_download, plain, show_chart, show_explanations, show_problem
from scoring import (gap_table, majority_class_accuracy, summary_rows,
                     uncertainty_table)

MODULES_DIR = Path(__file__).resolve().parent / "modules"
CSV_NAME = "data.csv"

# Every algorithm the app knows. The benchmark and the progress bar both step through
# the ones that can actually run (see active_algorithms), in this order, so the bar's
# plan and the work actually done cannot drift apart. Conditions are not listed here:
# they are whatever values a CSV's Condition column holds.
ALGORITHMS = ("Rocket", "MiniRocket", "catch22", "DTW", "fSFA")


def algorithm_available(name):
    """Whether the library behind an algorithm is installed."""
    return {"Rocket": SKTIME_AVAILABLE, "MiniRocket": SKTIME_AVAILABLE,
            "catch22": CATCH22_AVAILABLE, "DTW": DTW_AVAILABLE,
            "fSFA": FSFA_AVAILABLE}[name]


def run_algorithms():
    """What a run steps through: the installed algorithms, plus the dynamics-aware
    reference when the sidebar checkbox is on. With it off this is exactly
    active_algorithms(), so nothing about an ordinary run changes."""
    algorithms = active_algorithms()
    if st.session_state.get("include_dynamics"):
        algorithms += (DYNAMICS,)
    return algorithms


def active_algorithms():
    """The algorithms that can really run, in ALGORITHMS order.

    A missing library means its algorithms are skipped. They are never swapped for
    a stand-in (random guesses, a cruder method) reported under the same name,
    because a score that looks real but is not is worse than no score.
    """
    return tuple(name for name in ALGORITHMS if algorithm_available(name))


# ==========================================
# 1. CSV PARSING
# ==========================================
def parse_uploaded_csv(df):
    """
    Parses a CSV into datasets, grouped by condition.
    Expects columns: 'Condition' (any text; each distinct value is one condition),
    'Label' (0 or 1), and the remaining columns treated as time-series steps.

    Returns ``(datasets, report)``. ``datasets`` maps each condition, in the order it
    first appears in the CSV, to its list of ``(X, y)`` datasets (blocks of up to 50
    rows). A condition with no evaluable block maps to an empty list, so the caller
    can say so. ``report`` records the rows and chunks that had to be skipped, so the
    caller can surface them instead of scoring them silently.
    """
    report = {
        "missing_condition_rows": 0,
        "single_label_chunks": 0,
        "tiny_chunks": 0,
    }
    if "Condition" not in df.columns or "Label" not in df.columns:
        st.error("Uploaded CSV must contain 'Condition' and 'Label' columns alongside sequential features.")
        return {}, report

    datasets = {}

    # Isolate time series data columns
    ts_cols = [c for c in df.columns if c not in ["Condition", "Label"]]

    # A row with no condition belongs to no task, so it cannot be scored.
    missing = df["Condition"].isna() | (df["Condition"].astype(str).str.strip() == "")
    report["missing_condition_rows"] = int(missing.sum())
    df = df[~missing]

    # Split by condition, keeping the order the conditions first appear in the CSV.
    for cond, sub_df in df.groupby("Condition", sort=False):
        datasets[str(cond)] = []
        # Chunk rows arbitrarily into pseudo-individual 'datasets' to preserve the
        # comparison interface: groups of up to 50 samples per dataset iteration.
        chunk_size = 50
        chunks = [sub_df[i:i + chunk_size] for i in range(0, sub_df.shape[0], chunk_size)]

        for chunk in chunks:
            if chunk.shape[0] < 4:
                report["tiny_chunks"] += 1
                continue  # skip empty/tiny fragments
            X = chunk[ts_cols].to_numpy()
            y = chunk["Label"].to_numpy().astype(int)

            if len(set(y.tolist())) < 2:
                # A chunk holding one class cannot be scored: a constant predictor
                # returns 100% whatever algorithm is used, so reporting it as a
                # score would be misleading.
                report["single_label_chunks"] += 1
                continue

            datasets[str(cond)].append((X, y))

    return datasets, report


def show_parse_report(report):
    """Surface whatever parse_uploaded_csv had to skip, instead of failing silently."""
    if report["missing_condition_rows"]:
        st.warning(
            f"Ignored {report['missing_condition_rows']} row(s) with an empty 'Condition', "
            "because they belong to no task."
        )
    if report["single_label_chunks"]:
        st.warning(
            f"Skipped {report['single_label_chunks']} dataset chunk(s) that contained only one "
            "class. Accuracy on a single-class chunk is always 100% no matter which algorithm "
            "runs, so those chunks are excluded rather than reported as a result."
        )
    if report["tiny_chunks"]:
        st.info(f"Skipped {report['tiny_chunks']} row fragment(s) with fewer than 4 samples.")


# ==========================================
# 2. FEATURE EXTRACTION & ALGORITHM CORES
# ==========================================
def evaluate_rocket(X_train, X_test, y_train, y_test, mini=False):
    if not SKTIME_AVAILABLE:
        raise RuntimeError("sktime is not installed, so Rocket and MiniRocket cannot run.")

    start_time = time.time()
    X_train_df = pd.DataFrame({"dim_0": [pd.Series(x) for x in X_train]})
    X_test_df = pd.DataFrame({"dim_0": [pd.Series(x) for x in X_test]})

    transformer = MiniRocket(random_state=42) if mini else Rocket(num_kernels=500, random_state=42)
    X_train_transform = transformer.fit_transform(X_train_df)
    X_test_transform = transformer.transform(X_test_df)

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)
    preds = classifier.predict(X_test_transform)

    return accuracy_score(y_test, preds), time.time() - start_time


def evaluate_catch22(X_train, X_test, y_train, y_test):
    if not CATCH22_AVAILABLE:
        raise RuntimeError("pycatch22 is not installed, so catch22 cannot run.")
    start_time = time.time()

    def extract_features(X):
        feats = []
        for row in X:
            feats.append(pycatch22.catch22_all(list(row))['values'])
        return np.array(feats)
    X_train_feat = extract_features(X_train)
    X_test_feat = extract_features(X_test)

    classifier = RidgeClassifierCV()
    classifier.fit(X_train_feat, y_train)
    preds = classifier.predict(X_test_feat)
    return accuracy_score(y_test, preds), time.time() - start_time


def evaluate_dtw(X_train, X_test, y_train, y_test, on_test_row=None):
    """1-NN under DTW.

    ``on_test_row(scored, total)``, when given, is called after each test series
    is scored. DTW is by far the slowest algorithm here, so this is the only way
    a progress bar can move *within* one of its steps rather than sitting still
    for minutes.
    """
    if not DTW_AVAILABLE:
        raise RuntimeError("fastdtw is not installed, so DTW cannot run.")
    start_time = time.time()
    preds = []

    for scored, test_row in enumerate(X_test, start=1):
        best_dist = float('inf')
        best_label = 0
        for train_row, label in zip(X_train, y_train):
            # Absolute point diff lambda bypasses SciPy 1-D shape validation.
            dist, _ = fastdtw(test_row, train_row, dist=lambda a, b: np.abs(a - b))
            if dist < best_dist:
                best_dist = dist
                best_label = label
        preds.append(best_label)
        if on_test_row is not None:
            on_test_row(scored, len(X_test))

    return accuracy_score(y_test, preds), time.time() - start_time


def evaluate_fsfa(X_train, X_test, y_train, y_test, n_components=3):
    if not (FSFA_AVAILABLE and CATCH22_AVAILABLE):
        raise RuntimeError("fsfa (with sksfa and pycatch22) is not installed, so fSFA cannot run.")
    start_time = time.time()

    T = X_train.shape[1]
    window = max(10, T // 4)               # catch22 needs a reasonable window
    stride = max(1, window // 8)           # ~25+ windows, more than the 24 features

    def summarise(row):
        S = fSFA(np.asarray(row, dtype=float).reshape(-1, 1), window, stride,
                 featureset="catch24", n_components=n_components)
        S = np.nan_to_num(np.asarray(S))
        d = np.diff(S, axis=0)
        z = (S - S.mean(0)) / (S.std(0) + 1e-12)
        feats = [
            np.abs((z ** 3).mean(0)),                       # |skewness|, sign-invariant
            (z ** 4).mean(0),                               # kurtosis
            np.abs(d).mean(0),                              # mean absolute change
            np.abs((z[1:] * z[:-1]).mean(0)),               # |lag-1 autocorrelation|
            (np.diff(np.sign(z), axis=0) != 0).mean(0),     # zero-crossing rate
        ]
        return np.concatenate(feats)

    X_train_feat = np.array([summarise(r) for r in X_train])
    X_test_feat = np.array([summarise(r) for r in X_test])

    classifier = RidgeClassifierCV()
    classifier.fit(X_train_feat, y_train)
    preds = classifier.predict(X_test_feat)
    return accuracy_score(y_test, preds), time.time() - start_time


def evaluate_dynamics_timed(X_train, X_test, y_train, y_test):
    """dynamics.evaluate_dynamics, returning (accuracy, seconds) like the others."""
    start_time = time.time()
    return evaluate_dynamics(X_train, X_test, y_train, y_test), time.time() - start_time


# ==========================================
# 3. MODULES
# ==========================================
def discover_modules():
    """Every folder under modules/ holding both main.py and main.html, in name order."""
    if not MODULES_DIR.is_dir():
        return []
    found = []
    for folder in sorted(MODULES_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue
        main_py = folder / "main.py"
        main_html = folder / "main.html"
        if main_py.is_file() and main_html.is_file():
            found.append({"name": folder.name, "dir": folder, "py": main_py, "html": main_html})
    return found


def load_module_main(main_py):
    """Import a module's main.py by path, so modules need no __init__.py."""
    name = f"_module_{main_py.parent.name}"
    spec = importlib.util.spec_from_file_location(name, main_py)
    module = importlib.util.module_from_spec(spec)
    # @dataclass and similar look the module up in sys.modules while it executes.
    sys.modules[name] = module
    # Never drop __pycache__ folders into the tree for a module we only import.
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


@st.cache_resource(show_spinner=False)
def module_object(main_py: str, modified: float):
    """The imported module, keyed on the file's mtime so edits take effect."""
    return load_module_main(Path(main_py))


@st.cache_data(show_spinner=False)
def module_csv_bytes(folder: str, py_modified: float, csv_modified: float) -> bytes:
    """Read the module's committed CSV, or rebuild it from main.py when stale."""
    csv_path = Path(folder) / CSV_NAME
    if csv_modified and csv_modified >= py_modified:
        return csv_path.read_bytes()
    load_module_main(Path(folder) / "main.py").generate_csv(csv_path)
    return csv_path.read_bytes()


def describe_module(found):
    """Attach the optional TITLE and DESCRIPTION a module's main.py may define."""
    info = {"name": found["name"], "dir": found["dir"], "py": found["py"],
            "html": found["html"], "title": found["name"], "description": "", "error": None}
    try:
        module = module_object(str(found["py"]), found["py"].stat().st_mtime)
        info["title"] = getattr(module, "TITLE", found["name"])
        info["description"] = getattr(module, "DESCRIPTION", "")
    except Exception as exc:  # a broken module should not take the app down
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def module_frame(info):
    """The module's CSV as a DataFrame, rebuilt if its main.py is newer."""
    csv_path = info["dir"] / CSV_NAME
    payload = module_csv_bytes(
        str(info["dir"]),
        info["py"].stat().st_mtime,
        csv_path.stat().st_mtime if csv_path.exists() else 0.0,
    )
    return pd.read_csv(io.BytesIO(payload))


# ==========================================
# 4. BENCHMARK
# ==========================================
def run_benchmark(data_by_condition, num_datasets, test_size, on_progress=None,
                  algorithms=None):
    """Score the algorithms on the first num_datasets of each condition.

    ``data_by_condition`` maps each condition to its list of ``(X, y)`` datasets, as
    parse_uploaded_csv returns them. ``algorithms`` defaults to the ones whose
    libraries are installed.

    One step is one algorithm on one dataset, so a run is
    ``num_datasets * len(data_by_condition) * len(algorithms)`` steps in total.

    ``on_progress(completed, total, label, seconds)``, when given, is called as
    the run advances: before each step with ``seconds=None``, after each step with
    the time it took, and during a DTW step with a fractional ``completed`` as it
    scores test series. ``label`` describes the work in progress, so a caller can
    show what is running and how long the rest will take.
    """
    algorithms = active_algorithms() if algorithms is None else tuple(algorithms)
    total = num_datasets * len(data_by_condition) * len(algorithms)
    completed = 0
    results = []
    for i in range(num_datasets):
        for condition, data in data_by_condition.items():
            X, y = data[i]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i)

            label = f"Dataset {i + 1} of {num_datasets} · {condition}"

            def report_dtw_row(scored, rows, _label=label):
                # The last row needs no update of its own: the completion callback
                # that follows carries the same value. Leaving it out keeps
                # ``completed`` fractional for every in-step update, so a whole
                # number with no time always means "this step is starting".
                if on_progress is not None and scored < rows:
                    on_progress(completed + scored / rows, total,
                                f"{_label} · DTW · {scored} of {rows} test series", None)

            # Indexed by name so the loop below runs exactly ``algorithms``, which is
            # the order the progress bar's plan counts.
            runners = {
                "Rocket": lambda: evaluate_rocket(X_train, X_test, y_train, y_test, mini=False),
                "MiniRocket": lambda: evaluate_rocket(X_train, X_test, y_train, y_test, mini=True),
                "catch22": lambda: evaluate_catch22(X_train, X_test, y_train, y_test),
                "DTW": lambda: evaluate_dtw(X_train, X_test, y_train, y_test,
                                            on_test_row=report_dtw_row),
                "fSFA": lambda: evaluate_fsfa(X_train, X_test, y_train, y_test),
                DYNAMICS: lambda: evaluate_dynamics_timed(X_train, X_test, y_train, y_test),
            }

            for name in algorithms:
                step = f"{label} · {name}"
                if on_progress is not None:
                    on_progress(completed, total, step, None)
                acc, elapsed = runners[name]()
                results.append({"Dataset": i + 1, "Condition": condition,
                                "Algorithm": name, "Accuracy": acc, "Time (s)": elapsed,
                                "Chance": majority_class_accuracy(y_test)})
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total, step, elapsed)
    return pd.DataFrame(results)


def describe_duration(seconds):
    """A duration as '2h 5m', '3m 12s' or '14s' -- long runs need the hours."""
    total = int(round(max(seconds, 0.0)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def run_benchmark_plan(num_datasets, conditions, algorithms=None):
    """The steps run_benchmark will take, in order: (dataset number, condition, algorithm)."""
    algorithms = active_algorithms() if algorithms is None else tuple(algorithms)
    return [(i + 1, condition, name)
            for i in range(num_datasets)
            for condition in conditions
            for name in algorithms]


def make_progress_reporter(plan, bar, timing, started):
    """Draw the progress bar, the running step and the elapsed/remaining caption.

    Returns the callback handed to run_benchmark. ``plan`` gives the step order,
    which is both what the caption counts and what the remaining-time estimate
    adds up.
    """
    total = len(plan)
    observed = {}   # algorithm -> its finished step durations, for the estimate
    planned = {name for _, _, name in plan}   # the algorithms this run will time

    def report(completed, _total, label, seconds):
        if seconds is not None:
            # A completed step, so this is the algorithm that just finished. The
            # fractional updates inside a DTW step report no time and are not
            # charged as done work here.
            observed.setdefault(plan[int(completed) - 1][2], []).append(seconds)

        bar.progress(min(completed / total, 1.0),
                     text=f"{label} ({int(completed)} of {total} steps)")

        # Only estimate once every algorithm has a measured step: before the first
        # DTW has finished, an estimate would price the remaining DTW steps at the
        # cost of the Rocket ones and promise minutes for an hour of work.
        means = {name: sum(times) / len(times) for name, times in observed.items() if times}
        if len(means) == len(planned):
            remaining = sum(means[name] for _, _, name in plan[int(completed):])
            left = f" · about {describe_duration(remaining)} left"
        else:
            left = " · estimating the remaining time"
        timing.caption(f"Elapsed {describe_duration(time.time() - started)}{left}.")

    return report


def show_gap(df_results):
    """The gap table, only for the conditions where the dynamics reference was run."""
    gaps = gap_table(df_results)
    if gaps.empty:
        return
    st.subheader("↕️ Gap to the dynamics-aware reference")
    st.dataframe(plain(gaps).style.format(
        {"Dynamics reference": "{:.1%}", "Best shape/feature method": "{:.1%}", "Gap": "{:+.1%}",
         "Gap uncertainty (± std. error)": "{:.1%}", "fSFA": "{:.1%}"}, na_rep="–"),
        hide_index=True)
    st.caption("Gap = the dynamics reference minus the best shape or feature method, per "
               "dataset, then averaged. fSFA is shown beside it, not included.")


def show_results(df_results, label="results"):
    # Every figure below is a mean over the datasets the user chose to evaluate,
    # never a single train/test split, so the winner is decided on that average.
    # Conditions are listed in the order the benchmark first ran them, which is the
    # order they appear in the CSV.
    conditions = list(df_results["Condition"].unique())
    summary = (df_results
               .assign(Condition=pd.Categorical(df_results["Condition"], categories=conditions, ordered=True))
               .groupby(["Condition", "Algorithm"], observed=True)[["Accuracy", "Time (s)"]]
               .mean().reset_index())
    summary["Condition"] = summary["Condition"].astype(str)
    # Extra columns only: the standard error across datasets, the chance level and
    # whether the mean clears it. The accuracy and time columns are left untouched.
    summary = summary.merge(uncertainty_table(df_results), on=["Condition", "Algorithm"], how="left")
    n_datasets = int(df_results["Dataset"].nunique())
    basis = "1 dataset" if n_datasets == 1 else f"{n_datasets} datasets"

    show_chart(summary)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Performance Benchmarks Summary")
        st.dataframe(plain(summary, mean_accuracy=True).style.format(
            {"Mean accuracy": "{:.2%}", "Time (s)": "{:.4f}s",
             "Uncertainty (± std. error)": "{:.2%}", "Chance level": "{:.1%}"}))
        st.caption(f"Mean over {basis}; each dataset is a block of up to 50 rows with "
                   "its own train/test split.")
        csv_download(plain(summary, mean_accuracy=True), label, "results_table", key=f"dl:{label}:table")

    with col2:
        st.subheader("💡 Key Takeaways")
        for cond in conditions:
            cond_df = summary[summary["Condition"] == cond]
            if not cond_df.empty:
                best_acc = cond_df["Accuracy"].max()
                # Algorithms separated by less than float noise are not "the winner",
                # so a tie is reported as a tie instead of crowning whichever sorted first.
                top = cond_df.loc[np.isclose(cond_df["Accuracy"], best_acc), "Algorithm"].tolist()
                if len(top) == 1:
                    st.write(f"🔹 Over {basis}, **{top[0]}** yielded peak performance on "
                             f"**{cond}** at **{best_acc:.1%}** accuracy.")
                else:
                    names = ", ".join(f"**{a}**" for a in top[:-1]) + f" and **{top[-1]}**"
                    st.write(f"🔹 Over {basis}, {names} tied at the top on **{cond}** at "
                             f"**{best_acc:.1%}** accuracy.")

    show_gap(df_results)

    st.subheader("📊 Granular Dataset Evaluation Matrix")
    st.dataframe(plain(df_results))
    csv_download(plain(df_results), label, "per_dataset_results", key=f"dl:{label}:datasets")
    show_explanations()


def render_benchmark(df, key, test_size, note=None, title=None):
    """Parse a CSV, then offer the dataset count and the big Execute Benchmark button."""
    datasets, report = parse_uploaded_csv(df)
    show_parse_report(report)

    unusable = [cond for cond, found in datasets.items() if not found]
    if unusable:
        st.warning("No evaluable dataset under the condition(s) " + ", ".join(unusable)
                   + ", so they are left out.")
    datasets = {cond: found for cond, found in datasets.items() if found}

    if not datasets:
        st.error(
            "No evaluable datasets found. Each 50-row block must sit under one 'Condition' "
            "and contain both Label 0 and Label 1 rows, because a block with a single "
            "class is always scored at 100%."
        )
        return

    counts = {cond: len(found) for cond, found in datasets.items()}
    max_available = min(counts.values())
    if len(set(counts.values())) > 1:
        st.caption("Datasets per condition: "
                   + ", ".join(f"{cond}: {n}" for cond, n in counts.items())
                   + f". Every condition is scored on the same number, so at most {max_available}.")

    if note:
        st.caption(note)

    algorithms = run_algorithms()
    if not active_algorithms():
        st.error("None of the algorithms can run, because sktime, pycatch22 and fastdtw "
                 "are all missing. Install them (see README.md) and rerun.")
        return
    skipped = [name for name in ALGORITHMS if name not in algorithms]
    if skipped:
        st.warning("Not installed, so not run: " + ", ".join(skipped) + ". The results "
                   "below cover only the algorithms that ran.")

    if max_available == 1:
        # st.slider rejects min_value == max_value, which a single-dataset CSV produces.
        num_datasets = 1
        st.caption("Datasets to evaluate: 1")
    else:
        num_datasets = st.slider(f"Datasets to evaluate (max {max_available})",
                                 1, max_available, max_available, key=f"{key}:n")

    if st.button("Execute Benchmark", type="primary", use_container_width=True, key=f"{key}:run"):
        # One step per algorithm per dataset, and a real bar over them: the spinner
        # this replaces showed nothing for up to an hour on the fractal module.
        plan = run_benchmark_plan(num_datasets, list(datasets), algorithms)
        started = time.time()
        with st.status(f"Running {', '.join(algorithms)} ({len(plan)} steps)...",
                       expanded=True) as status:
            bar = st.progress(0.0, text=f"Starting the first of {len(plan)} steps...")
            timing = st.empty()
            try:
                results = run_benchmark(datasets, num_datasets, test_size,
                                        on_progress=make_progress_reporter(plan, bar, timing, started),
                                        algorithms=algorithms)
            except Exception:
                # Reachable: a CSV of 8-sample records parses and then makes MiniRocket
                # raise (n_timepoints must be >= 9). Streamlit already flips the box to
                # its error state when the exception leaves this block, so the label is
                # the part worth setting -- it would otherwise still read "Running
                # Rocket, MiniRocket, catch22 and DTW (N steps)..." beneath a red icon.
                status.update(label="Benchmark failed", state="error")
                raise
            elapsed = describe_duration(time.time() - started)
            bar.progress(1.0, text=f"All {len(plan)} steps finished")
            timing.caption(f"Elapsed {elapsed}.")
            status.update(label=f"Benchmark complete: {len(plan)} steps in {elapsed}",
                          state="complete")
        # Kept for the Summary tab, which outlives the tab switch that clears this view.
        st.session_state.setdefault("summaries", {})[key] = (title or "Uploaded CSV", results)
        show_results(results, label=title or key)


def run_all_modules(modules, test_size, max_datasets):
    """Benchmark every module in turn behind one progress bar.

    A module that cannot be loaded or that fails is reported and skipped, and the others
    still run. Each finished module is saved to ``st.session_state`` straight away, so
    stopping the run part way keeps what was done. Returns ``[(title, message, details)]``
    for the modules that did not run.
    """
    algorithms = run_algorithms()
    problems, jobs = [], []
    for info in modules:
        if info["error"]:
            problems.append((info["title"], "its main.py could not be loaded", info["error"]))
            continue
        try:
            datasets, _ = parse_uploaded_csv(module_frame(info))
            datasets = {cond: found for cond, found in datasets.items() if found}
            if not datasets:
                raise ValueError("no 50-row block holds both labels")
            jobs.append((info, datasets, min(max_datasets, min(len(f) for f in datasets.values()))))
        except Exception as exc:
            problems.append((info["title"], f"its data could not be prepared ({exc})",
                             traceback.format_exc()))
    total = sum(n * len(datasets) * len(algorithms) for _, datasets, n in jobs) or 1
    bar = st.progress(0.0, text=f"Starting {len(jobs)} modules...")
    done = 0
    for position, (info, datasets, n) in enumerate(jobs, start=1):
        steps = n * len(datasets) * len(algorithms)

        def report(completed, _total, label, seconds, _done=done, _info=info, _position=position):
            bar.progress(min((_done + completed) / total, 1.0),
                         text=f"Module {_position} of {len(jobs)}: {_info['title']} · {label}")

        try:
            results = run_benchmark(datasets, n, test_size, on_progress=report, algorithms=algorithms)
        except Exception as exc:
            problems.append((info["title"], f"the benchmark stopped part way ({exc})",
                             traceback.format_exc()))
        else:
            st.session_state.setdefault("summaries", {})[f"module:{info['name']}"] = (
                info["title"], results)
        done += steps
    bar.progress(1.0, text=f"Finished {len(jobs)} module(s)" + (f", {len(problems)} skipped" if problems else ""))
    return problems


def show_summary(modules, test_size):
    """One row per module and condition from every benchmark run in this session."""
    st.markdown("Run each module's benchmark in its own tab, or run them all here; the "
                "results are collected in the table below. Turn on the dynamics-aware "
                "reference in the sidebar first to fill the Gap columns.")
    usable = [m for m in modules if not m["error"]]
    if active_algorithms() and usable:
        per_module = st.slider("Datasets per module", 1, 10, 2, key="all:n",
                               help="DTW on long series is slow: about 3 minutes per dataset "
                                    "on the Fractal module. Start with 1 or 2.")
        if st.button("Run all modules", type="primary", key="all:run"):
            for title, message, details in run_all_modules(modules, test_size, per_module):
                show_problem(f"{title} was skipped: {message}.", details)
    stored = st.session_state.get("summaries", {})
    if not stored:
        st.info("Nothing to summarise yet. Open a module tab and press **Execute Benchmark**.")
        return
    rows = [row for title, results in stored.values() for row in summary_rows(title, results)]
    table = pd.DataFrame(rows)
    ordered = [name for name in (*ALGORITHMS, DYNAMICS) if name in table.columns]
    table = plain(table[["Module", "Condition", *ordered, "Chance", "Gap", "Gap SE", "Datasets"]])
    formats = {name: "{:.1%}" for name in (*ordered, "Chance level", "Gap uncertainty (± std. error)")}
    formats["Gap"] = "{:+.1%}"
    st.dataframe(table.style.format(formats, na_rep="–"), hide_index=True)
    st.caption("Mean accuracy over the datasets each run used. The gap is blank when the "
               "reference was not run for that module.")
    show_explanations()
    csv_download(table, "all_modules", "summary", key="dl:summary")


# ==========================================
# 5. MODULE PAGE
# ==========================================
def show_module_page(html_path):
    """Show the module's main.html verbatim, sized to its own content.

    st.iframe reads a local HTML file and measures it, so the page needs no
    wrapper, no injected resize script and no guessed height, and a module
    author's markup is rendered exactly as written.
    """
    st.iframe(html_path, height="content")


# ==========================================
# 6. STREAMLIT APPLICATION INTERFACE
# ==========================================
st.set_page_config(page_title="Time-Series Framework Benchmark", layout="wide")

st.sidebar.header("⚙️ Evaluation Settings")
test_size = st.sidebar.slider("Test Set Split Proportion", 0.1, 0.5, 0.3)

st.sidebar.checkbox(
    "Include dynamics-aware reference (NN)", key="include_dynamics",
    help="A simple reference, not a proposed best method: it scores how predictable "
         "each series is one step ahead from its nearest neighbours in delay-vector "
         "space. It is low for a scrambled order and high for deterministic dynamics, "
         "and it is fragile to observation noise.")

st.sidebar.header("📦 System Library Status")
NOT_INSTALLED = "❌ not installed, skipped"
st.sidebar.write(f"{'✅' if SKTIME_AVAILABLE else NOT_INSTALLED} sktime (Rocket, MiniRocket)")
st.sidebar.write(f"{'✅' if CATCH22_AVAILABLE else NOT_INSTALLED} pycatch22")
st.sidebar.write(f"{'✅' if DTW_AVAILABLE else NOT_INSTALLED} fastdtw")
st.sidebar.write(f"{'✅' if FSFA_AVAILABLE else NOT_INSTALLED} fsfa (sksfa)")

st.title("⏱️ Time-Series Algorithm Benchmarking Suite")
st.markdown("""
Compare how **Rocket**, **MiniRocket**, **DTW** and **catch22** classify each module's labelled
time series, under each of the conditions that module defines.
""")

modules = [describe_module(found) for found in discover_modules()]

home_tab, *module_tabs, summary_tab = st.tabs(
    ["Home"] + [m["title"] for m in modules] + ["Summary"])

with home_tab:
    st.markdown("### Modules", anchors=False)
    st.caption("One tab per module. Each module owns the CSV its benchmark runs on.")
    if modules:
        for info in modules:
            st.markdown(f"**{info['title']}** &nbsp;·&nbsp; `modules/{info['name']}`")
            if info["error"]:
                st.error(f"main.py failed to load: {info['error']}")
            elif info["description"]:
                st.caption(info["description"])
    else:
        st.info("No modules found. Add a folder under `modules/` containing `main.py` "
                "and `main.html`.")
    st.caption("To add your own, see the Modules section of README.md.")

    st.markdown("### Upload your own CSV", anchors=False)
    uploaded_file = st.file_uploader("Upload Time-Series CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("📋 **Uploaded Data Sample Preview:**")
        st.dataframe(raw_df.head(4))
        render_benchmark(raw_df, key="home", test_size=test_size)
    else:
        st.info("👋 Upload a CSV to benchmark it, or pick a module tab above.")

for info, tab in zip(modules, module_tabs):
    with tab:
        if info["error"]:
            st.error(f"This module's main.py failed to load: {info['error']}")
            continue
        show_module_page(info["html"])

        try:
            frame = module_frame(info)
        except Exception as exc:
            st.error(f"Could not build this module's CSV: {type(exc).__name__}: {exc}")
            continue

        render_benchmark(
            frame, key=f"module:{info['name']}", test_size=test_size,
            note=f"`modules/{info['name']}/{CSV_NAME}` · {len(frame)} rows × {frame.shape[1]} columns",
            title=info["title"],
        )

with summary_tab:
    show_summary(modules, test_size)
