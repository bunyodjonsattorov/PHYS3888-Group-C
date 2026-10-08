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
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Specialised time-series libraries, with fallbacks when they are not installed.
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

MODULES_DIR = Path(__file__).resolve().parent / "modules"
CSV_NAME = "data.csv"


# ==========================================
# 1. CSV PARSING
# ==========================================
def parse_uploaded_csv(df):
    """
    Parses a CSV into structured stationary and non-stationary list variations.
    Expects columns: 'Condition' ('Constant Parameter' or 'Non-Stationary'),
    'Label' (0 or 1), and the remaining columns treated as time-series steps.

    Returns ``(stationary, non_stationary, report)``; ``report`` records the rows
    and chunks that had to be skipped, so the caller can surface them instead of
    scoring them silently.
    """
    report = {
        "unknown_rows": 0,
        "unknown_conditions": [],
        "single_label_chunks": 0,
        "tiny_chunks": 0,
    }
    if "Condition" not in df.columns or "Label" not in df.columns:
        st.error("Uploaded CSV must contain 'Condition' and 'Label' columns alongside sequential features.")
        return [], [], report

    stat_list = []
    non_stat_list = []

    # Isolate time series data columns
    ts_cols = [c for c in df.columns if c not in ["Condition", "Label"]]

    # Split by condition types
    for cond, sub_df in df.groupby("Condition"):
        if cond not in ("Constant Parameter", "Non-Stationary"):
            report["unknown_conditions"].append(str(cond))
            report["unknown_rows"] += int(sub_df.shape[0])
            continue
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

            if cond == "Constant Parameter":
                stat_list.append((X, y))
            else:
                non_stat_list.append((X, y))

    return stat_list, non_stat_list, report


def show_parse_report(report):
    """Surface whatever parse_uploaded_csv had to skip, instead of failing silently."""
    if report["unknown_rows"]:
        names = ", ".join(sorted(set(report["unknown_conditions"])))
        st.warning(
            f"Ignored {report['unknown_rows']} row(s) whose 'Condition' is not 'Constant Parameter' "
            f"or 'Non-Stationary' (found: {names})."
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
        time.sleep(0.01)
        return accuracy_score(y_test, np.random.binomial(1, 0.5, len(y_test))), 0.01

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
    start_time = time.time()
    if not CATCH22_AVAILABLE:
        def custom_catch22_features(X):
            features = []
            for row in X:
                features.append([
                    np.mean(row), np.std(row), np.max(row) - np.min(row),
                    np.median(row), np.var(row), np.percentile(row, 75),
                    np.percentile(row, 25), np.sum(np.abs(np.diff(row)))
                ])
            return np.array(features)
        X_train_feat = custom_catch22_features(X_train)
        X_test_feat = custom_catch22_features(X_test)
    else:
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


def evaluate_dtw(X_train, X_test, y_train, y_test):
    start_time = time.time()
    preds = []

    for test_row in X_test:
        best_dist = float('inf')
        best_label = 0
        for train_row, label in zip(X_train, y_train):
            if DTW_AVAILABLE:
                # Absolute point diff lambda bypasses SciPy 1-D shape validation.
                dist, _ = fastdtw(test_row, train_row, dist=lambda a, b: np.abs(a - b))
            else:
                dist = np.linalg.norm(test_row - train_row)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        preds.append(best_label)

    return accuracy_score(y_test, preds), time.time() - start_time


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
    spec = importlib.util.spec_from_file_location(f"_module_{main_py.parent.name}", main_py)
    module = importlib.util.module_from_spec(spec)
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
def run_benchmark(stat_data, non_stat_data, num_datasets, test_size):
    """Score all four algorithms on the first num_datasets of each condition."""
    results = []
    for i in range(num_datasets):
        for condition, data in (("Constant Parameter", stat_data),
                                ("Non-Stationary", non_stat_data)):
            X, y = data[i]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i)

            for name, run in (
                ("Rocket", lambda: evaluate_rocket(X_train, X_test, y_train, y_test, mini=False)),
                ("MiniRocket", lambda: evaluate_rocket(X_train, X_test, y_train, y_test, mini=True)),
                ("catch22", lambda: evaluate_catch22(X_train, X_test, y_train, y_test)),
                ("DTW", lambda: evaluate_dtw(X_train, X_test, y_train, y_test)),
            ):
                acc, elapsed = run()
                results.append({"Dataset": i + 1, "Condition": condition,
                                "Algorithm": name, "Accuracy": acc, "Time (s)": elapsed})
    return pd.DataFrame(results)


def show_results(df_results):
    # Every figure below is a mean over the datasets the user chose to evaluate,
    # never a single train/test split, so the winner is decided on that average.
    summary = df_results.groupby(["Condition", "Algorithm"])[["Accuracy", "Time (s)"]].mean().reset_index()
    n_datasets = int(df_results["Dataset"].nunique())
    basis = "1 dataset" if n_datasets == 1 else f"{n_datasets} datasets"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Performance Benchmarks Summary")
        st.dataframe(summary.style.format({"Accuracy": "{:.2%}", "Time (s)": "{:.4f}s"}))
        st.caption(f"Mean over {basis}; each dataset is a block of up to 50 rows with "
                   "its own train/test split.")

    with col2:
        st.subheader("💡 Key Takeaways")
        for cond in ["Constant Parameter", "Non-Stationary"]:
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

    st.subheader("📊 Granular Dataset Evaluation Matrix")
    st.dataframe(df_results)


def render_benchmark(df, key, test_size, note=None):
    """Parse a CSV, then offer the dataset count and the big Execute Benchmark button."""
    stat_data, non_stat_data, report = parse_uploaded_csv(df)
    show_parse_report(report)

    max_available = min(len(stat_data), len(non_stat_data))
    if max_available == 0:
        st.error(
            "No evaluable datasets found. Each 50-row block must sit under one 'Condition' "
            "('Constant Parameter' or 'Non-Stationary') and contain both Label 0 and Label 1 "
            "rows, because a block with a single class is always scored at 100%."
        )
        return

    if note:
        st.caption(note)

    if max_available == 1:
        # st.slider rejects min_value == max_value, which a single-dataset CSV produces.
        num_datasets = 1
        st.caption("Datasets to evaluate: 1")
    else:
        num_datasets = st.slider(f"Datasets to evaluate (max {max_available})",
                                 1, max_available, max_available, key=f"{key}:n")

    if st.button("Execute Benchmark", type="primary", use_container_width=True, key=f"{key}:run"):
        with st.spinner("Running Rocket, MiniRocket, catch22 and DTW..."):
            results = run_benchmark(stat_data, non_stat_data, num_datasets, test_size)
        show_results(results)


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

st.sidebar.header("📦 System Library Status")
st.sidebar.write(f"{'✅' if SKTIME_AVAILABLE else '⚠️ Proxy'} sktime (Rocket)")
st.sidebar.write(f"{'✅' if CATCH22_AVAILABLE else '⚠️ Proxy'} pycatch22")
st.sidebar.write(f"{'✅' if DTW_AVAILABLE else '⚠️ Proxy'} fastdtw")

st.title("⏱️ Time-Series Algorithm Benchmarking Suite")
st.markdown("""
Compare **Constant Parameter (Stationary)** and **Non-Stationary** data performance across
**Rocket**, **MiniRocket**, **DTW**, and **catch22**.
""")

modules = [describe_module(found) for found in discover_modules()]

home_tab, *module_tabs = st.tabs(["Home"] + [m["title"] for m in modules])

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
        )
