# Time-series benchmarking app

A Streamlit app that scores four classifiers on labelled time-series CSVs: Rocket,
MiniRocket, DTW and catch22. There is one tab per module, plus a Home tab for
uploading your own CSV.

## Running

```
python3 -m pip install streamlit numpy pandas scikit-learn scikit-base sktime pycatch22 fastdtw
streamlit run app.py
```

Only the first half of that list is required. Without sktime, pycatch22 or fastdtw
the app still runs, and the sidebar marks the affected algorithms as proxies.

## Modules

Every folder under `modules/` holding both of these becomes a tab:

```
main.py     defines generate_csv(path), which writes the module's CSV
main.html   the page shown in that tab
```

`main.py` may also define `TITLE` and `DESCRIPTION` strings to label the tab and
the Home listing. The folder name is used when they are absent.

The CSV lands beside them as `data.csv`. It is committed, so tabs open without
waiting, and the app rebuilds it whenever `main.py` is newer than the file. That
is what happens after you edit a generator; there is nothing to run by hand.

Shipping today:

```
modules/exponent_dispersion   a dial angle held frozen against one allowed to drift
modules/synthetic             AR(1) pairs, and random walks whose drift flips sign
```

### Adding a module

1. `mkdir modules/my_task`
2. Add `main.py` defining `generate_csv(path)`.
3. Add `main.html`. Any self-contained page works, and scripts and canvases run.
4. `python3 modules/my_task/main.py` writes `data.csv`.

Rerun the app and the tab is there. Nothing else needs editing.

## CSV format

One row per sample, with the series itself in the columns after the first two:

```
Condition,Label,t0,t1,t2,...
Constant Parameter,0,0.13,-0.42,...
Constant Parameter,1,1.07,0.55,...
```

`Condition` has to be `Constant Parameter` or `Non-Stationary`, and `Label` has to
be 0 or 1. Rows are scored in blocks of 50, and every block has to hold both
labels. A block with one label is scored at 100% by any algorithm, so the app
skips it and says so instead of reporting that as a result.

## Worth knowing

DTW compares every test series against every training series, so it dominates the
runtime while the other three finish almost instantly.

Accuracy comes from a single train/test split of a 50-row block, so on the small
modules the gap between the best and worst algorithm is largely noise. The
"peak performance" line names whichever won that split, which is why it can move
between runs and between conditions.
