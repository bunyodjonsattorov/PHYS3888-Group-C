# Time-series benchmarking app

A Streamlit app that scores classifiers on labelled time-series CSVs: Rocket,
MiniRocket, DTW, catch22 and fSFA, plus an optional dynamics-aware reference. There is
one tab per module, a Home tab for uploading your own CSV, and a Summary tab.

## Running

Python 3.10 or newer is required (3.11 is tested); the app uses `st.iframe`, which
needs a recent Streamlit.

```
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Only streamlit, numpy, pandas and scikit-learn are required. Without sktime, pycatch22 or fastdtw
the app still runs, but it skips the algorithms that need the missing library. The
sidebar marks them as not installed and the results cover only what actually ran.
The app never substitutes random guesses or a cruder method for a missing algorithm,
because a made-up score reported under a real name could be mistaken for a result.

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
modules/fractal_timeseries    a dial angle held frozen against one allowed to drift
modules/synthetic             AR(1) pairs, and random walks whose drift flips sign
modules/chaos_sampling        chaotic logistic-map series against matched fakes, recorded at every step and every 4th step
modules/chaos_multisystem     the same chaos-vs-fake task on logistic, Lorenz and Rossler, each recorded finely and coarsely; add your own system in main.py
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

`Condition` names the task a row belongs to and can be any text. Every distinct value
is its own condition, shown in the order it first appears in the file; the existing
modules use `Constant Parameter` and `Non-Stationary`, and a module can define its own.
Every column other than `Condition` and `Label` is read as a time step, so a CSV must not carry extra metadata columns. `Label` has to be 0 or 1. Within each condition, rows are scored in blocks of 50, and
every block has to hold both labels. A block with one label is scored at 100% by any
algorithm, so the app skips it and says so instead of reporting that as a result.
Every condition is scored on the same number of datasets, the smallest count any of
them has, and a condition with no usable block is left out with a warning.

## Reading the results

Every run reports, per condition and algorithm, the mean accuracy over the datasets
you chose together with:

- **Std. error**: the standard error of that mean across datasets
  (`std / sqrt(n)`). It needs at least two datasets.
- **Chance**: the accuracy of always predicting the majority class of a test set,
  averaged over the datasets, about 0.5 on balanced blocks.
- **Vs chance**: "not clearly above chance" when `mean - 2 * SE <= chance`.

None of these change an accuracy; they sit beside it.

### Dynamics-aware reference

The sidebar has an off-by-default checkbox, "Include dynamics-aware reference (NN)".
It adds one algorithm, `Dynamics (NN)` (`dynamics.py`), to the run. For each series and
each embedding dimension d in 1, 2, 3 it builds delay vectors and predicts the next
value from the one belonging to the nearest other vector. The mean absolute one-step
error is low for a deterministic process and high when the order has been scrambled,
and a random forest is trained on the three errors. It is a simple reference, fragile
to observation noise, and not a proposed best method. With the checkbox off, a run is
exactly what it was without this feature.

### Gap

When the reference was run, a table shows, per condition,

```
gap = accuracy(Dynamics (NN)) - best accuracy among Rocket, MiniRocket, catch22, DTW
```

worked out on each dataset and then averaged, with its standard error. fSFA is listed
beside the gap but is not part of the maximum. A positive gap means a method that
looks at the dynamics beats every shape or feature method on that problem.

### Summary tab

Collects, for every module run in this session, one row per condition: the mean
accuracy of each algorithm, the chance level, and the gap. Run each module first (with
the reference on to fill the gap), or press **Run all modules** there to run them in turn
behind one progress bar. A module that fails is reported and skipped, and the others still
run. Start with 1 or 2 datasets per module, because DTW on long series is slow. Every results
table and the Summary can be downloaded as CSV, named with the module and a timestamp.
The chart colours are fixed in `ui.py`: blue and grey for the shape and feature methods,
vermilion for the dynamics reference, amber for fSFA. Results live in the
session, so they are gone when the page is reloaded.

## Tests

```
pip install pytest
python -m pytest tests
```

## Worth knowing

DTW compares every test series against every training series, so it dominates the
runtime while the other three finish almost instantly. On the fractal module's
1024-sample records one 50-row dataset costs roughly 3 minutes of DTW, so its full
ten-dataset run takes about an hour. The dataset slider is the dial for that: one
or two gives a quick look, ten gives the steadiest answer.

While a run is going the app shows a progress bar over one step per algorithm per
dataset, naming the dataset, condition and algorithm in flight, with the elapsed
time and an estimate of what is left. The estimate appears once every algorithm
has been measured, because until a first DTW has finished there is nothing to
price the remaining DTW steps against. DTW also reports how many test series it
has scored, so the bar keeps moving through its share rather than stalling for
minutes at a time.

That fractal CSV is about 10 MB, which is what ten blocks of 1024-sample records
cost on disk. `PER_CONDITION` in its `main.py` is the single number to change if
that trade is not worth making.

Accuracy per dataset comes from a single train/test split of a block of up to 50
rows, and every figure the app reports is the mean over the datasets you chose
rather than over one split. A lone dataset can crown a different winner on each run, and
averaging across the ten the fractal module offers is what steadies it. The
"peak performance" line states the basis it averaged over.
