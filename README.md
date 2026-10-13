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
