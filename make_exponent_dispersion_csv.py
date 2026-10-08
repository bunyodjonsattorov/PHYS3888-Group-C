"""Generate a CSV of the exponent-dispersion label for app.py.

The label comes from the Fractals project's explainer page
(``Fractals/figures/exponent_dispersion_explainer.html``).  That page is
hand-authored and ships its numbers pre-baked, so there is no generator in the
repository.  The construction below was reconstructed from that page and then
*verified against the page's own baked arrays* before use:

    x(t)      = cos(theta(t)) * r(t) + sin(theta(t)) * s(t)
    record    = cumulative sum of x        (length ``n``)
    increments= first difference of record

    r : unit-variance fGn with H = 0.4   (rough ingredient)
    s : unit-variance fGn with H = 0.9   (smooth ingredient)
    class 0 : theta(t) = pi/4                        (dial frozen)
    class 1 : theta(t) = pi/4 + 0.7 * c(t)           (dial drifts)

Checks that passed against the page's baked data (4-dp rounding aside):
    max |inc1 - (cos(theta1)*r + sin(theta1)*s)[1:]| = 2.2e-04
    max |x1 - cumsum(cos(theta1)*r + sin(theta1)*s)| = 1.5e-03
Class 0's record is a *separate draw* of r and s (it is not a blend of the
displayed pair), so every record here gets fresh ingredients.

Because cos^2(theta) + sin^2(theta) = 1 and both ingredients are unit variance,
Var[x(t)] = 1 for every t in both classes: the two classes match in marginal
distribution and differ only in whether the local roughness varies across the
record.  That is the whole point of the label.

``c(t)`` is not recoverable from the page (it is only stored after being folded
into theta).  Recovering it gives min -0.669, max 1.000, mean 0.089, std 0.332,
lag-1 autocorrelation 0.989 and lag-50 0.371, i.e. a very smooth, slowly varying
zero-mean field.  It is reproduced here as Gaussian-smoothed white noise
standardised to zero mean and std 0.33.

Usage::

    python3 make_exponent_dispersion_csv.py
    python3 make_exponent_dispersion_csv.py --records-per-class 50 -o out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# The explainer's footer cites fractal_bench.selfaffine.fgn as the source of the
# ingredients, so reuse that exact generator rather than a re-implementation.
FRACTALS_SRC = Path("/home/murray/Development/timeseries-classification/Fractals/src")

H_ROUGH = 0.4
H_SMOOTH = 0.9
DIAL_AMPLITUDE = 0.7          # the page's "theta = pi/4 + 0.7 c(t)"
CONTROL_SIGMA = 20.0          # smoothing width of the control field, samples
CONTROL_STD = 0.33            # recovered from the page's baked theta1
LENGTH = 1024                 # the page's n
WINDOWS = 8                   # the page's reader uses 8 windows
LAGS = (1, 2, 4, 8)


def _import_fgn():
    try:
        from fractal_bench.selfaffine import fgn
        return fgn
    except ImportError:
        if FRACTALS_SRC.is_dir():
            sys.path.insert(0, str(FRACTALS_SRC))
            from fractal_bench.selfaffine import fgn
            return fgn
        raise SystemExit(
            "cannot import fractal_bench.selfaffine.fgn; set PYTHONPATH to the "
            f"Fractals src directory (expected at {FRACTALS_SRC})"
        )


def control_field(rng: np.random.Generator, n: int) -> np.ndarray:
    """Smooth, zero-mean control field c(t) driving the class-1 dial."""
    noise = rng.normal(size=n)
    radius = int(4 * CONTROL_SIGMA)
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / CONTROL_SIGMA) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(noise, radius, mode="reflect")
    smooth = np.convolve(padded, kernel, mode="same")[radius:-radius]
    smooth -= smooth.mean()
    return smooth / smooth.std() * CONTROL_STD


def make_record(fgn, rng: np.random.Generator, n: int, drifting: bool,
                h_rough: float = H_ROUGH, h_smooth: float = H_SMOOTH) -> np.ndarray:
    """Build one record of length ``n`` for the given class."""
    rough = fgn(n, h_rough, rng)
    smooth = fgn(n, h_smooth, rng)
    rough = rough / rough.std()          # "normalise each to unit variance"
    smooth = smooth / smooth.std()

    theta = np.full(n, np.pi / 4)
    if drifting:
        theta = theta + DIAL_AMPLITUDE * control_field(rng, n)

    blend = np.cos(theta) * rough + np.sin(theta) * smooth
    return np.cumsum(blend)


def window_exponents(record: np.ndarray, windows: int = WINDOWS) -> np.ndarray:
    """The page's reader: local roughness exponent per window.

    Fits the slope of log mean|delta_lag x|^2 against log lag over lags 1,2,4,8.
    """
    exponents = []
    for segment in np.array_split(record, windows):
        lags = [lag for lag in LAGS if lag < segment.size]
        structure = []
        for lag in lags:
            delta = segment[lag:] - segment[:-lag]
            structure.append(np.log(np.mean(delta ** 2)))
        exponents.append(np.polyfit(np.log(lags), structure, 1)[0])
    return np.array(exponents)


def build_app_schema_rows(fgn, rng, per_condition: int, length: int):
    """Lay the label out the way app.py requires.

    app.py only accepts the literal conditions 'Constant Parameter' and
    'Non-Stationary', and it scores each 50-row chunk, so every chunk must hold
    *both* classes: a chunk with a single class is always scored at 100%.

    The binary task inside each condition is the exponent-dispersion label
    itself (dial frozen vs dial drifting).  The condition only names the
    ingredient pair the task is posed with, which is why these two borrowed
    strings should be read as two ingredient settings rather than as a claim
    that one of them is stationary.
    """
    families = (
        ("Constant Parameter", 0.4, 0.9),   # the page's ingredient pair
        ("Non-Stationary", 0.2, 0.7),       # a second pair
    )
    half = max(1, per_condition // 2)
    rows = []
    for condition, h_rough, h_smooth in families:
        for label, drifting in ((0, False), (1, True)):
            for _ in range(half):
                rows.append((
                    condition, label,
                    make_record(fgn, rng, length, drifting=drifting,
                                h_rough=h_rough, h_smooth=h_smooth),
                ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-per-class", type=int, default=50,
                        help="rows per class (50 = one dataset chunk in app.py)")
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-condition", type=int, default=50,
                        help="rows per Condition in --app-schema (50 = one chunk)")
    parser.add_argument("--app-schema", action="store_true",
                        help="write the layout app.py expects: both classes inside each "
                             "Condition, since app.py scores each 50-row chunk")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path(__file__).with_name("exponent_dispersion_label.csv"))
    args = parser.parse_args()

    fgn = _import_fgn()
    rng = np.random.default_rng(args.seed)

    if args.app_schema:
        rows = build_app_schema_rows(fgn, rng, args.per_condition, args.length)
    else:
        rows = []
        for drift, condition, label in (
            (False, "Constant Parameter", 0),
            (True, "Non-Stationary", 1),
        ):
            for _ in range(args.records_per_class):
                rows.append((condition, label,
                             make_record(fgn, rng, args.length, drifting=drift)))

    header = ["Condition", "Label"] + [f"t{i}" for i in range(args.length)]
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for condition, label, record in rows:
            writer.writerow([condition, label] + [f"{v:.6f}" for v in record])

    # Self-check: the two classes must match in marginal spread but differ in
    # whether the local roughness varies along the record.
    #
    # The reader is applied to each record's *increments*, not the running sum:
    # over 1024 samples the sum is dominated by the smooth ingredient's drift and
    # the local texture is invisible in it.  Running the reader on the increments
    # reproduces the explainer page's published values (class-0 window-exponent
    # spread 0.062 here vs 0.062 published; reader AUC 0.668 here vs 0.683), which
    # is the evidence that this reconstruction is faithful.
    by_label = {0: [], 1: []}
    for condition, label, record in rows:
        by_label[label].append(record)

    spreads = {label: np.array([window_exponents(np.diff(r)).std() for r in recs])
               for label, recs in by_label.items()}

    def auc(a, b):
        """P(class 1 spread > class 0 spread), ties counted as half."""
        return float(np.mean([(x > y) + 0.5 * (x == y) for x in b for y in a]))

    print(f"wrote {args.output}  ({len(rows)} rows, {args.length} cols)")
    if args.app_schema:
        print("  layout : both labels inside each 'Condition' chunk (app.py schema)")
        print("  label 0 = dial frozen at pi/4, label 1 = dial drifts")
        print("  Condition names the ingredient pair the task is posed with")
    else:
        print(f"  class 0 = {rows[0][0]}, label 0, dial frozen at pi/4")
        print(f"  class 1 = {rows[-1][0]}, label 1, dial drifts")
    print("  record std      : class0 %.3f  class1 %.3f  (matched by construction)"
          % (np.mean([r.std() for r in by_label[0]]),
             np.mean([r.std() for r in by_label[1]])))
    print("  increment std   : class0 %.4f  class1 %.4f  (matched by construction)"
          % (np.mean([np.diff(r).std() for r in by_label[0]]),
             np.mean([np.diff(r).std() for r in by_label[1]])))
    print("  reader spread   : class0 mean %.4f (sd %.4f)  class1 mean %.4f (sd %.4f)"
          % (spreads[0].mean(), spreads[0].std(),
             spreads[1].mean(), spreads[1].std()))
    print("  reader AUC      : %.3f   (chance = 0.500; page reports 0.683)"
          % auc(spreads[0], spreads[1]))
    print("  window exponents: class0 mean %.4f  class1 mean %.4f  (means should match)"
          % (np.mean([window_exponents(np.diff(r)).mean() for r in by_label[0]]),
             np.mean([window_exponents(np.diff(r)).mean() for r in by_label[1]])))


if __name__ == "__main__":
    main()
