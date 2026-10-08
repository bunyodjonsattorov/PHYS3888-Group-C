"""Exponent-dispersion label: two fractal ingredients mixed at a dial angle.

The construction, reconstructed from the companion explainer page (``main.html``)
and verified against that page's own baked arrays:

    x(t)       = cos(theta(t)) * r(t) + sin(theta(t)) * s(t)
    record     = cumulative sum of x          (length ``n``)
    r          = unit-variance fGn, H = 0.4   (rough ingredient)
    s          = unit-variance fGn, H = 0.9   (smooth ingredient)
    label 0    = theta(t) = pi/4              (dial frozen)
    label 1    = theta(t) = pi/4 + 0.7 * c(t) (dial drifts)

Because cos^2 + sin^2 = 1 and both ingredients are unit variance, Var[x(t)] = 1
at every t in both classes. The two classes therefore match in marginal
distribution and differ only in whether the local roughness varies along the
record, which is what makes the label hard: nothing about a single window gives it
away, only the dispersion of roughness *between* windows.

``c(t)`` survives in the page only folded into theta. Recovering it gives min
-0.669, max 1.000, mean 0.089, std 0.332 and lag-1 autocorrelation 0.989, so it
is a very smooth zero-mean field; here it is Gaussian-smoothed white noise
standardised to zero mean and std 0.33.

The reader on the page, and the only thing that separates the classes, is the
local exponent measured per window on the *increments*; on the running sum the
local texture is hidden by the smooth ingredient's drift.

Run standalone to refresh the CSV::

    python3 modules/exponent_dispersion/main.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TITLE = "Exponent dispersion"
DESCRIPTION = (
    "Two fractal ingredients blended at a dial angle that is either frozen or "
    "drifting. The classes match in every marginal statistic."
)

# app.py scores each block of 50 rows, so a block must never hold one class only.
CHUNK = 50
LENGTH = 1024
PER_CONDITION = 100
SEED = 0

H_ROUGH = 0.4
H_SMOOTH = 0.9
DIAL_AMPLITUDE = 0.7
CONTROL_SIGMA = 20.0
CONTROL_STD = 0.33

# The app accepts these two literal condition names. They name the ingredient
# pair each copy of the task is posed with, not a claim about stationarity.
FAMILIES = (
    ("Constant Parameter", 0.4, 0.9),
    ("Non-Stationary", 0.2, 0.7),
)


# --------------------------------------------------------------------------
# fractional Gaussian noise, circulant embedding
# --------------------------------------------------------------------------
def _autocovariance(n: int, hurst: float) -> np.ndarray:
    """Exact fGn autocovariance ``( |k-1|^2H - 2|k|^2H + |k+1|^2H ) / 2``."""
    k = np.arange(n, dtype=float)
    return 0.5 * (
        np.abs(k - 1.0) ** (2.0 * hurst)
        - 2.0 * np.abs(k) ** (2.0 * hurst)
        + np.abs(k + 1.0) ** (2.0 * hurst)
    )


def fgn(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Exact discrete fractional Gaussian noise of length ``n``."""
    gamma = _autocovariance(n, hurst)
    row = np.concatenate([gamma, [0.0], gamma[:0:-1]])
    spectrum = np.fft.fft(row).real
    # The true spectrum is non-negative; clipping only the floating-point
    # negatives keeps the square root defined without regularising anything.
    spectrum = np.clip(spectrum, 0.0, None)
    white = rng.normal(size=2 * n)
    return np.fft.ifft(np.sqrt(spectrum) * np.fft.fft(white)).real[:n]


# --------------------------------------------------------------------------
# the construction
# --------------------------------------------------------------------------
def control_field(rng: np.random.Generator, n: int) -> np.ndarray:
    """Smooth zero-mean field driving the drifting dial."""
    noise = rng.normal(size=n)
    radius = int(4 * CONTROL_SIGMA)
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / CONTROL_SIGMA) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(np.pad(noise, radius, mode="reflect"), kernel, mode="same")
    smoothed = smoothed[radius:-radius]
    smoothed -= smoothed.mean()
    return smoothed / smoothed.std() * CONTROL_STD


def make_record(rng: np.random.Generator, n: int, drifting: bool,
                h_rough: float, h_smooth: float) -> np.ndarray:
    """One record of length ``n`` for the given class."""
    rough = fgn(n, h_rough, rng)
    smooth = fgn(n, h_smooth, rng)
    rough = rough / rough.std()
    smooth = smooth / smooth.std()

    theta = np.full(n, np.pi / 4)
    if drifting:
        theta = theta + DIAL_AMPLITUDE * control_field(rng, n)

    return np.cumsum(np.cos(theta) * rough + np.sin(theta) * smooth)


def build_rows(rng: np.random.Generator, per_condition: int = PER_CONDITION,
               length: int = LENGTH):
    """Rows laid out so every 50-row block holds both labels."""
    rows = []
    for condition, h_rough, h_smooth in FAMILIES:
        remaining = per_condition
        while remaining > 0:
            size = min(CHUNK, remaining)
            half = size // 2
            for label, drifting in ((0, False), (1, True)):
                take = size - half if label else half
                for _ in range(take):
                    rows.append((condition, label,
                                 make_record(rng, length, drifting, h_rough, h_smooth)))
            remaining -= size
    return rows


def generate_csv(path, per_condition: int = PER_CONDITION, length: int = LENGTH,
                 seed: int = SEED):
    """Write the module's CSV in the format the app parses. Returns the path."""
    rows = build_rows(np.random.default_rng(seed), per_condition, length)
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Condition", "Label"] + [f"t{i}" for i in range(length)])
        for condition, label, record in rows:
            writer.writerow([condition, label] + [f"{v:.6f}" for v in record])
    return path


def main() -> None:
    path = generate_csv(Path(__file__).with_name("data.csv"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
