"""Chaos against a matched fake, recorded finely and coarsely.

Can an algorithm tell a chaotic series from a fake that has exactly the same values
and almost the same power spectrum, so that only the *order* of the values (the
determinism) can give it away? And does that survive coarser sampling?

    Fine sampling      every step of the system is recorded
    Coarse sampling    every 4th step is recorded (the same system and the same fakes)

Label 1 is a chaotic series and label 0 is its fake. A dataset is 50 rows, 25 of each
label. The real and fake rows of a dataset come from 50 DIFFERENT orbits, so a fake is
never in the same dataset as the series it was made from. (A random train/test split
inside a dataset could otherwise put a real series and its own twin on opposite sides.)

The system is the logistic map x -> r x (1 - x) with r drawn from {3.90, 3.95, 3.99},
where it is strongly chaotic. Real series are 100-sample windows chosen so that the
end matches the start, which keeps the fake's wrap-around seam from giving it away.
Fakes are IAAFT surrogates (Schreiber and Schmitz, 1996): the same values, a power
spectrum matched to within a few percent, and scrambled phases. Each fake is the best
spectral match out of CANDIDATES attempts. Every series is z-normalised.

Only numpy is needed. Run standalone to refresh the CSV::

    python3 modules/chaos_sampling/main.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TITLE = "Chaos vs matched fake"
DESCRIPTION = (
    "Logistic-map chaos against an IAAFT surrogate with the same values and power "
    "spectrum, recorded at every step and at every 4th step."
)

N_DATASETS = 10
SAMPLES = 50                 # rows per dataset: half chaotic, half fake
LENGTH = 100
SEED = 2026
R_VALUES = (3.90, 3.95, 3.99)
SEARCH = 150                 # how far into an orbit to look for an end-matched window
CANDIDATES = 20              # IAAFT attempts per fake; the best spectral match is kept
BURN_IN = 400                # steps discarded so the orbit has settled onto the attractor

# (condition name, stride): a stride of k records every k-th step of the system.
CONDITIONS = (("Fine sampling", 1), ("Coarse sampling", 4))


def _orbit(rng: np.random.Generator, r: float, stride: int) -> np.ndarray:
    """A logistic-map orbit long enough to cut one window at the given stride."""
    x = rng.uniform(0.05, 0.95)
    for _ in range(BURN_IN):
        x = r * x * (1.0 - x)
    out = np.empty(SEARCH + stride * (LENGTH + 2))
    for i in range(len(out)):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


def _window(orbit: np.ndarray, stride: int) -> np.ndarray:
    """The z-normalised window, sampled every ``stride`` steps, whose end best matches its start."""
    best, start = np.inf, 0
    for s in range(SEARCH):
        w = orbit[s:s + stride * (LENGTH + 2):stride]
        mismatch = (w[0] - w[LENGTH]) ** 2 + (w[1] - w[LENGTH + 1]) ** 2
        if mismatch < best:
            best, start = mismatch, s
    w = orbit[start:start + stride * LENGTH:stride]
    return (w - w.mean()) / w.std()


def _iaaft(x: np.ndarray, rng: np.random.Generator):
    """One IAAFT surrogate of ``x`` and its relative spectral error."""
    sorted_x = np.sort(x)
    amplitude = np.abs(np.fft.fft(x))
    s = rng.permutation(x).astype(float)
    previous = None
    for _ in range(2000):
        filtered = np.real(np.fft.ifft(amplitude * np.exp(1j * np.angle(np.fft.fft(s)))))
        ranks = np.argsort(np.argsort(filtered))
        s = sorted_x[ranks]
        if previous is not None and np.array_equal(ranks, previous):
            break
        previous = ranks
    target, got = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s))
    return s, float(np.linalg.norm(got - target) / np.linalg.norm(target))


def _surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The best of CANDIDATES surrogates of ``x``: the one whose spectrum matches most closely."""
    return min((_iaaft(x, rng) for _ in range(CANDIDATES)), key=lambda c: c[1])[0]


def datasets(n: int = N_DATASETS):
    """Yield ``(condition, X, y)`` for the first ``n`` datasets of each condition."""
    rng = np.random.default_rng(SEED)
    half = SAMPLES // 2
    for condition, stride in CONDITIONS:
        for _ in range(n):
            windows = [_window(_orbit(rng, float(rng.choice(R_VALUES)), stride), stride)
                       for _ in range(SAMPLES)]
            real = windows[:half]
            fake = [_surrogate(w, rng) for w in windows[half:]]
            X = np.array(real + fake)
            y = np.array([1] * half + [0] * half)
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
