"""One task, several chaotic systems: chaos against a matched fake.

The task is the same for every system. Label 1 is a series from a chaotic system and
label 0 is its fake: an IAAFT surrogate with exactly the same values and almost the same
power spectrum, so only the *order* of the values (the determinism) can give it away.

Each system is recorded at two sampling intervals, measured in Lyapunov units
(lambda * dt, how much a tiny error grows between two recorded points):

    fine sampling     lambda * dt = 0.5
    coarse sampling   lambda * dt = 2.0

Every condition in the CSV is "<system>, <fine|coarse> sampling", so the app shows one
row per system and sampling. A dataset is 50 rows, 25 of each label, and the real and
fake rows of a dataset come from 50 different orbits (a fake never shares a dataset with
the series it was made from). Real series are 100-sample windows whose end matches their
start, so the fake's wrap-around seam does not give it away. Every series is z-normalised.

ADD YOUR OWN SYSTEM
-------------------
Add one entry to SYSTEMS below.

    A flow (ODE):  _flow("My system", derivative, lam, h, start, observed, burn_in)
    Anything else: a System(...) with your own simulate(rng, n_orbits, n_samples, dt)
                   that returns an array of shape (n_orbits, n_samples).

``lam`` is the largest Lyapunov exponent per unit of time (per step for a map). Please
check it (e.g. from the dysts metadata) and make sure the system really is chaotic with
the parameters you use. Only numpy is needed. Refresh the CSV with::

    python3 modules/chaos_multisystem/main.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

TITLE = "Chaos vs matched fake, several systems"
DESCRIPTION = (
    "Logistic map, Lorenz and Rossler against IAAFT fakes with the same values and "
    "power spectrum, each recorded finely and coarsely."
)

N_DATASETS = 10
SAMPLES = 50                 # rows per dataset: half chaotic, half fake
LENGTH = 100
SEED = 2026
SEARCH = 60                  # how many start points to try for an end-matched window
CANDIDATES = 10              # IAAFT attempts per fake; the best spectral match is kept

# (label, lambda * dt): how coarsely each system is recorded.
SAMPLINGS = (("fine sampling", 0.5), ("coarse sampling", 2.0))


@dataclass
class System:
    name: str
    lam: float                                    # largest Lyapunov exponent, per time unit
    simulate: Callable                            # (rng, n_orbits, n_samples, dt) -> array
    dt_unit: float = 0.0                          # dt is rounded to a multiple of this (0 = no rounding)


def _flow(name, deriv, lam, h, start, observed, burn_in):
    """A System from an ODE ``deriv(state) -> d state / dt`` with state shape (n_orbits, dim)."""

    def rk4(s):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * h * k1)
        k3 = deriv(s + 0.5 * h * k2)
        k4 = deriv(s + h * k3)
        return s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(rng, n_orbits, n_samples, dt):
        s = start(rng, n_orbits)
        for _ in range(int(burn_in / h)):
            s = rk4(s)
        per_sample = max(1, round(dt / h))
        out = np.empty((n_orbits, n_samples))
        for i in range(n_samples):
            for _ in range(per_sample):
                s = rk4(s)
            out[:, i] = s[:, observed]
        return out

    return System(name, lam, simulate, dt_unit=h)


def _logistic_simulate(rng, n_orbits, n_samples, dt):
    """Logistic map x -> r x (1 - x), r drawn per orbit; ``dt`` is the number of steps between samples."""
    r = rng.choice((3.90, 3.95, 3.99), size=n_orbits)
    x = rng.uniform(0.05, 0.95, size=n_orbits)
    for _ in range(400):
        x = r * x * (1.0 - x)
    stride = int(round(dt))
    out = np.empty((n_orbits, n_samples))
    for i in range(n_samples):
        for _ in range(stride):
            x = r * x * (1.0 - x)
        out[:, i] = x
    return out


def _lorenz(s):
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return np.stack([10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z], axis=1)


def _rossler(s):
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return np.stack([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)], axis=1)


SYSTEMS = (
    System("Logistic map", 0.5, _logistic_simulate, dt_unit=1.0),   # lambda ~ 0.5 per step for r in 3.9-3.99
    _flow("Lorenz", _lorenz, lam=0.906, h=0.02,
          start=lambda rng, n: rng.uniform(-10, 10, size=(n, 3)) * np.array([1, 1, 0.5]) + [0, 0, 25],
          observed=0, burn_in=20),
    _flow("Rossler", _rossler, lam=0.0714, h=0.05,
          start=lambda rng, n: np.array([1.0, 1.0, 0.0]) + rng.uniform(-0.5, 0.5, size=(n, 3)),
          observed=0, burn_in=300),
)


def _window(series: np.ndarray) -> np.ndarray:
    """The z-normalised window of ``series`` whose end best matches its start."""
    best, start = np.inf, 0
    for s in range(SEARCH):
        w = series[s:s + LENGTH + 2]
        mismatch = (w[0] - w[LENGTH]) ** 2 + (w[1] - w[LENGTH + 1]) ** 2
        if mismatch < best:
            best, start = mismatch, s
    w = series[start:start + LENGTH]
    return (w - w.mean()) / w.std()


def _iaaft(x: np.ndarray, rng: np.random.Generator):
    """One IAAFT surrogate of ``x`` (Schreiber and Schmitz, 1996) and its relative spectral error."""
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


def _sample_interval(system: System, lam_dt: float) -> float:
    dt = lam_dt / system.lam
    if system.dt_unit:
        dt = max(system.dt_unit, round(dt / system.dt_unit) * system.dt_unit)
    return dt


def datasets(n: int = N_DATASETS):
    """Yield ``(condition, X, y)`` for the first ``n`` datasets of each condition."""
    rng = np.random.default_rng(SEED)
    half = SAMPLES // 2
    for system in SYSTEMS:
        for label, lam_dt in SAMPLINGS:
            dt = _sample_interval(system, lam_dt)
            for _ in range(n):
                orbits = system.simulate(rng, SAMPLES, SEARCH + LENGTH + 2, dt)
                windows = [_window(o) for o in orbits]
                real = windows[:half]
                fake = [_surrogate(w, rng) for w in windows[half:]]
                X = np.array(real + fake)
                y = np.array([1] * half + [0] * half)
                yield f"{system.name}, {label}", X, y


def generate_csv(path, n: int = N_DATASETS, length: int = LENGTH):
    """Write the module's CSV in the format the app parses. Returns the path."""
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Condition", "Label"] + [f"t{i}" for i in range(length)])
        for condition, X, y in datasets(n):
            for row, label in zip(X, y):
                writer.writerow([condition, int(label)] + [f"{v:.6f}" for v in row[:length]])
    return path


def main() -> None:
    path = generate_csv(Path(__file__).with_name("data.csv"))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
