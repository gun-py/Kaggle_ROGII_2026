"""Label-free synthetic wells for pretraining.

Real typewells and real trajectories, simulated TVT paths and simulated GR. No TVT label from any
well is used, so the pretrained weights can be shared by every evaluation split.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import lfilter

from rogii.attention.data import Typewell, make_sample

PHI = 0.998
SIGMA_RANGE = (-3.7, -2.7)
FAULTS_PER_FT = 1.0 / 3000.0


def _ar1(n, phi, sigma, rng, start=0.0):
    return lfilter([1.0], [1.0, -phi], rng.normal(0.0, sigma, n), zi=[phi * start])[0]


def tvt_path(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """TVT relative to row 0: slow random drift, rare faults, and the trajectory's own short wiggles."""
    n = len(z)
    rate = _ar1(n, PHI, 10 ** rng.uniform(*SIGMA_RANGE), rng, start=rng.normal(0.0, 0.003))
    drift = np.cumsum(rate)
    if rng.random() < 0.3:
        for at in np.flatnonzero(rng.random(n) < FAULTS_PER_FT):
            drift[at:] += rng.normal(0.0, 6.0)
    wiggle = z - uniform_filter1d(z, 301, mode="nearest")
    path = drift - wiggle
    return path - path[0]


def synthetic_gr(tvt: np.ndarray, tw: Typewell, rng: np.random.Generator, noise: bool = True) -> np.ndarray:
    pivot = float(np.mean(tvt))
    stretch = rng.uniform(0.92, 1.08) if noise else 1.0
    warped = pivot + (tvt - pivot) * stretch
    if noise:
        wavelength = rng.uniform(300.0, 3000.0)
        warped = warped + rng.uniform(0.0, 3.0) * np.sin(2 * np.pi * np.arange(len(tvt)) / wavelength)
    g = np.interp(warped, tw.tvt, tw.raw_gr)
    if not noise:
        return g
    g = gaussian_filter1d(g, rng.uniform(0.0, 3.0)) if rng.random() < 0.7 else g
    g = rng.uniform(0.7, 1.4) * g + rng.normal(0.0, 0.3 * tw.sd)
    g = g + _ar1(len(g), 0.9, rng.uniform(0.05, 0.4) * tw.sd * np.sqrt(1 - 0.81), rng)
    g = g + rng.normal(0.0, rng.uniform(0.02, 0.2) * tw.sd, len(g))
    spikes = rng.random(len(g)) < 0.002
    g[spikes] += rng.uniform(1.0, 3.0, spikes.sum()) * tw.sd
    g[rng.random(len(g)) < rng.uniform(0.0, 0.6)] = np.nan
    return g


class SyntheticWells:
    def __init__(self, typewells: list[Typewell], trajectories: list[dict], fractions: np.ndarray):
        self.typewells = typewells
        self.trajectories = trajectories
        self.fractions = np.asarray(fractions)

    @classmethod
    def from_wellset(cls, ws) -> SyntheticWells:
        trajectories = [{k: d[k] for k in ("md", "x", "y", "z")} for d in ws.wells.values()]
        return cls(list(ws.typewells.values()), trajectories, ws.fractions)

    def well(self, rng: np.random.Generator, noise: bool = True) -> tuple[dict, Typewell]:
        tw = self.typewells[rng.integers(len(self.typewells))]
        traj = self.trajectories[rng.integers(len(self.trajectories))]
        rel = tvt_path(traj["z"], rng) if noise else np.zeros(len(traj["z"]))
        lo, hi = tw.tvt[0] + 20.0, tw.tvt[-1] - 20.0
        span_lo, span_hi = lo - rel.min(), hi - rel.max()
        start = rng.uniform(span_lo, span_hi) if span_hi > span_lo else 0.5 * (span_lo + span_hi)
        tvt = start + rel
        d = dict(traj, tvt=tvt, gr=synthetic_gr(tvt, tw, rng, noise), tw_tvt=tw.tvt, tw_gr=tw.raw_gr)
        return d, tw

    def sample(self, rng: np.random.Generator, max_visible: int, noise: bool = True):
        d, tw = self.well(rng, noise)
        n = len(d["md"])
        frac = rng.choice(self.fractions) * rng.uniform(0.9, 1.1)
        ia = int(np.clip(round(frac * n) - 1, 100, n - 200))
        d["known"] = np.arange(n) <= ia
        return make_sample("synthetic", d, tw, ia, max_visible)
