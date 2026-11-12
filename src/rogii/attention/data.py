"""Token views of wells for the TRAIL model.

Every feature is relative to the anchor (the last visible row), so the model never sees a datum:
GR is mapped onto the typewell scale, depths and TVT enter only as differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rogii.config import HeelConfig
from rogii.geology.calibration import heel_calibrate
from rogii.structural.field import DUP_TOL, NB_CUTOFF, NB_MAX
from rogii.structural.wells import anchor_index

STRIDE = 16
TW_PAD = 64
TOKEN_FEATURES = 8
NEIGHBOUR_FEATURES = 4
DRIFT_SCALE = 50.0
CANDIDATE_RADIUS = 15000.0
MAX_CANDIDATES = 20
HEEL = HeelConfig()


@dataclass
class Typewell:
    lo: float
    gr: np.ndarray
    valid: np.ndarray
    mu: float
    sd: float
    tvt: np.ndarray
    raw_gr: np.ndarray

    def grid(self) -> np.ndarray:
        return np.column_stack([self.gr, self.valid]).astype(np.float32)


@dataclass
class Sample:
    wid: str
    feats: np.ndarray
    pos: np.ndarray
    hidden: np.ndarray
    drift: np.ndarray
    target: np.ndarray
    tw: np.ndarray
    u_anchor: float
    nb: np.ndarray
    nb_mask: np.ndarray
    centres: np.ndarray
    ia: int
    tvt_anchor: float

    def __len__(self) -> int:
        return len(self.centres)


def typewell_grid(tw_tvt, tw_gr, pad: int = TW_PAD) -> Typewell:
    """Typewell GR averaged onto a 1 ft TVT grid that starts on a whole foot."""
    t, g = np.asarray(tw_tvt, float), np.asarray(tw_gr, float)
    ok = np.isfinite(t) & np.isfinite(g)
    order = np.argsort(t[ok])
    t, g = t[ok][order], g[ok][order]
    lo = np.floor(t[0]) - pad
    grid = np.arange(lo, np.ceil(t[-1]) + pad + 1.0)
    area = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))])
    left, right = np.clip(grid - 0.5, t[0], t[-1]), np.clip(grid + 0.5, t[0], t[-1])
    width = right - left
    binned = (np.interp(right, t, area) - np.interp(left, t, area)) / np.maximum(width, 1e-9)
    mean = np.where(width > 1e-6, binned, np.interp(grid, t, g))
    valid = (grid >= t[0]) & (grid <= t[-1])
    mu, sd = float(g.mean()), float(g.std() + 1e-6)
    gr = np.where(valid, (mean - mu) / sd, 0.0).astype(np.float32)
    return Typewell(float(lo), gr, valid, mu, sd, t, g)


def calibrated_gr(gr, tvt, ia: int, tw: Typewell) -> np.ndarray:
    tvt_input = np.full(len(gr), np.nan)
    tvt_input[: ia + 1] = tvt[: ia + 1]
    out, _ = heel_calibrate(pd.DataFrame({"TVT_input": tvt_input}), tw.tvt, tw.raw_gr, gr, HEEL)
    return (out - tw.mu) / tw.sd


def token_centres(n: int, ia: int, max_visible: int) -> np.ndarray:
    back = min(ia // STRIDE, max_visible)
    centres = ia + STRIDE * np.arange(-back, (n - 1 - ia) // STRIDE + 1)
    return centres if centres[-1] == n - 1 else np.append(centres, n - 1)


def _window_stats(values, centres, half: int):
    ok = np.isfinite(values)
    v = np.where(ok, values, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    lo = np.clip(centres - half, 0, len(values))
    hi = np.clip(centres + half, 0, len(values))
    count = cn[hi] - cn[lo]
    mean = np.where(count > 0, (c1[hi] - c1[lo]) / np.maximum(count, 1), 0.0)
    var = np.where(count > 1, (c2[hi] - c2[lo]) / np.maximum(count, 1) - mean**2, 0.0)
    return mean, np.sqrt(np.maximum(var, 0.0)), count / np.maximum(hi - lo, 1)


def make_sample(wid, d, tw: Typewell, ia: int, max_visible: int, nb=None, nb_mask=None) -> Sample:
    n = len(d["md"])
    c = token_centres(n, ia, max_visible)
    z, md, tvt = d["z"], d["md"], d["tvt"]
    g_mean, g_std, g_frac = _window_stats(calibrated_gr(d["gr"], tvt, ia, tw), c, STRIDE // 2)
    lo, hi = np.maximum(c - STRIDE // 2, 0), np.minimum(c + STRIDE // 2, n - 1)
    incl = (z[hi] - z[lo]) / np.maximum(md[hi] - md[lo], 1.0)
    visible = c <= ia
    tvt_a = float(tvt[ia])
    target = tvt[c] - tvt_a
    known = np.where(visible, target, 0.0)
    feats = np.column_stack([
        g_mean, g_std, g_frac,
        (z[c] - z[ia]) / 100.0, (md[c] - md[ia]) / 1000.0, incl * 10.0,
        visible, known / DRIFT_SCALE,
    ]).astype(np.float32)  # fmt: skip
    if nb is None:
        nb = np.zeros((len(c), NB_MAX, NEIGHBOUR_FEATURES), np.float32)
        nb_mask = np.zeros((len(c), NB_MAX), bool)
    return Sample(
        wid=wid,
        feats=feats,
        pos=((c - ia) / STRIDE).astype(np.float32),
        hidden=~visible,
        drift=known.astype(np.float32),
        target=np.where(visible, known, target).astype(np.float32),
        tw=tw.grid(),
        u_anchor=tvt_a - tw.lo,
        nb=nb,
        nb_mask=nb_mask,
        centres=c,
        ia=ia,
        tvt_anchor=tvt_a,
    )


def rows_from_tokens(sample: Sample, drift: np.ndarray, z: np.ndarray) -> np.ndarray:
    """TVT on every row past the anchor, interpolating the smooth level S = TVT + Z between tokens."""
    c = sample.centres
    s = sample.tvt_anchor + drift + z[c]
    rows = np.arange(sample.ia + 1, len(z))
    return np.interp(rows, c, s) - z[rows]


class NeighbourIndex:
    """Nearest points of nearby train wells for every row of each query well.

    The anchor-dependent part of the neighbour rule in `Field.delta_s` then reduces to lookups.
    """

    def __init__(self, pool: dict[str, dict], queries: dict[str, dict]):
        self.s = {w: d["tvt"] + d["z"] for w, d in pool.items()}
        trees = {w: cKDTree(np.c_[d["x"], d["y"]]) for w, d in pool.items()}
        names = list(pool)
        centroids = cKDTree(np.array([[pool[w]["x"].mean(), pool[w]["y"].mean()] for w in names]))
        self.table = {}
        for q, dq in queries.items():
            xy = np.c_[dq["x"], dq["y"]]
            rows = []
            for k in centroids.query_ball_point(xy.mean(0), CANDIDATE_RADIUS):
                j = names[k]
                if j == q or trees[j].query(xy[::64])[0].min() > NB_CUTOFF:
                    continue
                dp, jp = trees[j].query(xy)
                med = float(np.median(dp))
                if DUP_TOL <= med <= NB_CUTOFF:
                    rows.append((med, j, dp.astype(np.float32), jp.astype(np.int32)))
            rows.sort(key=lambda r: r[0])
            self.table[q] = rows[:MAX_CANDIDATES]

    def features(self, q: str, dq: dict, centres: np.ndarray, ia: int, allowed):
        nb = np.zeros((len(centres), NB_MAX, NEIGHBOUR_FEATURES), np.float32)
        mask = np.zeros((len(centres), NB_MAX), bool)
        picks = []
        for med, j, dp, jp in self.table.get(q, ()):
            if j in allowed and dp[ia] <= NB_CUTOFF:
                picks.append((max(float(dp[ia]), med), med, j, dp, jp))
        picks.sort(key=lambda r: r[0])
        dz = dq["z"][centres] - dq["z"][ia]
        for m, (_, med, j, dp, jp) in enumerate(picks[:NB_MAX]):
            dist = dp[centres]
            implied = self.s[j][jp[centres]] - self.s[j][jp[ia]] - dz
            nb[:, m] = np.column_stack([
                implied / DRIFT_SCALE, np.log1p(dist / 300.0),
                np.full(len(centres), dp[ia] / 1000.0), np.full(len(centres), med / 1000.0),
            ])  # fmt: skip
            mask[:, m] = dist <= 2 * NB_CUTOFF
        return nb, mask


class WellSet:
    """Wells, typewell grids and the neighbour index shared by every sampler."""

    def __init__(self, train: dict[str, dict], test: dict[str, dict] | None = None):
        self.train_ids = sorted(train)
        self.wells = {**train, **(test or {})}
        self.typewells = {w: typewell_grid(d["tw_tvt"], d["tw_gr"]) for w, d in self.wells.items()}
        self.index = NeighbourIndex(train, self.wells)
        self.fractions = np.array(
            [(anchor_index(train[w]) + 1) / len(train[w]["md"]) for w in self.train_ids]
        )

    def sample(self, wid: str, allowed, ia: int | None = None, max_visible: int = 48, neighbours=True):
        d = self.wells[wid]
        ia = anchor_index(d) if ia is None else int(ia)
        centres = token_centres(len(d["md"]), ia, max_visible)
        nb = mask = None
        if neighbours:
            nb, mask = self.index.features(wid, d, centres, ia, allowed)
        return make_sample(wid, d, self.typewells[wid], ia, max_visible, nb, mask)

    def random_anchor(self, wid: str, rng: np.random.Generator, real_prob: float) -> int:
        d = self.wells[wid]
        n = len(d["md"])
        if rng.random() < real_prob:
            return anchor_index(d)
        frac = rng.choice(self.fractions) * rng.uniform(0.9, 1.1)
        return int(np.clip(round(frac * n) - 1, 100, n - 200))
