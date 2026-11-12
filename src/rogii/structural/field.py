from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree

NB_CUTOFF = 1200.0
NB_MAX = 8
WT_ANCHOR_MAX = 700.0
WT_CUTOFF = 1500.0
WT_SCALE = 300.0
PLANE_CUTOFF = 1100.0
PLANE_MAX = 10
DUP_TOL = 25.0


class Field:
    """Train wells indexed for datum-free structural queries.

    Each well carries S = TVT + Z, so differences of S between two points of the same
    well do not depend on that well's datum.
    """

    def __init__(self, wells: dict[str, dict]):
        self.pool = list(wells)
        self.xy, self.s, self.tree = {}, {}, {}
        for w in self.pool:
            d = wells[w]
            self.xy[w] = np.c_[d["x"], d["y"]]
            self.s[w] = d["tvt"] + d["z"]
            self.tree[w] = cKDTree(self.xy[w])
        centroids = np.array([[self.xy[w][:, 0].mean(), self.xy[w][:, 1].mean()] for w in self.pool])
        self.centroids = cKDTree(centroids)

    def near(self, xy, radius: float = 15000.0):
        return [self.pool[k] for k in self.centroids.query_ball_point(xy.mean(0), radius)]

    def delta_s(self, xy, ia, drop):
        """Median over near wells of S(p) - S(anchor); NaN where unsupported, plus the nearest support distance."""
        rows, dist = [], []
        for j in self.near(xy):
            if j in drop:
                continue
            da, ja = self.tree[j].query(xy[ia])
            if da > NB_CUTOFF:
                continue
            dp, jp = self.tree[j].query(xy)
            med = float(np.median(dp))
            if med < DUP_TOL or med > NB_CUTOFF:
                continue
            ds = self.s[j][jp] - self.s[j][ja]
            ds[dp > 2 * NB_CUTOFF] = np.nan
            rows.append(ds)
            dist.append(max(da, med))
        if not rows:
            return np.full(len(xy), np.nan), np.inf
        keep = np.argsort(dist)[:NB_MAX]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            merged = np.nanmedian(np.array([rows[k] for k in keep]), axis=0)
        return merged, float(np.min(dist))

    def delta_s_weighted(self, xy, ia, drop):
        """Per-row inverse-distance blend, so partially overlapping wells still count."""
        num = np.zeros(len(xy))
        den = np.zeros(len(xy))
        for j in self.near(xy):
            if j in drop:
                continue
            da, ja = self.tree[j].query(xy[ia])
            if da > WT_ANCHOR_MAX:
                continue
            dp, jp = self.tree[j].query(xy)
            if np.median(dp) < DUP_TOL:
                continue
            w = 1.0 / (1.0 + (dp / WT_SCALE) ** 2)
            w[dp > WT_CUTOFF] = 0.0
            if w.sum() <= 0:
                continue
            num += (self.s[j][jp] - self.s[j][ja]) * w
            den += w
        return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)


def _basis(u, v):
    return np.column_stack([np.ones_like(u), u, u * u, v, u * v])


def plane_delta_s(field: Field, d: dict, drop, ridge=3.0, prefix_w=3.0, min_wells=2, clamp=400.0):
    """Fit S_j = F(u, v) + k_j over the neighbours and the query prefix (k_query = 0), then read F along the well."""
    known = d["known"]
    ia = np.where(known)[0][-1]
    xy = np.c_[d["x"], d["y"]]
    S = d["tvt"] + d["z"]

    vec = xy[-1] - xy[max(0, ia - 400)]
    n = np.hypot(*vec)
    if n < 1:
        return np.full(len(xy), np.nan)
    e1 = vec / n
    e2 = np.array([-e1[1], e1[0]])
    rel = xy - xy[ia]
    u, v = (rel @ e1) / 1000.0, (rel @ e2) / 1000.0

    qt = cKDTree(xy)
    nk = int(known.sum())
    rows = [(u[known], v[known], S[known], np.full(nk, -1), np.full(nk, prefix_w))]
    used = 0
    for j in field.near(xy):
        if j in drop:
            continue
        da, _ = field.tree[j].query(xy[ia])
        if da > PLANE_CUTOFF:
            continue
        jxy = field.xy[j]
        dp, _ = qt.query(jxy)
        m = dp < PLANE_CUTOFF
        if m.sum() < 50 or np.median(dp[m]) < DUP_TOL:
            continue
        r = jxy[m] - xy[ia]
        rows.append((
            (r @ e1) / 1000.0,
            (r @ e2) / 1000.0,
            field.s[j][m],
            np.full(int(m.sum()), used),
            np.full(int(m.sum()), 1.0 / (1.0 + da / 500.0)),
        ))  # fmt: skip
        used += 1
        if used >= PLANE_MAX:
            break
    if used < min_wells:
        return np.full(len(xy), np.nan)

    uu, vv, ss, gg, ww = (np.concatenate([r[i] for r in rows]) for i in range(5))
    B = _basis(uu, vv)
    nb = B.shape[1]
    dummies = np.zeros((len(uu), used))
    for k in range(used):
        dummies[gg == k, k] = 1.0
    A = np.hstack([B, dummies]) * np.sqrt(ww)[:, None]
    reg = ridge * np.eye(nb + used)
    reg[0, 0] = 1e-6
    coef = np.linalg.solve(A.T @ A + reg, A.T @ (ss * np.sqrt(ww)))
    f = _basis(u, v) @ coef[:nb]
    return np.clip(f - f[ia], -clamp, clamp)
