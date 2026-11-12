from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

SUB = 4
GRID = np.arange(-460.0, 460.01, 0.5)

HMM_CFGS = [
    dict(sigma_gr=6.0, step_sd=0.15, prior_sd=25.0),
    dict(sigma_gr=9.0, step_sd=0.15, prior_sd=25.0),
    dict(sigma_gr=12.0, step_sd=0.15, prior_sd=25.0),
    dict(sigma_gr=6.0, step_sd=0.15, prior_sd=80.0),
    dict(sigma_gr=6.0, step_sd=0.05, prior_sd=25.0),
]
PLANE_CFG = dict(sigma_gr=6.0, step_sd=0.15, prior_sd=25.0)


def gr_calibration(gr, tvt, tw_tvt, tw_gr):
    """Affine map of horizontal GR onto the typewell GR scale, fitted on known rows."""
    ok = np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum() < 50:
        return 1.0, 0.0
    a, b = np.polyfit(gr[ok], np.interp(tvt[ok], tw_tvt, tw_gr), 1)
    return (a, b) if 0.2 < a < 5.0 else (1.0, 0.0)


def hmm_align(d: dict, prior, sigma_gr: float, step_sd: float, prior_sd: float, huber: float = 2.5):
    """Forward-backward over the structural drift dS: GR emission plus a soft structural prior."""
    known = d["known"]
    ia = np.where(known)[0][-1]
    ta, za = d["tvt"][ia], d["z"][ia]
    a, b = gr_calibration(d["gr"][known], d["tvt"][known], d["tw_tvt"], d["tw_gr"])

    unknown = np.where(~known)[0]
    use = unknown[::SUB]
    if use[-1] != unknown[-1]:
        use = np.r_[use, unknown[-1]]
    n, ns = len(use), len(GRID)

    cand = ta + GRID[None, :] - (d["z"][use] - za)[:, None]
    tw_gr_at = np.interp(cand, d["tw_tvt"], d["tw_gr"], left=np.nan, right=np.nan)
    r = np.abs((a * d["gr"][use] + b)[:, None] - tw_gr_at) / sigma_gr
    cost = np.where(r <= huber, 0.5 * r**2, huber * (r - 0.5 * huber))
    cost = np.where(np.isfinite(cost), cost, huber * 2.0)

    p = prior[use]
    has_prior = np.isfinite(p)
    if has_prior.any():
        cost[has_prior] += 0.5 * ((GRID[None, :] - p[has_prior][:, None]) / prior_sd) ** 2

    emit = np.exp(-(cost - cost.min(1, keepdims=True)))
    emit /= emit.sum(1, keepdims=True)

    dmd = np.diff(d["md"][use], prepend=d["md"][use][0])
    sd = np.maximum(step_sd * np.sqrt(np.maximum(dmd, 1.0)), 0.25) / (GRID[1] - GRID[0])

    fwd, bwd = np.empty((n, ns)), np.empty((n, ns))
    q = np.exp(-0.5 * (GRID / 4.0) ** 2) * emit[0]
    fwd[0] = q / q.sum()
    for t in range(1, n):
        q = gaussian_filter1d(fwd[t - 1], sd[t], mode="nearest") * emit[t]
        fwd[t] = q / q.sum() if q.sum() > 0 else emit[t]
    bwd[n - 1] = 1.0 / ns
    for t in range(n - 2, -1, -1):
        q = gaussian_filter1d(bwd[t + 1] * emit[t + 1], sd[t + 1], mode="nearest")
        bwd[t] = q / q.sum() if q.sum() > 0 else 1.0 / ns

    post = fwd * bwd
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    s = np.interp(np.arange(len(d["md"])), use, post @ GRID)
    return ta + s - (d["z"] - za)
