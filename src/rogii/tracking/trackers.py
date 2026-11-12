from __future__ import annotations

import numpy as np
import pandas as pd

from .kernels import pf_ancc_kernel, pf_lik_kernel, pf_z_kernel

ANCC_PARTICLES = 600
Z_PARTICLES = 600

Z_MOMENTUM, Z_VEL_NOISE, Z_POS_NOISE = 0.993, 0.005, 0.01
Z_GR_WINDOW, Z_GR_WEIGHT = 5, 0.3
Z_ROUGHEN_POS, Z_ROUGHEN_VEL = 0.2, 0.003
ANCC_ALPHA, ANCC_RATE_NOISE, ANCC_POS_NOISE = 0.998, 0.002, 0.005
ANCC_INIT_SPREAD, ANCC_ROUGHEN_POS, ANCC_ROUGHEN_RATE = 0.3, 0.1, 0.001
RESAMPLE_AT = 0.5
GR_SIGMA_MIN, GR_SIGMA_MAX, GR_SIGMA_DEFAULT = 10.0, 60.0, 30.0


def typewell_grid(tw_tvt, tw_gr, step: float = 0.2):
    """Dense regular-grid copy of the typewell GR for O(1) lookups inside the kernels."""
    lo, hi = float(tw_tvt.min()), float(tw_tvt.max())
    axis = np.arange(lo, hi + step, step)
    return np.interp(axis, tw_tvt, tw_gr).astype(np.float64), lo, float(step)


def gr_sigma(hw: pd.DataFrame, tw_tvt, tw_gr) -> float:
    known = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(known) < 20:
        return float(GR_SIGMA_DEFAULT)
    spread = np.std(known.GR.values - np.interp(known.TVT_input.values, tw_tvt, tw_gr))
    return float(np.clip(spread, GR_SIGMA_MIN, GR_SIGMA_MAX))


def _initial_rate(known: pd.DataFrame) -> float:
    tail = known.tail(30)
    d_tvt, d_z, d_md = (np.diff(tail[c].values) for c in ("TVT_input", "Z", "MD"))
    ok = d_md > 0
    return float(np.median((d_tvt + d_z)[ok] / d_md[ok])) if ok.sum() >= 3 else 0.0


def run_pf_ancc(hw, tw_tvt, tw_gr, n_particles: int = ANCC_PARTICLES):
    sigma = gr_sigma(hw, tw_tvt, tw_gr)
    known, unknown = hw[hw.TVT_input.notna()], hw[hw.TVT_input.isna()]
    if len(unknown) == 0:
        return np.array([]), np.array([])
    start_level = float(known.TVT_input.iloc[-1] + known.Z.iloc[-1])
    grid, grid_min, grid_step = typewell_grid(tw_tvt, tw_gr)
    path, spread = pf_ancc_kernel(
        unknown.MD.values.astype(np.float64), unknown.Z.values.astype(np.float64), unknown.GR.values.astype(np.float64),
        grid, grid_min, grid_step, sigma, start_level, _initial_rate(known), n_particles,
        ANCC_ALPHA, ANCC_RATE_NOISE, ANCC_POS_NOISE, ANCC_INIT_SPREAD, ANCC_ROUGHEN_POS, ANCC_ROUGHEN_RATE, RESAMPLE_AT,
    )  # fmt: skip
    return path.astype(np.float32), spread.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, n_particles: int = Z_PARTICLES):
    sigma = gr_sigma(hw, tw_tvt, tw_gr)
    tw_smooth = (
        pd.Series(tw_gr).rolling(Z_GR_WINDOW, center=True, min_periods=1).mean().values.astype(np.float32)
    )
    known, unknown = hw[hw.TVT_input.notna()], hw[hw.TVT_input.isna()]
    if len(unknown) == 0:
        return np.array([]), np.array([])

    d_z, d_tvt, d_md = (np.diff(known[c].values) for c in ("Z", "TVT_input", "MD"))
    ok = d_md > 0
    if ok.sum() >= 10:
        vz, vt = d_z[ok] / d_md[ok], d_tvt[ok] / d_md[ok]
        coef = np.linalg.lstsq(np.column_stack([vz, np.ones_like(vz)]), vt, rcond=None)[0]
        beta, intercept = float(coef[0]), float(coef[1])
        z_sigma = max(float(np.std(vt - (coef[0] * vz + coef[1]))), 0.001)
    else:
        beta, intercept, z_sigma = -1.0, 0.0, 0.1

    recent = known.tail(20)
    r_tvt, r_md = np.diff(recent.TVT_input.values), np.diff(recent.MD.values)
    ok2 = r_md > 0
    init_vel = float(np.median(r_tvt[ok2] / r_md[ok2])) if ok2.sum() >= 3 else 0.0

    grid, grid_min, grid_step = typewell_grid(tw_tvt, tw_gr)
    grid_smooth, _, _ = typewell_grid(tw_tvt, tw_smooth)
    gr_smooth = hw.GR.rolling(Z_GR_WINDOW, center=True, min_periods=1).mean()
    path, spread = pf_z_kernel(
        unknown.MD.values.astype(np.float64), unknown.Z.values.astype(np.float64), unknown.GR.values.astype(np.float64),
        gr_smooth.loc[unknown.index].values.astype(np.float64), grid, grid_smooth, grid_min, grid_step, sigma,
        float(known.TVT_input.iloc[-1]), init_vel, beta, intercept, z_sigma, n_particles,
        Z_MOMENTUM, Z_VEL_NOISE, Z_POS_NOISE, Z_GR_WEIGHT, Z_ROUGHEN_POS, Z_ROUGHEN_VEL, RESAMPLE_AT,
    )  # fmt: skip
    return path.astype(np.float32), spread.astype(np.float32)


def lik_pf(
    hw,
    tw,
    n_particles=500,
    n_seeds=128,
    scales=(3.0, 5.0, 8.0, 12.0),
    init_spread=4.5,
    seed_base=0,
    with_quality=False,
):
    """Likelihood-weighted ensemble of compiled filters.

    Returns ({pf_scale_X: path over unknown rows}, unknown-row index, quality stats).
    """
    tw = tw.sort_values("TVT")
    tw_tvt = tw.TVT.values.astype(float)
    tw_gr = tw.GR.fillna(tw.GR.mean()).values.astype(float)
    known, unknown = hw[hw.TVT_input.notna()], hw[hw.TVT_input.isna()]
    if len(unknown) == 0:
        return {}, np.array([]), {}

    last = known.iloc[-1]
    start_level = float(last.TVT_input) + float(last.Z)
    at_known = np.interp(known.TVT_input.values, tw_tvt, tw_gr)
    sigma = float(np.clip(np.nanstd(known.GR.fillna(0).values - at_known), 10.0, 60.0)) * 1.3
    grid, grid_min, grid_step = typewell_grid(tw_tvt, tw_gr)
    gr = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[unknown.index]

    preds, liks = pf_lik_kernel(
        unknown.MD.values.astype(float), unknown.Z.values.astype(float), gr, grid, grid_min, grid_step, sigma,
        start_level, _initial_rate(known), n_particles, n_seeds, seed_base,
        ANCC_ALPHA, ANCC_RATE_NOISE, ANCC_POS_NOISE, ANCC_ROUGHEN_POS, ANCC_ROUGHEN_RATE, RESAMPLE_AT, init_spread,
    )  # fmt: skip
    norm = liks - liks.max()
    out = {}
    for scale in scales:
        w = np.exp(norm / float(scale))
        w /= w.sum()
        out[f"pf_scale_{scale:g}"] = (w[:, None] * preds).sum(0)
    out["pf_mean"] = preds.mean(0)

    quality = {}
    if with_quality:
        quality = {
            "pf_best_ll": float(liks.max()) / len(unknown),
            "pf_ll_spread": float(liks.std()),
            "pf_pt_std": preds.std(0).astype(np.float32),
            "pf_gr_sig": sigma,
        }
    return out, unknown.index.values, quality
