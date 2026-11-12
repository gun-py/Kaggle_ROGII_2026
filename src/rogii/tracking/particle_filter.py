from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM = 0.998
RATE_NOISE = 0.002
POS_NOISE = 0.005
ROUGHEN_POS = 0.1
ROUGHEN_RATE = 0.001
RESAMPLE_AT = 0.5


def _typewell_curve(tw: pd.DataFrame):
    tw = tw.sort_values("TVT")
    tvt = tw["TVT"].to_numpy(dtype=float)
    gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(dtype=float)
    return tvt, gr


def run_particle_filter(hw: pd.DataFrame, tw: pd.DataFrame, n_particles: int = 500, seed: int = 42):
    """One seeded particle-filter pass over the unknown rows; returns (path, log-likelihood)."""
    tw_tvt, tw_gr = _typewell_curve(tw)
    known = hw[hw["TVT_input"].notna()]
    unknown = hw[hw["TVT_input"].isna()]
    if len(unknown) == 0:
        return hw["TVT_input"].to_numpy(dtype=float).copy(), 0.0

    last = known.iloc[-1]
    last_tvt, last_z, last_md = float(last["TVT_input"]), float(last["Z"]), float(last["MD"])

    tw_at_known = np.interp(known["TVT_input"].to_numpy(), tw_tvt, tw_gr)
    gr_sigma = float(np.clip(np.nanstd(known["GR"].fillna(0).to_numpy() - tw_at_known), 10.0, 60.0))

    tail = known.tail(30)
    d_tvt, d_z, d_md = (np.diff(tail[c].to_numpy()) for c in ("TVT_input", "Z", "MD"))
    pos_ok = d_md > 0
    init_rate = float(np.median((d_tvt + d_z)[pos_ok] / d_md[pos_ok])) if pos_ok.sum() >= 3 else 0.0

    n = n_particles
    rng = np.random.default_rng(seed)
    pos = last_tvt + last_z + 4.5 * rng.standard_normal(n)
    rate = init_rate + 0.01 * rng.standard_normal(n)
    w = np.ones(n) / n

    md = unknown["MD"].to_numpy(dtype=float)
    z = unknown["Z"].to_numpy(dtype=float)
    gr_filled = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr = gr_filled.to_numpy(dtype=float)[unknown.index]

    out = hw["TVT_input"].to_numpy(dtype=float).copy()
    path = np.empty(len(unknown))
    prev_md = last_md
    log_lik = 0.0

    for i in range(len(unknown)):
        step = max(md[i] - prev_md, 1.0)
        rate = MOMENTUM * rate + RATE_NOISE * rng.standard_normal(n)
        pos = pos + rate * step + POS_NOISE * rng.standard_normal(n)
        tvt = np.clip(pos - z[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt + z[i]

        expected = np.interp(tvt, tw_tvt, tw_gr)
        resid = (gr[i] - expected) / gr_sigma
        lik = np.maximum(np.exp(-0.5 * np.minimum(resid**2, 600.0)), 1e-300)
        log_lik += np.log(max(float((w * lik).sum()), 1e-300))
        w = w * lik
        total = w.sum()
        w = w / total if total > 0 else np.ones(n) / n

        if 1.0 / (w**2).sum() < RESAMPLE_AT * n:
            cum = np.cumsum(w)
            start = rng.uniform(0, 1.0 / n)
            pick = np.clip(np.searchsorted(cum, start + np.arange(n) / n), 0, n - 1)
            pos = pos[pick] + ROUGHEN_POS * rng.standard_normal(n)
            rate = rate[pick] + ROUGHEN_RATE * rng.standard_normal(n)
            w = np.ones(n) / n

        path[i] = float(np.dot(w, pos - z[i]))
        prev_md = md[i]

    out[list(unknown.index)] = path
    return out, log_lik


def seed_branches(preds: np.ndarray, log_lik_norm: np.ndarray, eval_mask: np.ndarray) -> dict | None:
    """Split the seed ensemble into a low and a high TVT branch (weighted 1-D two-means)."""
    if int(eval_mask.sum()) < 10:
        return None
    seed_weight = np.exp(log_lik_norm / 5.0)
    seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
    level = np.nanmedian(preds[:, eval_mask], axis=1)
    valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0)
    level, seed_weight = level[valid], seed_weight[valid]
    seed_weight = seed_weight / max(float(seed_weight.sum()), 1e-12)
    if len(level) < 4:
        return None

    order = np.argsort(level)
    x, w = level[order], seed_weight[order]
    cw, cx, cx2 = np.cumsum(w), np.cumsum(w * x), np.cumsum(w * x * x)
    total_w, total_x, total_x2 = float(cw[-1]), float(cx[-1]), float(cx2[-1])

    best = None
    for cut in range(1, len(x)):
        wl = float(cw[cut - 1])
        wr = total_w - wl
        if wl < 0.05 or wr < 0.05:
            continue
        xl = float(cx[cut - 1])
        xr = total_x - xl
        sse_l = float(cx2[cut - 1] - xl * xl / wl)
        sse_r = float(total_x2 - cx2[cut - 1] - xr * xr / wr)
        score = max(0.0, sse_l) + max(0.0, sse_r)
        if best is None or score < best[0]:
            best = (score, wl, wr, xl / wl, xr / wr)
    if best is None:
        return None

    _, mass_low, mass_high, centre_low, centre_high = best
    return dict(
        center_low=float(centre_low),
        center_high=float(centre_high),
        mass_low=float(mass_low),
        mass_high=float(mass_high),
        weighted_center=float(np.sum(seed_weight * level)),
        eval_rows=np.flatnonzero(eval_mask).astype(int).tolist(),
        seed_count=int(len(level)),
    )


def ensemble_by_scale(
    hw, tw, scales=(3.0, 5.0, 8.0, 12.0), n_particles=500, n_seeds=128, return_std=False, branch_stats=None
) -> dict[str, np.ndarray]:
    """Likelihood-weighted average of many seeded filters, one output per temperature scale."""
    runs = [run_particle_filter(hw, tw, n_particles=n_particles, seed=s) for s in range(n_seeds)]
    preds = np.stack([p for p, _ in runs], 0)
    log_lik = np.array([ll for _, ll in runs])
    log_lik_norm = log_lik - log_lik.max()

    out = {}
    for scale in scales:
        weights = np.exp(log_lik_norm / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    if return_std:
        out["pf_seed_std"] = preds.std(0)

    if branch_stats is not None:
        try:
            eval_mask = pd.to_numeric(hw["TVT_input"], errors="coerce").isna().to_numpy()
            found = seed_branches(preds, log_lik_norm, eval_mask)
            if found:
                branch_stats.update(found)
        except Exception as exc:
            branch_stats["error"] = repr(exc)
    return out
