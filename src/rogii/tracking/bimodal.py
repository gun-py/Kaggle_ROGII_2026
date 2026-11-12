from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.config import SelectorConfig
from rogii.geology.calibration import eval_mask as default_eval_mask
from rogii.geology.calibration import gr_scale, heel_calibrate, typewell_arrays

HEEL_KEYS = (
    "heel_calibrated",
    "heel_rows",
    "heel_alpha",
    "heel_beta",
    "heel_rmse_raw",
    "heel_rmse_calibrated",
    "heel_denoised",
)


def _finite_or_nan(x) -> float:
    return float(x) if np.isfinite(x) else np.nan


def _heel_fields(info: dict | None) -> dict:
    info = info or {}
    fields = {
        "heel_calibrated": bool(info.get("heel_calibrated", False)),
        "heel_rows": int(info.get("heel_rows", 0) or 0),
        "heel_denoised": bool(info.get("heel_denoised", False)),
    }
    for key in ("heel_alpha", "heel_beta", "heel_rmse_raw", "heel_rmse_calibrated"):
        fields[key] = _finite_or_nan(info.get(key, np.nan))
    return fields


def _calibrated_gr(hw, tw_tvt, tw_gr, cfg: SelectorConfig):
    raw = pd.to_numeric(hw["GR"], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
    return heel_calibrate(hw, tw_tvt, tw_gr, raw, cfg.heel)


def gr_misfit(hw, tw, tvt_path, cfg: SelectorConfig, mask=None):
    """Mean clipped squared GR residual of a TVT path; returns (score, rows used)."""
    tw_tvt, tw_gr = typewell_arrays(tw)
    if tw_tvt is None or hw is None or "GR" not in hw.columns or tvt_path is None:
        return np.nan, 0
    hgr, _ = _calibrated_gr(hw, tw_tvt, tw_gr, cfg)
    path = np.asarray(tvt_path, dtype=float)
    n = min(len(hgr), len(path))
    rows = default_eval_mask(hw, n) if mask is None else np.asarray(mask, dtype=bool)[:n]
    rows = rows & np.isfinite(hgr[:n]) & np.isfinite(path[:n])
    if int(rows.sum()) < cfg.bimodal.min_valid_gr_rows:
        return np.nan, int(rows.sum())
    resid = hgr[:n][rows] - np.interp(path[:n][rows], tw_tvt, tw_gr)
    scale = gr_scale(hw, tw_tvt, tw_gr, hgr[:n], fallback_resid=resid)
    z = np.clip(resid / scale, -6.0, 6.0)
    return float(np.nanmean(z * z)), int(rows.sum())


def lag1_autocorr(resid) -> float:
    x = np.asarray(resid, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    a = x[:-1] - np.nanmean(x[:-1])
    b = x[1:] - np.nanmean(x[1:])
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(np.sum(a * b) / denom, 0.0, 0.999))


def softmax_temperature(resid, n_valid: int, cfg: SelectorConfig):
    base = max(cfg.bimodal.temp, 1e-6)
    if not cfg.bimodal.adaptive_temp:
        return base, np.nan, np.nan
    rho = lag1_autocorr(resid)
    n_eff = float(max(n_valid, 1)) * (1.0 - rho) / max(1.0 + rho, 1e-6)
    temp = float(np.clip(2.0 / max(n_eff, 1.0), cfg.bimodal.temp_min, cfg.bimodal.temp_max))
    return temp, float(rho), float(n_eff)


def prefix_trust(hw, tw, base, delta_b, cfg: SelectorConfig):
    """How strongly the visible prefix prefers the base path over the shifted decoy."""
    b = cfg.bimodal
    if not b.prefix_trust_gate:
        return 1.0, np.nan, np.nan, np.nan, 0
    if hw is None or "TVT_input" not in hw.columns:
        return b.trust_fallback, np.nan, np.nan, np.nan, 0
    n = min(len(hw), len(base))
    prefix = hw["TVT_input"].notna().to_numpy()[:n]
    try:
        j_base, rows = gr_misfit(hw, tw, base, cfg, mask=prefix)
        j_decoy, _ = gr_misfit(hw, tw, np.asarray(base, dtype=float) + float(delta_b), cfg, mask=prefix)
    except Exception:
        return b.trust_fallback, np.nan, np.nan, np.nan, 0
    if int(rows) < b.trust_min_prefix_rows:
        return b.trust_fallback, j_base, j_decoy, np.nan, int(rows)
    if not (np.isfinite(j_base) and np.isfinite(j_decoy)):
        return b.trust_fallback, j_base, j_decoy, np.nan, int(rows)
    margin = float(j_decoy - j_base)
    denom = max(b.trust_kappa * max(float(j_base), 1e-12), 1e-12)
    return float(np.clip(margin / denom, 0.0, 1.0)), float(j_base), float(j_decoy), margin, int(rows)


def bimodal_scan(hw, tw, base_path, cfg: SelectorConfig, mask=None):
    """Scan datum shifts for a second GR-consistent branch near the base path."""
    b = cfg.bimodal
    tw_tvt, tw_gr = typewell_arrays(tw)
    if tw_tvt is None or hw is None or "GR" not in hw.columns or base_path is None:
        return None
    hgr, heel_info = _calibrated_gr(hw, tw_tvt, tw_gr, cfg)
    heel = _heel_fields(heel_info)
    base = np.asarray(base_path, dtype=float)
    n = min(len(hgr), len(base))
    rows = default_eval_mask(hw, n) if mask is None else np.asarray(mask, dtype=bool)[:n]
    rows = rows & np.isfinite(hgr[:n]) & np.isfinite(base[:n])
    if int(rows.sum()) < b.min_valid_gr_rows:
        return None

    h, path = hgr[:n][rows], base[:n][rows]
    resid0 = h - np.interp(path, tw_tvt, tw_gr)
    scale = gr_scale(hw, tw_tvt, tw_gr, hgr[:n], fallback_resid=resid0)

    step = b.dz_step if b.dz_step > 0 else 0.5
    deltas = np.arange(-b.dz_range, b.dz_range + 0.5 * step, step, dtype=float)
    scores = np.empty(len(deltas))
    for i, dz in enumerate(deltas):
        z = np.clip((h - np.interp(path + dz, tw_tvt, tw_gr)) / scale, -6.0, 6.0)
        scores[i] = float(np.nanmean(z * z))
    finite = np.isfinite(scores)
    if int(finite.sum()) < 3:
        return None

    best = int(np.nanargmin(scores))
    delta_a, score_a = float(deltas[best]), float(scores[best])
    sep = np.abs(deltas - delta_a)
    candidate = finite & (sep >= max(b.scan_min_sep, b.bundle_min)) & (sep <= b.bundle_max)
    resid_a = h - np.interp(path + delta_a, tw_tvt, tw_gr)
    temp, rho1, n_eff = softmax_temperature(resid_a, int(rows.sum()), cfg)
    common = dict(
        temperature=float(temp),
        rho1=_finite_or_nan(rho1),
        n_eff=_finite_or_nan(n_eff),
        score_a=score_a,
        valid_gr_rows=int(rows.sum()),
        **heel,
    )
    if not candidate.any():
        return dict(
            is_bimodal=False, delta_a=delta_a, delta_b=np.nan, delta_star=delta_a,
            p_base=1.0, p_eff=1.0, prefix_trust=1.0, j_prefix_base=np.nan,
            j_prefix_decoy=np.nan, prefix_trust_margin=np.nan, prefix_rows=0,
            score_b=np.nan, j_ratio=np.nan, dz_gap=np.nan, forced_midpoint=False, **common,
        )  # fmt: skip

    pool = np.flatnonzero(candidate)
    second = int(pool[np.nanargmin(scores[pool])])
    delta_b, score_b = float(deltas[second]), float(scores[second])
    is_bimodal = bool(score_b <= (1.0 + b.j_ratio_eps) * max(score_a, 1e-12))
    p_base = 1.0 / (1.0 + np.exp(np.clip((score_a - score_b) / max(temp, 1e-6), -50.0, 50.0)))
    trust, j_base, j_decoy, margin, prefix_rows = prefix_trust(hw, tw, base, delta_b, cfg)
    if b.force_midpoint:
        p_eff, forced = 0.5, True
    else:
        p_eff, forced = 0.5 + float(trust) * (float(p_base) - 0.5), False
    return dict(
        is_bimodal=is_bimodal,
        delta_a=delta_a,
        delta_b=delta_b,
        delta_star=float(p_eff * delta_a + (1.0 - p_eff) * delta_b),
        p_base=float(p_base),
        p_eff=float(p_eff),
        prefix_trust=float(trust),
        j_prefix_base=_finite_or_nan(j_base),
        j_prefix_decoy=_finite_or_nan(j_decoy),
        prefix_trust_margin=_finite_or_nan(margin),
        prefix_rows=int(prefix_rows),
        score_b=score_b,
        j_ratio=float(score_b / max(score_a, 1e-12)),
        dz_gap=float(abs(delta_b - delta_a)),
        forced_midpoint=forced,
        **common,
    )


def bimodal_hedge(base, beam, hw, tw, cfg: SelectorConfig):
    """Datum-shift hedge, or None when PF and beam agree or the scan finds a single mode."""
    b = cfg.bimodal
    base = np.asarray(base, dtype=float)
    beam = np.asarray(beam, dtype=float)
    n = min(len(base), len(beam))
    if n == 0:
        return None
    rows = default_eval_mask(hw, n)
    diff = np.abs(base[:n] - beam[:n])
    valid = rows & np.isfinite(diff)
    if int(valid.sum()) < b.min_valid_gr_rows:
        return None
    median = float(np.nanmedian(diff[valid]))
    p90 = float(np.nanquantile(diff[valid], 0.90))
    big_frac = float(np.nanmean(diff[valid] >= b.trigger_big_diff))
    if median < b.trigger_min_median_diff and p90 < b.trigger_min_p90_diff:
        return None
    if big_frac < b.trigger_min_big_diff_frac:
        return None

    scan = bimodal_scan(hw, tw, base, cfg, mask=rows)
    if not scan or not scan.get("is_bimodal"):
        return None
    return {
        "delta_star": float(scan["delta_star"]),
        "p_base": float(scan["p_base"]),
        "p_eff": float(scan.get("p_eff", scan["p_base"])),
        "prefix_trust": float(scan.get("prefix_trust", np.nan)),
        "j_prefix_base": float(scan.get("j_prefix_base", np.nan)),
        "j_prefix_decoy": float(scan.get("j_prefix_decoy", np.nan)),
        "prefix_trust_margin": float(scan.get("prefix_trust_margin", np.nan)),
        "prefix_rows": int(scan.get("prefix_rows", 0)),
        "temperature": float(scan.get("temperature", np.nan)),
        "rho1": float(scan.get("rho1", np.nan)),
        "n_eff": float(scan.get("n_eff", np.nan)),
        "forced_midpoint": bool(scan.get("forced_midpoint", False)),
        "score_base": float(scan["score_a"]),
        "score_second": float(scan["score_b"]),
        "delta_a": float(scan["delta_a"]),
        "delta_b": float(scan["delta_b"]),
        "j_ratio": float(scan["j_ratio"]),
        "dz_gap": float(scan["dz_gap"]),
        "median_abs_diff": median,
        "p90_abs_diff": p90,
        "big_diff_frac": big_frac,
        "valid_gr_rows": int(scan["valid_gr_rows"]),
        "heel_calibrated": bool(scan.get("heel_calibrated", False)),
        "heel_rows": int(scan.get("heel_rows", 0)),
        "heel_alpha": float(scan.get("heel_alpha", np.nan)),
        "heel_beta": float(scan.get("heel_beta", np.nan)),
        "heel_rmse_raw": float(scan.get("heel_rmse_raw", np.nan)),
        "heel_rmse_calibrated": float(scan.get("heel_rmse_calibrated", np.nan)),
        "heel_denoised": bool(scan.get("heel_denoised", False)),
    }
