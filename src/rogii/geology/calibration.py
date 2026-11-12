from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.config import HeelConfig


def typewell_arrays(tw: pd.DataFrame | None):
    if tw is None or "TVT" not in tw.columns or "GR" not in tw.columns:
        return None, None
    tw = tw.sort_values("TVT")
    tvt = pd.to_numeric(tw["TVT"], errors="coerce").to_numpy(dtype=float)
    gr = pd.to_numeric(tw["GR"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(tvt) & np.isfinite(gr)
    if int(ok.sum()) < 10:
        return None, None
    tvt, gr = tvt[ok], gr[ok]
    order = np.argsort(tvt)
    return tvt[order], gr[order]


def eval_mask(hw: pd.DataFrame | None, n: int) -> np.ndarray:
    if hw is not None and "TVT_input" in hw.columns:
        return hw["TVT_input"].isna().to_numpy()[:n]
    return np.ones(n, dtype=bool)


def gr_scale(hw, tw_tvt, tw_gr, hgr, fallback_resid=None) -> float:
    """Robust sigma of horizontal-vs-typewell GR, from the known prefix when there is one."""
    if hw is not None and "TVT_input" in hw.columns:
        known = hw["TVT_input"].notna().to_numpy()[: len(hgr)] & np.isfinite(hgr)
        if int(known.sum()) >= 25:
            known_tvt = pd.to_numeric(hw.loc[known, "TVT_input"], errors="coerce").to_numpy(dtype=float)
            ref = np.interp(known_tvt, tw_tvt, tw_gr)
            scale = np.nanmedian(np.abs(hgr[known] - ref)) * 1.4826
            if np.isfinite(scale) and scale >= 8.0:
                return float(scale)
    if fallback_resid is not None and len(fallback_resid):
        scale = np.nanmedian(np.abs(fallback_resid - np.nanmedian(fallback_resid))) * 1.4826
        if np.isfinite(scale) and scale >= 8.0:
            return float(scale)
    return 20.0


def notch_dominant_tone(hgr: np.ndarray):
    x = np.asarray(hgr, dtype=float).copy()
    finite = np.isfinite(x)
    if int(finite.sum()) < 64:
        return x, False
    idx = np.arange(len(x), dtype=float)
    filled = x.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    centre = np.nanmedian(filled)
    spec = np.fft.rfft(filled - centre)
    if len(spec) < 5:
        return x, False
    amp = np.abs(spec)
    amp[0] = 0.0
    peak = int(np.argmax(amp))
    if peak <= 0 or not np.isfinite(amp[peak]):
        return x, False
    spec[max(1, peak - 1) : min(len(spec), peak + 2)] = 0.0
    x[finite] = (np.fft.irfft(spec, n=len(filled)) + centre)[finite]
    return x, True


def heel_calibrate(hw, tw_tvt, tw_gr, hgr, cfg: HeelConfig):
    """Map horizontal GR onto the typewell scale using the visible heel."""
    info = {
        "heel_calibrated": False,
        "heel_rows": 0,
        "heel_alpha": np.nan,
        "heel_beta": np.nan,
        "heel_rmse_raw": np.nan,
        "heel_rmse_calibrated": np.nan,
        "heel_denoised": False,
    }
    prepared = np.asarray(hgr, dtype=float).copy()
    if cfg.fft_denoise:
        prepared, info["heel_denoised"] = notch_dominant_tone(prepared)
    if not cfg.enabled or hw is None or "TVT_input" not in hw.columns:
        return prepared, info

    n = min(len(hw), len(prepared))
    tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(dtype=float)[:n]
    obs_all = prepared[:n]
    mask = np.isfinite(tvt_input) & np.isfinite(obs_all)
    if int(mask.sum()) < cfg.min_rows:
        info["heel_rows"] = int(mask.sum())
        return prepared, info
    ref = np.interp(tvt_input[mask], tw_tvt, tw_gr)
    ok = np.isfinite(ref) & np.isfinite(obs_all[mask])
    if int(ok.sum()) < cfg.min_rows:
        info["heel_rows"] = int(ok.sum())
        return prepared, info
    ref, obs = ref[ok], obs_all[mask][ok]
    try:
        alpha, beta = np.linalg.lstsq(np.column_stack([ref, np.ones_like(ref)]), obs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return prepared, info
    alpha, beta = float(alpha), float(beta)
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        return prepared, info
    info["heel_rows"] = int(ok.sum())
    if not (cfg.alpha_min <= alpha <= cfg.alpha_max) or abs(beta) > cfg.beta_abs_max:
        info.update(heel_alpha=alpha, heel_beta=beta)
        return prepared, info
    raw_resid = obs - ref
    cal_resid = (obs - beta) / max(alpha, 1e-12) - ref
    info.update(
        heel_calibrated=True,
        heel_alpha=alpha,
        heel_beta=beta,
        heel_rmse_raw=float(np.sqrt(np.nanmean(raw_resid * raw_resid))),
        heel_rmse_calibrated=float(np.sqrt(np.nanmean(cal_resid * cal_resid))),
    )
    return (prepared - beta) / max(alpha, 1e-12), info
