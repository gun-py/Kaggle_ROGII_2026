from __future__ import annotations

import numpy as np
import pandas as pd

from .kernels import beam_kernel

# (beam size, move cost, GR error scale, smoothing radius)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]

# (beam size, move cost, GR error scale, smoothing radius, tag) used for the tabular features
FEATURE_BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]


def _nearest(arr, value) -> int:
    i = int(np.searchsorted(arr, value, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - value) <= abs(arr[i] - value):
        return i - 1
    return i


def _smooth_gr(values, fallback: float, radius: int) -> np.ndarray:
    s = pd.Series(values, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    out = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean() if radius > 0 else s
    return out.to_numpy(np.float32)


def jit_beam_search(gr_h, tw_tvt, tw_gr, start_tvt, beam_size, move_cost, gr_scale, radius):
    """Beam search over typewell depth; returns the typewell TVT along the best path (float32)."""
    start = _nearest(tw_tvt, start_tvt)
    smooth = _smooth_gr(gr_h, float(np.nanmean(tw_gr)), radius).astype(np.float64)
    path = beam_kernel(smooth, tw_gr.astype(np.float64), start, beam_size, float(move_cost), float(gr_scale))
    return tw_tvt[path].astype(np.float32)


def beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    known = hw[hw["TVT_input"].notna()]
    unknown = hw[hw["TVT_input"].isna()]
    if len(unknown) == 0:
        return hw["TVT_input"].to_numpy(dtype=float).copy()

    last_tvt = float(known.iloc[-1]["TVT_input"])
    tw = tw.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(dtype=float)
    gr = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(dtype=float)
    hgr = gr[unknown.index]

    paths = [jit_beam_search(hgr, tw_tvt, tw_gr, last_tvt, *cfg) for cfg in BEAM_CONFIGS]
    out = hw["TVT_input"].to_numpy(dtype=float).copy()
    out[list(unknown.index)] = np.stack(paths, 0).mean(0)
    return out
