from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def fade_in(md_since, tau: float):
    """Ramp 0 -> 1 over the first ~tau feet past the last known point."""
    if tau <= 1e-6:
        return 1.0
    return 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau)


def savgol_by_well(df: pd.DataFrame, col: str, window: int = 17, poly: int = 3) -> pd.DataFrame:
    df = df.copy()
    for _, part in df.groupby("well", sort=False):
        values = part[col].values
        wl = min(window, len(values))
        if wl % 2 == 0:
            wl -= 1
        if wl >= poly + 2:
            values = savgol_filter(values, wl, poly)
        df.loc[part.index, col] = values
    return df


def smooth_groups(pred: np.ndarray, wells: pd.Series, window: int, poly: int) -> np.ndarray:
    out = pred.copy()
    wells = wells.reset_index(drop=True)
    for _, idx in wells.groupby(wells, sort=False).groups.items():
        pos = wells.index.get_indexer(idx)
        values = pred[pos]
        wl = min(window, len(values))
        if wl % 2 == 0:
            wl -= 1
        if wl >= poly + 2:
            out[pos] = savgol_filter(values, wl, poly)
    return out
