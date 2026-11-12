from __future__ import annotations

import numpy as np
import pandas as pd

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def tvt_from_contacts(hw: pd.DataFrame, tw: pd.DataFrame, ref_col: str = "EGFDU") -> pd.Series:
    """TVT of a train well rebuilt from its formation contact depths."""
    picks = tw.dropna(subset=["Geology"])
    ref_tvt = picks.loc[picks["Geology"] == ref_col, "TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = picks["Geology"].iloc[0]
        ref_tvt = picks.loc[picks["Geology"] == ref_col, "TVT"].min()
    raw = ref_tvt - (hw["Z"] - hw[ref_col])
    return raw + (hw["TVT"] - raw).mean()


def contact_tvt(hw: pd.DataFrame, tw: pd.DataFrame, ref_col: str) -> np.ndarray | None:
    """Contact-derived TVT for one reference formation, or None when the well lacks it."""
    if ref_col not in hw.columns or not {"Geology", "TVT"} <= set(tw.columns):
        return None
    picks = tw.dropna(subset=["Geology", "TVT"])
    ref = picks.loc[picks["Geology"].astype(str) == str(ref_col), "TVT"]
    if ref.empty:
        return None
    raw = float(ref.min()) - (hw["Z"].to_numpy(dtype=float) - hw[ref_col].to_numpy(dtype=float))
    return raw + float(np.nanmean(hw["TVT"].to_numpy(dtype=float) - raw))


def md_lookup(md_src: np.ndarray, tvt_src: np.ndarray, min_rows: int):
    """Sorted (md, tvt) pairs of the finite rows, or None when too few survive."""
    ok = np.isfinite(md_src) & np.isfinite(tvt_src)
    if int(ok.sum()) < min_rows:
        return None
    order = np.argsort(md_src[ok])
    return md_src[ok][order], tvt_src[ok][order]
