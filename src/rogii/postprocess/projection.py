from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.io import split_ids


def robust_fit(s, y, degree: int = 5):
    """Polynomial fit that down-weights outliers (four reweighting passes)."""
    if len(s) < degree + 2:
        return y.copy()
    coef = np.polyfit(s, y, degree)
    for _ in range(4):
        resid = y - np.polyval(coef, s)
        scale = np.median(np.abs(resid)) * 1.4826 + 1e-6
        coef = np.polyfit(s, y, degree, w=1.0 / (1.0 + (resid / (2.0 * scale)) ** 2))
    return np.polyval(coef, s)


def project(sub: pd.DataFrame, data_dir: Path, degree: int = 3, weight: float = 0.75) -> pd.DataFrame:
    """Smooth each well's stratigraphic level U = TVT + Z against normalised MD past the last known row."""
    frame = split_ids(sub[["id", "tvt"]])
    out = dict(zip(frame["id"].values, frame["tvt"].astype(float).values))
    applied = 0
    for wid, part in frame.groupby("well"):
        try:
            hw = pd.read_csv(Path(data_dir) / "test" / f"{wid}__horizontal_well.csv")
            known = hw[hw["TVT_input"].notna()]
            if len(known) < 5:
                continue
            last = known.iloc[-1]
            anchor = float(last["TVT_input"]) + float(last["Z"])
            md_start, md_end = float(last["MD"]), float(hw["MD"].iloc[-1])

            part = part.sort_values("row_idx")
            rows = part["row_idx"].values
            z = hw["Z"].values[rows].astype(float)
            md = hw["MD"].values[rows].astype(float)
            s = (md - md_start) / max(md_end - md_start, 1e-6)
            tvt = part["tvt"].values.astype(float)

            fit = robust_fit(s, (tvt + z) - anchor, degree)
            smoothed = (1.0 - weight) * tvt + weight * ((anchor + fit) - z)
            if not np.all(np.isfinite(smoothed)):
                continue
            out.update(zip(part["id"].values, map(float, smoothed)))
            applied += 1
        except Exception as exc:
            print("projection fallback", wid, exc)
    print("projection applied to", applied, "wells")
    result = frame[["id"]].copy()
    result["tvt"] = result["id"].map(out).astype(float)
    return result
