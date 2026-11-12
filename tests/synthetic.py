"""Small synthetic competition folder: planar structure, shared typewell shape, GR derived from TVT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
TYPEWELL_TVT = np.arange(11000.0, 12000.0, 0.5)


def typewell_gr(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = (TYPEWELL_TVT - TYPEWELL_TVT[0]) / 50.0
    gr = 70 + 25 * np.sin(t) + 18 * np.sin(2.7 * t + 1.0) + 10 * np.sin(6.1 * t + 2.0)
    return gr + rng.normal(0, 1.0, len(t))


def make_well(
    wid: str,
    x0: float,
    y0: float,
    n_rows: int = 1400,
    known_frac: float = 0.45,
    seed: int = 0,
    drift: float = 0.0,
):
    """Return (horizontal frame with every column, typewell frame with Geology).

    TVT + Z = S(x, y) + k_well holds exactly for the plane S, so neighbour wells carry the structure.
    `drift` bends the true TVT away from the structural trend past the last known row.
    """
    rng = np.random.default_rng(seed)
    md = 10000.0 + np.arange(n_rows, dtype=float)
    x = x0 + 0.9 * (md - md[0])
    y = y0 + 0.3 * (md - md[0])
    z = -9000.0 + 0.12 * (md - md[0]) + 6.0 * np.sin((md - md[0]) / 300.0)
    k_well = rng.normal(0, 15)
    structure = 2640.0 + 0.004 * x + 0.002 * y + k_well
    tvt = structure - z
    n_known = int(n_rows * known_frac)
    tvt = tvt + drift * np.clip((np.arange(n_rows) - n_known) / max(n_rows - n_known, 1), 0, None) ** 2

    tw_gr = typewell_gr()
    gr = np.interp(tvt, TYPEWELL_TVT, tw_gr) * 1.1 + 3.0 + rng.normal(0, 1.5, n_rows)
    hw = pd.DataFrame({"MD": md, "X": x, "Y": y, "Z": z})
    for i, name in enumerate(FORMATIONS):
        hw[name] = tvt + z - (-60.0 - 45.0 * i)
    hw["TVT"] = tvt
    hw["GR"] = gr
    hw["TVT_input"] = np.where(np.arange(n_rows) < n_known, tvt, np.nan)

    geology = np.full(len(TYPEWELL_TVT), None, dtype=object)
    for i, name in enumerate(FORMATIONS):
        geology[int(np.searchsorted(TYPEWELL_TVT, 11200.0 + 120.0 * i))] = name
    tw = pd.DataFrame({"TVT": TYPEWELL_TVT, "GR": tw_gr, "Geology": geology})
    return hw, tw


def write_competition(
    root: Path, n_train: int = 12, n_test: int = 2, n_rows: int = 1400, seed: int = 0
) -> Path:
    """Write train/ test/ sample_submission.csv. Test wells are fresh wells close to the train cluster."""
    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ids = []
    for i in range(n_train):
        wid = f"tr{i:06d}"
        hw, tw = make_well(
            wid, 1000.0 + 600.0 * (i % 4), 500.0 + 700.0 * (i // 4), n_rows, 0.5, seed=seed + i
        )
        hw.to_csv(root / "train" / f"{wid}__horizontal_well.csv", index=False)
        tw.to_csv(root / "train" / f"{wid}__typewell.csv", index=False)
        ids.append(wid)

    sample_rows = []
    for j in range(n_test):
        wid = f"te{j:06d}"
        hw, tw = make_well(
            wid,
            1300.0 + 450.0 * j,
            800.0 + float(rng.integers(0, 300)),
            n_rows,
            0.45,
            seed=500 + j,
            drift=12.0,
        )
        hw[["MD", "X", "Y", "Z", "GR", "TVT_input"]].to_csv(
            root / "test" / f"{wid}__horizontal_well.csv", index=False
        )
        tw[["TVT", "GR"]].to_csv(root / "test" / f"{wid}__typewell.csv", index=False)
        hw.to_csv(root.parent / f"{root.name}_truth_{wid}.csv", index=False)
        sample_rows += [(f"{wid}_{r}", 0.0) for r in np.where(hw["TVT_input"].isna())[0]]
    pd.DataFrame(sample_rows, columns=["id", "tvt"]).to_csv(root / "sample_submission.csv", index=False)
    return root
