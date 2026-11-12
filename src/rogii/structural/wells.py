from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.io import list_wells


def load_well(data_dir: Path, wid: str, split: str) -> dict:
    """Array view of one well; `tvt` falls back to TVT_input when the split has no ground truth."""
    base = Path(data_dir) / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return dict(
        md=hw.MD.values, x=hw.X.values, y=hw.Y.values, z=hw.Z.values, gr=hw.GR.values,
        known=hw.TVT_input.notna().values,
        tvt=hw.TVT.values if "TVT" in hw else hw.TVT_input.values,
        tw_tvt=tw.TVT.values, tw_gr=tw.GR.values,
    )  # fmt: skip


def load_wells(data_dir: Path, split: str, ids: list[str] | None = None) -> dict[str, dict]:
    return {w: load_well(data_dir, w, split) for w in (ids or list_wells(data_dir, split))}


def anchor_index(d: dict) -> int:
    return int(np.where(d["known"])[0][-1])
