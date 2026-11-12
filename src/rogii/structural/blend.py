from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.io import list_wells
from rogii.metrics import rmsd

from .field import Field
from .variants import Variant
from .wells import load_wells


def gated_weight(rule, nearest: float) -> float:
    """Struct weight for a well from its neighbour distance (`schedule` is a list of (upper bound, weight))."""
    if not np.isfinite(nearest):
        return rule.no_neighbours
    for upper, weight in rule.schedule:
        if nearest < upper:
            return weight
    return rule.beyond


def struct_predictions(variant: Variant, data_dir: Path, index: pd.Index):
    """Structural TVT for every unknown test row, aligned to `index`, and the per-row gated weight."""
    train = load_wells(data_dir, "train")
    fld = Field(train)
    parts = []
    for wid in list_wells(data_dir, "test"):
        d = load_wells(data_dir, "test", [wid])[wid]
        unknown = np.where(~d["known"])[0]
        path, nearest = variant.predict(fld, d, drop={wid})
        weight = gated_weight(variant.blend, nearest) if variant.blend.mode == "gated" else np.nan
        parts.append(pd.DataFrame({"id": [f"{wid}_{i}" for i in unknown], "tvt": path[unknown], "w": weight}))
    table = pd.concat(parts, ignore_index=True).set_index("id")
    return table["tvt"].reindex(index), table["w"].reindex(index)


def blend(existing: pd.Series, struct: pd.Series, weights: pd.Series, rule, package: pd.Series | None = None):
    """Mix the structural prediction (and optionally the package) into `existing`; gaps fall back to `existing`."""
    if rule.mode == "gated":
        out = (1 - weights) * existing + weights * struct
        return out.where(struct.notna() & weights.notna(), existing)

    package_w = rule.package if package is not None else 0.0
    total = rule.stack + rule.struct + package_w
    out = (rule.stack / total) * existing + (rule.struct / total) * struct
    if package_w > 0:
        out = out + (package_w / total) * package
    return out.fillna(existing)


def report(existing: pd.Series, struct: pd.Series, package: pd.Series | None, rule) -> None:
    print(f"struct NaN before fill: {int(struct.isna().sum())} of {len(struct)}")
    print(f"RMSD(stack, struct): {rmsd(existing, struct):.3f}")
    if package is not None:
        print(
            f"package coverage: {float(package.notna().mean()):.4f}  RMSD(stack, package): {rmsd(existing, package):.3f}"
        )
    total = rule.stack + rule.struct + (rule.package if package is not None else 0.0)
    print(f"weights stack/struct/package: {rule.stack / total:.2f} {rule.struct / total:.2f} {(rule.package if package is not None else 0.0) / total:.2f}")  # fmt: skip
