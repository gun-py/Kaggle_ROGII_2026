from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.io import write_submission


def weighted(
    a: pd.DataFrame, b: pd.DataFrame, weight_a: float, weight_b: float | None = None
) -> pd.DataFrame:
    """`weight_a * a + weight_b * b` (weight_b defaults to the complement), matched on id in the row order of `a`."""
    weight_b = 1.0 - weight_a if weight_b is None else weight_b
    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise RuntimeError(f"blend id mismatch: {len(a)} vs {len(b)} rows, {len(merged)} matched")
    out = merged[["id"]].copy()
    out["tvt"] = weight_a * merged["tvt_a"] + weight_b * merged["tvt_b"]
    return out


def _clean(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame[["id", "tvt"]].copy()
    frame["id"] = frame["id"].astype(str)
    frame["tvt"] = frame["tvt"].astype(float)
    if not np.isfinite(frame["tvt"].to_numpy()).all():
        raise RuntimeError(f"non-finite values in {label} tvt")
    return frame


def sp45_learned(sp45, learned, selected: float, candidates, out_dir: Path) -> pd.DataFrame:
    """Blend the projected SP45 anchor with the learned trajectory and save every candidate weight."""
    out_dir = Path(out_dir)
    learned = _clean(learned, "learned")
    sp45 = _clean(sp45, "sp45")
    write_submission(learned, out_dir / "learned_trajectory_submission.csv")

    rows, chosen = [], None
    for w in candidates:
        candidate = weighted(sp45, learned, w)
        name = f"submission_sp45_learned_w{w:.2f}.csv"
        write_submission(candidate, out_dir / name)
        diff = candidate["tvt"].to_numpy(dtype=float) - sp45["tvt"].to_numpy(dtype=float)
        rows.append({
            "file": name,
            "w_sp45": float(w),
            "w_learned": float(1.0 - w),
            "rows": int(len(candidate)),
            "mean_tvt": float(candidate["tvt"].mean()),
            "std_tvt": float(candidate["tvt"].std()),
            "rmse_vs_sp45": float(np.sqrt(np.mean(diff * diff))),
            "p95_abs_vs_sp45": float(np.quantile(np.abs(diff), 0.95)),
        })  # fmt: skip
        if abs(w - selected) < 1e-12:
            chosen = candidate
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "sp45_learned_blend_report.csv", index=False)
    print(report.to_string(index=False), flush=True)
    return chosen
