from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.config import HedgeConfig


def _wells_to_skip(out_dir: Path) -> set[str]:
    skip: set[str] = set()
    for name in ("gold_contact_override_report.csv", "guarded_overlap_override_report.csv"):
        path = out_dir / name
        if not path.exists():
            continue
        try:
            report = pd.read_csv(path)
            if {"well", "status"} <= set(report.columns):
                active = report["status"].astype(str).str.lower().isin(["override", "applied"])
                skip.update(report.loc[active, "well"].astype(str))
        except Exception:
            pass
    moves = out_dir / "gold_prefix_moves_balanced.csv"
    if moves.exists():
        try:
            report = pd.read_csv(moves)
            alpha = pd.to_numeric(report.get("alpha", 0.0), errors="coerce").fillna(0.0)
            if "well" in report.columns:
                skip.update(report.loc[alpha > 0.0, "well"].astype(str))
        except Exception:
            pass
    return skip


def apply_hedge(sub: pd.DataFrame, branch_stats: dict, cfg: HedgeConfig, out_dir: Path) -> pd.DataFrame:
    """Nudge wells whose PF seeds split into two branches towards the branch midpoint (capped)."""
    out_dir = Path(out_dir)
    if list(sub.columns) != ["id", "tvt"]:
        raise RuntimeError("branch hedge expects id,tvt columns")
    sub = sub.copy()
    sub.to_csv(out_dir / "submission_before_branch_hedge.csv", index=False)
    skip = _wells_to_skip(out_dir) if cfg.skip_existing else set()

    ids = sub["id"].astype(str)
    well = ids.str.split("_", n=1).str[0]
    row = pd.to_numeric(ids.str.rsplit("_", n=1).str[-1], errors="coerce")
    tvt = pd.to_numeric(sub["tvt"], errors="coerce").to_numpy(dtype=float).copy()

    rows = []
    for wid, stats in sorted((branch_stats or {}).items()):
        reason, shift, moved = "not_qualified", 0.0, 0
        try:
            low, high = float(stats["center_low"]), float(stats["center_high"])
            mass_low, mass_high = float(stats["mass_low"]), float(stats["mass_high"])
            weighted = float(stats["weighted_center"])
            separation = abs(high - low)
            if str(wid) in skip:
                reason = "skip_existing_route"
            elif min(mass_low, mass_high) < cfg.min_mass:
                reason = "skip_minor_mass"
            elif not (cfg.sep_low <= separation <= cfg.sep_high):
                reason = "skip_separation"
            else:
                target = 0.5 * (low + high)
                shift = float(np.clip(cfg.strength * (target - weighted), -cfg.cap, cfg.cap))
                eval_rows = {int(x) for x in stats.get("eval_rows", [])}
                mask = (well == str(wid)).to_numpy().copy()
                if eval_rows:
                    mask &= row.isin(eval_rows).to_numpy()
                if abs(shift) >= 0.01 and bool(mask.any()):
                    tvt[mask] += shift
                    moved = int(mask.sum())
                    reason = "applied"
                else:
                    reason = "skip_zero_or_missing_rows"
            rows.append(dict(well=str(wid), reason=reason, center_low=low, center_high=high,
                             mass_low=mass_low, mass_high=mass_high, separation=separation,
                             weighted_center=weighted, shift=shift, moved_rows=moved))  # fmt: skip
        except Exception as exc:
            rows.append(dict(well=str(wid), reason="error", error=repr(exc)))
    if not np.isfinite(tvt).all():
        raise RuntimeError("branch hedge produced non-finite predictions")
    sub["tvt"] = tvt
    pd.DataFrame(rows).to_csv(out_dir / "pf_seed_branch_hedge_report.csv", index=False)
    counts = pd.Series([r["reason"] for r in rows], dtype=object).value_counts().to_dict()
    print("PF seed-branch hedge:", counts)
    return sub
