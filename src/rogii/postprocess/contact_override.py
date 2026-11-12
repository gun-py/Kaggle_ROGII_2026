from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.config import OverrideConfig
from rogii.geology.contacts import contact_tvt, md_lookup
from rogii.io import list_wells, split_ids
from rogii.metrics import masked_rmse


def reference_priority(cfg: OverrideConfig) -> tuple[str, ...]:
    refs = []
    for ref in cfg.ref_cols:
        ref = str(ref).strip()
        if ref and ref not in refs:
            refs.append(ref)
    primary = cfg.primary_ref.strip()
    if primary and primary not in refs:
        refs.insert(0, primary)
    return tuple(refs or ("EGFDU",))


def best_reconstruction(hw_test, hw_train, tw_train, cfg: OverrideConfig):
    """Pick the contact reference whose train-copy path reproduces the visible prefix.

    Returns (candidate, "") or (None, reason).
    """
    known = hw_test[hw_test["TVT_input"].notna()]
    best, reasons = None, []
    for ref in reference_priority(cfg):
        phys = contact_tvt(hw_train, tw_train, ref)
        if phys is None:
            reasons.append(f"{ref}:no_ref")
            continue
        md_raw = hw_train["MD"].to_numpy(dtype=float)
        valid_rows = int((np.isfinite(phys) & np.isfinite(md_raw)).sum())
        lookup = md_lookup(md_raw, phys, cfg.min_valid_rows)
        if lookup is None:
            reasons.append(f"{ref}:valid_rows={valid_rows}")
            continue
        md_tr, ph_tr = lookup
        comparable = known[(known["MD"] >= md_tr[0]) & (known["MD"] <= md_tr[-1])]
        if len(comparable) < cfg.min_known_rows:
            reasons.append(f"{ref}:known_rows={len(comparable)}")
            continue
        err = masked_rmse(
            np.interp(comparable["MD"].to_numpy(dtype=float), md_tr, ph_tr),
            comparable["TVT_input"].to_numpy(dtype=float),
        )
        if not np.isfinite(err):
            reasons.append(f"{ref}:rmse=nan")
            continue
        candidate = dict(ref_col=str(ref), rmse=float(err), md_tr=md_tr, ph_tr=ph_tr,
                         valid_phys_rows=valid_rows, comparable_known_rows=int(len(comparable)))  # fmt: skip
        if best is None or err < best["rmse"]:
            best = candidate
        if str(ref) == cfg.primary_ref.strip() and err <= cfg.prefix_rmse_limit:
            break
    if best is None:
        return None, ";".join(reasons) or "no_candidate"
    if best["rmse"] > cfg.prefix_rmse_limit:
        return None, f"best_ref={best['ref_col']} rmse={best['rmse']:.3f}>{cfg.prefix_rmse_limit:.3f}"
    return best, ""


def apply_override(
    sub: pd.DataFrame, data_dir: Path, cfg: OverrideConfig, report_path: Path | None = None
) -> pd.DataFrame:
    """Overwrite wells that also exist in train with their contact-derived TVT, when the prefix agrees."""
    data_dir = Path(data_dir)
    frame = split_ids(sub[["id", "tvt"]])
    pred = dict(zip(frame["id"].astype(str), frame["tvt"].astype(float)))
    train_wells = set(list_wells(data_dir, "train"))
    report, n_ok, n_skip = [], 0, 0

    for wid, part in frame.groupby("well"):
        if wid not in train_wells:
            continue
        try:
            hw_te = pd.read_csv(data_dir / "test" / f"{wid}__horizontal_well.csv")
            hw_tr = pd.read_csv(data_dir / "train" / f"{wid}__horizontal_well.csv")
            tw_tr = pd.read_csv(data_dir / "train" / f"{wid}__typewell.csv")
            best, reason = best_reconstruction(hw_te, hw_tr, tw_tr, cfg)
            if best is None:
                print(f"override skip {wid} {reason}")
                report.append({"well": wid, "status": "skip", "reason": reason, "rows_total": int(len(part))})
                n_skip += 1
                continue
            md_te = hw_te["MD"].to_numpy(dtype=float)
            rows = part["row_idx"].to_numpy(dtype=int)
            ids = part["id"].astype(str).to_numpy()
            in_table = (rows >= 0) & (rows < len(md_te))
            md = md_te[rows[in_table]]
            inside = (md >= best["md_tr"][0]) & (md <= best["md_tr"][-1])
            new_values = np.interp(md[inside], best["md_tr"], best["ph_tr"])
            pred.update(zip(ids[in_table][inside], map(float, new_values)))
            n_row = int(inside.sum())
            print(
                f"override ok   {wid} ref={best['ref_col']} prefix rmse={best['rmse']:.4f} rows={n_row}/{len(part)}"
            )
            report.append(dict(well=wid, status="override", ref_col=best["ref_col"],
                               known_prefix_rmse=float(best["rmse"]), valid_phys_rows=int(best["valid_phys_rows"]),
                               comparable_known_rows=int(best["comparable_known_rows"]), rows_overridden=n_row,
                               rows_total=int(len(part)), reason=""))  # fmt: skip
            n_ok += 1
        except Exception as exc:
            print(f"override fallback {wid}: {exc}")
            report.append({"well": wid, "status": "error", "reason": str(exc), "rows_total": int(len(part))})
            n_skip += 1

    out = frame[["id"]].copy()
    out["tvt"] = out["id"].astype(str).map(pred).astype(float)
    if not out["tvt"].notna().all():
        raise RuntimeError("contact override produced NaN")
    if report_path is not None:
        pd.DataFrame(report).to_csv(report_path, index=False)
    print(f"contact override: overridden={n_ok} skipped={n_skip}")
    return out


def probe_overlap(sub: pd.DataFrame, data_dir: Path, path: Path) -> pd.DataFrame:
    """Read-only report of which submission wells also appear in train and how long their prefix is."""
    data_dir = Path(data_dir)
    frame = split_ids(sub[["id", "tvt"]])
    train, test = set(list_wells(data_dir, "train")), set(list_wells(data_dir, "test"))
    rows = []
    for wid in sorted(frame["well"].unique()):
        known = 0
        if wid in test:
            hw = pd.read_csv(
                data_dir / "test" / f"{wid}__horizontal_well.csv", usecols=lambda c: c == "TVT_input"
            )
            known = int(hw["TVT_input"].notna().sum()) if "TVT_input" in hw.columns else 0
        rows.append(dict(well=wid, has_train_copy=wid in train, has_test_file=wid in test,
                         known_prefix_rows=known, submission_rows=int((frame["well"] == wid).sum())))  # fmt: skip
    probe = pd.DataFrame(rows)
    probe.to_csv(path, index=False)
    fire = int((probe["has_train_copy"] & (probe["known_prefix_rows"] > 0)).sum()) if not probe.empty else 0
    print(f"overlap probe: candidate same-well copies {fire}/{len(probe)}")
    return probe
