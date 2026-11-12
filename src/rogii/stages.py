from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.config import PipelineConfig
from rogii.geology.contacts import tvt_from_contacts
from rogii.io import list_wells, read_sample, read_well, split_ids
from rogii.tracking import selector
from rogii.tracking.beam import beam_ensemble
from rogii.tracking.particle_filter import ensemble_by_scale


def selector_stage(data_dir: Path, out_dir: Path, cfg: PipelineConfig):
    """Per-well selector prediction (PF ensemble + beam) for the test wells.

    Returns (predictions, seed-branch stats for the hedge). Wells that also exist in
    train are replaced by their contact-derived path.
    """
    sel = cfg.selector
    sample = split_ids(read_sample(data_dir))
    train_wells = set(list_wells(data_dir, "train"))
    rows, bimodal_rows, branch_stats = [], [], {}

    test_wells = list_wells(data_dir, "test")
    for i, wid in enumerate(test_wells, 1):
        print(f"\nselector {i}/{len(test_wells)}: {wid}", flush=True)
        hw, tw = read_well(data_dir, wid, "test")
        phys, tw_train = None, None
        if wid in train_wells:
            try:
                hw_train, tw_train = read_well(data_dir, wid, "train")
                hw["TVT_input"] = hw_train["TVT_input"].values
                phys = tvt_from_contacts(hw_train, tw_train)
                print("  physical model ok")
            except Exception as exc:
                print(f"  physical model failed: {exc}")
                phys = None

        code, variant, n_eval, z_span = selector.well_code(hw)
        tw_ref = tw_train if tw_train is not None else tw

        try:
            stats = {}
            pf_by_scale = ensemble_by_scale(
                hw, tw_ref, scales=sel.scales, n_particles=sel.n_particles, n_seeds=sel.n_seeds,
                return_std=sel.return_std, branch_stats=stats,
            )  # fmt: skip
            if stats:
                branch_stats[str(wid)] = stats
            tvt_pf = pf_by_scale["pf_scale_8"]
        except Exception as exc:
            print(f"  PF failed: {exc}")
            known = hw["TVT_input"].dropna()
            tvt_pf = hw["TVT_input"].fillna(float(known.iloc[-1]) if len(known) else 0.0).values.astype(float)
            pf_by_scale = {f"pf_scale_{s:g}": tvt_pf.copy() for s in sel.scales}

        try:
            beam = beam_ensemble(hw, tw_ref)
        except Exception as exc:
            print(f"  beam failed: {exc}")
            beam = tvt_pf.copy()

        known = hw["TVT_input"].dropna()
        last_known = float(known.iloc[-1]) if len(known) else float(np.nanmean(tvt_pf))
        path, info = selector.apply_variant(
            variant, pf_by_scale, beam, last_known, hw, tw_ref, sel, return_info=True
        )
        bimodal_rows.append(dict(well=wid, selector_code=int(code), selector_variant=variant, n_eval=float(n_eval),
                                 z_span=float(z_span), **info))  # fmt: skip
        print(f"  selector code={code} variant={variant} n_eval={n_eval:.0f} z_span={z_span:.3f}"
              + (f" bimodal_delta={info['delta_star']:.2f}" if info.get("bimodal_active") else ""))  # fmt: skip

        part = sample[sample["well"] == wid]
        idx = part["row_idx"].to_numpy(dtype=int)
        values = phys.to_numpy(dtype=float)[idx] if phys is not None else path[idx]
        rows.append(pd.DataFrame({"id": part["id"].values, "tvt": values}))
        print(f"  added {len(part)} rows")

    write_bimodal_reports(pd.DataFrame(bimodal_rows), out_dir)
    return pd.concat(rows, ignore_index=True), branch_stats


def write_bimodal_reports(report: pd.DataFrame, out_dir: Path) -> None:
    if report.empty:
        return
    report.to_csv(out_dir / "bimodal_selector_report.csv", index=False)
    if "bimodal_active" not in report.columns:
        return
    active = report["bimodal_active"].astype(str).str.lower().isin(["true", "1", "yes"])
    keep = [c for c in (
        "well", "selector_code", "selector_variant", "n_eval", "z_span", "delta_star", "delta_a", "delta_b",
        "p_base", "p_eff", "prefix_trust", "temperature", "rho1", "n_eff", "score_base", "score_second",
        "j_ratio", "prefix_rows", "j_prefix_base", "j_prefix_decoy", "prefix_trust_margin",
    ) if c in report.columns]  # fmt: skip
    report.loc[active, keep].to_csv(out_dir / "bimodal_active_wells.csv", index=False)


def anchor_blend(ridge: pd.DataFrame, selector_sub: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    merged = ridge.merge(selector_sub, on="id", suffixes=("_1", "_2"))
    out = merged[["id"]].copy()
    out["tvt"] = cfg.ridge_weight * merged["tvt_1"] + cfg.selector_weight * merged["tvt_2"]
    return out
