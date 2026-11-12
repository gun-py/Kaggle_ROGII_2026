from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.config import SelectorConfig
from rogii.io import list_wells, read_well
from rogii.structural.field import Field
from rogii.structural.variants import load_variant
from rogii.structural.wells import load_wells
from rogii.tracking import selector
from rogii.tracking.beam import beam_ensemble
from rogii.tracking.bimodal import bimodal_scan
from rogii.tracking.particle_filter import ensemble_by_scale

CV_PF_SEEDS = 24


def _sample_wells(data_dir: Path, n_wells: int, seed: int) -> list[str]:
    wells = list_wells(data_dir, "train")
    if 0 < n_wells < len(wells):
        rng = np.random.default_rng(seed)
        wells = sorted(rng.choice(wells, size=n_wells, replace=False).tolist())
    return wells


def _sse(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.nan, 0, 0.0
    err = y[ok] - p[ok]
    sse = float(np.sum(err * err))
    return float(np.sqrt(sse / ok.sum())), int(ok.sum()), sse


def _robust_polyfit(x, y, degree=6):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < max(4, degree + 2):
        return np.full_like(y, float(np.nanmean(y)), dtype=float)
    xx, yy = x[ok], y[ok]
    deg = min(int(degree), max(1, int(ok.sum()) - 2))
    coef = np.polyfit(xx, yy, deg)
    for _ in range(4):
        resid = yy - np.polyval(coef, xx)
        scale = np.nanmedian(np.abs(resid)) * 1.4826 + 1e-6
        coef = np.polyfit(xx, yy, deg, w=1.0 / (1.0 + (resid / (2.0 * scale)) ** 2))
    out = np.full_like(y, np.nan, dtype=float)
    out[ok] = np.polyval(coef, xx)
    return out


def oracle_ladder(data_dir: Path, out_dir: Path, n_wells: int, seed: int):
    """How well would oracle shapes (flat, constant, line, smooth fit of the truth) score? A floor for any model."""
    pooled = {k: {"sse": 0.0, "n": 0} for k in ("flat", "constant", "line", "smooth")}
    smooth_sse, rows = [], []
    for wid in _sample_wells(data_dir, n_wells, seed):
        hw, _ = read_well(data_dir, wid, "train")
        if "TVT" not in hw.columns or "TVT_input" not in hw.columns:
            continue
        unknown = hw["TVT_input"].isna().to_numpy()
        if int(unknown.sum()) < 10:
            continue
        y = pd.to_numeric(hw.loc[unknown, "TVT"], errors="coerce").to_numpy(dtype=float)
        md = pd.to_numeric(hw.loc[unknown, "MD"], errors="coerce").to_numpy(dtype=float)
        known = pd.to_numeric(hw.loc[hw["TVT_input"].notna(), "TVT_input"], errors="coerce").dropna()
        constant = np.full(len(y), float(np.nanmean(y)))
        flat = np.full(len(y), float(known.iloc[-1])) if len(known) else constant
        s = md - np.nanmin(md)
        span = np.nanmax(s)
        s = s / span if np.isfinite(span) and span > 0 else np.linspace(0.0, 1.0, len(y))
        finite = np.isfinite(y)
        line = np.polyval(np.polyfit(s[finite], y[finite], 1), s) if finite.sum() >= 3 else constant
        preds = {"flat": flat, "constant": constant, "line": line, "smooth": _robust_polyfit(s, y, 6)}
        row = {"well": wid, "eval_rows": int(len(y))}
        for name, pred in preds.items():
            rmse, n, sse = _sse(y, pred)
            pooled[name]["sse"] += sse
            pooled[name]["n"] += n
            row[f"{name}_rmse"] = rmse
            if name == "smooth":
                smooth_sse.append(sse)
        rows.append(row)

    summary = pd.DataFrame(
        [
            {"oracle": k, "pooled_rmse": float(np.sqrt(v["sse"] / max(v["n"], 1))), "rows": int(v["n"])}
            for k, v in pooled.items()
        ]
    )
    smooth_sse.sort(reverse=True)
    k = max(1, int(np.ceil(0.10 * len(smooth_sse)))) if smooth_sse else 0
    tail = float(sum(smooth_sse[:k]) / max(sum(smooth_sse), 1e-12)) if k else np.nan
    summary.to_csv(out_dir / "oracle_ladder_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "oracle_ladder_by_well.csv", index=False)
    pd.Series({"wells": len(rows), "worst_decile_smooth_sse_share": tail}).to_csv(
        out_dir / "oracle_tail_concentration.csv"
    )
    print(summary.to_string(index=False))
    print(f"worst_decile_smooth_sse_share={tail:.4f}")
    return summary


def selector_cv(data_dir: Path, out_dir: Path, n_wells: int, seed: int, cfg: SelectorConfig | None = None):
    """Score the selector (PF ensemble + beam + per-well variant) on train wells it never saw."""
    cfg = cfg or SelectorConfig()
    rows, total_sse, total_n = [], 0.0, 0
    for wid in _sample_wells(data_dir, n_wells, seed):
        try:
            hw, tw = read_well(data_dir, wid, "train")
            unknown = hw["TVT_input"].isna().to_numpy()
            if "TVT" not in hw.columns or int(unknown.sum()) < 10:
                continue
            code, variant, _, z_span = selector.well_code(hw)
            pf = ensemble_by_scale(
                hw, tw, scales=cfg.scales, n_particles=cfg.n_particles, n_seeds=CV_PF_SEEDS
            )
            path = pf.get("pf_scale_8", pf.get("pf_mean"))
            try:
                beam = beam_ensemble(hw, tw)
            except Exception:
                beam = path.copy()
            known = hw["TVT_input"].dropna()
            last = float(known.iloc[-1]) if len(known) else float(np.nanmean(path))
            pred, info = selector.apply_variant(variant, pf, beam, last, hw, tw, cfg, return_info=True)
            y = pd.to_numeric(hw.loc[unknown, "TVT"], errors="coerce").to_numpy(dtype=float)
            rmse, n, sse = _sse(y, np.asarray(pred, dtype=float)[unknown])
            if n <= 0 or not np.isfinite(rmse):
                continue
            total_sse += sse
            total_n += n
            rows.append({"well": wid, "eval_rows": n, "selector_variant": variant, "selector_code": int(code),
                         "z_span": float(z_span), "rmse": rmse, "sse": sse,
                         **{k: info.get(k) for k in ("bimodal_active", "delta_star", "p_base", "heel_calibrated", "heel_rows",
                                                     "heel_alpha", "heel_beta", "heel_rmse_raw", "heel_rmse_calibrated")}})  # fmt: skip
        except Exception as exc:
            rows.append({"well": wid, "error": repr(exc)})

    report = pd.DataFrame(rows)
    ok = (
        report[pd.to_numeric(report.get("eval_rows", 0), errors="coerce").fillna(0) > 0]
        if not report.empty
        else report
    )
    if not report.empty:
        report.to_csv(out_dir / "selector_cv_by_well.csv", index=False)
    rmse_values = (
        pd.to_numeric(ok["rmse"], errors="coerce").dropna().to_numpy() if not ok.empty else np.array([])
    )
    tail = np.nan
    if not ok.empty:
        worst = ok.sort_values("sse", ascending=False).head(max(1, int(np.ceil(0.10 * len(ok)))))
        tail = float(worst["sse"].sum() / max(ok["sse"].sum(), 1e-12))
    quant = lambda q: float(np.quantile(rmse_values, q)) if len(rmse_values) else np.nan  # noqa: E731
    summary = pd.Series({"wells": len(ok), "rows": total_n, "pf_seeds": CV_PF_SEEDS,
                         "pooled_rmse": float(np.sqrt(total_sse / max(total_n, 1))), "per_well_rmse_p50": quant(0.5),
                         "per_well_rmse_p90": quant(0.9), "per_well_rmse_p99": quant(0.99), "worst_decile_sse_share": tail,
                         "bimodal_active_wells": int(ok["bimodal_active"].sum()) if "bimodal_active" in ok else 0})  # fmt: skip
    summary.to_csv(out_dir / "selector_cv_summary.csv")
    print(summary.to_string())
    return summary


def heel_localization(data_dir: Path, out_dir: Path, n_wells: int, seed: int, tolerance: float = 2.0):
    """Does heel calibration move the GR datum scan closer to the true datum (shift 0) on train wells?"""
    rows = []
    for wid in _sample_wells(data_dir, n_wells, seed):
        hw, tw = read_well(data_dir, wid, "train")
        if not {"TVT", "TVT_input", "GR"} <= set(hw.columns):
            continue
        unknown = hw["TVT_input"].isna().to_numpy()
        if int(unknown.sum()) < 10:
            continue
        truth = pd.to_numeric(hw["TVT"], errors="coerce").to_numpy(dtype=float)
        for heel_on in (False, True):
            cfg = SelectorConfig()
            cfg = replace(cfg, heel=replace(cfg.heel, enabled=heel_on))
            scan = bimodal_scan(hw, tw, truth, cfg, mask=unknown)
            if not scan:
                rows.append({"well": wid, "heel_on": heel_on, "status": "scan_skip"})
                continue
            delta = float(scan.get("delta_a", np.nan))
            rows.append({"well": wid, "heel_on": heel_on, "status": "ok", "eval_rows": int(unknown.sum()), "delta_a": delta,
                         "abs_delta_a": abs(delta), "localized_within_tol": bool(abs(delta) <= tolerance),
                         **{k: scan.get(k) for k in ("score_a", "score_b", "j_ratio", "valid_gr_rows", "heel_calibrated",
                                                     "heel_rows", "heel_alpha", "heel_beta")}})  # fmt: skip
    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame()
    report.to_csv(out_dir / "heel_calibration_localization_report.csv", index=False)
    ok = report[report["status"].eq("ok")]
    summary = pd.DataFrame([
        {"heel_on": bool(on), "wells": len(g), "localized_tolerance_ft": tolerance,
         "localized_rate": float(g["localized_within_tol"].mean()),
         "median_abs_delta_a": float(g["abs_delta_a"].median()), "p90_abs_delta_a": float(g["abs_delta_a"].quantile(0.9)),
         "calibrated_rate": float(g["heel_calibrated"].astype(float).mean())}
        for on, g in ok.groupby("heel_on")
    ])  # fmt: skip
    summary.to_csv(out_dir / "heel_calibration_localization_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def struct_cv(data_dir: Path, out_dir: Path, n_wells: int, seed: int, variant_name: str = "v5"):
    """Leave-one-out score of a structural variant against its carry-forward and neighbour components."""
    variant = load_variant(variant_name)
    wells = load_wells(data_dir, "train")
    fld = Field(wells)
    sse = {"carry": 0.0, "neighbour": 0.0, variant_name: 0.0}
    n = 0
    for wid in _sample_wells(data_dir, n_wells, seed):
        d = wells[wid]
        unknown = ~d["known"]
        if not unknown.any():
            continue
        y = d["tvt"][unknown]
        from rogii.structural.components import components

        P, _ = components(fld, d, drop={wid}, sanitize=variant.sanitize)
        path, _ = variant.predict(fld, d, drop={wid})
        sse["carry"] += float(((P[unknown, -1] - y) ** 2).sum())
        sse["neighbour"] += float(((P[unknown, -3] - y) ** 2).sum())
        sse[variant_name] += float(((path[unknown] - y) ** 2).sum())
        n += int(unknown.sum())
    summary = pd.Series({k: round(float(np.sqrt(v / n)), 3) for k, v in sse.items()})
    summary.to_csv(out_dir / "struct_cv_summary.csv")
    print(summary.to_string())
    return summary


def run(kind: str, data_dir: Path, out_dir: Path, n_wells: int = 250, seed: int = 0, **kwargs):
    jobs = {"oracle": oracle_ladder, "selector": selector_cv, "heel": heel_localization, "struct": struct_cv}
    return jobs[kind](data_dir, out_dir, n_wells, seed, **kwargs)
