from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.audit import sha256_file
from rogii.config import OverrideConfig, SelectorConfig, VisiblePrefixConfig
from rogii.features.imputers import Imputers
from rogii.features.surfaces import segment_offsets
from rogii.geology.contacts import FORMATIONS, contact_tvt, md_lookup
from rogii.io import read_sample, split_ids, validate_submission, write_submission
from rogii.metrics import masked_rmse
from rogii.tracking import selector
from rogii.tracking.beam import beam_ensemble
from rogii.tracking.particle_filter import ensemble_by_scale

from .contact_override import apply_override, reference_priority

PROFILES = {
    "conservative": dict(min_gain=1.00, max_best=12.0, min_consistency=0.67, base=0.06, gain_scale=0.12, margin_scale=0.04, quality_bonus=0.02, cap=0.22, clip_base=8.0, clip_gain=3.0, clip_max=18.0, delta_soft=22.0, p95_hard=55.0),
    "balanced": dict(min_gain=1.00, max_best=12.0, min_consistency=0.80, min_margin=0.10, base=0.08, gain_scale=0.20, margin_scale=0.06, quality_bonus=0.04, cap=0.40, clip_base=10.0, clip_gain=4.5, clip_max=30.0, delta_soft=30.0, p95_hard=75.0),
    "aggressive": dict(min_gain=0.25, max_best=15.0, min_consistency=0.34, base=0.12, gain_scale=0.32, margin_scale=0.10, quality_bonus=0.06, cap=0.56, clip_base=14.0, clip_gain=7.0, clip_max=45.0, delta_soft=42.0, p95_hard=110.0),
}  # fmt: skip
STRUCTURAL_PREFIXES = ("surface_", "dense_", "poly_", "contact_")
SELECTOR_ALIASES = {
    "anchor": {"self_verified_anchor", "anchor", "self_verified"},
    "profile": {"profile", "gold_profile", "visible_prefix"},
}


@dataclass
class Context:
    """Everything a calibration run needs besides the per-well frames."""

    data_dir: Path
    out_dir: Path
    imputers: Imputers
    selector_cfg: SelectorConfig
    override_cfg: OverrideConfig


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def bimodal_guard_wells(out_dir: Path, requires_low_trust: bool, threshold: float):
    frames = []
    for name in ("bimodal_active_wells.csv", "bimodal_selector_report.csv"):
        path = out_dir / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print("could not read bimodal report:", path, exc)
            continue
        if "well" not in df.columns:
            continue
        active = (
            _truthy(df["bimodal_active"])
            if "bimodal_active" in df.columns
            else pd.Series(True, index=df.index)
        )
        if active.any():
            frames.append(df.loc[active].copy())
    if not frames:
        return set(), pd.DataFrame()
    report = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["well"], keep="first")
    if requires_low_trust:
        if "prefix_trust" in report.columns:
            trust = pd.to_numeric(report["prefix_trust"], errors="coerce")
            report = report[trust.fillna(0.0) <= float(threshold)].copy()
        else:
            report = report.iloc[0:0].copy()
    report["visible_prefix_action"] = "keep_bimodal_hedge"
    report["vp_requires_low_trust"] = bool(requires_low_trust)
    report["vp_low_trust_threshold"] = float(threshold)
    return set(report["well"].astype(str)), report


def robust_poly_predict(x_known, y_known, x_all, degree: int):
    x_known, y_known, x_all = (np.asarray(a, dtype=float) for a in (x_known, y_known, x_all))
    ok = np.isfinite(x_known) & np.isfinite(y_known)
    x_known, y_known = x_known[ok], y_known[ok]
    if len(x_known) < 3:
        fill = float(np.nanmedian(y_known)) if len(y_known) else 0.0
        return np.full_like(x_all, fill, dtype=float)
    degree = int(min(max(1, degree), len(x_known) - 1))
    x0 = float(x_known[0])
    span = float(np.nanmax(x_known) - np.nanmin(x_known))
    if not np.isfinite(span) or span < 1e-6:
        span = 1.0
    xk, xa = (x_known - x0) / span, (x_all - x0) / span
    try:
        coef = np.polyfit(xk, y_known, degree)
        for _ in range(5):
            resid = y_known - np.polyval(coef, xk)
            scale = 1.4826 * float(np.nanmedian(np.abs(resid - np.nanmedian(resid)))) + 1e-6
            coef = np.polyfit(xk, y_known, degree, w=1.0 / (1.0 + (resid / (2.5 * scale)) ** 2))
        return np.polyval(coef, xa).astype(float)
    except Exception:
        return np.full_like(x_all, float(np.nanmedian(y_known)), dtype=float)


def contact_candidates(wid: str, hw, ctx: Context) -> dict[str, np.ndarray]:
    out = {}
    base = ctx.data_dir / "train"
    try:
        if (
            not (base / f"{wid}__horizontal_well.csv").exists()
            or not (base / f"{wid}__typewell.csv").exists()
        ):
            return out
        hw_tr = pd.read_csv(base / f"{wid}__horizontal_well.csv")
        tw_tr = pd.read_csv(base / f"{wid}__typewell.csv")
        refs = reference_priority(ctx.override_cfg)
        for ref in refs:
            phys = contact_tvt(hw_tr, tw_tr, ref)
            lookup = md_lookup(hw_tr["MD"].to_numpy(dtype=float), phys, 100) if phys is not None else None
            if lookup is None:
                continue
            pred = np.interp(hw["MD"].to_numpy(dtype=float), *lookup, left=np.nan, right=np.nan)
            name = "contact_md_lookup" if ref == refs[0] else f"contact_md_lookup_{ref.lower()}"
            out[name] = pred.astype(float)
    except Exception:
        return out
    return out


def poly_candidates(hw_masked) -> dict[str, np.ndarray]:
    out = {}
    tvt, md, z = (hw_masked[c].to_numpy(dtype=float) for c in ("TVT_input", "MD", "Z"))
    known = np.flatnonzero(np.isfinite(tvt) & np.isfinite(md) & np.isfinite(z))
    if len(known) < 30:
        return out
    level = tvt + z
    for tail in (80, 160, 320, 640, 1000000):
        sel = known[-min(int(tail), len(known)) :]
        if len(sel) < 30:
            continue
        tag = "all" if tail >= 1000000 else f"tail{tail}"
        for deg in (1, 2, 3):
            if len(sel) < deg + 12:
                continue
            out[f"poly_u_deg{deg}_{tag}"] = (robust_poly_predict(md[sel], level[sel], md, deg) - z).astype(
                float
            )
    return out


def _surface_family(prefix: str, surface, tvt, z, good) -> dict[str, np.ndarray]:
    """Datum-shifted copies of one imputed surface (median, full, late, tail-weighted offsets)."""
    out = {}
    b_med = float(np.nanmedian(tvt[good] + z[good] - surface[good]))
    out[f"{prefix}_median"] = (-z + surface + b_med).astype(float)
    try:
        b_full, _, _, b_late, b_wls = segment_offsets(
            tvt[good].astype(np.float32), z[good].astype(np.float32), surface[good].astype(np.float32)
        )
        out[f"{prefix}_full"] = (-z + surface + float(b_full)).astype(float)
        out[f"{prefix}_late"] = (-z + surface + float(b_late)).astype(float)
        out[f"{prefix}_wls"] = (-z + surface + float(b_wls)).astype(float)
    except Exception:
        pass
    return out


def surface_candidates(hw_masked, wid: str, ctx: Context) -> dict[str, np.ndarray]:
    out = {}
    tvt, z = hw_masked["TVT_input"].to_numpy(dtype=float), hw_masked["Z"].to_numpy(dtype=float)
    xy = hw_masked[["X", "Y"]].to_numpy(dtype=float)
    known = np.isfinite(tvt) & np.isfinite(z) & np.isfinite(xy).all(axis=1)
    if int(known.sum()) < 30:
        return out
    try:
        form, _ = ctx.imputers.formation.impute(xy, self_wid=None)
        form = np.asarray(form, dtype=float)
        for i, name in enumerate(FORMATIONS[: form.shape[1]]):
            good = known & np.isfinite(form[:, i])
            if int(good.sum()) >= 30:
                out.update(_surface_family(f"surface_{name}", form[:, i], tvt, z, good))
    except Exception as exc:
        print("surface imputer skipped", wid, exc)
    try:
        dense, _, _ = ctx.imputers.dense.impute(xy, self_wid=None)
        dense = np.asarray(dense, dtype=float)
        good = known & np.isfinite(dense)
        if int(good.sum()) >= 30:
            out.update(_surface_family("dense_ancc", dense, tvt, z, good))
    except Exception as exc:
        print("dense imputer skipped", wid, exc)

    names = [n for n in out if n.startswith(("surface_", "dense_ancc"))]
    if len(names) >= 2:
        errs = np.array([masked_rmse(out[n][known], tvt[known]) for n in names], dtype=float)
        finite = np.isfinite(errs)
        if int(finite.sum()) >= 2:
            names = [n for n, ok in zip(names, finite) if ok]
            weights = 1.0 / np.maximum(errs[finite], 0.25) ** 2
            weights = weights / weights.sum()
            out["surface_weighted_prefix"] = (
                (weights[:, None] * np.vstack([out[n] for n in names])).sum(axis=0).astype(float)
            )
    out.update(contact_candidates(wid, hw_masked, ctx))
    return out


def pf_candidates(
    hw_masked, tw, variants, n_seeds: int, n_particles: int, ctx: Context
) -> dict[str, np.ndarray]:
    out = {}
    known = hw_masked[hw_masked["TVT_input"].notna()]
    unknown = hw_masked[hw_masked["TVT_input"].isna()]
    if len(known) < 30 or len(unknown) == 0:
        return out
    cfg = ctx.selector_cfg
    try:
        pf_by_scale = ensemble_by_scale(hw_masked, tw, scales=cfg.scales, n_particles=int(n_particles),
                                        n_seeds=int(n_seeds), return_std=cfg.return_std)  # fmt: skip
        try:
            beam = beam_ensemble(hw_masked, tw)
        except Exception:
            beam = pf_by_scale.get("pf_mean")
            if beam is None:
                beam = next(iter(pf_by_scale.values()))
        last = float(known["TVT_input"].iloc[-1])
        for variant in variants:
            try:
                pred = selector.apply_variant(variant, pf_by_scale, beam, last, hw_masked, tw, cfg)
                if pred is not None and len(pred) == len(hw_masked):
                    out["pf|" + variant] = np.asarray(pred, dtype=float)
            except Exception:
                pass
    except Exception as exc:
        print("PF calibration skipped:", exc)
    return out


def candidate_pool(wid, hw_masked, tw, variants, ctx: Context, include_pf=True, n_seeds=24, n_particles=350):
    pool = {**poly_candidates(hw_masked), **surface_candidates(hw_masked, wid, ctx)}
    if include_pf:
        pool.update(pf_candidates(hw_masked, tw, variants, n_seeds, n_particles, ctx))
    keep = {}
    for name, pred in pool.items():
        arr = np.asarray(pred, dtype=float)
        if len(arr) == len(hw_masked) and np.isfinite(arr).sum() >= max(20, len(hw_masked) // 20):
            keep[name] = arr
    return keep


def _default_name(hw) -> str:
    try:
        return "pf|" + selector.well_code(hw)[1]
    except Exception:
        return "pf|" + selector.GLOBAL_VARIANT


def calibrate_well(wid, hw, tw, variants, ctx: Context, cfg: VisiblePrefixConfig):
    """Back-test every candidate generator on hidden slices of the visible prefix."""
    tvt = hw["TVT_input"].to_numpy(dtype=float)
    is_known = np.isfinite(tvt)
    if not (~is_known).any():
        return None
    first_hidden = int(np.flatnonzero(~is_known)[0])
    prefix = np.flatnonzero(is_known & (np.arange(len(hw)) < first_hidden))
    if len(prefix) < 140:
        return dict(well=wid, status="skip_short_prefix", known_prefix=int(len(prefix)))

    cuts = []
    for frac in cfg.cut_fracs:
        pos = max(50, min(int(round(len(prefix) * float(frac))), len(prefix) - 35))
        if pos <= 0 or pos >= len(prefix):
            continue
        holdout = prefix[pos:]
        if len(holdout) >= 35:
            cuts.append((float(frac), int(prefix[pos - 1]), holdout))
    if not cuts:
        return dict(well=wid, status="skip_no_holdout", known_prefix=int(len(prefix)))

    scores, cut_rows = {}, []
    default_name = _default_name(hw)
    for frac, cutoff, holdout in cuts:
        masked = hw.copy(deep=True)
        masked.loc[masked.index > cutoff, "TVT_input"] = np.nan
        pool = candidate_pool(wid, masked, tw, variants, ctx, cfg.include_pf, cfg.cal_seeds, cfg.particles)
        truth = tvt[holdout]
        row = {"well": wid, "cut_frac": frac, "holdout_rows": int(len(holdout)), "candidates": int(len(pool))}
        local = []
        for name, pred in pool.items():
            err = masked_rmse(pred[holdout], truth)
            if np.isfinite(err):
                scores.setdefault(name, []).append(err)
                local.append((err, name))
        local.sort()
        if local:
            row["best_name"], row["best_rmse"] = local[0][1], float(local[0][0])
            row["default_rmse"] = (
                float(masked_rmse(pool[default_name][holdout], truth))
                if default_name in pool
                else float("nan")
            )
        cut_rows.append(row)
    if not scores:
        return dict(well=wid, status="skip_no_scores", known_prefix=int(len(prefix)))

    agg = {
        n: float(np.nanmedian(v) + 0.10 * np.nanstd(v))
        for n, v in ((k, np.asarray(vs, dtype=float)) for k, vs in scores.items())
    }
    ordered = sorted((v, k) for k, v in agg.items() if np.isfinite(v))
    if not ordered:
        return dict(well=wid, status="skip_nonfinite_scores", known_prefix=int(len(prefix)))
    best_score, best_name = ordered[0]
    second_score = ordered[1][0] if len(ordered) > 1 else best_score
    if default_name in agg:
        default_score = float(agg[default_name])
    else:
        pf_scores = [v for k, v in agg.items() if k.startswith("pf|")]
        default_score = float(np.nanmedian(pf_scores)) if pf_scores else float(second_score)

    wins, comparable = 0.0, 0
    for row in cut_rows:
        if np.isfinite(row.get("default_rmse", np.nan)):
            comparable += 1
            wins += float(row.get("best_rmse", float("inf")) <= row["default_rmse"] - 0.25)
    if comparable:
        consistency = wins / comparable
    else:
        winners = [r.get("best_name") for r in cut_rows if r.get("best_name")]
        consistency = float(sum(w == best_name for w in winners)) / max(1, len(winners))
    return dict(
        well=wid, status="ok", known_prefix=int(len(prefix)), cuts=int(len(cut_rows)), candidate_count=int(len(agg)),
        best_name=best_name, best_score=float(best_score), second_score=float(second_score), default_name=default_name,
        default_score=float(default_score), gain=float(default_score - best_score),
        rank_margin=float(second_score - best_score), consistency=float(consistency), cut_rows=cut_rows,
    )  # fmt: skip


def move_weight(report: dict, delta_rmse: float, delta_p95: float, profile: str) -> float:
    p = PROFILES[profile]
    if report.get("status") != "ok":
        return 0.0
    gain = float(report.get("gain", 0.0))
    best = float(report.get("best_score", float("inf")))
    margin = float(report.get("rank_margin", 0.0))
    consistency = float(report.get("consistency", 0.0))
    if (not np.isfinite(best) or best > p["max_best"] or gain < p["min_gain"]
            or consistency < p["min_consistency"] or margin < p.get("min_margin", 0.0)):  # fmt: skip
        return 0.0
    alpha = p["base"]
    alpha += p["gain_scale"] * min(max(gain, 0.0), 5.0) / 5.0
    alpha += p["margin_scale"] * min(max(margin, 0.0), 3.0) / 3.0
    if best <= 5.0:
        alpha += p["quality_bonus"]
    if str(report.get("best_name", "")).startswith(STRUCTURAL_PREFIXES) and consistency >= 0.67:
        alpha += 0.03 if profile != "aggressive" else 0.06
    if np.isfinite(delta_rmse) and delta_rmse > p["delta_soft"]:
        alpha *= max(0.20, p["delta_soft"] / max(delta_rmse, 1e-6))
    if np.isfinite(delta_p95) and delta_p95 > p["p95_hard"]:
        return 0.0
    return float(min(p["cap"], max(0.0, alpha * 1.30)))


def profile_output(base: pd.DataFrame, candidate_by_id: dict, reports: dict, profile: str):
    prof = PROFILES[profile]
    out = base.copy()
    moves = []
    for wid, rep in reports.items():
        ids = out.loc[out["well"] == wid, "id"].astype(str).tolist()
        if not ids:
            continue
        if rep.get("status") != "ok":
            moves.append({**rep, "profile": profile, "alpha": 0.0, "delta_rmse_vs_base": float("nan"),
                          "delta_p95_vs_base": float("nan"), "max_move_clip": 0.0, "applied_rows": 0,
                          "mean_abs_move": 0.0, "max_abs_move": 0.0, "apply_status": str(rep.get("status", "skipped"))})  # fmt: skip
            continue
        cand = np.array([candidate_by_id.get(i, np.nan) for i in ids], dtype=float)
        idx = out.index[out["well"] == wid].to_numpy()
        current = out.loc[idx, "tvt"].to_numpy(dtype=float)
        if not (np.isfinite(cand) & np.isfinite(current)).all():
            moves.append({**rep, "apply_status": "skip_nonfinite_candidate"})
            continue
        diff = cand - current
        delta_rmse = float(np.sqrt(np.mean(diff * diff))) if len(diff) else float("nan")
        delta_p95 = float(np.quantile(np.abs(diff), 0.95)) if len(diff) else float("nan")
        alpha = move_weight(rep, delta_rmse, delta_p95, profile)
        gain = max(0.0, float(rep.get("gain", 0.0)))
        max_move = min(prof["clip_max"], prof["clip_base"] + prof["clip_gain"] * np.sqrt(gain + 1e-9))
        ramp = 1.0 - np.exp(-np.arange(len(diff), dtype=float) / max(80.0, 0.12 * max(1, len(diff))))
        move = np.clip(alpha * ramp * diff, -max_move, max_move)
        out.loc[idx, "tvt"] = current + move
        moves.append({**rep, "profile": profile, "alpha": float(alpha), "delta_rmse_vs_base": delta_rmse,
                      "delta_p95_vs_base": delta_p95, "max_move_clip": float(max_move), "applied_rows": int(len(idx)),
                      "mean_abs_move": float(np.mean(np.abs(move))) if len(move) else 0.0,
                      "max_abs_move": float(np.max(np.abs(move))) if len(move) else 0.0,
                      "apply_status": "applied" if alpha > 0 else "kept_base"})  # fmt: skip
    return out, moves


def run(sub: pd.DataFrame, ctx: Context, cfg: VisiblePrefixConfig, contact_override: bool) -> pd.DataFrame:
    """Calibrate every well on its own prefix and let the best candidate make a bounded move."""
    if not cfg.enabled:
        print("visible-prefix calibration disabled")
        return sub
    t0 = time.time()
    profile = cfg.profile if cfg.profile in PROFILES else "balanced"
    sample = read_sample(ctx.data_dir)[["id"]]
    anchor = validate_submission(sub, sample, "visible-prefix input")
    anchor_path = ctx.out_dir / "submission_self_verified_anchor.csv"
    write_submission(anchor, anchor_path)
    base = split_ids(anchor)
    variants = selector.variant_grid()

    skip_wells, guard_report = set(), pd.DataFrame()
    if cfg.bimodal_guard:
        skip_wells, guard_report = bimodal_guard_wells(
            ctx.out_dir, cfg.guard_requires_low_trust, cfg.low_trust_threshold
        )
    if len(guard_report):
        guard_report.to_csv(ctx.out_dir / "gold_prefix_bimodal_guard_report.csv", index=False)
    if skip_wells:
        print("visible-prefix bimodal guard wells:", sorted(skip_wells), flush=True)

    reports, cut_reports, candidate_by_id = [], [], {}
    wells = list(base["well"].drop_duplicates())[: cfg.max_wells]
    for i, wid in enumerate(wells, 1):
        try:
            hw_path = ctx.data_dir / "test" / f"{wid}__horizontal_well.csv"
            tw_path = ctx.data_dir / "test" / f"{wid}__typewell.csv"
            if not hw_path.exists() or not tw_path.exists():
                reports.append(dict(well=wid, status="skip_missing_files"))
                continue
            hw, tw = pd.read_csv(hw_path), pd.read_csv(tw_path)
            if wid in skip_wells:
                print(f"[prefix {i}/{len(wells)}] keeping bimodal hedge for {wid}", flush=True)
                reports.append(dict(well=wid, status="skip_bimodal_hedge", bimodal_prefix_guard=True,
                                    reason="visible-prefix commit disabled for active bimodal selector well"))  # fmt: skip
                continue
            print(f"[prefix {i}/{len(wells)}] calibrating {wid}", flush=True)
            rep = calibrate_well(wid, hw, tw, variants, ctx, cfg) or dict(well=wid, status="skip_none")
            cut_reports.extend(rep.pop("cut_rows", []))
            if rep.get("status") == "ok":
                best = rep["best_name"]
                needs_pf = str(best).startswith("pf|")
                final = candidate_pool(wid, hw, tw, variants, ctx, needs_pf, cfg.final_seeds, cfg.particles)
                if best not in final and needs_pf:
                    final = candidate_pool(wid, hw, tw, variants, ctx, False, 0, cfg.particles)
                if best in final:
                    part = base[base["well"] == wid]
                    values = final[best]
                    for rid, ri in zip(part["id"].astype(str).values, part["row_idx"].astype(int).values):
                        if 0 <= ri < len(values) and np.isfinite(values[ri]):
                            candidate_by_id[rid] = float(values[ri])
                    rep["final_candidate_available"] = True
                else:
                    rep["final_candidate_available"] = False
                    rep["status"] = "skip_no_final_candidate"
            reports.append(rep)
            print(
                "  report:",
                {
                    k: rep.get(k)
                    for k in ("status", "best_name", "best_score", "default_score", "gain", "consistency")
                },
                flush=True,
            )
        except Exception as exc:
            print("prefix calibration fallback", wid, exc)
            reports.append(dict(well=wid, status="error", error=str(exc)))

    pd.DataFrame(reports).to_csv(ctx.out_dir / "gold_prefix_calibration_report.csv", index=False)
    if cut_reports:
        pd.DataFrame(cut_reports).to_csv(ctx.out_dir / "gold_prefix_cut_report.csv", index=False)
    by_well = {r["well"]: r for r in reports if isinstance(r, dict) and r.get("well")}

    summaries = {}
    outputs = {}
    for name in PROFILES:
        out, moves = profile_output(base, candidate_by_id, by_well, name)
        out = out[["id", "tvt"]].copy()
        if contact_override:
            out = apply_override(
                out, ctx.data_dir, ctx.override_cfg, ctx.out_dir / "gold_contact_override_report.csv"
            )
        out = validate_submission(out, sample, f"profile {name}")
        path = ctx.out_dir / f"submission_gold_prefix_{name}.csv"
        write_submission(out, path)
        outputs[name] = out
        move_df = pd.DataFrame(moves)
        move_df.to_csv(ctx.out_dir / f"gold_prefix_moves_{name}.csv", index=False)
        summaries[name] = dict(
            file=str(path), rows=int(len(out)), sha256=sha256_file(path),
            applied_wells=int((move_df.get("apply_status") == "applied").sum()) if "apply_status" in move_df else 0,
            mean_abs_move=float(move_df["mean_abs_move"].mean()) if "mean_abs_move" in move_df and len(move_df) else 0.0,
            max_abs_move=float(move_df["max_abs_move"].max()) if "max_abs_move" in move_df and len(move_df) else 0.0,
        )  # fmt: skip

    selection = cfg.final_selection.strip().lower()
    if selection in SELECTOR_ALIASES["anchor"]:
        chosen, chosen_path = anchor, anchor_path
    elif selection in SELECTOR_ALIASES["profile"]:
        chosen, chosen_path = outputs[profile], ctx.out_dir / f"submission_gold_prefix_{profile}.csv"
    else:
        raise ValueError(f"unknown visible-prefix final selection {cfg.final_selection!r}")

    final_path = ctx.out_dir / "submission.csv"
    write_submission(chosen, final_path)
    audit = dict(
        selected_profile=profile, selected_sha256=sha256_file(final_path), self_verified_anchor_sha256=sha256_file(anchor_path),
        elapsed_sec=float(time.time() - t0), wells=int(len(wells)), candidates_with_final_values=int(len(candidate_by_id)),
        bimodal_guard_enabled=bool(cfg.bimodal_guard), bimodal_guard_requires_low_trust=bool(cfg.guard_requires_low_trust),
        bimodal_guard_low_trust_threshold=float(cfg.low_trust_threshold), bimodal_guard_wells=sorted(skip_wells),
        contact_override_enabled=bool(contact_override), final_selection=selection, chosen_path=str(chosen_path),
        profiles=summaries,
    )  # fmt: skip
    (ctx.out_dir / "gold_prefix_submission_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    print("visible-prefix selected:", profile, selection, flush=True)
    return chosen
