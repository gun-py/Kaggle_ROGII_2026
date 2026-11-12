from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rogii.geology.contacts import FORMATIONS
from rogii.io import HW_SUFFIX, TW_SUFFIX, read_well
from rogii.tracking.beam import FEATURE_BEAMS, jit_beam_search
from rogii.tracking.trackers import lik_pf, run_pf_ancc, run_pf_z

from .imputers import Imputers
from .ncc import multi_scale_ncc
from .surfaces import affine_cal, robust_slope, segment_offsets

ANCHOR_OFFSETS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFSETS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
NCC_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFSETS = NCC_OFFSETS
GR_WINDOWS = (5, 21, 51, 101)
GR_LAGS = (1, 5, 15, 30)


def default_jobs() -> int:
    return min(8, multiprocessing.cpu_count())


def _f32(a) -> np.ndarray:
    return np.asarray(a).astype(np.float32)


def _gr_context(gr: pd.Series, rows: pd.Index) -> dict[str, np.ndarray]:
    out = {}
    for w in GR_WINDOWS:
        roll = gr.rolling(w, center=True, min_periods=1)
        out[f"grm{w}"] = _f32(roll.mean().iloc[rows].values)
        out[f"grs{w}"] = _f32(roll.std().fillna(0).iloc[rows].values)
    for lag in GR_LAGS:
        out[f"glag{lag}"] = _f32(gr.shift(lag).bfill().iloc[rows].values)
        out[f"glead{lag}"] = _f32(gr.shift(-lag).ffill().iloc[rows].values)
    return out


def build_well(hw_path, tw_path, is_train: bool, imputers: Imputers) -> pd.DataFrame | None:
    wid = Path(hw_path).stem.replace(HW_SUFFIX, "")
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    if is_train and "TVT" not in hw.columns:
        return None
    known, unknown = hw[hw.TVT_input.notna()], hw[hw.TVT_input.isna()]
    if len(unknown) == 0 or len(known) < 10:
        return None
    if is_train and hw.TVT.isna().all():
        return None
    tw_tvt = tw.TVT.to_numpy(np.float32)
    tw_gr = tw.GR.to_numpy(np.float32)
    if len(tw_tvt) < 3:
        return None

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0:
        return None
    pf_z, _ = run_pf_z(hw, tw_tvt, tw_gr)
    pf = _f32(pf_a)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    last = known.iloc[-1]
    last_tvt = float(last.TVT_input)
    gr_full = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[unknown.index[0] :].to_numpy(np.float32)
    kgr = gr_full.iloc[: len(known)].to_numpy(np.float32)

    beams = {
        tag: jit_beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
        for bs, mc, es, r, tag in FEATURE_BEAMS
    }
    beam_ref = (beams["cons"] + beams["sm5"]) / 2.0

    ktvt = known.TVT_input.to_numpy(np.float32)
    ncc, ncc_ens = multi_scale_ncc(kgr, ktvt, hgr, half_windows=(8, 15, 25), stride=3)
    (sc8, sc8s), (sc15, sc15s), (sc25, sc25s) = ncc
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(known) / 200.0, 0.0, 0.6))
    hybrid = (1 - sc_trust) * beam_ref + sc_trust * ncc_ens

    tw_at_known = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    cal_a, cal_b = affine_cal(kgr, tw_at_known)
    kmd, kz = known.MD.to_numpy(np.float32), known.Z.to_numpy(np.float32)
    prefix_rmse = float(np.sqrt(np.mean((kgr - tw_at_known) ** 2)))
    slope_all, slope_50, slope_z = (
        robust_slope(kmd, ktvt),
        robust_slope(kmd[-50:], ktvt[-50:]),
        robust_slope(kz, ktvt),
    )

    self_wid = wid if is_train else None
    xy_unknown = unknown[["X", "Y"]].to_numpy(np.float64)
    xy_known = known[["X", "Y"]].to_numpy(np.float64)
    form_unknown, knn_dist = imputers.formation.impute(xy_unknown, self_wid=self_wid)
    form_known, _ = imputers.formation.impute(xy_known, self_wid=self_wid)
    z_known, z_unknown = known.Z.to_numpy(np.float32), unknown.Z.to_numpy(np.float32)

    surface, form_rmse, per_form = {}, {}, []
    for i, name in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = segment_offsets(ktvt, z_known, form_known[:, i])
        tvt_f = _f32(-z_unknown + form_unknown[:, i] + b_full)
        surface[f"tvtF_{name}"] = tvt_f
        surface[f"tvtFw_{name}"] = _f32(-z_unknown + form_unknown[:, i] + b_wls)
        surface[f"tvtF50_{name}"] = _f32(-z_unknown + form_unknown[:, i] + b_late)
        surface[f"bw_{name}"] = np.float32(b_full)
        surface[f"bww_{name}"] = np.float32(b_wls)
        surface[f"bw50_{name}"] = np.float32(b_late)
        surface[f"bw_early_{name}"] = np.float32(b_early)
        surface[f"bw_mid_{name}"] = np.float32(b_mid)
        form_rmse[name] = float(np.sqrt(np.mean((ktvt - (-z_known + form_known[:, i] + b_full)) ** 2)))
        per_form.append(tvt_f)
    stack = np.stack(per_form, 1)

    d_ancc, d_std, d_dist = imputers.dense.impute(xy_unknown, self_wid=self_wid)
    d_known, d_std_known, _ = imputers.dense.impute(xy_known, self_wid=self_wid)
    _, _, _, b_dense_late, b_dense_wls = segment_offsets(ktvt, z_known, d_known)
    b_dense = float(np.median(ktvt + z_known - d_known))
    tvt_dense = _f32(-z_unknown + d_ancc + b_dense)
    tvt_dense_w = _f32(-z_unknown + d_ancc + b_dense_wls)
    tvt_dense_50 = _f32(-z_unknown + d_ancc + b_dense_late)
    dense_resid = ktvt + z_known - d_known

    signals = np.stack([pf, *beams.values(), sc8, sc15, sc25, ncc_ens, surface["tvtF_ANCC"], tvt_dense], 1)

    gr_series = pd.Series(gr_full.values)
    rows = unknown.index
    rolls = _gr_context(gr_series, rows)
    gr_d1 = _f32(gr_series.diff().fillna(0.0).iloc[rows].values)
    gr_d2 = _f32(gr_series.diff().diff().fillna(0.0).iloc[rows].values)
    gr_env = _f32(gr_series.rolling(21, center=True, min_periods=1).max().iloc[rows].values)
    gr_nrg = _f32(
        np.sqrt(np.maximum((gr_series**2).rolling(21, center=True, min_periods=1).mean(), 0.0))
        .iloc[rows]
        .values
    )

    md_since = unknown.MD.to_numpy(np.float32) - float(last.MD)
    slope_path_all = _f32(last_tvt + slope_all * md_since)
    slope_path_50 = _f32(last_tvt + slope_50 * md_since)

    md_step = hw.MD.diff().replace(0, np.nan)
    dz_dmd, dx_dmd, dy_dmd = (_f32((hw[c].diff() / md_step).iloc[rows].values) for c in ("Z", "X", "Y"))

    n = len(unknown)
    frac = _f32(np.arange(n) / max(n - 1, 1))

    def const(v):
        return np.full(n, np.float32(v), np.float32)

    def delta(path):
        return _f32(path - np.float32(last_tvt))

    beam_deltas = np.stack([p - last_tvt for p in beams.values()], 1)
    tw_at_last = np.float32(np.interp(last_tvt, tw_tvt, tw_gr))

    feats = {
        "well": wid,
        "id": [f"{wid}_{i}" for i in unknown.index],
        "last_known_tvt": const(last_tvt),
        "pf_ancc": pf,
        "pf_ancc_std": _f32(std_a),
        "pf_ancc_delta": _f32(pf - last_tvt),
        "pf_z": _f32(pf_z) if has_z else const(last_tvt),
        "pf_z_delta": _f32(pf_z - last_tvt) if has_z else const(0.0),
        "pf_vs_z": (pf - _f32(pf_z)) if has_z else const(0.0),
        **{f"beam_{tag}_d": delta(p) for tag, p in beams.items()},
        "beam_mean_d": _f32(beam_deltas.mean(1)),
        "beam_std_d": _f32(beam_deltas.std(1)),
        "beam_med_d": _f32(np.median(beam_deltas, 1)),
        "sc8_d": delta(sc8), "sc8_sc": sc8s,
        "sc15_d": delta(sc15), "sc15_sc": sc15s,
        "sc25_d": delta(sc25), "sc25_sc": sc25s,
        "sc_cons_d": delta(sc_cons),
        "sc_ens_d": delta(ncc_ens),
        "sc_trust": const(sc_trust),
        "hyb_d": delta(hybrid),
        "sig_std": _f32(signals.std(1)),
        "sig_mean_d": _f32(signals.mean(1) - last_tvt),
        **surface,
        **{f"frm_rmse_{name}": const(form_rmse[name]) for name in FORMATIONS},
        "form_mean_d": _f32(stack.mean(1) - last_tvt),
        "form_std_d": _f32(stack.std(1)),
        "form_rng_d": _f32(stack.max(1) - stack.min(1)),
        "spatial_ancc_d": form_unknown[:, 0] - tw_at_last,
        "spatial_knn_dist": knn_dist,
        "dense_ancc": d_ancc, "dense_std": d_std, "dense_dist": d_dist,
        "tvt_dense_d": _f32(tvt_dense - last_tvt),
        "tvt_densew_d": _f32(tvt_dense_w - last_tvt),
        "tvt_dense50_d": _f32(tvt_dense_50 - last_tvt),
        "dense_rmse": const(float(np.sqrt(np.mean(dense_resid**2)))),
        "dense_bias": const(float(np.mean(dense_resid))),
        "dense_nb_std": const(float(np.mean(d_std_known))),
        "pf_vs_spatial": _f32(pf - surface["tvtF_ANCC"]),
        "pf_vs_dense": _f32(pf - tvt_dense),
        "spatial_vs_dense": _f32(surface["tvtF_ANCC"] - tvt_dense),
        "beam_vs_spatial": _f32(beams["cons"] - surface["tvtF_ANCC"]),
        "sc_vs_beam": _f32(ncc_ens - beams["cons"]),
        "cal_a": const(cal_a), "cal_b": const(cal_b),
        "pfx_rmse": const(prefix_rmse),
        "known_len": const(len(known)),
        "eval_len": const(n),
        "slp_all": const(slope_all), "slp_50": const(slope_50), "slp_z": const(slope_z),
        "slp_b_d_all": _f32(slope_path_all - last_tvt),
        "slp_b_d_50": _f32(slope_path_50 - last_tvt),
        "ktvt_range": const(float(np.ptp(ktvt))),
        "ktvt_std": const(float(ktvt.std())),
        "md_since": md_since,
        "frac": frac, "frac2": frac**2, "sqrt_frac": np.sqrt(frac),
        "z": z_unknown,
        "dx": (unknown.X - float(last.X)).to_numpy(np.float32),
        "dy": (unknown.Y - float(last.Y)).to_numpy(np.float32),
        "dz": _f32(z_unknown - float(last.Z)),
        "dxy": np.sqrt((unknown.X - float(last.X)) ** 2 + (unknown.Y - float(last.Y)) ** 2).to_numpy(np.float32),
        "dzdmd": dz_dmd, "dxdmd": dx_dmd, "dydmd": dy_dmd,
        "gr": hgr, "gr_d1": gr_d1, "gr_d2": gr_d2, "gr_env": gr_env, "gr_nrg": gr_nrg,
        "gr_vs_tw_anc": hgr - tw_at_last,
        "gr_vs_slp_all": hgr - _f32(np.interp(slope_path_all, tw_tvt, tw_gr)),
        **{f"tda{int(o)}": hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCHOR_OFFSETS},
        **{f"tdbc{int(o)}": hgr - _f32(np.interp(beam_ref + o, tw_tvt, tw_gr)) for o in BEAM_OFFSETS},
        **{f"tdsc{int(o)}": hgr - _f32(np.interp(ncc_ens + o, tw_tvt, tw_gr)) for o in NCC_OFFSETS},
        **{f"tdpf{int(o)}": hgr - _f32(np.interp(pf + o, tw_tvt, tw_gr)) for o in PF_OFFSETS},
        "tw_range": const(float(np.ptp(tw_tvt))),
        "tw_gr_mean": const(float(tw_gr.mean())),
    }  # fmt: skip
    feats.update(rolls)
    frame = pd.DataFrame(feats)
    if is_train:
        if unknown.TVT.isna().all():
            return None
        frame["target"] = unknown.TVT.to_numpy(np.float32) - np.float32(last_tvt)
    return frame


def build_features(
    data_dir: Path, wells: list[str], split: str, imputers: Imputers, n_jobs: int | None = None
):
    is_train = split == "train"
    base = Path(data_dir) / split
    jobs = [
        (str(base / f"{w}{HW_SUFFIX}.csv"), str(base / f"{w}{TW_SUFFIX}.csv"))
        for w in wells
        if (base / f"{w}{TW_SUFFIX}.csv").exists()
    ]
    parts = Parallel(n_jobs=n_jobs or default_jobs(), prefer="threads")(
        delayed(build_well)(hp, tp, is_train, imputers) for hp, tp in jobs
    )
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _likpf_rows(data_dir: Path, wid: str, split: str, n_particles: int, n_seeds: int):
    hw, tw = read_well(data_dir, wid, split, sort_typewell=True)
    out, idx, _ = lik_pf(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    if not len(out):
        return None
    cols = {"id": [f"{wid}_{i}" for i in idx]}
    for key, value in out.items():
        cols["likpf_" + key.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = value.astype(
            np.float32
        )
    return pd.DataFrame(cols)


def build_likpf(
    data_dir: Path, wells: list[str], split: str, n_particles=500, n_seeds=128, n_jobs=None
) -> pd.DataFrame:
    # the kernel releases the GIL, so threads scale without pickling compiled code
    parts = Parallel(n_jobs=n_jobs or default_jobs(), prefer="threads")(
        delayed(_likpf_rows)(data_dir, w, split, n_particles, n_seeds) for w in wells
    )
    return pd.concat([p for p in parts if p is not None], ignore_index=True)


def add_likpf_features(df: pd.DataFrame, likpf: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(likpf, on="id", how="left")
    for col in [c for c in likpf.columns if c != "id"]:
        df[col] = df[col].fillna(df["last_known_tvt"])
        df[col + "_d"] = (df[col] - df["last_known_tvt"]).astype(np.float32)
    return df
