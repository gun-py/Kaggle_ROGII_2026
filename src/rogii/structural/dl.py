from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from rogii.io import list_wells

from .field import Field
from .hmm import gr_calibration
from .variants import DLRule
from .wells import load_wells

DRIFT_GRID = np.arange(-150.0, 150.01, 7.5)
SIGMA_GR = 8.0
BASE_STRIDE = 8
HEEL_CONTEXT = 250
MAX_STEPS = 1400

DEFAULT_NET = dict(hid=96, layers=1, drop=0.10, lr=2.0e-3)
NET_GRID = [
    dict(hid=64, layers=1, drop=0.10, lr=2.0e-3),
    dict(hid=96, layers=1, drop=0.10, lr=2.0e-3),
    dict(hid=96, layers=2, drop=0.20, lr=1.5e-3),
    dict(hid=128, layers=1, drop=0.15, lr=1.0e-3),
]


@dataclass
class DLResult:
    test: pd.Series
    oof_rmse: float
    carry_rmse: float
    backend: str


def sample_features(wid: str, d: dict, fld: Field, drop):
    """Strided per-row features (neighbour prior, GR emission over a drift grid, geometry) and the drift target."""
    md, z, gr, known = d["md"], d["z"], d["gr"], d["known"]
    if known.sum() == 0 or (~known).sum() == 0:
        return None
    ia = int(np.where(known)[0][-1])
    ta, za = float(d["tvt"][ia]), float(z[ia])

    a, b = gr_calibration(gr[known], d["tvt"][known], d["tw_tvt"], d["tw_gr"])
    grc = a * gr + b
    finite = np.isfinite(grc)
    if finite.sum() >= 2:
        grc = np.interp(md, md[finite], grc[finite])
    else:
        grc = np.nan_to_num(grc, nan=float(np.nanmedian(d["tw_gr"])))

    n = len(md)
    start = max(0, ia - HEEL_CONTEXT)
    stride = max(BASE_STRIDE, int(np.ceil((n - start) / MAX_STEPS)))
    idx = np.arange(start, n, stride)
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]

    med, nearest = fld.delta_s(np.c_[d["x"], d["y"]], ia, drop)
    neighbour_drift = med - (z - za)

    tw_at = np.interp(ta + DRIFT_GRID, d["tw_tvt"], d["tw_gr"], left=np.nan, right=np.nan)
    g = grc[idx]
    err = np.abs(g[:, None] - tw_at[None, :])
    in_range = np.isfinite(err)
    sim = np.where(in_range, np.exp(-0.5 * (err / SIGMA_GR) ** 2), 0.0)
    total = sim.sum(1)
    soft = np.where(total > 1e-9, (sim * DRIFT_GRID[None, :]).sum(1) / np.maximum(total, 1e-9), 0.0)
    err_m = np.where(in_range, err, 1e9)
    best = err_m.argmin(1)
    dstar = DRIFT_GRID[best]
    cost = np.minimum(np.take_along_axis(err_m, best[:, None], 1)[:, 0], 1e4)

    dmd = (md[idx] - md[ia]) / 1000.0
    dz = z[idx] - za
    nb = neighbour_drift[idx]
    feats = np.column_stack([
        np.nan_to_num(nb, nan=0.0), np.isfinite(nb).astype(float), np.full(len(idx), min(nearest, 3000.0) / 1000.0),
        dstar, soft, cost, sim.max(1), g,
        gaussian_filter1d(dstar, 6), gaussian_filter1d(soft, 6),
        dmd, dz, np.gradient(dz), np.gradient(g), (~known[idx]).astype(float),
    ]).astype(np.float32)  # fmt: skip

    return dict(feats=feats, target=(d["tvt"][idx] - ta).astype(np.float32), evalmask=~known[idx], idx=idx,
                ia=ia, ta=ta, nd=nearest, n=n, ev_rows=np.where(~known)[0], wid=wid)  # fmt: skip


def eval_rmse(samples: dict, oof: dict) -> float:
    errs = np.concatenate([oof[w][s["evalmask"]] - s["target"][s["evalmask"]] for w, s in samples.items()])
    return float(np.sqrt((errs**2).mean()))


def reconstruct(samples: dict, drift_by_well: dict, clip: float = 200.0) -> pd.Series:
    """Strided drifts -> TVT on the unknown rows, indexed by submission id."""
    parts = []
    for w, s in samples.items():
        drift = np.clip(np.nan_to_num(drift_by_well[w], nan=0.0), -clip, clip)
        tvt = s["ta"] + np.interp(np.arange(s["n"]), s["idx"], drift)
        rows = s["ev_rows"]
        parts.append(pd.DataFrame({"id": [f"{w}_{i}" for i in rows], "tvt": tvt[rows]}))
    return pd.concat(parts, ignore_index=True).set_index("id")["tvt"]


def train_hgbr(samples: dict, folds: int, verbose: bool = True):
    """CPU fallback when torch is unavailable: boosted trees over the same features."""
    wells = list(samples)
    lens = [len(samples[w]["feats"]) for w in wells]
    off = np.cumsum([0, *lens])
    X = np.vstack([samples[w]["feats"] for w in wells])
    y = np.concatenate([samples[w]["target"] for w in wells])
    owner = np.concatenate([[i] * lens[i] for i in range(len(wells))])

    def make():
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=2.0,
                                             min_samples_leaf=60, early_stopping=True, random_state=0)  # fmt: skip

    oof_rows = np.zeros(len(y))
    for fold, (tr, va) in enumerate(KFold(folds, shuffle=True, random_state=42).split(wells)):
        fit_mask, val_mask = np.isin(owner, tr), np.isin(owner, va)
        oof_rows[val_mask] = make().fit(X[fit_mask], y[fit_mask]).predict(X[val_mask])
        if verbose:
            print(f"    hgbr fold {fold + 1}/{folds} done", flush=True)
    full = make().fit(X, y)
    oof = {w: gaussian_filter1d(oof_rows[off[i] : off[i + 1]], 8) for i, w in enumerate(wells)}

    def predict(test_samples):
        return {w: gaussian_filter1d(full.predict(s["feats"]), 8) for w, s in test_samples.items()}

    return oof, predict


def fit(data_dir: Path, rule: DLRule) -> DLResult:
    """Build features for every well, train the sequence model (or the HGB fallback) and predict the test wells."""
    t0 = time.time()
    train_ids, test_ids = list_wells(data_dir, "train"), list_wells(data_dir, "test")
    wells = load_wells(data_dir, "train", train_ids)
    fld = Field(wells)
    train_samples = {}
    for k, w in enumerate(train_ids):
        s = sample_features(w, wells[w], fld, {w})
        if s is not None:
            train_samples[w] = s
        if (k + 1) % 150 == 0:
            print(f"  {k + 1}/{len(train_ids)}  {time.time() - t0:.0f}s", flush=True)
    test_samples = {}
    for w in test_ids:
        s = sample_features(w, load_wells(data_dir, "test", [w])[w], fld, {w})
        if s is not None:
            test_samples[w] = s

    carry = eval_rmse(train_samples, {w: np.zeros_like(s["target"]) for w, s in train_samples.items()})
    print(f"carry-forward OOF RMSE (strided) = {carry:.3f}")

    try:
        from . import dl_torch
    except ImportError:
        dl_torch = None

    def run(cfg, folds, seeds, epochs, verbose):
        if dl_torch is not None:
            try:
                return (*dl_torch.train(train_samples, cfg, folds, seeds, epochs, verbose), "torch")
            except Exception as exc:
                print("  torch path failed, using HGB fallback:", repr(exc)[:200], flush=True)
        return (*train_hgbr(train_samples, folds, verbose), "hgbr")

    cfg = DEFAULT_NET
    if rule.search and dl_torch is not None:
        best = np.inf
        for candidate in NET_GRID:
            oof, _, backend = run(candidate, 2, 1, 20, False)
            score = eval_rmse(train_samples, oof)
            print(f"    cfg {candidate} -> quick OOF {score:.3f}", flush=True)
            if score < best:
                best, cfg = score, candidate
            if backend == "hgbr":
                break
    oof, predict, backend = run(cfg, rule.folds, rule.seeds, rule.epochs, True)
    oof_rmse = eval_rmse(train_samples, oof)
    print(f"backend={backend}  DL OOF RMSE (strided) = {oof_rmse:.3f}  (carry-forward {carry:.3f})")
    return DLResult(reconstruct(test_samples, predict(test_samples)), oof_rmse, carry, backend)


def guarded_blend(existing: pd.Series, result: DLResult, rule: DLRule) -> pd.Series:
    """`(1 - w) * existing + w * DL`, forced back to `existing` when an OOF or divergence gate trips."""
    w = rule.weight
    if result.oof_rmse >= rule.guard_oof_ratio * result.carry_rmse:
        print(
            f"DL gate: OOF {result.oof_rmse:.3f} did not beat {rule.guard_oof_ratio:.2f}*carry {result.carry_rmse:.3f}"
        )
        return existing
    mine = result.test.reindex(existing.index)
    nan_frac = float(mine.isna().mean())
    gap = float((mine.fillna(existing) - existing).mean())
    print(
        f"RMSD(two_way, DL) = {float(np.sqrt(((existing - mine) ** 2).mean(skipna=True))):.3f} | gap {gap:+.2f} ft | NaN {nan_frac:.3f}"
    )
    if nan_frac > rule.guard_max_nan:
        print(f"DL gate: {nan_frac:.1%} of rows missing a prediction")
        return existing
    if abs(gap) > rule.guard_mean_ft:
        print(f"DL gate: |mean gap| {abs(gap):.1f} > {rule.guard_mean_ft} ft")
        return existing
    return ((1.0 - w) * existing + w * mine).fillna(existing)
