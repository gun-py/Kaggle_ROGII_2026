"""Holdout, cross-validation and test-split runs for TRAIL, written in the diagnostics long format."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.attention.config import TrailConfig, ablated
from rogii.attention.data import TEST_PREFIX, WellSet, rows_from_tokens
from rogii.attention.synthetic import SyntheticWells
from rogii.diagnostics.cv import _sample_wells
from rogii.io import read_sample, validate_submission
from rogii.structural.wells import anchor_index, load_wells


def wellset(data_dir: Path, cache: Path | None = None) -> WellSet:
    if cache is not None and Path(cache).is_file():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    ws = WellSet(load_wells(data_dir, "train"), load_wells(data_dir, "test"))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(ws, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ws


def inner_split(ids: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed + 17)
    val = set(rng.choice(ids, size=max(1, int(round(fraction * len(ids)))), replace=False).tolist())
    return [w for w in ids if w not in val], sorted(val)


def carry_scores(ws: WellSet, wells: list[str]) -> dict[str, tuple[float, int]]:
    out = {}
    for w in wells:
        d = ws.wells[w]
        ia = anchor_index(d)
        y = d["tvt"][ia + 1 :]
        out[w] = (float(((y - d["tvt"][ia]) ** 2).sum()), int(len(y)))
    return out


def _long(scores: dict, method: str) -> pd.DataFrame:
    return pd.DataFrame(
        [(w, method, s, n) for w, (s, n) in scores.items()], columns=["well", "method", "sse", "n"]
    )


def evaluate_wells(model, ws: WellSet, wells: list[str], cfg: TrailConfig, device):
    """Score held-out wells at their real anchors; every other train well may serve as a neighbour."""
    from rogii.attention.train import predict, score_rows

    everyone = frozenset(ws.train_ids)
    samples = [ws.sample(w, everyone - {w}, None, cfg.max_visible_tokens, cfg.use_neighbours) for w in wells]
    t0 = time.time()
    drifts = predict(model, samples, device, cfg.token_budget)
    seconds = (time.time() - t0) / max(len(wells), 1)
    rows = {
        s.wid: rows_from_tokens(s, dr, ws.wells[s.wid]["z"]) for s, dr in zip(samples, drifts, strict=True)
    }
    return score_rows(ws, samples, drifts), rows, seconds


def run_split(
    cfg: TrailConfig, ws: WellSet, train_ids, test_ids, device, init, out_dir: Path, method: str, log=print
):
    from rogii.attention.train import finetune, save

    fit_ids, val_ids = inner_split(train_ids, cfg.val_fraction, cfg.seed)
    assert not set(test_ids) & set(fit_ids + val_ids), "held-out well inside the training pool"
    model, history = finetune(cfg, ws, fit_ids, val_ids, device, init=init, log=log)
    scores, rows, seconds = evaluate_wells(model, ws, test_ids, cfg, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(model, out_dir / f"{method}.pt", train=len(fit_ids), val=len(val_ids), test=len(test_ids))
    pd.DataFrame(history).to_csv(out_dir / f"{method}_history.csv", index=False)
    np.savez_compressed(out_dir / f"{method}_rows.npz", **rows)
    return scores, seconds


def holdout(data_dir: Path, out_dir: Path, cfg: TrailConfig, ablation: str = "full", init_path=None,
            n_wells: int = 100, seed: int = 0, cache=None, log=print) -> pd.DataFrame:  # fmt: skip
    from rogii.attention.train import load, pick_device

    device = pick_device()
    cfg = ablated(cfg, ablation)
    ws = wellset(data_dir, cache)
    test_ids = _sample_wells(data_dir, n_wells, seed)
    train_ids = [w for w in ws.train_ids if w not in set(test_ids)]
    init = None
    if init_path and ablation != "no_pretrain":
        init = load(init_path, "cpu").state_dict()
    method = f"trail_{ablation}"
    scores, seconds = run_split(cfg, ws, train_ids, test_ids, device, init, Path(out_dir), method, log)
    frame = pd.concat([_long(scores, method), _long(carry_scores(ws, test_ids), "carry")], ignore_index=True)
    frame.to_csv(Path(out_dir) / f"{method}_wells.csv", index=False)
    summary = {"method": method, "wells": len(test_ids), "rows": int(frame.query("method == @method").n.sum()),
               "rmse": pooled(frame, method), "carry": pooled(frame, "carry"), "seconds_per_well": seconds}  # fmt: skip
    (Path(out_dir) / f"{method}_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))
    return frame


def cross_validate(data_dir: Path, out_dir: Path, cfg: TrailConfig, init_path=None, folds: int = 5,
                   seed: int = 0, cache=None, log=print) -> pd.DataFrame:  # fmt: skip
    from rogii.attention.train import load, pick_device

    device = pick_device()
    ws = wellset(data_dir, cache)
    ids = np.array(ws.train_ids)
    fold_of = np.random.default_rng(seed).permutation(len(ids)) % folds
    init = load(init_path, "cpu").state_dict() if init_path else None
    frames = []
    for k in range(folds):
        test_ids = sorted(ids[fold_of == k].tolist())
        train_ids = sorted(ids[fold_of != k].tolist())
        log(f"fold {k + 1}/{folds}: {len(train_ids)} train, {len(test_ids)} scored")
        scores, _ = run_split(cfg, ws, train_ids, test_ids, device, init, Path(out_dir), f"trail_cv{k}", log)
        frames.append(_long(scores, "trail_cv").assign(fold=k))
        frames.append(_long(carry_scores(ws, test_ids), "carry").assign(fold=k))
        pd.concat(frames, ignore_index=True).to_csv(Path(out_dir) / "trail_cv_wells.csv", index=False)
    frame = pd.concat(frames, ignore_index=True)
    log(
        json.dumps({"cv_rmse": pooled(frame, "trail_cv"), "carry": pooled(frame, "carry"), "wells": len(ids)})
    )
    return frame


def pretrain_run(data_dir: Path, out_path: Path, cfg: TrailConfig, steps=None, cache=None, log=print):
    from rogii.attention.train import pick_device, pretrain, save

    ws = wellset(data_dir, cache)
    model = pretrain(cfg, SyntheticWells.from_wellset(ws), pick_device(), steps, log)
    save(model, out_path, steps=steps or cfg.pretrain_steps)
    return model


def predict_test(data_dir: Path, checkpoint: Path, out_dir: Path, cache=None) -> pd.DataFrame:
    """Write submission.csv for the test split; every train well is a candidate neighbour."""
    from rogii.attention.train import load, pick_device, predict

    device = pick_device()
    model = load(checkpoint, device)
    ws = wellset(data_dir, cache)
    everyone = frozenset(ws.train_ids)
    rows = []
    for w in ws.test_ids:
        s = ws.sample(w, everyone, None, model.cfg.max_visible_tokens, model.cfg.use_neighbours)
        drift = predict(model, [s], device, model.cfg.token_budget)[0]
        tvt = rows_from_tokens(s, drift, ws.wells[w]["z"])
        rows += [(f"{w.removeprefix(TEST_PREFIX)}_{s.ia + 1 + i}", float(v)) for i, v in enumerate(tvt)]
    sample = read_sample(data_dir)
    sub = sample[["id"]].merge(pd.DataFrame(rows, columns=["id", "tvt"]), on="id", how="left")
    sub = validate_submission(sub, sample, "attention")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sub.to_csv(Path(out_dir) / "submission.csv", index=False)
    return sub


def pooled(frame: pd.DataFrame, method: str) -> float:
    part = frame[frame.method == method]
    return float(np.sqrt(part.sse.sum() / part.n.sum()))


def bootstrap(frame: pd.DataFrame, base: str, resamples: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Pooled RMSE per method with 95% well-bootstrap intervals and paired differences against `base`."""
    sse = frame.pivot_table(index="well", columns="method", values="sse", aggfunc="sum")
    n = frame.pivot_table(index="well", columns="method", values="n", aggfunc="sum")
    sse, n = sse.dropna(), n.loc[sse.dropna().index]
    idx = np.random.default_rng(seed).integers(0, len(sse), size=(resamples, len(sse)))

    def rmse(col, take=None):
        s, c = sse[col].to_numpy(), n[col].to_numpy()
        return np.sqrt(s.sum() / c.sum()) if take is None else np.sqrt(s[take].sum(1) / c[take].sum(1))

    ref = rmse(base, idx)
    rows = []
    for m in sse.columns:
        boot = rmse(m, idx)
        per_well = np.sqrt(sse[m] / n[m])
        rows.append(dict(method=m, wells=len(sse), rows=int(n[m].sum()), rmse=rmse(m),
                         lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975),
                         diff=rmse(m) - rmse(base), diff_lo=np.quantile(boot - ref, 0.025),
                         diff_hi=np.quantile(boot - ref, 0.975), median_well=float(per_well.median())))  # fmt: skip
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
