from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from rogii.attention.config import TrailConfig
from rogii.attention.data import NEIGHBOUR_FEATURES, TOKEN_FEATURES, Sample, WellSet, rows_from_tokens
from rogii.attention.model import Trail, trail_loss
from rogii.attention.synthetic import SyntheticWells


def pick_device() -> torch.device:
    if os.environ.get("ROGII_DEVICE"):
        return torch.device(os.environ["ROGII_DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


SKIPPED = [0]


def collate(samples: Sequence[Sample], device) -> dict:
    B = len(samples)
    L = max(len(s) for s in samples)
    G = max(len(s.tw) for s in samples)
    M = samples[0].nb.shape[1]
    feats = np.zeros((B, L, TOKEN_FEATURES), np.float32)
    pos = np.zeros((B, L), np.float32)
    pad = np.ones((B, L), bool)
    hidden = np.zeros((B, L), bool)
    drift = np.zeros((B, L), np.float32)
    target = np.zeros((B, L), np.float32)
    tmask = np.zeros((B, L), bool)
    tw = np.zeros((B, G, 2), np.float32)
    tw_pad = np.ones((B, G), bool)
    g_hi = np.zeros(B, np.int64)
    u = np.zeros(B, np.float32)
    nb = np.zeros((B, L, M, NEIGHBOUR_FEATURES), np.float32)
    nb_mask = np.zeros((B, L, M), bool)
    for i, s in enumerate(samples):
        n, g = len(s), len(s.tw)
        ok = s.hidden & np.isfinite(s.target)
        feats[i, :n], pos[i, :n], pad[i, :n] = s.feats, s.pos, False
        hidden[i, :n], drift[i, :n] = s.hidden, s.drift
        target[i, :n], tmask[i, :n] = np.where(ok, s.target, 0.0), ok
        tw[i, :g], tw_pad[i, :g], g_hi[i], u[i] = s.tw, False, g - 1, s.u_anchor
        nb[i, :n], nb_mask[i, :n] = s.nb, s.nb_mask
    arrays = dict(feats=feats, pos=pos, pad=pad, hidden=hidden, drift=drift, target=target, tmask=tmask,
                  tw=tw, tw_pad=tw_pad, g_hi=g_hi, u_anchor=u, nb=nb, nb_mask=nb_mask)  # fmt: skip
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}


def packed(samples: list[Sample], budget: int, rng: np.random.Generator | None = None) -> list[list[Sample]]:
    """Group samples of similar length so each batch stays under `budget` padded tokens."""
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]))
    batches, current, longest = [], [], 0
    for i in order:
        n = len(samples[i])
        if current and max(longest, n) * (len(current) + 1) > budget:
            batches.append(current)
            current, longest = [], 0
        current.append(samples[i])
        longest = max(longest, n)
    if current:
        batches.append(current)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def _optimiser(model, cfg: TrailConfig, total_steps: int):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(1, min(500, total_steps // 20))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def _step(model, opt, sched, batch, cfg):
    model.train()
    paths, info = model(batch)
    loss, mse = trail_loss(paths, info, batch, cfg)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    sched.step()
    if not torch.isfinite(norm):
        SKIPPED[0] += 1
        return float(loss.detach()), float(mse)
    opt.step()
    return float(loss.detach()), float(mse)


def pretrain(cfg: TrailConfig, synth: SyntheticWells, device, steps: int | None = None,
             log: Callable[[str], None] = print, per_draw: int = 24) -> Trail:  # fmt: skip
    """Fit TRAIL on synthetic wells with the neighbour stream left empty."""
    steps = steps or cfg.pretrain_steps
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = Trail(cfg).to(device)
    opt, sched = _optimiser(model, cfg, steps)
    step, t0, running = 0, time.time(), []
    while step < steps:
        draw = [synth.sample(rng, cfg.max_visible_tokens) for _ in range(per_draw)]
        for chunk in packed(draw, cfg.token_budget, rng):
            _, mse = _step(model, opt, sched, collate(chunk, device), cfg)
            running.append(mse)
            step += 1
            if step % 100 == 0:
                log(
                    f"pretrain step {step}/{steps}  rmse {np.sqrt(np.mean(running)):.2f} ft  "
                    f"skipped {SKIPPED[0]}  {time.time() - t0:.0f}s"
                )
                running = []
            if step >= steps:
                break
    return model


@torch.no_grad()
def predict(model: Trail, samples: list[Sample], device, budget: int, iterations: int | None = None):
    """Final-iteration drift per sample, in input order."""
    model.eval()
    out = [None] * len(samples)
    index = {id(s): i for i, s in enumerate(samples)}
    for chunk in packed(samples, budget):
        paths, _ = model(collate(chunk, device), iterations)
        final = paths[-1].float().cpu().numpy()
        for row, s in enumerate(chunk):
            out[index[id(s)]] = final[row, : len(s)]
    return out


def score_rows(ws: WellSet, samples: list[Sample], drifts) -> dict[str, tuple[float, int]]:
    """Per-well SSE and row count over the rows past each sample's anchor."""
    scores = {}
    for s, drift in zip(samples, drifts, strict=True):
        d = ws.wells[s.wid]
        pred = rows_from_tokens(s, drift, d["z"])
        truth = d["tvt"][s.ia + 1 :]
        ok = np.isfinite(truth)
        scores[s.wid] = (float(((pred[ok] - truth[ok]) ** 2).sum()), int(ok.sum()))
    return scores


def pooled_rmse(scores) -> float:
    sse = sum(v[0] for v in scores.values())
    n = sum(v[1] for v in scores.values())
    return float(np.sqrt(sse / max(n, 1)))


def finetune(cfg: TrailConfig, ws: WellSet, train_ids: list[str], val_ids: list[str], device,
             init: Trail | dict | None = None, log: Callable[[str], None] = print,
             epochs: int | None = None) -> tuple[Trail, list[dict]]:  # fmt: skip
    """Train on real wells with random anchors; keep the weights with the best inner-validation RMSE.

    Neighbour pools hold training wells only, so no validation or held-out label reaches the weights.
    """
    epochs = epochs or cfg.finetune_epochs
    rng = np.random.default_rng(cfg.seed + 1)
    torch.manual_seed(cfg.seed + 1)
    model = Trail(cfg).to(device)
    if isinstance(init, Trail):
        model.load_state_dict(init.state_dict(), strict=False)
    elif isinstance(init, dict):
        model.load_state_dict(init, strict=False)
    pool = frozenset(train_ids)
    val = [ws.sample(w, pool - {w}, None, cfg.max_visible_tokens, cfg.use_neighbours) for w in val_ids]
    steps_per_epoch = len(
        packed([ws.sample(w, pool, None, cfg.max_visible_tokens, False) for w in train_ids], cfg.token_budget)
    )
    opt, sched = _optimiser(model, cfg, epochs * steps_per_epoch)
    best, best_state, history, stale = np.inf, None, [], 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        draw = [
            ws.sample(w, pool - {w}, ws.random_anchor(w, rng, cfg.real_anchor_prob), cfg.max_visible_tokens, cfg.use_neighbours)
            for w in train_ids
        ]  # fmt: skip
        mses = [
            _step(model, opt, sched, collate(chunk, device), cfg)[1]
            for chunk in packed(draw, cfg.token_budget, rng)
        ]
        val_rmse = pooled_rmse(score_rows(ws, val, predict(model, val, device, cfg.token_budget)))
        history.append({"epoch": epoch, "train_rmse": float(np.sqrt(np.mean(mses))), "val_rmse": val_rmse,
                        "seconds": time.time() - t0})  # fmt: skip
        log(
            f"epoch {epoch}/{epochs}  train {history[-1]['train_rmse']:.2f}  val {val_rmse:.2f} ft  "
            f"skipped {SKIPPED[0]}  {time.time() - t0:.0f}s"
        )
        if val_rmse < best - 1e-3:
            best, best_state, stale = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save(model: Trail, path: Path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "config": model.cfg.to_dict(), "meta": meta}, path)


def load(path: Path, device=None) -> Trail:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = Trail(TrailConfig(**blob["config"]))
    model.load_state_dict(blob["state"])
    return model.to(device or pick_device())
