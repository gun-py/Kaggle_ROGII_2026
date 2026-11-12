from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

BATCH = 16
PATIENCE = 10


class GeoSeqNet(nn.Module):
    def __init__(self, n_features, hidden, layers, dropout):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)  # fmt: skip
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x, lengths):
        h = self.inp(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(out).squeeze(-1)


def train(samples: dict, cfg: dict, folds: int, seeds: int, epochs: int, verbose: bool = True):
    """Out-of-fold drift predictions per well and a test-time predictor averaging every fold/seed model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wells = list(samples)
    n_features = samples[wells[0]]["feats"].shape[1]
    stacked = np.vstack([samples[w]["feats"] for w in wells])
    mu, sd = stacked.mean(0), stacked.std(0) + 1e-6

    def scale(f):
        return ((f - mu) / sd).astype(np.float32)

    def batches(names, shuffle):
        order = np.arange(len(names))
        if shuffle:
            np.random.shuffle(order)
        for i in range(0, len(order), BATCH):
            chunk = [names[j] for j in order[i : i + BATCH]]
            lengths = [len(samples[w]["feats"]) for w in chunk]
            longest = max(lengths)
            X = np.zeros((len(chunk), longest, n_features), np.float32)
            T = np.zeros((len(chunk), longest), np.float32)
            M = np.zeros((len(chunk), longest), np.float32)
            for k, w in enumerate(chunk):
                s, n = samples[w], len(samples[w]["feats"])
                X[k, :n], T[k, :n], M[k, :n] = (
                    scale(s["feats"]),
                    s["target"],
                    s["evalmask"].astype(np.float32),
                )
            yield (torch.from_numpy(X).to(device), torch.from_numpy(T).to(device),
                   torch.from_numpy(M).to(device), torch.tensor(lengths))  # fmt: skip

    def predict_one(model, s):
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(scale(s["feats"]))[None].to(device)
            return model(x, torch.tensor([len(s["feats"])]))[0].cpu().numpy()

    def fit_one(fit_wells, val_wells, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GeoSeqNet(n_features, cfg["hid"], cfg["layers"], cfg["drop"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
        loss_fn = nn.SmoothL1Loss(beta=5.0, reduction="none")
        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            model.train()
            for X, T, M, L in batches(fit_wells, True):
                opt.zero_grad()
                loss = (loss_fn(model(X, L), T) * M).sum() / M.sum().clamp_min(1.0)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sse, count = 0.0, 0
            for w in val_wells:
                s = samples[w]
                m = s["evalmask"]
                sse += float(((predict_one(model, s)[m] - s["target"][m]) ** 2).sum())
                count += int(m.sum())
            score = np.sqrt(sse / max(count, 1))
            sched.step(score)
            if score < best - 1e-4:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    oof, models = {}, []
    for fold, (tr, va) in enumerate(KFold(folds, shuffle=True, random_state=42).split(wells)):
        train_wells, oof_wells = [wells[i] for i in tr], [wells[i] for i in va]
        cut = max(1, int(0.10 * len(train_wells)))
        val_wells, fit_wells = train_wells[:cut], train_wells[cut:]
        acc = {w: [] for w in oof_wells}
        for seed in range(seeds):
            model = fit_one(fit_wells, val_wells, seed)
            models.append(model)
            for w in oof_wells:
                acc[w].append(predict_one(model, samples[w]))
        for w in oof_wells:
            oof[w] = np.mean(acc[w], axis=0)
        if verbose:
            print(f"    fold {fold + 1}/{folds} done ({len(oof_wells)} oof wells)", flush=True)

    def predict(test_samples):
        return {w: np.mean([predict_one(m, s) for m in models], axis=0) for w, s in test_samples.items()}

    return oof, predict
