from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import torch
import torch.nn as nn

STRIDE = 8


class GRUHead(nn.Module):
    def __init__(self, n_features, hidden=96, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(out).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1, self.bn2 = nn.BatchNorm1d(channels), nn.BatchNorm1d(channels)
        self.drop, self.act = nn.Dropout(dropout), nn.ReLU()

    def forward(self, x):
        y = self.drop(self.act(self.bn1(self.c1(x))))
        y = self.drop(self.act(self.bn2(self.c2(y))))
        return self.act(x + y)


class TCN(nn.Module):
    def __init__(self, n_features, channels=64):
        super().__init__()
        self.inp = nn.Conv1d(n_features, channels, 1)
        self.blocks = nn.ModuleList([TCNBlock(channels, d) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Sequential(nn.Conv1d(channels, 32, 1), nn.ReLU(), nn.Conv1d(32, 1, 1))

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(1)


@lru_cache(maxsize=1)
def _load():
    path = resources.files("rogii.structural") / "weights" / "cv3_seq.pt"
    with path.open("rb") as f:
        art = torch.load(f, map_location="cpu", weights_only=False)
    n_features = int(art["NF"])
    gru, tcn = GRUHead(n_features), TCN(n_features)
    gru.load_state_dict(art["gru"])
    tcn.load_state_dict(art["tcn"])
    return gru.eval(), tcn.eval(), art["mu"], art["sd"]


def member_paths(P, base_drift, nearest, md_since, dz, ta, unknown):
    """GRU and TCN corrections on top of the linear drift `base_drift`, as full-length TVT paths.

    The nets see every STRIDE-th unknown row and are interpolated back, as they were trained.
    """
    gru, tcn, mu, sd = _load()
    n = len(md_since)
    pick = unknown[::STRIDE]
    if len(pick) == 0 or pick[-1] != unknown[-1]:
        pick = np.r_[pick, unknown[-1]]
    nb = (min(nearest, 3000.0) / 1000.0) if np.isfinite(nearest) else 3.0
    feats = np.column_stack([
        P[pick] - ta, base_drift[pick], dz[pick], md_since[pick] / 1000.0, np.full(len(pick), nb),
    ]).astype(np.float32)  # fmt: skip
    x = torch.tensor((feats - mu) / sd)[None]
    with torch.no_grad():
        g = gru(x, torch.tensor([len(pick)]))[0].numpy()
        t = tcn(x.transpose(1, 2))[0].numpy()
    rows = np.arange(n)
    return np.interp(rows, pick, base_drift[pick] + g), np.interp(rows, pick, base_drift[pick] + t)
