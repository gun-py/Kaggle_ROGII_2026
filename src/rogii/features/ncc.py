from __future__ import annotations

import numpy as np
import pandas as pd


def _smooth5(values) -> np.ndarray:
    return pd.Series(values).rolling(5, center=True, min_periods=1).mean().to_numpy().astype(np.float32)


def multi_scale_ncc(known_gr, known_tvt, hidden_gr, half_windows=(8, 15, 25), stride=3):
    """Slide normalised GR windows of the visible heel over the hidden section at several scales.

    Returns the per-scale (tvt, score) pairs and a softmax-weighted ensemble of the scales.
    """
    per_scale = []
    for half in half_windows:
        win = 2 * half + 1
        nk, nh = len(known_gr), len(hidden_gr)
        if nk < win + 1 or nh == 0:
            per_scale.append((np.full(nh, known_tvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg, hg = _smooth5(known_gr), _smooth5(hidden_gr)
        starts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(starts) == 0:
            per_scale.append((np.full(nh, known_tvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        cand = kg[starts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        cand = (cand - cand.mean(1, keepdims=True)) / (cand.std(1, keepdims=True) + 1e-6)
        padded = np.pad(hg, half, mode="edge")
        hid = padded[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        hid = (hid - hid.mean(1, keepdims=True)) / (hid.std(1, keepdims=True) + 1e-6)
        ncc = hid @ cand.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        per_scale.append((known_tvt[np.clip(starts[best] + half, 0, nk - 1)].astype(np.float32), score))

    tvts = np.stack([p[0] for p in per_scale], 1)
    scores = np.stack([p[1] for p in per_scale], 1)
    weight = np.exp(3.0 * scores)
    weight /= weight.sum(1, keepdims=True) + 1e-9
    return per_scale, (tvts * weight).sum(1).astype(np.float32)
