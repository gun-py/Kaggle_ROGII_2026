from __future__ import annotations

import numpy as np


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def masked_rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("inf")
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def rmsd(u, v) -> float:
    """RMS difference over rows where both series are present."""
    ok = u.notna() & v.notna()
    return float(np.sqrt(((u[ok] - v[ok]) ** 2).mean()))
