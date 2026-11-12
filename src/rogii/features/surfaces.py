from __future__ import annotations

import numpy as np


def robust_slope(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def affine_cal(known_gr, typewell_at_known, min_pts: int = 20):
    ok = np.isfinite(known_gr) & np.isfinite(typewell_at_known)
    if ok.sum() < min_pts or np.std(typewell_at_known[ok]) < 1e-6:
        offset = float(np.nanmean(known_gr) - np.nanmean(typewell_at_known)) if ok.any() else 0.0
        return 1.0, offset
    a, b = np.polyfit(typewell_at_known[ok], known_gr[ok], 1)
    return float(a), float(b)


def segment_offsets(known_tvt, known_z, surface):
    """Datum offset of a known prefix against one surface: full, early, mid, late and tail-weighted."""
    b = known_tvt + known_z - surface
    n = len(b)
    full = float(np.median(b))
    late = float(np.median(b[max(0, n - 50) :])) if n >= 5 else full
    t1, t2 = n // 3, 2 * n // 3
    early = float(np.median(b[: max(1, t1)])) if t1 > 0 else full
    mid = float(np.median(b[t1 : max(t1 + 1, t2)])) if t2 > t1 else full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    return full, early, mid, late, float(np.dot(w, b))
