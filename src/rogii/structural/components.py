from __future__ import annotations

import numpy as np

from .field import Field, plane_delta_s
from .hmm import HMM_CFGS, PLANE_CFG, hmm_align

N_COMPONENTS = len(HMM_CFGS) + 6


def components(field: Field, d: dict, drop, sanitize: bool = True):
    """Eleven TVT paths for the unknown rows and the distance to the nearest supporting neighbour.

    Order: one HMM per HMM_CFGS entry, plane-primed HMM, weighted-primed HMM, plane,
    weighted neighbour, median neighbour, carry-forward. With `sanitize`, non-finite
    entries fall back to carry-forward instead of poisoning the row.
    """
    ia = np.where(d["known"])[0][-1]
    ta, za = d["tvt"][ia], d["z"][ia]
    xy = np.c_[d["x"], d["y"]]
    dz = d["z"] - za

    med, nearest = field.delta_s(xy, ia, drop)
    weighted = field.delta_s_weighted(xy, ia, drop)
    plane = plane_delta_s(field, d, drop)
    both = np.isfinite(med) & np.isfinite(plane)
    combined = np.where(
        both, 0.5 * (np.nan_to_num(med) + np.nan_to_num(plane)), np.where(np.isfinite(plane), plane, med)
    )

    carry = np.full(len(xy), ta)
    cols = [hmm_align(d, med, **cfg) for cfg in HMM_CFGS]
    cols.append(hmm_align(d, combined, **PLANE_CFG))
    cols.append(hmm_align(d, weighted, **PLANE_CFG))
    cols.append(np.where(np.isfinite(plane), ta + plane - dz, carry))
    cols.append(np.where(np.isfinite(weighted), ta + weighted - dz, carry))
    cols.append(np.where(np.isfinite(med), ta + med - dz, carry))
    cols.append(carry)

    P = np.column_stack(cols)
    if sanitize:
        bad = ~np.isfinite(P)
        if bad.any():
            P[bad] = np.repeat(carry[:, None], P.shape[1], 1)[bad]
    return P, nearest
