from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def resample(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N)
    na = np.empty(N)
    ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def beam_kernel(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr)
    nt = len(tw_gr)
    MAX = BS * 6
    bidx = np.zeros(BS, np.int64)
    bidx[0] = si
    bcost = np.full(BS, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64)
    hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64)
    cC = np.full(MAX, 1e30)
    cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


@njit(cache=True)
def pf_ancc_kernel(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N)
    rate = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = resample(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va**0.5
        pm = md_v[i]
    return pts, std_


@njit(cache=True)
def pf_z_kernel(
    md_v,
    z_v,
    gr_v,
    gr_sm_v,
    gg_p,
    gg_s,
    vmin,
    step,
    gs,
    ip,
    iv,
    beta,
    icpt,
    zsig,
    N,
    MOM,
    VN,
    PN,
    GR_WT,
    RP,
    RV,
    RESAMP,
):
    pos = np.empty(N)
    vel = np.empty(N)
    w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v))
    std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0
    pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    lsm = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * lsm
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz
            ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = resample(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm
        va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va**0.5
        pm = md_v[i]
        pz = z_v[i]
    return pts, std_


@njit(cache=True, nogil=True)
def pf_lik_kernel(
    md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base, MOM, VN, PN, RP, RR, RESAMP, init_spr
):
    n = len(md_v)
    preds = np.empty((n_seeds, n))
    liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N)
        rate = np.empty(N)
        w = np.ones(N) / N
        for j in range(N):
            pos[j] = ls + init_spr * np.random.randn()
            rate[j] = ir + 0.01 * np.random.randn()
        log_lik = 0.0
        prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0:
                dm = 1.0
            for j in range(N):
                rate[j] = MOM * rate[j] + VN * np.random.randn()
                pos[j] += rate[j] * dm + PN * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.0:
                    tvt_j = vmin - 100.0
                if tvt_j > tmax + 100.0:
                    tvt_j = tmax + 100.0
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                dd = d * d
                if dd > 600.0:
                    dd = 600.0
                lk = np.exp(-0.5 * dd)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += w[j] * lk
                w[j] = w[j] * lk
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N):
                ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
            neff = 0.0
            for j in range(N):
                neff += w[j] * w[j]
            neff = 1.0 / neff
            if neff < RESAMP * N:
                cum = np.empty(N)
                c = 0.0
                for j in range(N):
                    c += w[j]
                    cum[j] = c
                u0 = np.random.uniform(0.0, 1.0 / N)
                newpos = np.empty(N)
                newrate = np.empty(N)
                ci = 0
                for j in range(N):
                    u = u0 + j / N
                    while ci < N - 1 and cum[ci] < u:
                        ci += 1
                    newpos[j] = pos[ci] + RP * np.random.randn()
                    newrate[j] = rate[ci] + RR * np.random.randn()
                for j in range(N):
                    pos[j] = newpos[j]
                    rate[j] = newrate[j]
                    w[j] = 1.0 / N
            est = 0.0
            for j in range(N):
                est += w[j] * (pos[j] - z_v[i])
            preds[s, i] = est
            prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


def warmup() -> None:
    """Trigger JIT compilation on tiny inputs so later timings are representative."""
    md = np.linspace(1, 50, 20)
    z = np.zeros(20)
    gr = np.full(20, 50.0)
    grid = np.linspace(45, 55, 100)
    pf_ancc_kernel(md, z, gr, grid, 45.0, 0.1, 20.0, 50.0, 0.0, 8, 0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
    pf_z_kernel(
        md,
        z,
        gr,
        gr,
        grid,
        grid,
        45.0,
        0.1,
        20.0,
        50.0,
        0.0,
        -1.0,
        0.0,
        0.1,
        8,
        0.993,
        0.005,
        0.01,
        0.3,
        0.2,
        0.003,
        0.5,
    )
    beam_kernel(np.random.randn(30), np.random.randn(50), 25, 8, 15.0, 100.0)
    pf_lik_kernel(
        md, z, gr, grid, 45.0, 0.1, 20.0, 50.0, 0.0, 64, 4, 0, 0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5
    )
