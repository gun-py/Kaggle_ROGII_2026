from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

SLUG = "rogii-wellbore-geology-prediction"
DATA = next(p for p in [Path("/kaggle/input") / SLUG,
                        Path("/kaggle/input/competitions") / SLUG,
                        Path(SLUG), Path(".")] if (p / "sample_submission.csv").exists())
OUT = Path("/kaggle/working")
DATA, OUT

NB_CUTOFF = 1200.0
NB_MAX = 8
WT_ANCHOR_MAX = 700.0
WT_CUTOFF = 1500.0
WT_SCALE = 300.0
PLANE_CUTOFF = 1100.0
PLANE_MAX = 10
DUP_TOL = 25.0
SUB = 4
GRID = np.arange(-460.0, 460.01, 0.5)

HMM_CFGS = [
    dict(sigma_gr=6.0, step_sd=0.15, prior_sd=25.0),
    dict(sigma_gr=9.0, step_sd=0.15, prior_sd=25.0),
    dict(sigma_gr=12.0, step_sd=0.15, prior_sd=25.0),
    dict(sigma_gr=6.0, step_sd=0.15, prior_sd=80.0),
    dict(sigma_gr=6.0, step_sd=0.05, prior_sd=25.0),
]
PLANE_CFG = dict(sigma_gr=6.0, step_sd=0.15, prior_sd=25.0)

# Column order: HMM_CFGS, plane-primed HMM, weighted-primed HMM,
#               plane, weighted neighbour, median neighbour, carry-forward.
#
# v4 used one weight vector per neighbour-distance bin. Those weights are now
# also a function of how far past the anchor the row sits: just past it every
# component is still pinned to the anchor and the mix is barely identifiable,
# while 3000 ft out the components have genuinely diverged and carry-forward
# earns a real shrinkage weight (0.03 -> 0.22 in the close-neighbour bin).
#
# Fitted on the simplex (non-negative, sum to 1) over 773 leave-one-out train
# wells. 5-fold GroupKFold over wells, 3.78M rows:
#     carry-forward                     15.910
#     v4 weights                        10.082
#     refit, same structure              9.961
#     refit, 4 MD bands                  9.849   <- this
# Free (unconstrained) least squares and six extra neighbour-gating columns
# were both tried and both overfit: 13.27 and 9.970 respectively.
NB_EDGES = [500.0, 1200.1]
MD_EDGES = [500.0, 1500.0, 3000.0]
WEIGHTS = np.array([
    [[0.0000, 0.0168, 0.0281, 0.0050, 0.0000, 0.0000, 0.0839, 0.2569, 0.2574, 0.3203, 0.0315],
     [0.0000, 0.1015, 0.1750, 0.0000, 0.0000, 0.0286, 0.2710, 0.0792, 0.3250, 0.0000, 0.0198],
     [0.0000, 0.1263, 0.1045, 0.0000, 0.0000, 0.0000, 0.3527, 0.0499, 0.2834, 0.0000, 0.0832],
     [0.0000, 0.0000, 0.0000, 0.0265, 0.0000, 0.1422, 0.1931, 0.0591, 0.3544, 0.0000, 0.2247]],
    [[0.0290, 0.0360, 0.0000, 0.0000, 0.0000, 0.0583, 0.0740, 0.1365, 0.0590, 0.5050, 0.1022],
     [0.0000, 0.1809, 0.1526, 0.0881, 0.0000, 0.1592, 0.1653, 0.0000, 0.0567, 0.0612, 0.1360],
     [0.2440, 0.0000, 0.1098, 0.2020, 0.0000, 0.0000, 0.0851, 0.0308, 0.0012, 0.1590, 0.1681],
     [0.3290, 0.0000, 0.0000, 0.1864, 0.0000, 0.0087, 0.0000, 0.0455, 0.1574, 0.0000, 0.2729]],
    [[0.0000, 0.0194, 0.1097, 0.0000, 0.0000, 0.1586, 0.0000, 0.2329, 0.0438, 0.2177, 0.2177],
     [0.1879, 0.0527, 0.0612, 0.1879, 0.0908, 0.0000, 0.0000, 0.1559, 0.0000, 0.1318, 0.1318],
     [0.2575, 0.0000, 0.0000, 0.2575, 0.0000, 0.0000, 0.0000, 0.1018, 0.1277, 0.1277, 0.1277],
     [0.0000, 0.2307, 0.0135, 0.0000, 0.0488, 0.0376, 0.0000, 0.0426, 0.2089, 0.2089, 0.2089]],
])


def load_well(wid, split):
    hw = pd.read_csv(DATA / split / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA / split / f"{wid}__typewell.csv").sort_values("TVT")
    return dict(
        md=hw.MD.values, x=hw.X.values, y=hw.Y.values, z=hw.Z.values, gr=hw.GR.values,
        known=hw.TVT_input.notna().values,
        tvt=hw.TVT.values if "TVT" in hw else hw.TVT_input.values,
        tw_tvt=tw.TVT.values, tw_gr=tw.GR.values)


def well_ids(split):
    return sorted(p.name.split("__")[0] for p in (DATA / split).glob("*__horizontal_well.csv"))


train_ids = well_ids("train")
test_ids = well_ids("test")
train = {w: load_well(w, "train") for w in train_ids}
len(train_ids), len(test_ids)


class Field:
    """Neighbour wells indexed for datum-free structural queries."""

    def __init__(self, pool):
        self.pool = list(pool)
        self.xy, self.s, self.tree = {}, {}, {}
        for w in self.pool:
            d = train[w]
            self.xy[w] = np.c_[d["x"], d["y"]]
            self.s[w] = d["tvt"] + d["z"]
            self.tree[w] = cKDTree(self.xy[w])
        c = np.array([[self.xy[w][:, 0].mean(), self.xy[w][:, 1].mean()] for w in self.pool])
        self.centroids = cKDTree(c)

    def near(self, xy, radius=15000.0):
        return [self.pool[k] for k in self.centroids.query_ball_point(xy.mean(0), radius)]

    def delta_s(self, xy, ia, drop):
        """Median over near wells of S(p) - S(anchor). NaN where unsupported."""
        rows, dist = [], []
        for j in self.near(xy):
            if j in drop:
                continue
            da, ja = self.tree[j].query(xy[ia])
            if da > NB_CUTOFF:
                continue
            dp, jp = self.tree[j].query(xy)
            med = float(np.median(dp))
            if med < DUP_TOL or med > NB_CUTOFF:
                continue
            ds = self.s[j][jp] - self.s[j][ja]
            ds[dp > 2 * NB_CUTOFF] = np.nan
            rows.append(ds)
            dist.append(max(da, med))
        if not rows:
            return np.full(len(xy), np.nan), np.inf
        keep = np.argsort(dist)[:NB_MAX]
        return np.nanmedian(np.array([rows[k] for k in keep]), axis=0), float(np.min(dist))

    def delta_s_weighted(self, xy, ia, drop):
        """Per-row inverse-distance blend, so partially overlapping wells still count."""
        num = np.zeros(len(xy))
        den = np.zeros(len(xy))
        for j in self.near(xy):
            if j in drop:
                continue
            da, ja = self.tree[j].query(xy[ia])
            if da > WT_ANCHOR_MAX:
                continue
            dp, jp = self.tree[j].query(xy)
            if np.median(dp) < DUP_TOL:
                continue
            w = 1.0 / (1.0 + (dp / WT_SCALE) ** 2)
            w[dp > WT_CUTOFF] = 0.0
            if w.sum() <= 0:
                continue
            num += (self.s[j][jp] - self.s[j][ja]) * w
            den += w
        return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)


def _basis(u, v):
    return np.column_stack([np.ones_like(u), u, u * u, v, u * v])


def plane_delta_s(field, d, drop, ridge=3.0, prefix_w=3.0, min_wells=2, clamp=400.0):
    """S_j = F(u,v) + k_j fitted over neighbours plus the query prefix (k_q = 0)."""
    known = d["known"]
    ia = np.where(known)[0][-1]
    xy = np.c_[d["x"], d["y"]]
    S = d["tvt"] + d["z"]

    vec = xy[-1] - xy[max(0, ia - 400)]
    n = np.hypot(*vec)
    if n < 1:
        return np.full(len(xy), np.nan)
    e1 = vec / n
    e2 = np.array([-e1[1], e1[0]])
    rel = xy - xy[ia]
    u, v = (rel @ e1) / 1000.0, (rel @ e2) / 1000.0

    qt = cKDTree(xy)
    nk = int(known.sum())
    rows = [(u[known], v[known], S[known], np.full(nk, -1), np.full(nk, prefix_w))]
    used = 0
    for j in field.near(xy):
        if j in drop:
            continue
        da, _ = field.tree[j].query(xy[ia])
        if da > PLANE_CUTOFF:
            continue
        jxy = field.xy[j]
        dp, _ = qt.query(jxy)
        m = dp < PLANE_CUTOFF
        if m.sum() < 50 or np.median(dp[m]) < DUP_TOL:
            continue
        r = jxy[m] - xy[ia]
        rows.append(((r @ e1) / 1000.0, (r @ e2) / 1000.0, field.s[j][m],
                     np.full(int(m.sum()), used),
                     np.full(int(m.sum()), 1.0 / (1.0 + da / 500.0))))
        used += 1
        if used >= PLANE_MAX:
            break
    if used < min_wells:
        return np.full(len(xy), np.nan)

    uu, vv, ss, gg, ww = (np.concatenate([r[i] for r in rows]) for i in range(5))
    B = _basis(uu, vv)
    nb = B.shape[1]
    dummies = np.zeros((len(uu), used))
    for k in range(used):
        dummies[gg == k, k] = 1.0
    A = np.hstack([B, dummies]) * np.sqrt(ww)[:, None]
    reg = ridge * np.eye(nb + used)
    reg[0, 0] = 1e-6
    coef = np.linalg.solve(A.T @ A + reg, A.T @ (ss * np.sqrt(ww)))
    f = _basis(u, v) @ coef[:nb]
    return np.clip(f - f[ia], -clamp, clamp)


def gr_calibration(gr, tvt, tw_tvt, tw_gr):
    """Affine map of horizontal GR onto the typewell GR scale, fitted on known rows."""
    m = np.isfinite(gr) & np.isfinite(tvt)
    if m.sum() < 50:
        return 1.0, 0.0
    a, b = np.polyfit(gr[m], np.interp(tvt[m], tw_tvt, tw_gr), 1)
    return (a, b) if 0.2 < a < 5.0 else (1.0, 0.0)


def hmm_align(d, prior, sigma_gr, step_sd, prior_sd, huber=2.5):
    """Forward-backward over dS, GR emission plus a soft structural prior."""
    known = d["known"]
    ia = np.where(known)[0][-1]
    ta, za = d["tvt"][ia], d["z"][ia]
    a, b = gr_calibration(d["gr"][known], d["tvt"][known], d["tw_tvt"], d["tw_gr"])

    ev = np.where(~known)[0]
    use = ev[::SUB]
    if use[-1] != ev[-1]:
        use = np.r_[use, ev[-1]]
    n, ns = len(use), len(GRID)

    cand = ta + GRID[None, :] - (d["z"][use] - za)[:, None]
    gtw = np.interp(cand, d["tw_tvt"], d["tw_gr"], left=np.nan, right=np.nan)
    r = np.abs((a * d["gr"][use] + b)[:, None] - gtw) / sigma_gr
    cost = np.where(r <= huber, 0.5 * r ** 2, huber * (r - 0.5 * huber))
    cost = np.where(np.isfinite(cost), cost, huber * 2.0)

    p = prior[use]
    ok = np.isfinite(p)
    if ok.any():
        cost[ok] += 0.5 * ((GRID[None, :] - p[ok][:, None]) / prior_sd) ** 2

    em = np.exp(-(cost - cost.min(1, keepdims=True)))
    em /= em.sum(1, keepdims=True)

    dmd = np.diff(d["md"][use], prepend=d["md"][use][0])
    sd = np.maximum(step_sd * np.sqrt(np.maximum(dmd, 1.0)), 0.25) / (GRID[1] - GRID[0])

    fwd, bwd = np.empty((n, ns)), np.empty((n, ns))
    q = np.exp(-0.5 * (GRID / 4.0) ** 2) * em[0]
    fwd[0] = q / q.sum()
    for t in range(1, n):
        q = gaussian_filter1d(fwd[t - 1], sd[t], mode="nearest") * em[t]
        fwd[t] = q / q.sum() if q.sum() > 0 else em[t]
    bwd[n - 1] = 1.0 / ns
    for t in range(n - 2, -1, -1):
        q = gaussian_filter1d(bwd[t + 1] * em[t + 1], sd[t + 1], mode="nearest")
        bwd[t] = q / q.sum() if q.sum() > 0 else 1.0 / ns

    post = fwd * bwd
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    s = np.interp(np.arange(len(d["md"])), use, post @ GRID)
    return ta + s - (d["z"] - za)


def components(field, d, drop):
    ia = np.where(d["known"])[0][-1]
    ta, za = d["tvt"][ia], d["z"][ia]
    xy = np.c_[d["x"], d["y"]]
    dz = d["z"] - za

    med, nd = field.delta_s(xy, ia, drop)
    wt = field.delta_s_weighted(xy, ia, drop)
    pln = plane_delta_s(field, d, drop)
    both = np.isfinite(med) & np.isfinite(pln)
    comb = np.where(both, 0.5 * (np.nan_to_num(med) + np.nan_to_num(pln)),
                    np.where(np.isfinite(pln), pln, med))

    carry = np.full(len(xy), ta)
    cols = [hmm_align(d, med, **c) for c in HMM_CFGS]
    cols.append(hmm_align(d, comb, **PLANE_CFG))
    cols.append(hmm_align(d, wt, **PLANE_CFG))
    cols.append(np.where(np.isfinite(pln), ta + pln - dz, carry))
    cols.append(np.where(np.isfinite(wt), ta + wt - dz, carry))
    cols.append(np.where(np.isfinite(med), ta + med - dz, carry))
    cols.append(carry)

    P = np.column_stack(cols)
    bad = ~np.isfinite(P)
    if bad.any():                       # a non-finite column must not poison the row
        P[bad] = np.repeat(carry[:, None], P.shape[1], 1)[bad]
    return P, nd


def predict(field, d, drop):
    P, nd = components(field, d, drop)
    ia = np.where(d["known"])[0][-1]
    b = int(np.digitize(nd, NB_EDGES))
    band = np.digitize(d["md"] - d["md"][ia], MD_EDGES)
    return np.einsum("ij,ij->i", P, WEIGHTS[b][band])


# ---- blend -------------------------------------------------------------------
# W_STACK  : the pipeline above (LB 9.349)
# W_STRUCT : neighbour-structure + GR alignment (LB 10.963 at v4 weights)
# W_PKG    : saved model package, trees only, TCN dropped (labelled LB 7.335)
# Weights are renormalised.
#
# THE PACKAGE IS OFF BY DEFAULT. v4 shipped (0.30, 0.20, 0.50) and scored 9.162,
# which those three component scores cannot produce: a convex blend obeys
#     RMSE <= 0.30*9.349 + 0.20*10.963 + 0.50*7.335 = 8.665,
# and the tighter Jensen bound is 8.784. Both are below 9.162, so at least one
# component was far worse in that run than its label. Solving the identity, with
# the stack and struct taken as labelled, puts the package at RMSE >= 16.8 -- not
# 7.335. (The reverse assumption needs the stack at >= 27.0, an upstream
# collapse.) A silently-NaN package cannot be the explanation: build_features
# raises on any missing sample id, so it would have crashed rather than degrade.
#
# So the package's 7.335 label is not reproducible and 50% weight on it cost
# ~1.8 ft. Until the run diagnostics below identify which component broke, the
# only internally consistent configuration is the measured two-way blend.
#
# Two-way optimum from the LB-solved rho_AB = 0.0411, with struct retrained
# (expected LB ~10.72 from OOF 9.849 plus struct's measured +0.87 OOF->LB gap):
#     stack 0.57 / struct 0.43 -> ~7.19   (v3 measured 7.319 at 0.65/0.35)
W_STACK, W_STRUCT, W_PKG = 0.57, 0.43, 0.00

PKG_ROOTS = ["/kaggle/input/rogii-model-package",
             "/kaggle/input/datasets/pilkwang/rogii-model-package"]

existing = pd.read_csv(OUT / "submission.csv").set_index("id")["tvt"]
field = Field(train_ids)

parts = []
for wid in test_ids:
    d = load_well(wid, "test")
    ev = np.where(~d["known"])[0]
    p = predict(field, d, drop={wid})
    parts.append(pd.DataFrame({"id": [f"{wid}_{i}" for i in ev], "tvt": p[ev]}))
struct = pd.concat(parts, ignore_index=True).set_index("id")["tvt"].reindex(existing.index)


def package_trees():
    """Saved package with the TCN dropped: 44 features are all-NaN at test and
    the TCN cannot absorb that, while the trees handle NaN natively.

    Weights and postprocess are the shipped ones. Refitting them for the
    tree-only path was tried and loses out-of-fold (10.971 -> 11.007 over 773
    wells), and alpha genuinely wants to stay at 1.05 even without the TCN.
    """
    import sys, json as js
    from scipy.signal import savgol_filter
    root = next((Path(p) for p in PKG_ROOTS if (Path(p) / "metadata").exists()), None)
    if root is None:
        return None
    sys.path.insert(0, str(root / "feature_builders"))
    import build_features as bf
    man = js.loads((root / "metadata" / "model_package_manifest.json").read_text())
    cfg = js.loads((root / man["blend_config"]).read_text())
    frame = bf.build_features(data_dir=DATA, package_root=root, manifest=man)
    cols = js.loads((root / man["feature_columns"]).read_text())
    cols = cols["drift_ncc_v1"] if isinstance(cols, dict) else cols

    entries = [e for e in man["models"] if e["model_type"] != "torch_tcn"]
    total = sum(float(cfg["weights"].get(e["prediction_column"], 0.0)) for e in entries)
    delta = np.zeros(len(frame))
    for e in entries:
        w = float(cfg["weights"].get(e["prediction_column"], 0.0)) / total
        if w <= 0:
            continue
        path = root / e["path"]
        kind = e["model_type"]
        if kind == "xgboost_json":
            import xgboost as xgb
            m = xgb.Booster(); m.load_model(str(path))
        elif kind == "lightgbm_booster":
            import lightgbm as lgb
            m = lgb.Booster(model_file=str(path))
        elif kind == "catboost_cbm":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(); m.load_model(str(path))
        else:
            import joblib
            m = joblib.load(path)
        X = frame[cols].replace([np.inf, -np.inf], np.nan)
        if e.get("fillna") is not None:
            X = X.fillna(float(e["fillna"]))
        if kind == "xgboost_json":
            import xgboost as xgb
            p = m.predict(xgb.DMatrix(X.to_numpy(np.float32)))
        else:
            p = m.predict(X)
        delta += w * np.asarray(p, dtype=float)

    post = cfg.get("postprocess", {})
    delta *= float(post.get("alpha", 1.0))
    tvt = frame["last_known_TVT"].to_numpy(float) + delta
    win = int(post.get("savgol_window", 0) or 0)
    if win > 2:
        win += 1 - win % 2
        poly = int(post.get("savgol_poly", 2))
        df = pd.DataFrame({"w": frame["well_id"].astype(str).to_numpy(),
                           "o": pd.to_numeric(frame["row_index"]).to_numpy(float),
                           "t": tvt, "p": np.arange(len(tvt))})
        for _, g in df.groupby("w", sort=False):
            g = g.sort_values("o")
            if len(g) < max(win, poly + 2):
                continue
            k = min(win, len(g) if len(g) % 2 else len(g) - 1)
            tvt[g["p"].to_numpy()] = savgol_filter(g["t"].to_numpy(float), k,
                                                   min(poly, k - 1), mode="interp")
    return pd.Series(tvt, index=frame["id"].astype(str)).reindex(existing.index)


pkg = package_trees() if W_PKG > 0 else None
if pkg is None and W_PKG > 0:
    print("!" * 70)
    print("MODEL PACKAGE NOT FOUND - attach the rogii-model-package dataset.")
    print("Falling back to a two-way blend; expect ~7.3 instead of ~5-7.")
    print("!" * 70)
    W_PKG = 0.0

# Diagnostics that would have caught the v4 blow-up. Each component's disagreement
# with the stack is predictable from the LB scores, so a component that has
# drifted from its label shows up here before it costs a submission.
#   expected RMSD(stack, struct) = sqrt(9.349^2 + 10.963^2 - 2*0.0411*9.349*10.963) = 14.1
#   expected RMSD(stack, pkg) with pkg at 7.335 ~ 10 (rho 0.3); at 16.8 it is ~17
def _rmsd(u, v):
    m = u.notna() & v.notna()
    return float(np.sqrt(((u[m] - v[m]) ** 2).mean()))


print("RMSD(stack, struct): %6.3f   (expect ~14.1 if both match their LB labels)"
      % _rmsd(existing, struct))
if pkg is not None:
    cover = float(pkg.notna().mean())
    print("pkg coverage:        %6.4f   (%d of %d rows)"
          % (cover, int(pkg.notna().sum()), len(pkg)))
    print("RMSD(stack, pkg):    %6.3f   (expect ~10 if pkg is really 7.335; ~17 if it is ~16.8)"
          % _rmsd(existing, pkg))
    if cover < 0.99:
        print("!" * 70)
        print("PACKAGE COVERS ONLY %.1f%% OF ROWS; the rest fall back to the stack," % (100 * cover))
        print("so any LB score attributed to the package is really a blend.")
        print("!" * 70)

s = W_STACK + W_STRUCT + W_PKG
out = (W_STACK / s) * existing + (W_STRUCT / s) * struct
if W_PKG > 0:
    out = out + (W_PKG / s) * pkg
out = out.fillna(existing)

print("weights stack/struct/pkg: %.2f %.2f %.2f" % (W_STACK / s, W_STRUCT / s, W_PKG / s))
print("struct NaN before fill:  %d of %d" % (int(struct.isna().sum()), len(struct)))

assert len(out) == len(existing), "row count changed"
assert out.notna().all(), "submission still contains NaN after fillna"
assert np.isfinite(out.to_numpy()).all(), "submission contains inf"

out.rename("tvt").reset_index().to_csv(OUT / "submission.csv", index=False)
print("wrote submission.csv rows=%d mean=%.2f" % (len(out), out.mean()))
