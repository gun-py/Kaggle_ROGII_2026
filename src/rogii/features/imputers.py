from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rogii.geology.contacts import FORMATIONS

PLANE_K = 10
DENSE_SAMPLES_PER_WELL = 60
DENSE_K = 20


def _safe_scale(xy: np.ndarray) -> np.ndarray:
    sd = xy.std(0)
    return np.where(sd < 1e-3, 1.0, sd)


class FormationPlaneKNN:
    """Locally weighted plane fit of every formation surface through the nearest train wells."""

    def __init__(self, well_ids, data_dir: Path):
        rows = []
        for wid in well_ids:
            try:
                df = pd.read_csv(
                    Path(data_dir) / f"{wid}__horizontal_well.csv", usecols=["X", "Y", *FORMATIONS]
                ).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            row.update({f"{c}_m": float(df[c].median()) for c in FORMATIONS})
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.index_of = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = _safe_scale(xy)
        self.tree = cKDTree(xy / self.scale)
        self.xs = self.df["x"].to_numpy()
        self.ys = self.df["y"].to_numpy()
        self.surfaces = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_query, self_wid=None, k: int = PLANE_K):
        q = xy_query / self.scale
        n_fetch = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=n_fetch, workers=-1)
        if self_wid in self.index_of:
            dist = np.where(idx == self.index_of[self_wid], np.inf, dist)
        order = np.argpartition(dist, min(k - 1, n_fetch - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, order, 1)
        ik = np.take_along_axis(idx, order, 1)
        valid = np.isfinite(dk)
        w = np.where(valid, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn, yn, fn = self.xs[ik], self.ys[ik], self.surfaces[ik]
        wx, wy = w * xn, w * yn

        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1)
        A[:, 0, 1] = (wx * yn).sum(1)
        A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(1)
        A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = w.sum(1)
        for d in range(3):
            A[:, d, d] += 1e-9
        rhs = np.stack(
            [(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1
        )
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except np.linalg.LinAlgError:
                    pass

        pred = (
            xy_query[:, 0][:, None] * coef[:, 0, :] + xy_query[:, 1][:, None] * coef[:, 1, :] + coef[:, 2, :]
        ).astype(np.float32)
        pred[~valid.any(1)] = self.surfaces.mean(0)
        return pred, np.where(valid, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    """Inverse-distance ANCC surface from a thinned point cloud of the train wells."""

    def __init__(self, well_ids, data_dir: Path, samples_per_well: int = DENSE_SAMPLES_PER_WELL):
        xs, ys, ancc, owners = [], [], [], []
        for wid in well_ids:
            try:
                df = pd.read_csv(
                    Path(data_dir) / f"{wid}__horizontal_well.csv", usecols=["X", "Y", "ANCC"]
                ).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            pick = np.linspace(0, len(df) - 1, min(samples_per_well, len(df)), dtype=int)
            part = df.iloc[pick]
            xs.append(part["X"].values)
            ys.append(part["Y"].values)
            ancc.append(part["ANCC"].values)
            owners.extend([wid] * len(part))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(ancc).astype(np.float32)
        self.owners = np.array(owners)
        self.scale = _safe_scale(self.xy)
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_query, self_wid=None, k: int = DENSE_K, n_fetch: int = 5000):
        xy_query = np.atleast_2d(xy_query)
        q = xy_query / self.scale
        n_fetch = min(n_fetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=n_fetch, workers=-1)
        if self_wid:
            dist = np.where(self.owners[idx] == self_wid, np.inf, dist)
        order = np.argpartition(dist, min(k - 1, n_fetch - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, order, 1)
        ik = np.take_along_axis(idx, order, 1)
        valid = np.isfinite(dk)
        w = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
        total = w.sum(1)
        safe = np.where(total < 1e-9, 1.0, total)
        vals = self.ancc[ik]
        mean = (vals * w).sum(1) / safe
        mean = np.where(total < 1e-9, float(self.ancc.mean()), mean)
        var = ((vals - mean[:, None]) ** 2 * w).sum(1) / safe
        nearest = np.where(valid, dk, np.inf).min(1)
        return (
            mean.astype(np.float32),
            np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
            nearest.astype(np.float32),
        )


class Imputers:
    def __init__(self, well_ids, data_dir: Path):
        self.formation = FormationPlaneKNN(well_ids, data_dir)
        self.dense = DenseANCCImputer(well_ids, data_dir)
