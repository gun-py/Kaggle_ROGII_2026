from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from rogii.config import PostprocessConfig
from rogii.features.builder import add_likpf_features, build_features, build_likpf
from rogii.features.imputers import Imputers
from rogii.io import list_wells, read_sample
from rogii.metrics import rmse
from rogii.postprocess.smoothing import fade_in, smooth_groups

N_SPLITS = 5


def detect_device(preference: str = "auto") -> str:
    if preference in {"cpu", "gpu"}:
        return preference
    try:
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
            return "gpu"
    except OSError:
        pass
    return "cpu"


def lgb_configs(device: str, fast: bool = False) -> list[dict]:
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if device == "gpu":
        base.update(device_type="gpu", gpu_use_dp=False)
    n = 600 if fast else 5000
    slow = dict(num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1, colsample_bytree=0.393,
                reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24, learning_rate=0.0093, n_estimators=min(2 * n, 10000))  # fmt: skip
    return [
        dict(
            **base,
            num_leaves=255,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            reg_alpha=0.05,
            learning_rate=0.03,
            n_estimators=n,
            seed=123,
        ),  # fmt: skip
        dict(**base, **slow, random_state=0),
        dict(**base, **slow, random_state=29),
    ]


def cb_configs(device: str, fast: bool = False) -> list[dict]:
    base = dict(depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE",
                task_type="GPU" if device == "gpu" else "CPU", od_type="Iter", od_wait=300, verbose=0,
                iterations=800 if fast else 8000)  # fmt: skip
    return [dict(base, learning_rate=0.02, random_seed=7), dict(base, learning_rate=0.03, random_seed=123)]


def train_stack(train_df, test_df, features, device="cpu", fast=False):
    """Out-of-fold LightGBM + CatBoost models stacked by a positive ridge; returns (oof, test) drifts."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    X = train_df[features].values.astype(np.float32)
    y = train_df["target"].values.astype(np.float32)
    groups = train_df["well"].values
    X_test = test_df[features].values.astype(np.float32)
    cv = GroupKFold(N_SPLITS)
    oof_cols, test_cols = {}, {}

    def run(name, make, fit_kw, is_lgb):
        oof, test = np.zeros(len(train_df)), np.zeros(len(test_df))
        for tr, va in cv.split(X, y, groups=groups):
            model = make()
            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **fit_kw)
            if is_lgb:
                best = model.best_iteration_
                oof[va] = model.predict(X[va], num_iteration=best)
                test += model.predict(X_test, num_iteration=best) / N_SPLITS
            else:
                oof[va] = model.predict(X[va])
                test += model.predict(X_test) / N_SPLITS
        oof_cols[name], test_cols[name] = oof, test
        print(f"  {name}: oof rmse {rmse(y, oof):.4f}", flush=True)

    lgb_fit = dict(eval_metric="rmse", callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
    for i, params in enumerate(lgb_configs(device, fast)):
        run(f"lgb{i}", lambda p=params: LGBMRegressor(**p), lgb_fit, True)
    for i, params in enumerate(cb_configs(device, fast)):
        run(
            f"cb{i}",
            lambda p=params: CatBoostRegressor(**p),
            dict(early_stopping_rounds=250, use_best_model=True),
            False,
        )

    oof_df, test_df_ = pd.DataFrame(oof_cols), pd.DataFrame(test_cols)
    ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True)
    meta = np.zeros(len(train_df))
    for tr, va in cv.split(oof_df.values, y, groups=groups):
        ridge.fit(oof_df.values[tr], y[tr])
        meta[va] = ridge.predict(oof_df.values[va])
    ridge.fit(oof_df.values, y)
    print(f"  ridge stack: oof rmse {rmse(y, meta):.4f}")
    return meta, ridge.predict(test_df_.values)


def make_prediction(df: pd.DataFrame, model_delta, cfg: PostprocessConfig) -> np.ndarray:
    """Blend the learned drift with the likelihood-weighted PF, fade it in, then smooth per well."""
    last = df["last_known_tvt"].values.astype(float)
    pf_delta = df["pf_ancc"].values.astype(float) - last
    lik_delta = df[f"likpf_{cfg.lik_scale}"].values.astype(float) - last
    learned = (
        cfg.alpha
        * fade_in(df["md_since"].values.astype(float), cfg.tau)
        * (model_delta * (1 - cfg.pf_weight) + pf_delta * cfg.pf_weight)
    )
    pred = last + cfg.learned_weight * learned + (1 - cfg.learned_weight) * lik_delta
    return smooth_groups(pred, df["well"], cfg.sg_window, cfg.sg_poly)


def find_models(roots, fallback: Path | None = None) -> Path | None:
    candidates = [Path(r) for r in roots]
    if fallback is not None:
        candidates.append(fallback)
    for d in candidates:
        if (d / "features.json").exists() and list(d.glob("lgb*.pkl")):
            return d
    return None


def find_precomputed(roots, sample_ids):
    """A CSV whose ids exactly cover the sample, searched in the learned-model roots."""
    ids = pd.Series(sample_ids, dtype=str).reset_index(drop=True)
    first = [
        "fleongg_pretrained_submission.csv",
        "pretrained_submission.csv",
        "submission_fleongg.csv",
        "submission.csv",
    ]
    seen, paths = set(), []
    for root in map(Path, roots):
        if not root.exists():
            continue
        for p in [root / n for n in first] + sorted(root.glob("*.csv")):
            if p.exists() and p not in seen:
                seen.add(p)
                paths.append(p)
    for path in paths:
        try:
            df = pd.read_csv(path)
            if not {"id", "tvt"} <= set(df.columns):
                continue
            frame = df[["id", "tvt"]].astype({"id": str})
            if frame["id"].duplicated().any():
                continue
            aligned = pd.DataFrame({"id": ids}).merge(frame, on="id", how="left")
            aligned["tvt"] = pd.to_numeric(aligned["tvt"], errors="coerce")
            if aligned["tvt"].isna().any() or not np.isfinite(aligned["tvt"].to_numpy(dtype=float)).all():
                print("precomputed submission skipped (id mismatch):", path)
                continue
            return path, aligned[["id", "tvt"]]
        except Exception as exc:
            print("precomputed submission skipped:", path, exc)
    return None, None


def predict(
    data_dir: Path,
    cfg: PostprocessConfig,
    model_roots=(),
    local_models: Path | None = None,
    n_train_wells: int = 0,
    fast: bool = False,
    device: str = "auto",
    n_particles: int = 500,
    n_seeds: int = 128,
    n_jobs: int | None = None,
    imputers: Imputers | None = None,
):
    """Learned-trajectory submission (`id`, `tvt`) and, when trained from scratch, the pooled CV score."""
    train_ids = list_wells(data_dir, "train")
    test_ids = list_wells(data_dir, "test")
    if n_train_wells:
        train_ids = train_ids[:n_train_wells]
    print(f"train wells: {len(train_ids)} | test wells: {len(test_ids)}")
    imputers = imputers or Imputers(train_ids, Path(data_dir) / "train")

    likpf_test = build_likpf(data_dir, test_ids, "test", n_particles, n_seeds, n_jobs)
    test_df = add_likpf_features(build_features(data_dir, test_ids, "test", imputers, n_jobs), likpf_test)
    test_df = test_df.reset_index(drop=True)

    sample = read_sample(data_dir)[["id"]]
    models_dir = find_models(model_roots, local_models)
    cv_score = None

    if models_dir is not None:
        print(f"inference with saved boosters from {models_dir}")
        features = json.loads((models_dir / "features.json").read_text())
        models = [joblib.load(p) for p in sorted(models_dir.glob("lgb*.pkl"))]
        for col in features:
            if col not in test_df.columns:
                test_df[col] = 0.0
        X_test = test_df[features].values.astype(np.float32)
        meta_test = np.mean([m.predict(X_test) for m in models], axis=0)
        fallback = float(test_df["last_known_tvt"].mean())
    else:
        path, precomputed = find_precomputed(model_roots, sample["id"])
        if precomputed is not None:
            print(f"using id-exact precomputed submission from {path}")
            return precomputed, None
        likpf_train = build_likpf(data_dir, train_ids, "train", n_particles, n_seeds, n_jobs)
        train_df = add_likpf_features(
            build_features(data_dir, train_ids, "train", imputers, n_jobs), likpf_train
        )
        features = [
            c for c in train_df.columns
            if c not in {"well", "id", "target"}
            and not (c.startswith("likpf_scale_") or c == "likpf_mean")
            and c in test_df.columns
        ]  # fmt: skip
        print(f"features: {len(features)} | train rows: {len(train_df)} | test rows: {len(test_df)}")
        meta_oof, meta_test = train_stack(train_df, test_df, features, detect_device(device), fast)
        y = train_df["target"].values.astype(float)
        cv_score = rmse(train_df["last_known_tvt"].values + y, make_prediction(train_df, meta_oof, cfg))
        print(f"pooled CV rmse (TVT): {cv_score:.4f}")
        fallback = float(train_df["last_known_tvt"].mean() + y.mean())

    pred = make_prediction(test_df, meta_test, cfg)
    sub = sample.copy()
    sub["tvt"] = sub["id"].map(dict(zip(test_df["id"], pred))).fillna(fallback)
    return sub, cv_score
