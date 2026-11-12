from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold

from rogii.features.builder import build_features
from rogii.features.imputers import Imputers
from rogii.io import list_wells, read_sample
from rogii.postprocess.smoothing import fade_in, savgol_by_well

from .trainer import Trainer, ensure_koolbox

N_SPLITS = 5
NON_FEATURES = {"well", "id", "target"}
RIDGE_PARAMS = dict(
    random_state=42, alpha=1.6602834637650032, tol=0.0005030247295617308, positive=True, fit_intercept=True
)
PP_PARAMS = dict(alpha=1.0, tau=85, w_pf=0.09)

_LGB_BASE = dict(
    n_jobs=-1, verbose=-1, reg_alpha=10.788188919840913, subsample=0.47437582748953966, num_leaves=64,
    reg_lambda=95.75401894533888, n_estimators=10000, boosting_type="gbdt", learning_rate=0.00934485794382918,
    colsample_bytree=0.39283351290380497, min_child_weight=0.24081152127177283, min_child_samples=40, device="gpu",
)  # fmt: skip
LGB_PARAMS = [
    dict(
        boosting_type="gbdt", num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, objective="regression", verbose=-1, n_jobs=-1,
        device_type="gpu", gpu_use_dp=False, max_bin=255, learning_rate=0.030, n_estimators=5000, seed=123,
    ),
    dict(_LGB_BASE, random_state=0),
    dict(_LGB_BASE, random_state=29),
]  # fmt: skip
_CB_BASE = dict(
    iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, loss_function="RMSE",
    task_type="GPU", devices="0", od_type="Iter", od_wait=300, verbose=0,
)  # fmt: skip
CB_PARAMS = [
    dict(_CB_BASE, learning_rate=0.020, random_seed=7),
    dict(_CB_BASE, learning_rate=0.030, random_seed=123),
]


def cpu_params(params: dict) -> dict:
    """Swap GPU-only options for their CPU equivalents."""
    out = {k: v for k, v in params.items() if k not in {"device_type", "gpu_use_dp", "device", "devices"}}
    if out.get("task_type") == "GPU":
        out["task_type"] = "CPU"
    return out


def apply_postprocess(df, model_delta, pf_delta, alpha, tau, w_pf):
    delta = model_delta * (1 - w_pf) + pf_delta * w_pf
    if tau:
        delta = delta * fade_in(df["md_since"].values, tau)
    return delta * alpha


def _load_or_fit(name: str, make_estimator, folder: Path, X, y, groups, fit_args):
    saved = list(folder.glob("*.pkl")) if folder.exists() else []
    if saved:
        trainer = joblib.load(saved[0])
        print(f"loaded {name} (oof rmse {trainer.overall_score:.4f})")
        return trainer
    trainer = Trainer(
        make_estimator(),
        task="regression",
        metric=root_mean_squared_error,
        cv=GroupKFold(n_splits=N_SPLITS),
        cv_args={"groups": groups},
        use_early_stopping=True,
        verbose=True,
        save=True,
        save_path=folder,
    )
    return trainer.fit(X, y, fit_args=fit_args)


def _fit_args_lgb():
    from lightgbm import early_stopping, log_evaluation

    return {
        "eval_metric": "rmse",
        "callbacks": [log_evaluation(period=250), early_stopping(stopping_rounds=250)],
    }


def train_or_load_models(train_df, X_test, features, artifacts: Path, use_cpu: bool):
    """Fit (or restore from the artifact folder) the LightGBM and CatBoost fold models."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    X, y, groups = train_df[features], train_df["target"], train_df["well"]
    oof, test = {}, {}
    scores = {}
    fixup = cpu_params if use_cpu else (lambda p: p)

    for i, params in enumerate(LGB_PARAMS, 1):
        trainer = _load_or_fit(
            f"lightgbm-{i}", lambda p=params: LGBMRegressor(**fixup(p)), artifacts / "models" / f"lightgbm-{i}",
            X, y, groups, _fit_args_lgb(),
        )  # fmt: skip
        oof[f"lightgbm-{i}"], test[f"lightgbm-{i}"], scores[f"lightgbm-{i}"] = (
            trainer.oof_preds, trainer.predict(X_test), trainer.overall_score,
        )  # fmt: skip
    cb_fit = {"verbose": 250, "early_stopping_rounds": 250, "use_best_model": True}
    for i, params in enumerate(CB_PARAMS, 1):
        trainer = _load_or_fit(
            f"catboost-{i}", lambda p=params: CatBoostRegressor(**fixup(p)), artifacts / "models" / f"catboost-{i}",
            X, y, groups, cb_fit,
        )  # fmt: skip
        oof[f"catboost-{i}"], test[f"catboost-{i}"], scores[f"catboost-{i}"] = (
            trainer.oof_preds, trainer.predict(X_test), trainer.overall_score,
        )  # fmt: skip
    return pd.DataFrame(oof), pd.DataFrame(test), scores


def load_train_frame(data_dir: Path, artifacts: Path | None, imputers: Imputers, n_jobs=None) -> pd.DataFrame:
    cached = artifacts / "data" / "train.csv" if artifacts else None
    if cached is not None and cached.exists():
        return pd.read_csv(cached, low_memory=False)
    return build_features(data_dir, list_wells(data_dir, "train"), "train", imputers, n_jobs)


def predict(
    data_dir: Path,
    artifacts: Path | None,
    koolbox_roots=(),
    use_cpu: bool = True,
    n_jobs=None,
    imputers: Imputers | None = None,
) -> pd.DataFrame:
    """Ridge-stacked TVT prediction for every sample id (`id`, `tvt`)."""
    ensure_koolbox(koolbox_roots)
    imputers = imputers or Imputers(list_wells(data_dir, "train"), Path(data_dir) / "train")
    train_df = load_train_frame(data_dir, artifacts, imputers, n_jobs)
    test_df = build_features(data_dir, list_wells(data_dir, "test"), "test", imputers, n_jobs)

    features = [c for c in train_df.columns if c not in NON_FEATURES]
    y, groups = train_df["target"], train_df["well"]
    oof, test, _ = train_or_load_models(
        train_df, test_df[features], features, artifacts or Path("artifacts/none"), use_cpu
    )

    stacker = Trainer(Ridge(**RIDGE_PARAMS), task="regression", metric=root_mean_squared_error,
                      cv=GroupKFold(n_splits=N_SPLITS), cv_args={"groups": groups}, verbose=True)  # fmt: skip
    stacker.fit(oof, y)
    ridge_test = stacker.predict(test)

    pf_delta = test_df["pf_ancc"].values - test_df["last_known_tvt"].values
    test_df = test_df.copy()
    test_df["pred"] = test_df["last_known_tvt"].values + apply_postprocess(
        test_df, ridge_test, pf_delta, **PP_PARAMS
    )
    test_df = savgol_by_well(test_df, "pred")

    sample = read_sample(data_dir)
    out = sample[["id"]].merge(test_df[["id", "pred"]].rename(columns={"pred": "tvt"}), on="id", how="left")
    out["tvt"] = out["tvt"].fillna(float(train_df["last_known_tvt"].mean() + train_df["target"].mean()))
    return out
