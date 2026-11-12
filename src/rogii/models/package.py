from __future__ import annotations

import importlib.util
import inspect
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from rogii.config import PackageConfig
from rogii.io import align_to_sample

GROUP_COLUMNS = ["well_id", "well", "WELL"]
ORDER_COLUMNS = ["row_index", "row", "sample_index", "MD"]
MD_SINCE_COLUMNS = ["md_since_ps", "md_since", "md_delta", "MD_since", "md_from_start"]


def _read_json(path: Path):
    with Path(path).open() as f:
        return json.load(f)


def _first_column(frame: pd.DataFrame, names) -> str | None:
    return next((n for n in names if n in frame.columns), None)


def find_package_root(roots, auto_search: bool = False) -> Path | None:
    for root in map(Path, roots):
        for candidate in (root, root / "rogii_model_package", root / "rogii_artifacts"):
            if (candidate / "metadata" / "model_package_manifest.json").exists():
                return candidate
    if auto_search and Path("/kaggle/input").exists():
        for manifest in Path("/kaggle/input").glob("**/metadata/model_package_manifest.json"):
            return manifest.parents[1]
    return None


def _prediction_column(entry: dict) -> str:
    if entry.get("prediction_column"):
        return str(entry["prediction_column"])
    if not entry.get("branch_name") or not entry.get("model_name"):
        raise RuntimeError(f"model entry needs prediction_column or branch_name/model_name: {entry}")
    return f"pred_delta_{entry['branch_name']}_{entry['model_name']}"


def _load_feature_builder(root: Path):
    feature_dir = root / "feature_builders"
    for path in (root, feature_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.modules.pop("rogii_model_package_feature_builder", None)
    for file in (feature_dir / "build_features.py", feature_dir / "feature_builder.py"):
        if not file.exists():
            continue
        spec = importlib.util.spec_from_file_location("rogii_model_package_feature_builder", file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in ("build_features", "build_tail_features", "make_features"):
            if hasattr(module, name):
                return getattr(module, name), file
    raise RuntimeError("model package has no feature builder")


def _build_frame(builder, data_dir: Path, sample: pd.DataFrame, root: Path, manifest: dict) -> pd.DataFrame:
    available = dict(
        data_dir=data_dir, competition_root=data_dir, sample_submission=sample, sample=sample,
        package_root=root, manifest=manifest, config=manifest,
    )  # fmt: skip
    params = inspect.signature(builder).parameters
    features = builder(**{k: v for k, v in available.items() if k in params})
    if not isinstance(features, pd.DataFrame) or "id" not in features.columns:
        raise RuntimeError("feature builder must return a DataFrame with an id column")
    features = features.copy()
    features["id"] = features["id"].astype(str)
    if features["id"].duplicated().any():
        raise RuntimeError("feature frame contains duplicate ids")
    ids = sample["id"].astype(str)
    missing, extra = set(ids) - set(features["id"]), set(features["id"]) - set(ids)
    if missing or extra:
        raise RuntimeError(f"feature frame id mismatch: missing={len(missing)} extra={len(extra)}")
    return sample[["id"]].astype({"id": str}).merge(features, on="id", how="left")


def _columns_for(config, entry: dict) -> list[str]:
    if isinstance(entry.get("feature_columns"), list):
        return list(entry["feature_columns"])
    if isinstance(config, list):
        return list(config)
    if isinstance(config, dict):
        feature_set = entry.get("feature_set")
        if feature_set and isinstance(config.get(feature_set), list):
            return list(config[feature_set])
        if isinstance(config.get("columns"), list):
            return list(config["columns"])
    raise RuntimeError(f"cannot resolve feature columns for {entry}")


def _load_model(root: Path, entry: dict):
    kind, path = entry.get("model_type"), root / entry["path"]
    if kind == "lightgbm_booster":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    if kind == "xgboost_json":
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    if kind == "catboost_cbm":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(path))
        return model
    if kind in {"lightgbm_sklearn_pickle", "xgboost_pickle", "sklearn_pickle"}:
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)
    if kind == "torch_tcn":
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    raise RuntimeError(f"unsupported model_type {kind!r}")


def _feature_matrix(frame, columns, entry, manifest) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RuntimeError(f"feature frame missing {len(missing)} columns, e.g. {missing[:5]}")
    X = frame[columns].replace([np.inf, -np.inf], np.nan)
    policy = str(entry.get("missing_value_policy", manifest.get("missing_value_policy", "native"))).lower()
    if entry.get("fillna") is not None:
        return X.fillna(float(entry["fillna"]))
    if policy in {"native", "none", "null"}:
        return X
    if policy in {"zero", "fill_zero"}:
        return X.fillna(0.0)
    raise RuntimeError(f"unsupported missing_value_policy {policy!r}")


def _tcn_module(nn, n_features: int, config: dict):
    class Block(nn.Module):
        def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
            super().__init__()
            pad = dilation * (kernel - 1) // 2
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
            self.act = nn.GELU()
            self.drop = nn.Dropout(float(dropout))
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

        def forward(self, x):
            residual = self.skip(x)
            y = self.drop(self.act(self.conv1(x)))
            y = self.drop(self.act(self.conv2(y)))
            if y.shape[-1] != residual.shape[-1]:
                n = min(y.shape[-1], residual.shape[-1])
                y, residual = y[..., :n], residual[..., :n]
            return self.act(y + residual)

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            channels = int(config.get("channels", 64))
            kernel = int(config.get("kernel_size", 5))
            dropout = float(config.get("dropout", 0.0))
            blocks, in_ch = [], int(n_features)
            for i in range(int(config.get("blocks", 6))):
                blocks.append(Block(in_ch, channels, kernel, 2**i, dropout))
                in_ch = channels
            self.net = nn.Sequential(*blocks)
            self.head = nn.Conv1d(channels, 1, 1)

        def forward(self, x):
            return self.head(self.net(x)).squeeze(1)

    return Regressor()


def _predict_tcn(payload: dict, frame: pd.DataFrame, columns: list[str], entry: dict) -> np.ndarray:
    import torch
    from torch import nn

    X = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    std = payload.get("standardizer", {}) or {}
    mean, scale = (np.asarray(std.get(k), dtype=np.float32) for k in ("mean", "scale"))
    if mean.shape[0] != X.shape[1] or scale.shape[0] != X.shape[1]:
        raise RuntimeError(f"torch_tcn standardizer shape mismatch for {entry.get('prediction_column')}")
    X = np.nan_to_num(
        (X - mean.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), 1e-6), nan=0.0, posinf=0.0, neginf=0.0
    )
    X = X.astype(np.float32)

    group_col = entry.get("sequence_group_column") or _first_column(frame, GROUP_COLUMNS)
    order_col = entry.get("sequence_order_column") or _first_column(frame, ORDER_COLUMNS)
    tmp = pd.DataFrame({"_pos": np.arange(len(frame), dtype=int)})
    if group_col and group_col in frame.columns:
        tmp["_group"] = frame[group_col].astype(str).to_numpy()
    else:
        tmp["_group"] = frame["id"].astype(str).str.rsplit("_", n=1).str[0].to_numpy()
    if order_col and order_col in frame.columns:
        tmp["_order"] = pd.to_numeric(frame[order_col], errors="coerce").to_numpy(dtype=float)
    else:
        tmp["_order"] = np.arange(len(frame), dtype=float)

    model = _tcn_module(nn, len(columns), payload.get("config", {}) or {})
    model.load_state_dict(payload["state_dict"])
    model.eval()
    pred = np.full(len(frame), np.nan, dtype=np.float32)
    with torch.no_grad():
        for _, part in tmp.groupby("_group", sort=False):
            idx = part.sort_values("_order")["_pos"].to_numpy(dtype=int)
            out = model(torch.from_numpy(X[idx].T[None, :, :].copy()))
            pred[idx] = out.detach().cpu().numpy().reshape(-1)[: len(idx)].astype(np.float32)
    if not np.isfinite(pred).all():
        raise RuntimeError(f"torch_tcn produced non-finite predictions for {entry.get('prediction_column')}")
    return pred.astype(float)


def _predict_entry(model, kind: str, frame, columns, entry, manifest) -> np.ndarray:
    X = _feature_matrix(frame, columns, entry, manifest)
    if kind == "torch_tcn":
        pred = _predict_tcn(model, frame, columns, entry)
    elif kind == "xgboost_json":
        import xgboost as xgb

        pred = model.predict(xgb.DMatrix(X.to_numpy(dtype=np.float32)))
    else:
        pred = model.predict(X)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim > 1:
        pred = pred.reshape(len(frame), -1)[:, 0]
    if len(pred) != len(frame) or not np.isfinite(pred).all():
        raise RuntimeError(f"model {entry.get('prediction_column')} returned a bad prediction vector")
    return pred


def _weights(blend: dict) -> dict[str, float]:
    for key in ("weights", "model_weights"):
        if isinstance(blend.get(key), dict):
            return {str(k): float(v) for k, v in blend[key].items()}
    rows = {str(r["prediction_column"]): float(r["weight"]) for r in blend.get("models", []) if "weight" in r}
    if rows:
        return rows
    for holder in (blend, blend.get("stacker")):
        if isinstance(holder, dict) and "result_keys" in holder and "coef" in holder:
            if len(holder["result_keys"]) != len(holder["coef"]):
                raise RuntimeError("blend config result_keys and coef differ in length")
            return {str(k): float(v) for k, v in zip(holder["result_keys"], holder["coef"])}
    raise RuntimeError("blend config has no weights")


def _intercept(blend: dict) -> float:
    for holder in (blend, blend.get("stacker")):
        if isinstance(holder, dict):
            for key in ("intercept", "bias", "blend_intercept"):
                if key in holder:
                    return float(holder[key])
    return 0.0


def _fade_and_scale(delta, blend: dict, frame: pd.DataFrame) -> np.ndarray:
    post = blend.get("postprocess", {}) or {}
    out = delta.astype(float).copy()
    tau = post.get("fade_tau_md", post.get("tau"))
    if tau is not None:
        col = _first_column(frame, MD_SINCE_COLUMNS)
        if col is None:
            raise RuntimeError("postprocess.fade_tau_md was set but no md_since column exists")
        md_since = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        out *= 1.0 - np.exp(-np.maximum(md_since, 0.0) / float(tau))
    return out * float(post.get("alpha", 1.0))


def _savgol_groups(tvt, blend: dict, frame: pd.DataFrame) -> np.ndarray:
    post = blend.get("postprocess", {}) or {}
    window = int(post.get("savgol_window", 0) or 0)
    if window <= 2:
        return tvt
    window += 1 - window % 2
    poly = int(post.get("savgol_poly", 2) or 2)
    group_col = _first_column(frame, GROUP_COLUMNS)
    row_col = _first_column(frame, ["row_index", "row", "sample_index"])
    tmp = pd.DataFrame({"_pos": np.arange(len(tvt)), "_tvt": tvt})
    tmp["_group"] = (
        frame[group_col].astype(str).to_numpy()
        if group_col
        else frame["id"].str.rsplit("_", n=1).str[0].to_numpy()
    )
    tmp["_order"] = (
        pd.to_numeric(frame[row_col], errors="coerce").to_numpy(dtype=float)
        if row_col
        else np.arange(len(tvt), dtype=float)
    )
    out = tvt.astype(float).copy()
    for _, part in tmp.groupby("_group", sort=False):
        if len(part) < max(window, poly + 2):
            continue
        part = part.sort_values("_order")
        w = min(window, len(part) if len(part) % 2 == 1 else len(part) - 1)
        if w < poly + 2 or w <= 2:
            continue
        out[part["_pos"].to_numpy(dtype=int)] = savgol_filter(
            part["_tvt"].to_numpy(dtype=float), w, min(poly, w - 1), mode="interp"
        )
    return out


def package_submission(root: Path, data_dir: Path, sample: pd.DataFrame, cfg: PackageConfig, skip_kinds=()):
    """Run every model in the package and blend them into a TVT submission.

    Returns (submission, per-model report, weight report, info). `skip_kinds` drops model types
    (e.g. ``("torch_tcn",)``) and renormalises the remaining weights.
    """
    manifest = _read_json(root / "metadata" / "model_package_manifest.json")
    blend = _read_json(root / manifest.get("blend_config", "stacking/blend_config.json"))
    columns_config = _read_json(
        root / manifest.get("feature_columns", "feature_builders/feature_columns.json")
    )
    builder, builder_path = _load_feature_builder(root)
    frame = _build_frame(builder, Path(data_dir), sample, root, manifest)

    entries = [e for e in manifest.get("models", []) if e.get("model_type") not in skip_kinds]
    preds = pd.DataFrame({"id": frame["id"].to_numpy()})
    report = []
    for entry in entries:
        name, kind = _prediction_column(entry), entry.get("model_type")
        if kind == "direct_feature":
            source = entry.get("feature_column")
            if source not in frame.columns:
                raise RuntimeError(f"direct_feature column is missing: {source}")
            pred = pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=float)
        else:
            pred = _predict_entry(
                _load_model(root, entry), kind, frame, _columns_for(columns_config, entry), entry, manifest
            )
        if not np.isfinite(pred).all():
            raise RuntimeError(f"non-finite predictions from {name}")
        preds[name] = pred
        report.append(dict(prediction_column=name, model_type=kind, pred_mean=float(pred.mean()),
                           pred_std=float(pred.std()), pred_min=float(pred.min()), pred_max=float(pred.max())))  # fmt: skip

    weights = _weights(blend)
    if skip_kinds:
        names = {_prediction_column(e) for e in entries}
        weights = {k: v for k, v in weights.items() if k in names}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    missing = [c for c in weights if c not in preds.columns]
    if missing:
        raise RuntimeError(f"blend config references missing prediction columns: {missing}")

    value = np.full(len(preds), _intercept(blend), dtype=float)
    for col, w in weights.items():
        value += w * preds[col].to_numpy(dtype=float)

    space = (
        blend.get("target_space") or blend.get("prediction_space") or manifest.get("target_space", "delta")
    )
    if space == "delta":
        if "last_known_TVT" not in frame.columns:
            raise RuntimeError("delta-space blend requires last_known_TVT")
        tvt = frame["last_known_TVT"].to_numpy(dtype=float) + _fade_and_scale(value, blend, frame)
    elif space == "tvt":
        tvt = value
    else:
        raise RuntimeError(f"unsupported target_space {space!r}")
    tvt = _savgol_groups(tvt, blend, frame)

    lo = cfg.clip_min if cfg.clip_min is not None else blend.get("tvt_clip_min")
    hi = cfg.clip_max if cfg.clip_max is not None else blend.get("tvt_clip_max")
    if lo is not None or hi is not None:
        tvt = np.clip(tvt, -np.inf if lo is None else float(lo), np.inf if hi is None else float(hi))

    submission = align_to_sample(
        pd.DataFrame({"id": frame["id"].to_numpy(), "tvt": tvt}), sample, "model_package"
    )
    info = dict(package_root=str(root), feature_builder=str(builder_path), target_space=space,
                weight_sum=float(sum(weights.values())), postprocess=json.dumps(blend.get("postprocess", {}) or {}))  # fmt: skip
    weight_report = pd.DataFrame([{"prediction_column": k, "weight": v} for k, v in weights.items()])
    return submission, pd.DataFrame(report), weight_report, info


def gated_blend(base: pd.DataFrame, pkg: pd.DataFrame, cfg: PackageConfig, out_dir: Path, info: dict):
    """Pull the base submission towards the package, strongest where the two already agree.

    Writes one candidate file per allowed max weight and returns the selected blend
    (or the untouched base when the p95 disagreement guard trips).
    """
    merged = base.rename(columns={"tvt": "tvt_base"}).merge(
        pkg.rename(columns={"tvt": "tvt_model_package"}), on="id", how="inner"
    )
    if len(merged) != len(base):
        raise RuntimeError("model package blend id mismatch")
    base_v = merged["tvt_base"].to_numpy(dtype=float)
    pkg_v = merged["tvt_model_package"].to_numpy(dtype=float)
    signed = pkg_v - base_v
    diff = np.abs(signed)
    p95 = float(np.quantile(diff, 0.95))
    diff_rmse = float(np.sqrt(np.mean(signed * signed)))
    active = bool(np.isfinite(diff_rmse) and diff_rmse > 1e-9)

    pd.Series({
        "model_package_available": True,
        "active_signal": active,
        "mean_signed_model_package_diff": float(signed.mean()),
        "mean_abs_model_package_diff": float(diff.mean()),
        "p95_abs_model_package_diff": p95,
        "max_abs_model_package_diff": float(diff.max()),
        "model_package_diff_rmse": diff_rmse,
        "model_package_corr_with_base": float(np.corrcoef(base_v, pkg_v)[0, 1]) if len(base_v) > 1 else float("nan"),
    }).to_csv(out_dir / "model_package_active_signal_summary.csv")  # fmt: skip

    disabled = cfg.diff_p95_disable is not None and p95 > float(cfg.diff_p95_disable)
    candidates = {*(round(x, 12) for x in cfg.candidates), round(cfg.max_weight, 12)}
    rows, chosen = [], None
    for gmax in sorted(candidates):
        gate = float(gmax) / (1.0 + (diff / cfg.scale) ** 2)
        out = merged[["id"]].copy()
        out["tvt"] = (1.0 - gate) * base_v + gate * pkg_v
        move = np.abs(out["tvt"].to_numpy(dtype=float) - base_v)
        name = f"submission_model_package_gated_{int(round(gmax * 1000)):03d}.csv"
        out.to_csv(out_dir / name, index=False)
        selected = abs(gmax - cfg.max_weight) < 1e-12
        if selected:
            chosen = (name, out)
        rows.append(dict(
            file=name, gated_max_weight=float(gmax), scale=float(cfg.scale),
            selected_for_submission_csv=bool(selected and not disabled),
            gate_mean=float(gate.mean()), gate_p95=float(np.quantile(gate, 0.95)), gate_max=float(gate.max()),
            mean_abs_model_package_diff=float(diff.mean()), p95_abs_model_package_diff=p95,
            max_abs_model_package_diff=float(diff.max()), mean_abs_final_move=float(move.mean()),
            p95_abs_final_move=float(np.quantile(move, 0.95)), max_abs_final_move=float(move.max()),
            disabled_by_diff_guard=bool(disabled),
        ))  # fmt: skip
    pd.DataFrame(rows).to_csv(out_dir / "model_package_correction_report.csv", index=False)
    pd.Series({**info, "model_package_available": True, "active_signal": active,
               "mean_abs_model_package_diff": float(diff.mean()), "p95_abs_model_package_diff": p95,
               "model_package_diff_rmse": diff_rmse, "disabled_by_diff_guard": bool(disabled)}).to_csv(out_dir / "model_package_correction_summary.csv")  # fmt: skip

    if disabled:
        print(f"model package disabled: p95 diff {p95:.3f} > {float(cfg.diff_p95_disable):.3f}")
        return base[["id", "tvt"]].copy()
    name, final = chosen
    print("model package blend selected:", name, final.shape)
    return final


def apply_correction(
    base: pd.DataFrame, data_dir: Path, sample: pd.DataFrame, roots, cfg: PackageConfig, out_dir: Path
):
    root = find_package_root(roots)
    base = align_to_sample(base, sample, "base before model package")
    base.to_csv(out_dir / "submission_before_model_package.csv", index=False)
    if root is None:
        if cfg.require:
            raise RuntimeError("model package dataset was not found")
        pd.Series({"model_package_available": False}).to_csv(out_dir / "model_package_correction_summary.csv")
        print("model package not found; correction skipped")
        return base
    pkg, report, weights, info = package_submission(root, data_dir, sample, cfg)
    pkg.to_csv(out_dir / "submission_model_package_only.csv", index=False)
    report.to_csv(out_dir / "model_package_prediction_report.csv", index=False)
    weights.to_csv(out_dir / "model_package_blend_weights.csv", index=False)
    return gated_blend(base, pkg, cfg, out_dir, info)
