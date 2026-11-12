from __future__ import annotations

import glob
import subprocess
import sys
import types
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _score(metric, y_true, y_pred) -> float:
    try:
        return (
            float(metric(y_true, y_pred))
            if callable(metric)
            else float(root_mean_squared_error(y_true, y_pred))
        )
    except Exception:
        return float(root_mean_squared_error(y_true, y_pred))


_FIT_ONLY_KEYS = (
    "callbacks",
    "eval_metric",
    "eval_set",
    "early_stopping_rounds",
    "use_best_model",
    "verbose",
)


class Trainer:
    """Cross-validated fitter with the surface of the `koolbox` Trainer used by the saved artifacts."""

    def __init__(self, estimator, task="regression", metric=None, cv=None, cv_args=None,
                 use_early_stopping=False, verbose=False, save=False, save_path=None):  # fmt: skip
        self.estimator = estimator
        self.task = task
        self.metric = metric or root_mean_squared_error
        self.cv = cv
        self.cv_args = cv_args or {}
        self.use_early_stopping = bool(use_early_stopping)
        self.verbose = bool(verbose)
        self.save = bool(save)
        self.save_path = save_path
        self.models = []
        self.oof_preds = None
        self.fold_scores = []
        self.overall_score = None

    def _splits(self, X, y):
        groups = self.cv_args.get("groups")
        cv = self.cv
        if cv is None:
            cv = (
                GroupKFold(n_splits=5)
                if groups is not None
                else KFold(n_splits=5, shuffle=True, random_state=42)
            )
        try:
            return list(cv.split(X, y, groups=groups))
        except TypeError:
            return list(cv.split(X, y))

    def _fit_one(self, estimator, X_tr, y_tr, X_va=None, y_va=None, fit_args=None):
        kwargs = dict(fit_args or {})
        if self.use_early_stopping and X_va is not None and y_va is not None:
            kind = (estimator.__class__.__module__ + estimator.__class__.__name__).lower()
            if "lightgbm" in kind or "lgbm" in kind:
                kwargs.setdefault("eval_set", [(X_va, y_va)])
            elif "catboost" in kind:
                kwargs.setdefault("eval_set", (X_va, y_va))
        try:
            estimator.fit(X_tr, y_tr, **kwargs)
        except TypeError:
            estimator.fit(X_tr, y_tr, **{k: v for k, v in kwargs.items() if k not in _FIT_ONLY_KEYS})
        return estimator

    def fit(self, X, y, fit_args=None):
        y_arr = np.asarray(y, dtype=float)
        oof = np.full(len(y_arr), np.nan)
        self.models, self.fold_scores = [], []
        for fold, (tr, va) in enumerate(self._splits(X, y_arr), start=1):
            est = self._fit_one(
                clone(self.estimator), _take(X, tr), y_arr[tr], _take(X, va), y_arr[va], fit_args
            )
            pred = np.asarray(est.predict(_take(X, va)), dtype=float)
            oof[va] = pred
            self.fold_scores.append(_score(self.metric, y_arr[va], pred))
            self.models.append(est)
            if self.verbose:
                print(f"fold {fold}: {self.fold_scores[-1]:.5f}")
        if not np.isfinite(oof).all():
            raise RuntimeError("incomplete out-of-fold predictions")
        self.oof_preds = oof
        self.overall_score = _score(self.metric, y_arr, oof)
        if self.save and self.save_path:
            out = Path(self.save_path)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, out / "trainer.pkl")
        return self

    def predict(self, X):
        if not self.models:
            raise RuntimeError("Trainer has no fitted fold models")
        return np.mean([np.asarray(m.predict(X), dtype=float) for m in self.models], axis=0)


def _wheel_matches_runtime(path) -> bool:
    name = Path(path).name
    if " (" in name or not name.endswith(".whl"):
        return False
    parts = name[:-4].split("-")
    if len(parts) < 5:
        return False
    py_tag, abi_tag = parts[-3], parts[-2]
    runtime = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if py_tag.startswith("cp") and py_tag != runtime:
        return False
    if abi_tag.startswith("cp") and abi_tag != runtime:
        return False
    return py_tag in {"py2.py3", "py3", runtime} or py_tag.startswith(runtime)


def _find_koolbox_root(roots) -> Path | None:
    for root in roots:
        if Path(root).exists():
            return Path(root)
    hits = []
    for pattern in ("/kaggle/input/**/koolbox*.whl", "/kaggle/input/**/koolbox*"):
        hits += [
            Path(h).parent if h.endswith(".whl") else Path(h) for h in glob.glob(pattern, recursive=True)
        ]
    return next((p for p in sorted(set(hits)) if p.exists()), None)


def _install_or_add_to_path(root: Path) -> None:
    wheels = [w for w in sorted(root.glob("**/*.whl")) if _wheel_matches_runtime(w)]
    if wheels:
        for wheel in wheels:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(wheel)], check=False
            )
        return
    sys.path.insert(0, str(root))
    for sub in root.iterdir():
        if sub.is_dir():
            sys.path.insert(0, str(sub))


def ensure_koolbox(roots=()) -> str:
    """Make `import koolbox` resolve, installing offline wheels or falling back to the local Trainer."""
    if "koolbox" in sys.modules:
        return "loaded"
    try:
        root = _find_koolbox_root(roots)
        if root is not None:
            _install_or_add_to_path(root)
        import koolbox  # noqa: F401

        return "external"
    except Exception:
        shim = types.ModuleType("koolbox")
        Trainer.__module__ = "koolbox"
        shim.Trainer = Trainer
        sys.modules["koolbox"] = shim
        return "fallback"
