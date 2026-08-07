# Shared utilities for the ROGII score-focused notebook pipeline.

from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import gc
import math
import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)

warnings.filterwarnings("ignore", category=PerformanceWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

KAGGLE_DATA_DIRS = [
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
]
LOCAL_DATA_DIR = Path(os.environ.get("ROGII_DATA_DIR", "."))
CANDIDATE_DATA_DIRS = KAGGLE_DATA_DIRS + [LOCAL_DATA_DIR]
DATA_DIR = next(
    (
        p
        for p in CANDIDATE_DATA_DIRS
        if (p / "train").exists() and (p / "sample_submission.csv").exists()
    ),
    LOCAL_DATA_DIR,
)
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
KAGGLE_WORKING_DIR = Path("/kaggle/working")
KAGGLE_NOTEBOOK_RUN = KAGGLE_WORKING_DIR.exists()
OUTPUT_DIR = KAGGLE_WORKING_DIR if KAGGLE_NOTEBOOK_RUN else Path(os.environ.get("ROGII_OUTPUT_DIR", DATA_DIR))
GPU_MODE = os.environ.get("ROGII_GPU_MODE", "auto").strip().lower()
LOCAL_NVIDIA_GPU_AVAILABLE = shutil.which("nvidia-smi") is not None
if GPU_MODE not in {"auto", "gpu", "cpu"}:
    raise ValueError("ROGII_GPU_MODE must be one of: auto, gpu, cpu")

# LightGBM GPU uses OpenCL, which can be absent on CUDA-only container images.
# Keep LGB on CPU by default while allowing XGBoost/CatBoost to use CUDA.
LGB_DEVICE_MODE = os.environ.get("ROGII_LGB_DEVICE", "cpu").strip().lower()
XGB_DEVICE_MODE = os.environ.get("ROGII_XGB_DEVICE", GPU_MODE).strip().lower()
CATBOOST_TASK_MODE = os.environ.get(
    "ROGII_CATBOOST_TASK_TYPE",
    "GPU" if (GPU_MODE == "gpu" or (GPU_MODE == "auto" and LOCAL_NVIDIA_GPU_AVAILABLE)) else "CPU",
).strip().upper()
if LGB_DEVICE_MODE not in {"auto", "gpu", "cpu"}:
    raise ValueError("ROGII_LGB_DEVICE must be one of: auto, gpu, cpu")
if XGB_DEVICE_MODE not in {"auto", "gpu", "cpu", "cuda"}:
    raise ValueError("ROGII_XGB_DEVICE must be one of: auto, gpu, cuda, cpu")
if CATBOOST_TASK_MODE not in {"AUTO", "GPU", "CUDA", "CPU"}:
    raise ValueError("ROGII_CATBOOST_TASK_TYPE must be one of: AUTO, GPU, CUDA, CPU")
LIGHTGBM_DEVICE_TYPE = "gpu" if (LGB_DEVICE_MODE == "gpu" or (LGB_DEVICE_MODE == "auto" and KAGGLE_NOTEBOOK_RUN)) else "cpu"
XGBOOST_USE_CUDA = XGB_DEVICE_MODE in {"gpu", "cuda"} or (XGB_DEVICE_MODE == "auto" and LOCAL_NVIDIA_GPU_AVAILABLE)
CATBOOST_TASK_TYPE = "GPU" if (CATBOOST_TASK_MODE in {"GPU", "CUDA"} or (CATBOOST_TASK_MODE == "AUTO" and LOCAL_NVIDIA_GPU_AVAILABLE)) else "CPU"


def lightgbm_accelerator_params(device_type: str | None = None) -> dict:
    device_type = device_type or LIGHTGBM_DEVICE_TYPE
    if device_type == "gpu":
        memory_safe = bool(globals().get("KAGGLE_MEMORY_SAFE_MODE", False))
        return {
            "device_type": "gpu",
            "gpu_use_dp": False,
            "max_bin": 127 if memory_safe else 255,
        }
    return {
        "device_type": "cpu",
        "force_col_wise": True,
    }


def well_id_from_path(path: Path) -> str:
    name = path.name
    if "__" in name:
        return name.split("__")[0]
    return path.stem


train_horizontal_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
train_typewell_files = sorted(TRAIN_DIR.glob("*__typewell.csv"))
train_png_files = sorted(TRAIN_DIR.glob("*.png"))
test_horizontal_files = sorted(TEST_DIR.glob("*__horizontal_well.csv")) if TEST_DIR.exists() else []
test_typewell_files = sorted(TEST_DIR.glob("*__typewell.csv")) if TEST_DIR.exists() else []

train_h_ids = {well_id_from_path(p) for p in train_horizontal_files}
train_t_ids = {well_id_from_path(p) for p in train_typewell_files}
train_png_ids = {p.stem for p in train_png_files}
test_h_ids = {well_id_from_path(p) for p in test_horizontal_files}
test_t_ids = {well_id_from_path(p) for p in test_typewell_files}


def prediction_zone_info(df: pd.DataFrame) -> dict:
    mask = df["TVT_input"].isna().to_numpy()
    null_indices = np.flatnonzero(mask)
    groups = 0
    in_group = False
    for value in mask:
        if value and not in_group:
            groups += 1
            in_group = True
        elif not value:
            in_group = False
    return {
        "n_rows": len(df),
        "known_rows": int((~mask).sum()),
        "prediction_rows": int(mask.sum()),
        "first_prediction_index": int(null_indices[0]) if len(null_indices) else None,
        "last_prediction_index": int(null_indices[-1]) if len(null_indices) else None,
        "n_missing_groups": groups,
    }



# Construct leakage-safe row-level tail features and residual targets for one well.

def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    if np.nanstd(x[valid]) < 1e-9:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def normalize_required_feature_columns(required_feature_columns) -> set[str] | None:
    if required_feature_columns is None:
        return None
    return {str(col) for col in required_feature_columns}


def needs_any(required_columns: set[str] | None, prefixes=(), exact=()) -> bool:
    if required_columns is None:
        return True
    prefixes = tuple(prefixes)
    exact = set(exact)
    for col in required_columns:
        if col in exact or any(col.startswith(prefix) for prefix in prefixes):
            return True
    return False




def typewell_gr_at_tvt(typewell_df: pd.DataFrame, tvt_values: np.ndarray) -> np.ndarray:
    if not {'TVT', 'GR'}.issubset(typewell_df.columns):
        return np.full(len(tvt_values), np.nan)
    tw = typewell_df[['TVT', 'GR']].dropna().sort_values('TVT')
    if len(tw) < 2:
        return np.full(len(tvt_values), np.nan)
    x = tw['TVT'].to_numpy(dtype=float)
    y = tw['GR'].to_numpy(dtype=float)
    return np.interp(np.asarray(tvt_values, dtype=float), x, y, left=np.nan, right=np.nan)



def sequence_filter_feature_names(prefix: str = 'pf_tvt') -> list[str]:
    return [
        f'{prefix}_delta',
        f'{prefix}_std',
        f'{prefix}_p10_delta',
        f'{prefix}_p90_delta',
        f'{prefix}_width',
        f'{prefix}_mode_delta',
        f'{prefix}_confidence',
        f'{prefix}_entropy',
        f'{prefix}_innovation',
        f'{prefix}_step',
        f'{prefix}_beam_cons_diff',
        f'{prefix}_beam_loose_diff',
    ]


def ancc_filter_feature_names(prefix: str = 'pf_ancc') -> list[str]:
    return [
        f'{prefix}_delta',
        f'{prefix}_std',
        f'{prefix}_plane_diff',
        f'{prefix}_row_diff',
        f'{prefix}_confidence',
        f'{prefix}_state_delta',
        f'{prefix}_state_std',
        f'{prefix}_state_vs_mean',
        f'{prefix}_state_confidence',
    ]


def z_filter_feature_names(prefix: str = 'pf_z') -> list[str]:
    return [
        f'{prefix}_delta',
        f'{prefix}_std',
        f'{prefix}_velocity_md',
        f'{prefix}_velocity_z',
        f'{prefix}_confidence',
        f'{prefix}_vs_tvt',
        f'{prefix}_vs_ancc',
        f'{prefix}_vs_beam',
    ]


def typewell_sequence_filter_features(
    typewell_df: pd.DataFrame,
    gr_values: np.ndarray,
    baseline_tvt: np.ndarray,
    last_known_tvt: float,
    prefix: str = 'pf_tvt',
    offsets: np.ndarray | None = None,
    emit_scale: float = 42.0,
    transition_sigma: float = 10.0,
) -> dict[str, np.ndarray]:
    """Small grid filter over TVT residual offset using hidden GR and typewell GR.

    This is an offline, target-free sequence prior. It uses the full hidden GR sequence,
    so it belongs in offline feature sets, not strict geosteering features.
    """
    gr_values = np.asarray(gr_values, dtype=float)
    baseline_tvt = np.asarray(baseline_tvt, dtype=float)
    n = len(gr_values)
    names = sequence_filter_feature_names(prefix)
    out = {name: np.full(n, np.nan, dtype=np.float32) for name in names}
    if n == 0 or not {'TVT', 'GR'}.issubset(typewell_df.columns) or not np.isfinite(last_known_tvt):
        return out
    tw = typewell_df[['TVT', 'GR']].dropna().sort_values('TVT')
    if len(tw) < 3:
        return out
    tw_tvt = tw['TVT'].to_numpy(dtype=float)
    tw_gr = tw['GR'].to_numpy(dtype=float)
    if offsets is None:
        offsets = np.linspace(-180.0, 180.0, 181, dtype=float)
    else:
        offsets = np.asarray(offsets, dtype=float)
    if len(offsets) < 3:
        return out
    step = float(np.nanmedian(np.diff(offsets))) if len(offsets) > 1 else 1.0
    prior_sigma = 55.0
    weights = np.exp(-0.5 * (offsets / prior_sigma) ** 2)
    weights = weights / max(float(weights.sum()), 1e-12)
    kernel_radius = max(2, int(np.ceil(3.0 * transition_sigma / max(abs(step), 1e-6))))
    kernel_offsets = np.arange(-kernel_radius, kernel_radius + 1) * step
    kernel = np.exp(-0.5 * (kernel_offsets / max(transition_sigma, 1e-6)) ** 2)
    kernel = kernel / max(float(kernel.sum()), 1e-12)
    means = np.full(n, np.nan, dtype=float)
    stds = np.full(n, np.nan, dtype=float)
    p10s = np.full(n, np.nan, dtype=float)
    p90s = np.full(n, np.nan, dtype=float)
    modes = np.full(n, np.nan, dtype=float)
    conf = np.full(n, np.nan, dtype=float)
    entropy = np.full(n, np.nan, dtype=float)
    innovation = np.full(n, np.nan, dtype=float)
    for i in range(n):
        weights = np.convolve(weights, kernel, mode='same')
        sw = float(weights.sum())
        weights = weights / max(sw, 1e-12)
        base = baseline_tvt[i] if np.isfinite(baseline_tvt[i]) else last_known_tvt
        candidate_tvt = base + offsets
        tw_gr_grid = np.interp(candidate_tvt, tw_tvt, tw_gr, left=np.nan, right=np.nan)
        valid = np.isfinite(tw_gr_grid)
        if np.isfinite(gr_values[i]) and valid.any():
            resid = gr_values[i] - tw_gr_grid
            likelihood = np.ones_like(weights)
            likelihood[valid] = np.exp(-0.5 * (resid[valid] / max(emit_scale, 1e-6)) ** 2)
            likelihood[~valid] = 0.0
            weights = weights * likelihood
            if float(weights.sum()) <= 1e-18:
                weights = np.where(valid, 1.0, 0.0)
            weights = weights / max(float(weights.sum()), 1e-12)
            innovation[i] = float(np.sqrt(np.nansum(weights[valid] * resid[valid] ** 2)))
        mean_offset = float(np.sum(weights * offsets))
        var_offset = float(np.sum(weights * (offsets - mean_offset) ** 2))
        cdf = np.cumsum(weights)
        p10_offset = float(offsets[min(int(np.searchsorted(cdf, 0.10, side='left')), len(offsets) - 1)])
        p90_offset = float(offsets[min(int(np.searchsorted(cdf, 0.90, side='left')), len(offsets) - 1)])
        mode_offset = float(offsets[int(np.argmax(weights))])
        base = baseline_tvt[i] if np.isfinite(baseline_tvt[i]) else last_known_tvt
        means[i] = base + mean_offset - last_known_tvt
        stds[i] = np.sqrt(max(var_offset, 0.0))
        p10s[i] = base + p10_offset - last_known_tvt
        p90s[i] = base + p90_offset - last_known_tvt
        modes[i] = base + mode_offset - last_known_tvt
        conf[i] = float(np.max(weights))
        entropy[i] = float(-np.sum(weights * np.log(weights + 1e-12)) / np.log(len(weights)))
    out[f'{prefix}_delta'] = means.astype(np.float32)
    out[f'{prefix}_std'] = stds.astype(np.float32)
    out[f'{prefix}_p10_delta'] = p10s.astype(np.float32)
    out[f'{prefix}_p90_delta'] = p90s.astype(np.float32)
    out[f'{prefix}_width'] = (p90s - p10s).astype(np.float32)
    out[f'{prefix}_mode_delta'] = modes.astype(np.float32)
    out[f'{prefix}_confidence'] = conf.astype(np.float32)
    out[f'{prefix}_entropy'] = entropy.astype(np.float32)
    out[f'{prefix}_innovation'] = innovation.astype(np.float32)
    out[f'{prefix}_step'] = np.r_[np.nan, np.diff(means)].astype(np.float32)
    return out



def self_corr_feature_names(prefix: str = 'sc') -> list[str]:
    return [
        f'{prefix}_delta',
        f'{prefix}_score',
        f'{prefix}_trust',
        'hyb_delta',
    ]


def self_corr_tvt(
    known_gr: np.ndarray,
    known_tvt: np.ndarray,
    hidden_gr: np.ndarray,
    half_window: int = 15,
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Align hidden GR windows to known-prefix GR windows from the same well.

    This is an offline, target-free sequence feature: it uses hidden GR but no
    hidden TVT. It captures repeated GR motifs within the well and complements
    the external typewell beam/path features.
    """
    known_gr = np.asarray(known_gr, dtype=float)
    known_tvt = np.asarray(known_tvt, dtype=float)
    hidden_gr = np.asarray(hidden_gr, dtype=float)
    n_hidden = len(hidden_gr)
    fallback_tvt = float(known_tvt[np.isfinite(known_tvt)][-1]) if np.isfinite(known_tvt).any() else np.nan
    if n_hidden == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    window = int(2 * half_window + 1)
    valid_known_tvt = np.isfinite(known_tvt)
    if len(known_gr) < window + 1 or not valid_known_tvt.any():
        return np.full(n_hidden, fallback_tvt, dtype=np.float32), np.zeros(n_hidden, dtype=np.float32)

    known_fill = float(np.nanmean(known_gr)) if np.isfinite(known_gr).any() else 0.0
    hidden_fill = float(np.nanmean(hidden_gr)) if np.isfinite(hidden_gr).any() else known_fill
    known_s = pd.Series(known_gr, dtype='float64').interpolate(limit_direction='both').fillna(known_fill)
    hidden_s = pd.Series(hidden_gr, dtype='float64').interpolate(limit_direction='both').fillna(hidden_fill)
    known_smooth = known_s.rolling(5, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
    hidden_smooth = hidden_s.rolling(5, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)

    starts = np.arange(0, len(known_smooth) - window + 1, max(int(stride), 1), dtype=np.int32)
    if len(starts) == 0:
        return np.full(n_hidden, fallback_tvt, dtype=np.float32), np.zeros(n_hidden, dtype=np.float32)
    idx = starts[:, None] + np.arange(window, dtype=np.int32)[None, :]
    candidates = known_smooth[idx].astype(np.float32)
    candidates = (candidates - candidates.mean(axis=1, keepdims=True)) / (candidates.std(axis=1, keepdims=True) + 1e-6)

    hidden_padded = np.pad(hidden_smooth, half_window, mode='edge')
    hidden_idx = np.arange(n_hidden, dtype=np.int32)[:, None] + np.arange(window, dtype=np.int32)[None, :]
    hidden_windows = hidden_padded[hidden_idx].astype(np.float32)
    hidden_windows = (hidden_windows - hidden_windows.mean(axis=1, keepdims=True)) / (hidden_windows.std(axis=1, keepdims=True) + 1e-6)

    ncc = hidden_windows @ candidates.T / float(window)
    best = np.argmax(ncc, axis=1)
    score = np.max(ncc, axis=1).astype(np.float32)
    centers = np.clip(starts[best] + half_window, 0, len(known_tvt) - 1)
    path = known_tvt[centers].astype(np.float32)
    path[~np.isfinite(path)] = fallback_tvt
    return path, score

def safe_corr(a: np.ndarray, b: np.ndarray, min_points: int = 30) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < min_points:
        return np.nan
    aa = a[valid]
    bb = b[valid]
    if np.nanstd(aa) < 1e-9 or np.nanstd(bb) < 1e-9:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def recent_mean_diff(values: np.ndarray, window: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[-(window + 1):]
    if len(values) < 2:
        return np.nan
    return float(np.diff(values).mean())


def recent_slope_window(y_values: np.ndarray, x_values: np.ndarray, window: int) -> float:
    y_values = np.asarray(y_values, dtype=float)[-window:]
    x_values = np.asarray(x_values, dtype=float)[-window:]
    valid = np.isfinite(y_values) & np.isfinite(x_values)
    if valid.sum() < 2:
        return np.nan
    x = x_values[valid]
    y = y_values[valid]
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(centered_x, y - y.mean()) / denominator)


def nearest_sorted_index(sorted_values: np.ndarray, target: float) -> int:
    values = np.asarray(sorted_values, dtype=float)
    if len(values) == 0:
        return 0
    idx = int(np.searchsorted(values, target, side='left'))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(values[idx - 1] - target) <= abs(values[idx] - target):
        return idx - 1
    return idx


def smooth_gr_for_beam(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    series = pd.Series(values, dtype='float64').interpolate(limit_direction='both').fillna(fallback)
    if radius <= 0:
        return series.to_numpy(dtype=float)
    return series.rolling(radius * 2 + 1, center=True, min_periods=1).mean().to_numpy(dtype=float)


def beam_typewell_path(
    gr_values: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    beam_size: int = 10,
    move_cost: float = 20.0,
    emit_scale: float = 144.0,
    radius: int = 2,
) -> np.ndarray:
    tw_tvt = np.asarray(tw_tvt, dtype=float)
    tw_gr = np.asarray(tw_gr, dtype=float)
    valid_tw = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[valid_tw]
    tw_gr = tw_gr[valid_tw]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    n = len(gr_values)
    if n == 0 or len(tw_tvt) < 2 or not np.isfinite(start_tvt):
        return np.full(n, np.nan)

    fallback = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    smoothed_gr = smooth_gr_for_beam(gr_values, fallback=fallback, radius=radius)
    start_idx = nearest_sorted_index(tw_tvt, start_tvt)
    states = {start_idx: 0.0}
    backpointers: list[dict[int, int]] = []

    for gr_value in smoothed_gr:
        candidates: dict[int, float] = {}
        parents: dict[int, int] = {}
        if not np.isfinite(gr_value):
            gr_value = fallback
        for idx, cost in states.items():
            for delta in (-1, 0, 1):
                next_idx = idx + delta
                if next_idx < 0 or next_idx >= len(tw_tvt):
                    continue
                emit_cost = ((gr_value - tw_gr[next_idx]) ** 2) / max(emit_scale, 1e-6)
                total_cost = cost + emit_cost + move_cost * abs(delta)
                if next_idx not in candidates or total_cost < candidates[next_idx]:
                    candidates[next_idx] = float(total_cost)
                    parents[next_idx] = idx
        kept = sorted(candidates.items(), key=lambda item: item[1])[:beam_size]
        if not kept:
            return np.full(n, np.nan)
        states = {idx: cost for idx, cost in kept}
        backpointers.append({idx: parents[idx] for idx, _ in kept})

    final_idx = min(states, key=states.get)
    path = [final_idx]
    for step in range(len(backpointers) - 1, 0, -1):
        path.append(backpointers[step][path[-1]])
    path.reverse()
    return tw_tvt[np.asarray(path, dtype=int)]


def offline_beam_feature_names(prefix: str = 'tw_beam') -> list[str]:
    names = [
        f'{prefix}_tight_delta',        # compatibility alias for the very-conservative beam
        f'{prefix}_vcons_delta',
        f'{prefix}_conservative_delta',
        f'{prefix}_sm5_delta',
        f'{prefix}_loose_delta',
        f'{prefix}_vloose_delta',
        f'{prefix}_gap',
        f'{prefix}_spread',
        f'{prefix}_mean_delta',
        f'{prefix}_median_delta',
        f'{prefix}_std_delta',
        f'{prefix}_tight_step',
        f'{prefix}_vcons_step',
        f'{prefix}_conservative_step',
        f'{prefix}_sm5_step',
        f'{prefix}_loose_step',
        f'{prefix}_vloose_step',
        f'{prefix}_gr_at_conservative',
        f'{prefix}_gr_at_loose',
        f'{prefix}_gr_minus_conservative',
        f'{prefix}_gr_minus_loose',
    ]
    if prefix == 'tw_beam':
        names += [
            'beam_tight_delta',        # compatibility alias for beam_vcons_delta
            'beam_vcons_delta',
            'beam_cons_delta',
            'beam_sm5_delta',
            'beam_loose_delta',
            'beam_vloose_delta',
            'beam_mean_delta',
            'beam_med_delta',
            'beam_std_delta',
            'beam_spread',
            'beam_gap',
            'gr_minus_tw_beam_cons',
            'gr_minus_tw_beam_loose',
        ]
    return names

def safe_stat(series, func) -> float:
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.nan
    return float(func(values[valid]))


def safe_affine_fit(x: np.ndarray, y: np.ndarray, min_points: int = 30) -> tuple[float, float]:
    """Fit y ~= a*x + b using only finite prefix points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < min_points or np.nanstd(x[valid]) < 1e-9:
        return np.nan, np.nan
    a, b = np.polyfit(x[valid], y[valid], 1)
    return float(a), float(b)


def apply_affine(x: np.ndarray | float, a: float, b: float):
    if np.isfinite(a) and np.isfinite(b):
        return a * x + b
    return np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)



def local_typewell_offset_features(
    typewell_df: pd.DataFrame,
    baseline_tvt: np.ndarray,
    horizontal_gr: np.ndarray,
    offsets: np.ndarray,
    affine_a: float | None = None,
    affine_b: float | None = None,
    score_scale: float = 20.0,
    prefix: str = 'typewell_local',
) -> dict[str, np.ndarray]:
    baseline_tvt = np.asarray(baseline_tvt, dtype=float)
    horizontal_gr = np.asarray(horizontal_gr, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    n = len(baseline_tvt)
    if n == 0 or len(offsets) == 0:
        return {
            f'{prefix}_best_delta': np.full(n, np.nan),
            f'{prefix}_best_abs_resid': np.full(n, np.nan),
            f'{prefix}_zero_abs_resid': np.full(n, np.nan),
            f'{prefix}_top2_gap': np.full(n, np.nan),
            f'{prefix}_soft_delta_mean': np.full(n, np.nan),
            f'{prefix}_best_gr': np.full(n, np.nan),
        }
    candidate_tvt = baseline_tvt[:, None] + offsets[None, :]
    candidate_gr = typewell_gr_at_tvt(typewell_df, candidate_tvt.reshape(-1)).reshape(n, len(offsets))
    if affine_a is not None and affine_b is not None and np.isfinite(affine_a) and np.isfinite(affine_b):
        candidate_gr = affine_a * candidate_gr + affine_b
    abs_resid = np.abs(horizontal_gr[:, None] - candidate_gr)
    valid = np.isfinite(abs_resid)
    masked = np.where(valid, abs_resid, np.inf)
    any_valid = valid.any(axis=1)
    best_idx = np.argmin(masked, axis=1)
    best_abs = np.where(any_valid, masked[np.arange(n), best_idx], np.nan)
    best_delta = np.where(any_valid, offsets[best_idx], np.nan)
    best_gr = np.where(any_valid, candidate_gr[np.arange(n), best_idx], np.nan)
    zero_idx = int(np.argmin(np.abs(offsets)))
    zero_abs = abs_resid[:, zero_idx] if len(offsets) else np.full(n, np.nan)
    if len(offsets) >= 2:
        sorted_abs = np.sort(masked, axis=1)
        top2_gap = np.full(n, np.nan)
        top2_ok = any_valid & np.isfinite(sorted_abs[:, 1])
        top2_gap[top2_ok] = sorted_abs[top2_ok, 1] - sorted_abs[top2_ok, 0]
    else:
        top2_gap = np.full(n, np.nan)
    weights = np.zeros_like(abs_resid, dtype=float)
    weights[valid] = np.exp(-abs_resid[valid] / score_scale)
    denom = weights.sum(axis=1)
    soft_delta = np.full(n, np.nan)
    denom_ok = denom > 0
    soft_delta[denom_ok] = (weights[denom_ok] * offsets[None, :]).sum(axis=1) / denom[denom_ok]
    return {
        f'{prefix}_best_delta': best_delta,
        f'{prefix}_best_abs_resid': best_abs,
        f'{prefix}_zero_abs_resid': zero_abs,
        f'{prefix}_top2_gap': top2_gap,
        f'{prefix}_soft_delta_mean': soft_delta,
        f'{prefix}_best_gr': best_gr,
    }

CANDIDATE_PATH_ENDPOINTS = np.array([-60.0, -40.0, -25.0, -15.0, -8.0, 0.0, 8.0, 15.0, 25.0, 40.0, 60.0])


def candidate_endpoint_label(endpoint: float) -> str:
    sign = 'm' if endpoint < 0 else 'p'
    value = str(int(abs(endpoint))) if float(endpoint).is_integer() else str(abs(endpoint)).replace('.', 'p')
    return f'{sign}{value}'




def typewell_offset_diff_feature_names(prefix: str, offsets: np.ndarray) -> list[str]:
    return [f'{prefix}_{candidate_endpoint_label(float(offset))}' for offset in np.asarray(offsets, dtype=float)]

def candidate_path_feature_names(prefix: str = 'tw_path', endpoints: np.ndarray = CANDIDATE_PATH_ENDPOINTS) -> list[str]:
    names: list[str] = []
    for endpoint in endpoints:
        label = candidate_endpoint_label(float(endpoint))
        names.extend([
            f'{prefix}_gr_diff_{label}',
            f'{prefix}_gr_absdiff_{label}',
            f'{prefix}_boundary_count_{label}',
            f'{prefix}_boundary_nearest_{label}',
        ])
    names.extend([
        f'{prefix}_min_absdiff',
        f'{prefix}_best_endpoint',
        f'{prefix}_best_endpoint_centered',
        f'{prefix}_top2_absdiff_gap',
        f'{prefix}_soft_endpoint_mean',
        f'{prefix}_best_gr_resid',
    ])
    return names


def typewell_boundary_tvt(typewell_df: pd.DataFrame) -> np.ndarray:
    if 'Geology' not in typewell_df.columns or 'TVT' not in typewell_df.columns:
        return np.array([], dtype=float)
    tw = typewell_df[['TVT', 'Geology']].dropna().sort_values('TVT')
    if len(tw) < 2:
        return np.array([], dtype=float)
    tvt = pd.to_numeric(tw['TVT'], errors='coerce').to_numpy(dtype=float)
    geology = tw['Geology'].astype(str).to_numpy()
    valid = np.isfinite(tvt)
    tvt = tvt[valid]
    geology = geology[valid]
    if len(tvt) < 2:
        return np.array([], dtype=float)
    change_idx = np.flatnonzero(geology[1:] != geology[:-1]) + 1
    return tvt[change_idx]




def typewell_interval_boundaries(typewell_df: pd.DataFrame) -> np.ndarray:
    if 'TVT' not in typewell_df.columns:
        return np.array([], dtype=float)
    tvt = pd.to_numeric(typewell_df['TVT'], errors='coerce').dropna().to_numpy(dtype=float)
    if len(tvt) < 2:
        return np.array([], dtype=float)
    boundaries = typewell_boundary_tvt(typewell_df)
    bounds = np.unique(np.r_[np.nanmin(tvt), boundaries, np.nanmax(tvt)])
    return np.sort(bounds[np.isfinite(bounds)])


def typewell_interval_context_features(typewell_df: pd.DataFrame, target_tvt, prefix: str) -> dict[str, np.ndarray]:
    target = np.asarray(target_tvt, dtype=float)
    original_shape = target.shape
    target_flat = target.reshape(-1)
    bounds = typewell_interval_boundaries(typewell_df)
    out = {
        f'{prefix}_prev_boundary_dist': np.full(len(target_flat), np.nan),
        f'{prefix}_next_boundary_dist': np.full(len(target_flat), np.nan),
        f'{prefix}_interval_thickness': np.full(len(target_flat), np.nan),
        f'{prefix}_interval_phase': np.full(len(target_flat), np.nan),
        f'{prefix}_interval_phase_sin': np.full(len(target_flat), np.nan),
        f'{prefix}_interval_phase_cos': np.full(len(target_flat), np.nan),
        f'{prefix}_boundary_balance': np.full(len(target_flat), np.nan),
        f'{prefix}_boundary_proximity': np.full(len(target_flat), np.nan),
    }
    if len(bounds) < 2:
        return {k: v.reshape(original_shape) for k, v in out.items()}
    valid = np.isfinite(target_flat)
    if valid.any():
        idx = np.searchsorted(bounds, target_flat[valid], side='right') - 1
        idx = np.clip(idx, 0, len(bounds) - 2)
        top = bounds[idx]
        base = bounds[idx + 1]
        thickness = base - top
        ok = np.isfinite(thickness) & (np.abs(thickness) > 1e-9)
        valid_positions = np.flatnonzero(valid)
        rows = valid_positions[ok]
        phase = (target_flat[rows] - top[ok]) / thickness[ok]
        prev_dist = target_flat[rows] - top[ok]
        next_dist = base[ok] - target_flat[rows]
        out[f'{prefix}_prev_boundary_dist'][rows] = prev_dist
        out[f'{prefix}_next_boundary_dist'][rows] = next_dist
        out[f'{prefix}_interval_thickness'][rows] = thickness[ok]
        out[f'{prefix}_interval_phase'][rows] = phase
        out[f'{prefix}_interval_phase_sin'][rows] = np.sin(np.pi * phase)
        out[f'{prefix}_interval_phase_cos'][rows] = np.cos(np.pi * phase)
        out[f'{prefix}_boundary_balance'][rows] = (next_dist - prev_dist) / (thickness[ok] + 1e-6)
        out[f'{prefix}_boundary_proximity'][rows] = np.minimum(prev_dist, next_dist) / (thickness[ok] + 1e-6)
    return {k: v.reshape(original_shape) for k, v in out.items()}


def typewell_interval_context_feature_names(prefix: str) -> list[str]:
    return [
        f'{prefix}_prev_boundary_dist',
        f'{prefix}_next_boundary_dist',
        f'{prefix}_interval_thickness',
        f'{prefix}_interval_phase',
        f'{prefix}_interval_phase_sin',
        f'{prefix}_interval_phase_cos',
        f'{prefix}_boundary_balance',
        f'{prefix}_boundary_proximity',
    ]

def boundary_count_between(boundaries: np.ndarray, start_tvt: float, target_tvt: np.ndarray) -> np.ndarray:
    target_tvt = np.asarray(target_tvt, dtype=float)
    if len(boundaries) == 0 or not np.isfinite(start_tvt):
        return np.full(len(target_tvt), np.nan)
    lo = np.minimum(start_tvt, target_tvt)
    hi = np.maximum(start_tvt, target_tvt)
    valid = np.isfinite(lo) & np.isfinite(hi)
    counts = np.full(len(target_tvt), np.nan)
    if valid.any():
        counts[valid] = ((boundaries[None, :] > lo[valid, None]) & (boundaries[None, :] <= hi[valid, None])).sum(axis=1)
    return counts


def nearest_boundary_distance(boundaries: np.ndarray, target_tvt: np.ndarray) -> np.ndarray:
    target_tvt = np.asarray(target_tvt, dtype=float)
    if len(boundaries) == 0:
        return np.full(len(target_tvt), np.nan)
    dist = np.full(len(target_tvt), np.nan)
    valid = np.isfinite(target_tvt)
    if valid.any():
        dist[valid] = np.min(np.abs(target_tvt[valid, None] - boundaries[None, :]), axis=1)
    return dist


def typewell_candidate_path_features(
    typewell_df: pd.DataFrame,
    last_known_tvt: float,
    tail_frac: np.ndarray,
    horizontal_gr: np.ndarray,
    endpoints: np.ndarray = CANDIDATE_PATH_ENDPOINTS,
    prefix: str = 'tw_path',
    score_scale: float = 20.0,
) -> dict[str, np.ndarray]:
    tail_frac = np.asarray(tail_frac, dtype=float)
    horizontal_gr = np.asarray(horizontal_gr, dtype=float)
    endpoints = np.asarray(endpoints, dtype=float)
    n = len(tail_frac)
    if n == 0 or len(endpoints) == 0 or not np.isfinite(last_known_tvt):
        return {name: np.full(n, np.nan) for name in candidate_path_feature_names(prefix, endpoints)}

    candidate_tvt = last_known_tvt + tail_frac[:, None] * endpoints[None, :]
    candidate_gr = typewell_gr_at_tvt(typewell_df, candidate_tvt.reshape(-1)).reshape(n, len(endpoints))
    gr_diff = horizontal_gr[:, None] - candidate_gr
    absdiff = np.abs(gr_diff)
    valid = np.isfinite(absdiff)
    masked = np.where(valid, absdiff, np.inf)
    any_valid = valid.any(axis=1)
    best_idx = np.argmin(masked, axis=1)
    best_abs = np.where(any_valid, masked[np.arange(n), best_idx], np.nan)
    best_endpoint = np.where(any_valid, endpoints[best_idx], np.nan)
    best_gr = np.where(any_valid, candidate_gr[np.arange(n), best_idx], np.nan)

    if len(endpoints) >= 2:
        sorted_abs = np.sort(masked, axis=1)
        top2_gap = np.full(n, np.nan)
        top2_ok = any_valid & np.isfinite(sorted_abs[:, 1])
        top2_gap[top2_ok] = sorted_abs[top2_ok, 1] - sorted_abs[top2_ok, 0]
    else:
        top2_gap = np.full(n, np.nan)

    weights = np.zeros_like(absdiff, dtype=float)
    weights[valid] = np.exp(-absdiff[valid] / score_scale)
    denom = weights.sum(axis=1)
    soft_endpoint = np.full(n, np.nan)
    denom_ok = denom > 0
    soft_endpoint[denom_ok] = (weights[denom_ok] * endpoints[None, :]).sum(axis=1) / denom[denom_ok]

    boundaries = typewell_boundary_tvt(typewell_df)
    max_endpoint = np.nanmax(np.abs(endpoints)) if len(endpoints) else np.nan
    features: dict[str, np.ndarray] = {}
    for j, endpoint in enumerate(endpoints):
        label = candidate_endpoint_label(float(endpoint))
        features[f'{prefix}_gr_diff_{label}'] = gr_diff[:, j]
        features[f'{prefix}_gr_absdiff_{label}'] = absdiff[:, j]
        features[f'{prefix}_boundary_count_{label}'] = boundary_count_between(boundaries, last_known_tvt, candidate_tvt[:, j])
        features[f'{prefix}_boundary_nearest_{label}'] = nearest_boundary_distance(boundaries, candidate_tvt[:, j])
    features[f'{prefix}_min_absdiff'] = best_abs
    features[f'{prefix}_best_endpoint'] = best_endpoint
    features[f'{prefix}_best_endpoint_centered'] = best_endpoint / max_endpoint if np.isfinite(max_endpoint) and max_endpoint > 0 else np.nan
    features[f'{prefix}_top2_absdiff_gap'] = top2_gap
    features[f'{prefix}_soft_endpoint_mean'] = soft_endpoint
    features[f'{prefix}_best_gr_resid'] = horizontal_gr - best_gr
    return features


FORMATION_TOP_COLUMNS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
FORMATION_LABELS = [name.lower() for name in FORMATION_TOP_COLUMNS]


def formation_feature_names(include_row: bool = True) -> list[str]:
    plane_names = [f'formation_plane_{label}' for label in FORMATION_LABELS]
    per_formation_formula_names: list[str] = []
    for label in FORMATION_LABELS:
        per_formation_formula_names.extend([
            f'formation_plane_anchor_b_{label}',
            f'formation_plane_anchor_b50_{label}',
            f'formation_plane_prefix_rmse_{label}',
            f'formation_plane_prefix_mae_{label}',
            f'formation_plane_tvt_formula_{label}',
            f'formation_plane_delta_formula_{label}',
            f'formation_plane_delta_formula50_{label}',
        ])
    names = [
        *plane_names,
        'formation_plane_min_dist',
        'formation_plane_anchor_b',
        'formation_plane_prefix_rmse',
        'formation_plane_prefix_mae',
        'formation_plane_tvt_formula',
        'formation_plane_delta_formula',
        'formation_plane_delta_from_slope_last200',
        *per_formation_formula_names,
        'formation_plane_formula_mean_delta',
        'formation_plane_formula_std_delta',
        'formation_plane_formula_min_delta',
        'formation_plane_formula_max_delta',
    ]
    if include_row:
        names += [
            'formation_row_ancc',
            'formation_row_ancc_std',
            'formation_row_min_dist',
            'formation_row_anchor_b',
            'formation_row_anchor_b50',
            'formation_row_prefix_rmse',
            'formation_row_prefix_mae',
            'formation_row_prefix_bias',
            'formation_row_tvt_formula',
            'formation_row_delta_formula',
            'formation_row_delta_formula50',
            'formation_row_delta_from_plane',
            'formation_formula_mean_delta',
            'formation_formula_abs_gap',
            'dense_ancc',
            'dense_std',
            'dense_dist',
            'dense_rmse',
            'dense_bias',
            'dense_nb_std',
            'tvt_dense_delta',
            'tvt_dense50_delta',
            'spatial_vs_dense',
            'pf_vs_spatial',
            'pf_vs_dense',
            'beam_vs_spatial',
            'beam_vs_dense',
            'beam_vs_pf_ancc',
        ]
    return names


def _safe_weighted_plane_predict(xy_query: np.ndarray, xy_neighbors: np.ndarray, values_neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xy_query = np.asarray(xy_query, dtype=float)
    n_query = len(xy_query)
    n_targets = values_neighbors.shape[2]
    out = np.full((n_query, n_targets), np.nan, dtype=np.float32)
    for r in range(n_query):
        valid = np.isfinite(weights[r]) & (weights[r] > 0)
        if valid.sum() < 3:
            if valid.any():
                sw = weights[r, valid].sum()
                out[r] = (values_neighbors[r, valid] * weights[r, valid, None]).sum(axis=0) / max(sw, 1e-12)
            continue
        x = xy_neighbors[r, valid, 0]
        y = xy_neighbors[r, valid, 1]
        w = weights[r, valid]
        X = np.column_stack([x, y, np.ones(len(x))])
        WX = X * w[:, None]
        ata = X.T @ WX
        ata.flat[::4] += 1e-8
        atb = X.T @ (values_neighbors[r, valid] * w[:, None])
        try:
            coef = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(ata) @ atb
        qx, qy = xy_query[r]
        out[r] = (qx * coef[0] + qy * coef[1] + coef[2]).astype(np.float32)
    return out


class FormationPlaneKNN:
    """Spatial plane-fit imputer for train-only formation top columns."""

    def __init__(self, well_ids, split_dir: Path, formations=FORMATION_TOP_COLUMNS, k: int = 10):
        self.formations = list(formations)
        self.k = int(k)
        rows = []
        for well_id in sorted(well_ids):
            path = split_dir / f'{well_id}__horizontal_well.csv'
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=lambda c: c in {'X', 'Y', *self.formations}).dropna(subset=['X', 'Y'])
            except Exception:
                continue
            if df.empty or not set(self.formations).issubset(df.columns):
                continue
            valid = df[['X', 'Y', *self.formations]].dropna()
            if valid.empty:
                continue
            row = {
                'well_id': well_id,
                'x': float(valid['X'].median()),
                'y': float(valid['Y'].median()),
            }
            for col in self.formations:
                row[col] = float(valid[col].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.ready = cKDTree is not None and len(self.df) >= 3
        if not self.ready:
            self.well_to_idx = {}
            self.xy = np.empty((0, 2), dtype=float)
            self.values = np.empty((0, len(self.formations)), dtype=float)
            self.scale = np.ones(2, dtype=float)
            self.tree = None
            return
        self.well_to_idx = {well_id: i for i, well_id in enumerate(self.df['well_id'].to_numpy())}
        self.xy = self.df[['x', 'y']].to_numpy(dtype=float)
        self.values = self.df[self.formations].to_numpy(dtype=float)
        self.scale = np.nanstd(self.xy, axis=0)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_query, self_wid=None, k: int | None = None):
        xy_query = np.atleast_2d(np.asarray(xy_query, dtype=float))
        n = len(xy_query)
        if not self.ready:
            return np.full((n, len(self.formations)), np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
        k = int(k or self.k)
        n_fetch = min(max(k + 5, k), len(self.df))
        dist, idx = self.tree.query(xy_query / self.scale, k=n_fetch)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if dist.shape[0] != n:
            dist = dist.reshape(n, -1)
            idx = idx.reshape(n, -1)
        if self_wid is not None and self_wid in self.well_to_idx:
            self_idx = self.well_to_idx[self_wid]
            dist = np.where(idx == self_idx, np.inf, dist)
        valid = np.isfinite(dist)
        safe_order_k = min(max(k - 1, 0), dist.shape[1] - 1)
        order = np.argpartition(np.where(valid, dist, np.inf), kth=safe_order_k, axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        weights = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0)
        xy_neighbors = self.xy[idx_k]
        value_neighbors = self.values[idx_k]
        pred = _safe_weighted_plane_predict(xy_query, xy_neighbors, value_neighbors, weights)
        no_neighbor = weights.sum(axis=1) <= 1e-12
        if no_neighbor.any():
            pred[no_neighbor] = np.nanmean(self.values, axis=0).astype(np.float32)
        min_dist = np.where(valid_k, d_k, np.inf).min(axis=1).astype(np.float32)
        min_dist[~np.isfinite(min_dist)] = np.nan
        return pred.astype(np.float32), min_dist


class RowANCCKNN:
    """Sampled row-level KNN imputer for ANCC only."""

    def __init__(self, well_ids, split_dir: Path, samples_per_well: int = 400, seed: int = 42):
        xs, ys, vals, wids = [], [], [], []
        rng = np.random.default_rng(seed)
        for well_id in sorted(well_ids):
            path = split_dir / f'{well_id}__horizontal_well.csv'
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=lambda c: c in {'X', 'Y', 'ANCC'}).dropna()
            except Exception:
                continue
            if df.empty:
                continue
            if len(df) > samples_per_well:
                idx = np.sort(rng.choice(df.index.to_numpy(), size=samples_per_well, replace=False))
                df = df.loc[idx]
            xs.append(df['X'].to_numpy(dtype=float))
            ys.append(df['Y'].to_numpy(dtype=float))
            vals.append(df['ANCC'].to_numpy(dtype=float))
            wids.extend([well_id] * len(df))
        self.ready = cKDTree is not None and bool(xs)
        if not self.ready:
            self.xy = np.empty((0, 2), dtype=float)
            self.ancc = np.empty(0, dtype=float)
            self.wids = np.empty(0, dtype=object)
            self.scale = np.ones(2, dtype=float)
            self.tree = None
            return
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(vals).astype(np.float32)
        self.wids = np.asarray(wids, dtype=object)
        self.scale = np.nanstd(self.xy, axis=0)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_query, self_wid=None, k: int = 20, extra_fetch: int = 450):
        xy_query = np.atleast_2d(np.asarray(xy_query, dtype=float))
        n = len(xy_query)
        if not self.ready:
            return (
                np.full(n, np.nan, dtype=np.float32),
                np.full(n, np.nan, dtype=np.float32),
                np.full(n, np.nan, dtype=np.float32),
            )
        n_fetch = min(max(k + extra_fetch, k), len(self.ancc))
        dist, idx = self.tree.query(xy_query / self.scale, k=n_fetch, workers=-1)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        if dist.shape[0] != n:
            dist = dist.reshape(n, -1)
            idx = idx.reshape(n, -1)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        valid = np.isfinite(dist)
        safe_order_k = min(max(k - 1, 0), dist.shape[1] - 1)
        order = np.argpartition(np.where(valid, dist, np.inf), kth=safe_order_k, axis=1)[:, :k]
        d_k = np.take_along_axis(dist, order, axis=1)
        idx_k = np.take_along_axis(idx, order, axis=1)
        valid_k = np.isfinite(d_k)
        weights = np.where(valid_k, 1.0 / (d_k + 1e-3), 0.0)
        sw = weights.sum(axis=1)
        safe = np.where(sw <= 1e-12, 1.0, sw)
        neighbor_vals = self.ancc[idx_k]
        pred = (neighbor_vals * weights).sum(axis=1) / safe
        no_neighbor = sw <= 1e-12
        if no_neighbor.any():
            pred[no_neighbor] = float(np.nanmean(self.ancc))
        var = (((neighbor_vals - pred[:, None]) ** 2) * weights).sum(axis=1) / safe
        std = np.sqrt(np.maximum(var, 0.0))
        min_dist = np.where(valid_k, d_k, np.inf).min(axis=1)
        min_dist[~np.isfinite(min_dist)] = np.nan
        return pred.astype(np.float32), std.astype(np.float32), min_dist.astype(np.float32)


def make_formation_imputers(well_ids, split_dir: Path, need_row_ancc: bool = False, seed: int = 42):
    plane = FormationPlaneKNN(well_ids, split_dir=split_dir, k=10)
    row = RowANCCKNN(well_ids, split_dir=split_dir, samples_per_well=400, seed=seed) if need_row_ancc else None
    return plane, row


def make_tail_features_for_well(
    well_id: str,
    split_dir: Path,
    include_target: bool = True,
    use_beam_features: bool | None = None,
    required_feature_columns=None,
    formation_plane_imputer=None,
    row_ancc_imputer=None,
    exclude_query_well_from_formation: bool = True,
) -> pd.DataFrame:
    if use_beam_features is None:
        use_beam_features = bool(globals().get('ENABLE_OFFLINE_BEAM_FEATURES', False))
    required_feature_columns = normalize_required_feature_columns(required_feature_columns)
    need_trailing_gr_roll = needs_any(required_feature_columns, prefixes=('gr_roll_',))
    need_offline_gr_context = needs_any(
        required_feature_columns,
        prefixes=('gr_center_',),
        exact=('gr_cumsum_since_ps', 'gr_cumsum_anom_since_ps'),
    )
    need_typewell_features = needs_any(
        required_feature_columns,
        prefixes=(
            'typewell_', 'gr_minus_typewell_', 'prefix_horizontal_vs_typewell_',
            'calibrated_', 'gr_minus_calibrated_', 'prefix_typewell_',
            'tw_path_', 'tw_path_ease_', 'tw_beam_', 'beam_', 'beam_vs_', 'pf_tvt_', 'pf_z_', 'tw_sc_anchor_gr_diff_',
        ),
    )
    need_typewell_slope_features = needs_any(
        required_feature_columns,
        prefixes=('gr_minus_typewell_slope_',),
        exact=('typewell_gr_at_slope_baseline_all', 'typewell_gr_at_slope_baseline_last200'),
    )
    need_local_typewell = needs_any(required_feature_columns, prefixes=('typewell_local_last200_',))
    need_calibrated_typewell = needs_any(
        required_feature_columns,
        prefixes=('calibrated_', 'gr_minus_calibrated_'),
        exact=(
            'prefix_typewell_gr_affine_a',
            'prefix_typewell_gr_affine_b',
            'prefix_horizontal_vs_calibrated_typewell_gr_mae',
            'prefix_horizontal_vs_calibrated_typewell_gr_rmse',
            'typewell_calibrated_gr_at_last_known_tvt',
            'typewell_calibrated_gr_at_slope_baseline_all',
            'typewell_calibrated_gr_at_slope_baseline_last200',
        ),
    )
    need_local_calibrated_typewell = needs_any(required_feature_columns, prefixes=('calibrated_typewell_local_last200_',))
    need_typewell_interval_context = needs_any(
        required_feature_columns,
        prefixes=('typewell_last_geo_', 'typewell_baseline_last200_geo_'),
    )
    need_typewell_anchor_offsets = needs_any(required_feature_columns, prefixes=('typewell_anchor_gr_diff_',))
    need_calibrated_anchor_offsets = needs_any(required_feature_columns, prefixes=('calibrated_typewell_anchor_gr_diff_',))
    need_tw_beam_anchor_offsets = needs_any(required_feature_columns, prefixes=('tw_beam_cons_anchor_gr_diff_',))
    need_tw_sc_anchor_offsets = needs_any(required_feature_columns, prefixes=('tw_sc_anchor_gr_diff_',))
    need_candidate_path = needs_any(required_feature_columns, prefixes=('tw_path_', 'tw_path_ease_'))
    need_beam = bool(use_beam_features and needs_any(required_feature_columns, prefixes=('tw_beam_', 'beam_', 'beam_vs_')))
    need_pf_tvt = needs_any(required_feature_columns, prefixes=('pf_tvt_',))
    need_pf_z = needs_any(required_feature_columns, prefixes=('pf_z_',))
    need_self_corr = needs_any(required_feature_columns, prefixes=('sc_', 'hyb_', 'tw_sc_anchor_gr_diff_'))
    need_pf_ancc = needs_any(required_feature_columns, prefixes=('pf_ancc_',))
    need_formation_plane = needs_any(required_feature_columns, prefixes=('formation_plane_', 'spatial_vs_', 'pf_vs_spatial', 'beam_vs_spatial'))
    need_row_ancc = needs_any(required_feature_columns, prefixes=('formation_row_', 'formation_formula_', 'dense_', 'tvt_dense_', 'spatial_vs_', 'pf_vs_dense', 'beam_vs_dense')) or need_pf_ancc
    need_formation_features = need_formation_plane or need_row_ancc or need_pf_ancc
    path = split_dir / f'{well_id}__horizontal_well.csv'
    df = pd.read_csv(path)
    mask = df['TVT_input'].isna().to_numpy()
    pred_idx = np.flatnonzero(mask)
    if len(pred_idx) == 0:
        return pd.DataFrame()
    ps = int(pred_idx[0])
    tail = df.iloc[pred_idx].copy()
    prefix = df.iloc[:ps].copy()
    last_known_tvt = float(prefix['TVT_input'].iloc[-1])
    last_known_md = float(prefix['MD'].iloc[-1])
    ps_md = float(df['MD'].iloc[ps])
    ps_x = float(df['X'].iloc[ps])
    ps_y = float(df['Y'].iloc[ps])
    ps_z = float(df['Z'].iloc[ps])
    prefix_tvt = pd.to_numeric(prefix['TVT_input'], errors='coerce')
    prefix_gr = pd.to_numeric(prefix['GR'], errors='coerce')
    tail_gr = pd.to_numeric(tail['GR'], errors='coerce').to_numpy(dtype=float)
    prefix_gr_mean = safe_stat(prefix_gr, np.mean)
    prefix_gr_std = safe_stat(prefix_gr, np.std)
    prefix_gr_last5_mean = safe_stat(prefix_gr.tail(5), np.mean)
    prefix_gr_last5_std = safe_stat(prefix_gr.tail(5), np.std)
    prefix_gr_last20_mean = safe_stat(prefix_gr.tail(20), np.mean)
    prefix_gr_last20_std = safe_stat(prefix_gr.tail(20), np.std)
    prefix_gr_last100_mean = safe_stat(prefix_gr.tail(100), np.mean)
    prefix_gr_last100_std = safe_stat(prefix_gr.tail(100), np.std)
    prefix_tvt_values = prefix_tvt.to_numpy(dtype=float)
    prefix_md_values = pd.to_numeric(prefix['MD'], errors='coerce').to_numpy(dtype=float)
    prefix_z_values = pd.to_numeric(prefix['Z'], errors='coerce').to_numpy(dtype=float)
    prefix_tvt_slope_all = safe_slope(prefix_md_values, prefix_tvt_values)
    prefix_tvt_slope_last200 = safe_slope(prefix['MD'].tail(200).to_numpy(), prefix_tvt.tail(200).to_numpy())
    prefix_tvt_step20 = recent_mean_diff(prefix_tvt_values, 20)
    prefix_tvt_step100 = recent_mean_diff(prefix_tvt_values, 100)
    prefix_tvt_md_slope100 = recent_slope_window(prefix_tvt_values, prefix_md_values, 100)
    prefix_tvt_z_slope100 = recent_slope_window(prefix_tvt_values, prefix_z_values, 100)
    
    # Use only the known prefix to estimate orientation. Full-well endpoint azimuth would use future tail information.
    if len(prefix) >= 2:
        dx_prefix = float(prefix['X'].iloc[-1] - prefix['X'].iloc[0])
        dy_prefix = float(prefix['Y'].iloc[-1] - prefix['Y'].iloc[0])
        prefix_azimuth = (np.degrees(np.arctan2(dx_prefix, dy_prefix)) + 360.0) % 360.0
    else:
        prefix_azimuth = np.nan
    
    md_since_ps = tail['MD'].to_numpy(dtype=float) - ps_md
    tail_len = len(tail)
    tail_row_number = np.arange(tail_len, dtype=float)
    tail_frac = tail_row_number / max(tail_len - 1, 1)
    n_rows = len(df)
    row_frac = pred_idx.astype(float) / max(n_rows - 1, 1)
    md_tail_span = float(tail['MD'].iloc[-1] - ps_md) if len(tail) else np.nan
    md_tail_frac = md_since_ps / md_tail_span if np.isfinite(md_tail_span) and abs(md_tail_span) > 1e-9 else np.full(tail_len, np.nan)
    tail_gr_missing_rate = float(pd.to_numeric(tail['GR'], errors='coerce').isna().mean())
    tail_frac2 = tail_frac ** 2
    tail_frac3 = tail_frac ** 3
    sqrt_tail_frac = np.sqrt(np.clip(tail_frac, 0.0, None))
    log1p_tail_row = np.log1p(tail_row_number)
    sin_tail_frac_pi = np.sin(np.pi * tail_frac)
    sin_tail_frac_2pi = np.sin(2.0 * np.pi * tail_frac)
    cos_tail_frac_3pi = np.cos(3.0 * np.pi * tail_frac)
    gap_x_delta = float(tail['X'].iloc[-1] - tail['X'].iloc[0]) if len(tail) else np.nan
    gap_y_delta = float(tail['Y'].iloc[-1] - tail['Y'].iloc[0]) if len(tail) else np.nan
    gap_z_delta = float(tail['Z'].iloc[-1] - tail['Z'].iloc[0]) if len(tail) else np.nan
    gap_xy_span = float(np.hypot(gap_x_delta, gap_y_delta)) if np.isfinite(gap_x_delta) and np.isfinite(gap_y_delta) else np.nan
    gap_z_over_xy = gap_z_delta / (gap_xy_span + 1.0) if np.isfinite(gap_z_delta) and np.isfinite(gap_xy_span) else np.nan
    finite_tail_gr = tail_gr[np.isfinite(tail_gr)]
    if len(finite_tail_gr):
        gap_gr_mean = float(np.mean(finite_tail_gr))
        gap_gr_std = float(np.std(finite_tail_gr))
        gap_gr_min = float(np.min(finite_tail_gr))
        gap_gr_max = float(np.max(finite_tail_gr))
        gap_gr_q = {q: float(np.quantile(finite_tail_gr, q)) for q in [0.05, 0.25, 0.50, 0.75, 0.95]}
    else:
        gap_gr_mean = gap_gr_std = gap_gr_min = gap_gr_max = np.nan
        gap_gr_q = {q: np.nan for q in [0.05, 0.25, 0.50, 0.75, 0.95]}
    x_delta_ps = tail['X'].to_numpy(dtype=float) - ps_x
    y_delta_ps = tail['Y'].to_numpy(dtype=float) - ps_y
    z_delta_ps = tail['Z'].to_numpy(dtype=float) - ps_z
    safe_md_since_ps = np.where(np.abs(md_since_ps) > 1e-9, md_since_ps, np.nan)
    dist_xyz_ps = np.sqrt(x_delta_ps ** 2 + y_delta_ps ** 2 + z_delta_ps ** 2)
    slope_delta_all = prefix_tvt_slope_all * md_since_ps if np.isfinite(prefix_tvt_slope_all) else np.full(len(tail), np.nan)
    slope_delta_last200 = prefix_tvt_slope_last200 * md_since_ps if np.isfinite(prefix_tvt_slope_last200) else np.full(len(tail), np.nan)
    slope_baseline_tvt_all = last_known_tvt + slope_delta_all
    slope_baseline_tvt_last200 = last_known_tvt + slope_delta_last200
    prefix_valid_gr = prefix_gr.dropna()
    if len(prefix_valid_gr):
        prefix_last_valid_gr = float(prefix_valid_gr.iloc[-1])
        prefix_last_valid_gr_index = int(prefix_valid_gr.index[-1])
        prefix_last_valid_gr_md = float(df['MD'].iloc[prefix_last_valid_gr_index])
    else:
        prefix_last_valid_gr = np.nan
        prefix_last_valid_gr_index = -1
        prefix_last_valid_gr_md = np.nan
    gr_prefix_z = (tail_gr - prefix_gr_mean) / (prefix_gr_std + 1e-6) if np.isfinite(prefix_gr_std) else np.full(len(tail), np.nan)
    
    md_full = pd.to_numeric(df['MD'], errors='coerce')
    x_full = pd.to_numeric(df['X'], errors='coerce')
    y_full = pd.to_numeric(df['Y'], errors='coerce')
    z_full = pd.to_numeric(df['Z'], errors='coerce')
    gr_full_numeric = pd.to_numeric(df['GR'], errors='coerce')
    md_step_1 = md_full.diff(1).iloc[pred_idx].to_numpy(dtype=float)
    x_step_1 = x_full.diff(1).iloc[pred_idx].to_numpy(dtype=float)
    y_step_1 = y_full.diff(1).iloc[pred_idx].to_numpy(dtype=float)
    z_step_1 = z_full.diff(1).iloc[pred_idx].to_numpy(dtype=float)
    md_step_5 = md_full.diff(5).iloc[pred_idx].to_numpy(dtype=float)
    x_step_5 = x_full.diff(5).iloc[pred_idx].to_numpy(dtype=float)
    y_step_5 = y_full.diff(5).iloc[pred_idx].to_numpy(dtype=float)
    z_step_5 = z_full.diff(5).iloc[pred_idx].to_numpy(dtype=float)
    x_d2_1 = x_full.diff(1).diff(1).iloc[pred_idx].to_numpy(dtype=float)
    y_d2_1 = y_full.diff(1).diff(1).iloc[pred_idx].to_numpy(dtype=float)
    z_d2_1 = z_full.diff(1).diff(1).iloc[pred_idx].to_numpy(dtype=float)
    gr_diff_1 = gr_full_numeric.diff(1).iloc[pred_idx].to_numpy(dtype=float)
    gr_diff_5 = gr_full_numeric.diff(5).iloc[pred_idx].to_numpy(dtype=float)
    gr_diff_10 = gr_full_numeric.diff(10).iloc[pred_idx].to_numpy(dtype=float)
    safe_md_step_1 = np.where(np.abs(md_step_1) > 1e-9, md_step_1, np.nan)
    safe_md_step_5 = np.where(np.abs(md_step_5) > 1e-9, md_step_5, np.nan)
    trajectory_step_1 = np.sqrt(x_step_1 ** 2 + y_step_1 ** 2 + z_step_1 ** 2)
    trajectory_step_5 = np.sqrt(x_step_5 ** 2 + y_step_5 ** 2 + z_step_5 ** 2)
    trajectory_curvature_1 = np.sqrt(x_d2_1 ** 2 + y_d2_1 ** 2 + z_d2_1 ** 2) / (trajectory_step_1 + 1.0)
    azimuth_5 = np.arctan2(y_step_5, x_step_5)
    trajectory_step_5_safe = np.where(trajectory_step_5 > 1e-9, trajectory_step_5, np.nan)
    
    out = pd.DataFrame({
        'well_id': well_id,
        'row_index': pred_idx,
        'id': [f'{well_id}_{i}' for i in pred_idx],
        'MD': tail['MD'].to_numpy(),
        'X': tail['X'].to_numpy(),
        'Y': tail['Y'].to_numpy(),
        'Z': tail['Z'].to_numpy(),
        'GR': tail_gr,
        'GR_isna': tail['GR'].isna().astype(int).to_numpy(),
        'GR_prefix_z': gr_prefix_z,
        'gr_diff_1': gr_diff_1,
        'gr_diff_5': gr_diff_5,
        'gr_diff_10': gr_diff_10,
        'gr_slope_md_1': gr_diff_1 / safe_md_step_1,
        'gr_slope_md_5': gr_diff_5 / safe_md_step_5,
        'md_step_1': md_step_1,
        'md_step_5': md_step_5,
        'x_step_1': x_step_1,
        'y_step_1': y_step_1,
        'z_step_1': z_step_1,
        'x_step_5': x_step_5,
        'y_step_5': y_step_5,
        'z_step_5': z_step_5,
        'x_d2_1': x_d2_1,
        'y_d2_1': y_d2_1,
        'z_d2_1': z_d2_1,
        'trajectory_step_1': trajectory_step_1,
        'trajectory_step_5': trajectory_step_5,
        'trajectory_curvature_1': trajectory_curvature_1,
        'trajectory_azimuth_sin_5': np.sin(azimuth_5),
        'trajectory_azimuth_cos_5': np.cos(azimuth_5),
        'trajectory_inclination_sin_5': z_step_5 / trajectory_step_5_safe,
        'z_slope_md_1': z_step_1 / safe_md_step_1,
        'z_slope_md_5': z_step_5 / safe_md_step_5,
        'last_known_TVT': last_known_tvt,
        'last_known_MD': last_known_md,
        'tail_len': tail_len,
        'tail_row_number': tail_row_number,
        'tail_frac': tail_frac,
        'tail_frac2': tail_frac2,
        'tail_frac3': tail_frac3,
        'sqrt_tail_frac': sqrt_tail_frac,
        'log1p_tail_row': log1p_tail_row,
        'sin_tail_frac_pi': sin_tail_frac_pi,
        'sin_tail_frac_2pi': sin_tail_frac_2pi,
        'cos_tail_frac_3pi': cos_tail_frac_3pi,
        'n_rows': n_rows,
        'row_frac': row_frac,
        'md_tail_span': md_tail_span,
        'md_tail_frac': md_tail_frac,
        'gap_md_span': md_tail_span,
        'gap_x_delta': gap_x_delta,
        'gap_y_delta': gap_y_delta,
        'gap_z_delta': gap_z_delta,
        'gap_xy_span': gap_xy_span,
        'gap_z_over_xy': gap_z_over_xy,
        'gap_gr_mean': gap_gr_mean,
        'gap_gr_std': gap_gr_std,
        'gap_gr_min': gap_gr_min,
        'gap_gr_p05': gap_gr_q[0.05],
        'gap_gr_p25': gap_gr_q[0.25],
        'gap_gr_p50': gap_gr_q[0.50],
        'gap_gr_p75': gap_gr_q[0.75],
        'gap_gr_p95': gap_gr_q[0.95],
        'gap_gr_max': gap_gr_max,
        'tail_gr_missing_rate': tail_gr_missing_rate,
        'md_since_ps': md_since_ps,
        'x_delta_ps': x_delta_ps,
        'y_delta_ps': y_delta_ps,
        'z_delta_ps': z_delta_ps,
        'xy_dist_ps': np.hypot(x_delta_ps, y_delta_ps),
        'dist_xyz_ps': dist_xyz_ps,
        'dx_per_md_since_ps': x_delta_ps / safe_md_since_ps,
        'dy_per_md_since_ps': y_delta_ps / safe_md_since_ps,
        'dz_per_md_since_ps': z_delta_ps / safe_md_since_ps,
        'prefix_len': len(prefix),
        'prefix_azimuth_deg': prefix_azimuth,
        'prefix_gr_missing_rate': float(prefix_gr.isna().mean()),
        'prefix_gr_mean': prefix_gr_mean,
        'prefix_gr_std': prefix_gr_std,
        'prefix_gr_last5_mean': prefix_gr_last5_mean,
        'prefix_gr_last5_std': prefix_gr_last5_std,
        'prefix_gr_last20_mean': prefix_gr_last20_mean,
        'prefix_gr_last20_std': prefix_gr_last20_std,
        'prefix_gr_last100_mean': prefix_gr_last100_mean,
        'prefix_gr_last100_std': prefix_gr_last100_std,
        'prefix_gr_min': safe_stat(prefix_gr, np.min),
        'prefix_gr_max': safe_stat(prefix_gr, np.max),
        'prefix_last_valid_gr': prefix_last_valid_gr,
        'rows_since_prefix_last_valid_gr': pred_idx - prefix_last_valid_gr_index if prefix_last_valid_gr_index >= 0 else np.nan,
        'md_since_prefix_last_valid_gr': tail['MD'].to_numpy(dtype=float) - prefix_last_valid_gr_md,
        'gr_minus_prefix_last_valid_gr': tail_gr - prefix_last_valid_gr,
        'gr_minus_prefix_gr_mean': tail_gr - prefix_gr_mean,
        'prefix_tvt_min': safe_stat(prefix_tvt, np.min),
        'prefix_tvt_max': safe_stat(prefix_tvt, np.max),
        'prefix_tvt_range': safe_stat(prefix_tvt, np.max) - safe_stat(prefix_tvt, np.min),
        'prefix_tvt_mean': safe_stat(prefix_tvt, np.mean),
        'prefix_tvt_std': safe_stat(prefix_tvt, np.std),
        'prefix_tvt_slope_md_all': prefix_tvt_slope_all,
        'prefix_tvt_slope_md_last200': prefix_tvt_slope_last200,
        'prefix_tvt_step20': prefix_tvt_step20,
        'prefix_tvt_step100': prefix_tvt_step100,
        'prefix_tvt_md_slope100': prefix_tvt_md_slope100,
        'prefix_tvt_z_slope100': prefix_tvt_z_slope100,
        'slope_baseline_delta_all': slope_delta_all,
        'slope_baseline_delta_last200': slope_delta_last200,
        'slope_baseline_tvt_all': slope_baseline_tvt_all,
        'slope_baseline_tvt_last200': slope_baseline_tvt_last200,
    })

    # Z-velocity PF-style prior: prefix-calibrated TVT-vs-Z and TVT-vs-MD motion.
    if need_pf_z:
        tail_z_for_pf = tail['Z'].to_numpy(dtype=float)
        z_slope_last200 = safe_slope(prefix['Z'].tail(200).to_numpy(dtype=float), prefix_tvt.tail(200).to_numpy(dtype=float))
        z_slope_all = safe_slope(prefix_z_values, prefix_tvt_values)
        z_velocity = z_slope_last200 if np.isfinite(z_slope_last200) else z_slope_all
        z_delta_model = z_velocity * (tail_z_for_pf - ps_z) if np.isfinite(z_velocity) else np.full(len(tail), np.nan)
        md_delta_model = slope_delta_last200 if np.isfinite(prefix_tvt_slope_last200) else slope_delta_all
        pf_z_delta = np.where(np.isfinite(z_delta_model), z_delta_model, md_delta_model)
        both_valid = np.isfinite(z_delta_model) & np.isfinite(md_delta_model)
        pf_z_delta = np.where(both_valid, 0.65 * z_delta_model + 0.35 * md_delta_model, pf_z_delta)
        prefix_tail_z = prefix['Z'].tail(200).to_numpy(dtype=float)
        prefix_tail_tvt = prefix_tvt.tail(200).to_numpy(dtype=float)
        if np.isfinite(z_velocity) and len(prefix_tail_z) >= 10:
            z_fit_resid = prefix_tail_tvt - (prefix_tail_tvt[-1] + z_velocity * (prefix_tail_z - prefix_tail_z[-1]))
            pf_z_base_std = float(np.nanstd(z_fit_resid))
        else:
            pf_z_base_std = np.nan
        pf_z_std = np.full(len(tail), pf_z_base_std if np.isfinite(pf_z_base_std) else 25.0, dtype=float) + 0.02 * np.abs(md_since_ps)
        out['pf_z_delta'] = pf_z_delta
        out['pf_z_std'] = pf_z_std
        out['pf_z_velocity_md'] = prefix_tvt_slope_last200
        out['pf_z_velocity_z'] = z_velocity
        out['pf_z_confidence'] = 1.0 / (1.0 + np.nan_to_num(pf_z_std, nan=50.0) / 20.0)
    
    # Rolling GR features are trailing only for strict mode.
    gr_full = df['GR'].copy()
    if need_trailing_gr_roll:
        for window in [25, 100, 300]:
            roll = gr_full.rolling(window, min_periods=max(5, window // 5))
            out[f'gr_roll_mean_{window}'] = roll.mean().iloc[pred_idx].to_numpy()
            out[f'gr_roll_std_{window}'] = roll.std().iloc[pred_idx].to_numpy()
            out[f'gr_roll_min_{window}'] = roll.min().iloc[pred_idx].to_numpy()
            out[f'gr_roll_max_{window}'] = roll.max().iloc[pred_idx].to_numpy()
            out[f'gr_roll_range_{window}'] = (roll.max() - roll.min()).iloc[pred_idx].to_numpy()
    
    # Offline features use target-free covariates from the full provided horizontal file.
    gr_full_filled = None
    if need_offline_gr_context:
        center_roll_cache = {}
        for window in [5, 21, 51, 151, 301]:
            center_roll = gr_full.rolling(window, center=True, min_periods=max(1, min(5, window // 2)))
            roll_mean = center_roll.mean()
            roll_std = center_roll.std()
            roll_min = center_roll.min()
            roll_max = center_roll.max()
            center_roll_cache[window] = (roll_mean, roll_std)
            out[f'gr_center_roll_mean_{window}'] = roll_mean.iloc[pred_idx].to_numpy()
            out[f'gr_center_roll_std_{window}'] = roll_std.iloc[pred_idx].to_numpy()
            out[f'gr_center_roll_min_{window}'] = roll_min.iloc[pred_idx].to_numpy()
            out[f'gr_center_roll_max_{window}'] = roll_max.iloc[pred_idx].to_numpy()
            out[f'gr_center_roll_range_{window}'] = (roll_max - roll_min).iloc[pred_idx].to_numpy()
        fallback_gr = prefix_gr_mean if np.isfinite(prefix_gr_mean) else 0.0
        gr_full_filled = gr_full_numeric.interpolate(limit_direction='both').fillna(fallback_gr)
        gr_cumsum = gr_full_filled.cumsum()
        gr_anom_cumsum = (gr_full_filled - fallback_gr).cumsum()
        cumsum_anchor = float(gr_cumsum.iloc[ps - 1]) if ps > 0 else 0.0
        anom_cumsum_anchor = float(gr_anom_cumsum.iloc[ps - 1]) if ps > 0 else 0.0
        out['gr_center_grad_1'] = gr_full_filled.diff().fillna(0.0).iloc[pred_idx].to_numpy()
        out['gr_center_grad_5'] = (gr_full_filled.shift(-5).ffill() - gr_full_filled.shift(5).bfill()).iloc[pred_idx].to_numpy() / 10.0
        out['gr_center_lag1'] = gr_full_filled.shift(1).bfill().iloc[pred_idx].to_numpy()
        out['gr_center_lead1'] = gr_full_filled.shift(-1).ffill().iloc[pred_idx].to_numpy()
        out['gr_center_lag5'] = gr_full_filled.shift(5).bfill().iloc[pred_idx].to_numpy()
        out['gr_center_lead5'] = gr_full_filled.shift(-5).ffill().iloc[pred_idx].to_numpy()
        out['gr_center_lag10'] = gr_full_filled.shift(10).bfill().iloc[pred_idx].to_numpy()
        out['gr_center_lead10'] = gr_full_filled.shift(-10).ffill().iloc[pred_idx].to_numpy()
        out['gr_center_lag20'] = gr_full_filled.shift(20).bfill().iloc[pred_idx].to_numpy()
        out['gr_center_lead20'] = gr_full_filled.shift(-20).ffill().iloc[pred_idx].to_numpy()
        out['gr_cumsum_since_ps'] = (gr_cumsum.iloc[pred_idx] - cumsum_anchor).to_numpy()
        out['gr_cumsum_anom_since_ps'] = (gr_anom_cumsum.iloc[pred_idx] - anom_cumsum_anchor).to_numpy()
        for window in [5, 21, 51]:
            roll_mean, roll_std = center_roll_cache[window]
            highfreq = gr_full_filled - roll_mean
            out[f'gr_center_highfreq_{window}'] = highfreq.iloc[pred_idx].to_numpy()
            out[f'gr_center_z_{window}'] = (highfreq / (roll_std + 1e-6)).iloc[pred_idx].to_numpy()
    
    # Formation top features use train-only surfaces only as auxiliary labels for a spatial imputer.
    if need_formation_features:
        self_wid = well_id if exclude_query_well_from_formation else None
        prefix_xy = prefix[['X', 'Y']].to_numpy(dtype=float)
        tail_xy = tail[['X', 'Y']].to_numpy(dtype=float)
        prefix_z = prefix['Z'].to_numpy(dtype=float)
        tail_z = tail['Z'].to_numpy(dtype=float)
        plane_delta = np.full(len(tail), np.nan, dtype=float)
        if need_formation_plane and formation_plane_imputer is not None and getattr(formation_plane_imputer, 'ready', False):
            plane_prefix, _ = formation_plane_imputer.impute(prefix_xy, self_wid=self_wid)
            plane_tail, plane_dist = formation_plane_imputer.impute(tail_xy, self_wid=self_wid)
            out['formation_plane_min_dist'] = plane_dist
            plane_delta_stack = []
            for label_idx, label in enumerate(FORMATION_LABELS):
                plane_prefix_surface = plane_prefix[:, label_idx]
                plane_tail_surface = plane_tail[:, label_idx]
                out[f'formation_plane_{label}'] = plane_tail_surface
                valid_b = np.isfinite(prefix_tvt_values) & np.isfinite(prefix_z) & np.isfinite(plane_prefix_surface)
                anchor_series = prefix_tvt_values + prefix_z - plane_prefix_surface
                anchor_b = float(np.nanmedian(anchor_series[valid_b])) if valid_b.any() else np.nan
                if valid_b.any():
                    valid_idx = np.flatnonzero(valid_b)
                    tail_idx = valid_idx[-min(50, len(valid_idx)):]
                    anchor_b50 = float(np.nanmedian(anchor_series[tail_idx]))
                else:
                    anchor_b50 = np.nan
                tvt_formula = -tail_z + plane_tail_surface + anchor_b
                tvt_formula50 = -tail_z + plane_tail_surface + anchor_b50
                prefix_formula = -prefix_z + plane_prefix_surface + anchor_b
                prefix_valid_formula = valid_b & np.isfinite(prefix_formula)
                prefix_resid = prefix_tvt_values[prefix_valid_formula] - prefix_formula[prefix_valid_formula]
                delta_formula = tvt_formula - last_known_tvt
                out[f'formation_plane_anchor_b_{label}'] = anchor_b
                out[f'formation_plane_anchor_b50_{label}'] = anchor_b50
                out[f'formation_plane_prefix_rmse_{label}'] = float(np.sqrt(np.mean(prefix_resid ** 2))) if len(prefix_resid) else np.nan
                out[f'formation_plane_prefix_mae_{label}'] = float(np.mean(np.abs(prefix_resid))) if len(prefix_resid) else np.nan
                out[f'formation_plane_tvt_formula_{label}'] = tvt_formula
                out[f'formation_plane_delta_formula_{label}'] = delta_formula
                out[f'formation_plane_delta_formula50_{label}'] = tvt_formula50 - last_known_tvt
                plane_delta_stack.append(delta_formula)
                if label == 'ancc':
                    out['formation_plane_anchor_b'] = anchor_b
                    out['formation_plane_prefix_rmse'] = out[f'formation_plane_prefix_rmse_{label}']
                    out['formation_plane_prefix_mae'] = out[f'formation_plane_prefix_mae_{label}']
                    out['formation_plane_tvt_formula'] = tvt_formula
                    out['formation_plane_delta_formula'] = delta_formula
                    out['formation_plane_delta_from_slope_last200'] = (tvt_formula - slope_baseline_tvt_last200)
            if plane_delta_stack:
                plane_delta_matrix = np.vstack(plane_delta_stack)
                out['formation_plane_formula_mean_delta'] = np.nanmean(plane_delta_matrix, axis=0)
                out['formation_plane_formula_std_delta'] = np.nanstd(plane_delta_matrix, axis=0)
                out['formation_plane_formula_min_delta'] = np.nanmin(plane_delta_matrix, axis=0)
                out['formation_plane_formula_max_delta'] = np.nanmax(plane_delta_matrix, axis=0)
            plane_delta = np.asarray(out.get('formation_plane_delta_formula', np.full(len(tail), np.nan)), dtype=float)
        row_delta = np.full(len(tail), np.nan, dtype=float)
        if need_row_ancc and row_ancc_imputer is not None and getattr(row_ancc_imputer, 'ready', False):
            row_prefix_ancc, row_prefix_std, _ = row_ancc_imputer.impute(prefix_xy, self_wid=self_wid)
            row_tail_ancc, row_tail_std, row_dist = row_ancc_imputer.impute(tail_xy, self_wid=self_wid)
            valid_b = np.isfinite(prefix_tvt_values) & np.isfinite(prefix_z) & np.isfinite(row_prefix_ancc)
            row_anchor_series = prefix_tvt_values + prefix_z - row_prefix_ancc
            row_anchor_b = float(np.nanmedian(row_anchor_series[valid_b])) if valid_b.any() else np.nan
            if valid_b.any():
                valid_idx = np.flatnonzero(valid_b)
                tail_idx = valid_idx[-min(50, len(valid_idx)):]
                row_anchor_b50 = float(np.nanmedian(row_anchor_series[tail_idx]))
            else:
                row_anchor_b50 = np.nan
            row_tvt_formula = -tail_z + row_tail_ancc + row_anchor_b
            row_tvt_formula50 = -tail_z + row_tail_ancc + row_anchor_b50
            row_prefix_formula = -prefix_z + row_prefix_ancc + row_anchor_b
            prefix_valid_formula = valid_b & np.isfinite(row_prefix_formula)
            row_prefix_resid = prefix_tvt_values[prefix_valid_formula] - row_prefix_formula[prefix_valid_formula]
            out['formation_row_ancc'] = row_tail_ancc
            out['formation_row_ancc_std'] = row_tail_std
            out['formation_row_min_dist'] = row_dist
            out['formation_row_anchor_b'] = row_anchor_b
            out['formation_row_anchor_b50'] = row_anchor_b50
            out['formation_row_prefix_rmse'] = float(np.sqrt(np.mean(row_prefix_resid ** 2))) if len(row_prefix_resid) else np.nan
            out['formation_row_prefix_mae'] = float(np.mean(np.abs(row_prefix_resid))) if len(row_prefix_resid) else np.nan
            out['formation_row_prefix_bias'] = float(np.mean(row_prefix_resid)) if len(row_prefix_resid) else np.nan
            out['formation_row_tvt_formula'] = row_tvt_formula
            out['formation_row_delta_formula'] = row_tvt_formula - last_known_tvt
            out['formation_row_delta_formula50'] = row_tvt_formula50 - last_known_tvt
            row_delta = np.asarray(out['formation_row_delta_formula'], dtype=float)
            out['formation_row_delta_from_plane'] = row_delta - plane_delta
            out['dense_ancc'] = row_tail_ancc
            out['dense_std'] = row_tail_std
            out['dense_dist'] = row_dist
            out['dense_rmse'] = out['formation_row_prefix_rmse']
            out['dense_bias'] = out['formation_row_prefix_bias']
            out['dense_nb_std'] = row_tail_std
            out['tvt_dense_delta'] = row_delta
            out['tvt_dense50_delta'] = out['formation_row_delta_formula50']
            out['spatial_vs_dense'] = plane_delta - row_delta
            stacked_delta = np.vstack([plane_delta, row_delta])
            finite_count = np.isfinite(stacked_delta).sum(axis=0)
            summed_delta = np.nansum(stacked_delta, axis=0)
            out['formation_formula_mean_delta'] = np.divide(
                summed_delta,
                finite_count,
                out=np.full(len(tail), np.nan, dtype=float),
                where=finite_count > 0,
            )
            out['formation_formula_abs_gap'] = np.abs(row_delta - plane_delta)
        if need_pf_ancc:
            stacked_delta = np.vstack([plane_delta, row_delta])
            finite_count = np.isfinite(stacked_delta).sum(axis=0)
            summed_delta = np.nansum(stacked_delta, axis=0)
            pf_ancc_delta = np.divide(
                summed_delta,
                finite_count,
                out=np.full(len(tail), np.nan, dtype=float),
                where=finite_count > 0,
            )
            pf_ancc_std = np.nanstd(stacked_delta, axis=0)
            out['pf_ancc_delta'] = pf_ancc_delta
            out['pf_ancc_std'] = pf_ancc_std
            out['pf_ancc_plane_diff'] = pf_ancc_delta - plane_delta
            out['pf_ancc_row_diff'] = pf_ancc_delta - row_delta
            out['pf_vs_spatial'] = pf_ancc_delta - plane_delta
            out['pf_vs_dense'] = pf_ancc_delta - row_delta
            plane_dist = np.asarray(out['formation_plane_min_dist'], dtype=float) if 'formation_plane_min_dist' in out else np.full(len(tail), np.nan)
            row_dist = np.asarray(out['formation_row_min_dist'], dtype=float) if 'formation_row_min_dist' in out else np.full(len(tail), np.nan)
            dist_conf = np.nanmin(np.vstack([plane_dist, row_dist]), axis=0)
            out['pf_ancc_confidence'] = 1.0 / (1.0 + np.nan_to_num(dist_conf, nan=1e6) / 5000.0 + np.nan_to_num(pf_ancc_std, nan=50.0) / 20.0)
            # Deterministic ANCC-state filter over A = TVT + Z. This is a lightweight
            # particle-filter surrogate that smooths spatial/dense ANCC observations through MD.
            tail_z_filter = tail['Z'].to_numpy(dtype=float)
            tail_md_filter = tail['MD'].to_numpy(dtype=float)
            prefix_a = prefix_tvt_values + prefix_z_values
            prefix_a_slope = safe_slope(prefix_md_values, prefix_a)
            state_a = last_known_tvt + ps_z
            state_var = 25.0
            prev_md = ps_md
            state_delta = np.full(len(tail), np.nan, dtype=float)
            state_std = np.full(len(tail), np.nan, dtype=float)
            obs_a = last_known_tvt + pf_ancc_delta + tail_z_filter
            obs_std = np.nan_to_num(pf_ancc_std, nan=30.0) + 5.0
            for j in range(len(tail)):
                dmd = tail_md_filter[j] - prev_md if np.isfinite(tail_md_filter[j]) and np.isfinite(prev_md) else 0.0
                if np.isfinite(prefix_a_slope):
                    state_a = state_a + prefix_a_slope * dmd
                state_var = state_var + 4.0 + 0.01 * abs(dmd)
                if np.isfinite(obs_a[j]):
                    meas_var = max(float(obs_std[j]) ** 2, 4.0)
                    gain = state_var / (state_var + meas_var)
                    state_a = state_a + gain * (obs_a[j] - state_a)
                    state_var = (1.0 - gain) * state_var
                state_delta[j] = state_a - tail_z_filter[j] - last_known_tvt if np.isfinite(tail_z_filter[j]) else np.nan
                state_std[j] = np.sqrt(max(state_var, 0.0))
                prev_md = tail_md_filter[j]
            out['pf_ancc_state_delta'] = state_delta
            out['pf_ancc_state_std'] = state_std
            out['pf_ancc_state_vs_mean'] = state_delta - pf_ancc_delta
            out['pf_ancc_state_confidence'] = 1.0 / (1.0 + np.nan_to_num(state_std, nan=50.0) / 20.0)
            if 'pf_z_delta' in out:
                out['pf_z_vs_ancc'] = np.asarray(out['pf_z_delta'], dtype=float) - state_delta
    

    # Same-well self-correlation aligns hidden GR windows to known-prefix GR windows.
    if need_self_corr:
        if gr_full_filled is None:
            fallback_gr = prefix_gr_mean if np.isfinite(prefix_gr_mean) else 0.0
            gr_full_filled = gr_full_numeric.interpolate(limit_direction='both').fillna(fallback_gr)
        hidden_gr_for_sc = gr_full_filled.iloc[pred_idx].to_numpy(dtype=float)
        prefix_gr_for_sc = gr_full_filled.iloc[:ps].to_numpy(dtype=float)
        sc_tvt, sc_score = self_corr_tvt(prefix_gr_for_sc, prefix_tvt_values, hidden_gr_for_sc, half_window=15, stride=3)
        sc_delta = sc_tvt.astype(float) - last_known_tvt
        sc_trust = np.clip((np.nan_to_num(sc_score, nan=0.0) - 0.05) / 0.45, 0.0, 1.0)
        slope_delta_ref = slope_baseline_tvt_last200 - last_known_tvt
        out['sc_delta'] = sc_delta
        out['sc_score'] = sc_score
        out['sc_trust'] = sc_trust
        out['hyb_delta'] = sc_trust * sc_delta + (1.0 - sc_trust) * slope_delta_ref


    # Typewell features are available reference-well features. They are evaluated only at prefix-derived TVT baselines.
    tw_path = split_dir / f'{well_id}__typewell.csv'
    if need_typewell_features and tw_path.exists():
        tw = pd.read_csv(tw_path)
    else:
        tw = pd.DataFrame(columns=['TVT', 'GR'])
    if need_typewell_features and {'TVT', 'GR'}.issubset(tw.columns) and tw[['TVT', 'GR']].dropna().shape[0] >= 2:
        tw_tvt = pd.to_numeric(tw['TVT'], errors='coerce')
        tw_gr = pd.to_numeric(tw['GR'], errors='coerce')
        tw_last_known_gr = typewell_gr_at_tvt(tw, np.array([last_known_tvt]))[0]
        tw_slope_all_gr = typewell_gr_at_tvt(tw, slope_baseline_tvt_all) if need_typewell_slope_features else np.full(len(tail), np.nan)
        tw_slope_last200_gr = typewell_gr_at_tvt(tw, slope_baseline_tvt_last200) if (need_typewell_slope_features or need_local_typewell or need_local_calibrated_typewell or need_calibrated_typewell) else np.full(len(tail), np.nan)
        prefix_tw_gr = typewell_gr_at_tvt(tw, prefix_tvt.to_numpy(dtype=float))
        prefix_gr_values = prefix_gr.to_numpy(dtype=float)
        valid_prefix_align = np.isfinite(prefix_gr_values) & np.isfinite(prefix_tw_gr)
        prefix_align_residual = prefix_gr_values - prefix_tw_gr
        prefix_align_absdiff = (
            float(np.mean(np.abs(prefix_align_residual[valid_prefix_align])))
            if valid_prefix_align.any()
            else np.nan
        )
        prefix_align_rmse = (
            float(np.sqrt(np.mean(prefix_align_residual[valid_prefix_align] ** 2)))
            if valid_prefix_align.any()
            else np.nan
        )
        calib_a = calib_b = np.nan
        prefix_calib_mae = prefix_calib_rmse = np.nan
        tw_last_known_gr_calibrated = np.nan
        tw_slope_all_gr_calibrated = np.full(len(tail), np.nan)
        tw_slope_last200_gr_calibrated = np.full(len(tail), np.nan)
        if need_calibrated_typewell or need_local_calibrated_typewell or need_calibrated_anchor_offsets:
            calib_a, calib_b = safe_affine_fit(prefix_tw_gr, prefix_gr_values)
            prefix_tw_gr_calibrated = apply_affine(prefix_tw_gr, calib_a, calib_b)
            tw_last_known_gr_calibrated = apply_affine(np.array([tw_last_known_gr]), calib_a, calib_b)[0]
            tw_slope_all_gr_calibrated = apply_affine(tw_slope_all_gr, calib_a, calib_b)
            tw_slope_last200_gr_calibrated = apply_affine(tw_slope_last200_gr, calib_a, calib_b)
            valid_prefix_calib = np.isfinite(prefix_gr_values) & np.isfinite(prefix_tw_gr_calibrated)
            prefix_calib_residual = prefix_gr_values - prefix_tw_gr_calibrated
            prefix_calib_mae = (
                float(np.mean(np.abs(prefix_calib_residual[valid_prefix_calib])))
                if valid_prefix_calib.any()
                else np.nan
            )
            prefix_calib_rmse = (
                float(np.sqrt(np.mean(prefix_calib_residual[valid_prefix_calib] ** 2)))
                if valid_prefix_calib.any()
                else np.nan
            )
        local_offsets = np.arange(-60.0, 60.1, 5.0)
        local_raw = {}
        if need_local_typewell:
            local_raw = local_typewell_offset_features(
                tw,
                slope_baseline_tvt_last200,
                tail_gr,
                local_offsets,
                prefix='typewell_local_last200',
            )
        local_calibrated = {}
        if need_local_calibrated_typewell:
            local_calibrated = local_typewell_offset_features(
                tw,
                slope_baseline_tvt_last200,
                tail_gr,
                local_offsets,
                affine_a=calib_a,
                affine_b=calib_b,
                prefix='calibrated_typewell_local_last200',
            )
        out['typewell_tvt_min'] = safe_stat(tw_tvt, np.min)
        out['typewell_tvt_max'] = safe_stat(tw_tvt, np.max)
        out['typewell_tvt_range'] = safe_stat(tw_tvt, np.max) - safe_stat(tw_tvt, np.min)
        out['typewell_gr_mean'] = safe_stat(tw_gr, np.mean)
        out['typewell_gr_std'] = safe_stat(tw_gr, np.std)
        out['typewell_gr_at_last_known_tvt'] = tw_last_known_gr
        out['typewell_gr_at_slope_baseline_all'] = tw_slope_all_gr
        out['typewell_gr_at_slope_baseline_last200'] = tw_slope_last200_gr
        out['gr_minus_typewell_last_known_tvt'] = tail_gr - tw_last_known_gr
        out['gr_minus_typewell_slope_baseline_all'] = tail_gr - tw_slope_all_gr
        out['gr_minus_typewell_slope_baseline_last200'] = tail_gr - tw_slope_last200_gr
        out['prefix_horizontal_vs_typewell_gr_corr'] = safe_corr(prefix_gr_values, prefix_tw_gr)
        out['prefix_horizontal_vs_typewell_gr_mae'] = prefix_align_absdiff
        out['prefix_horizontal_vs_typewell_gr_rmse'] = prefix_align_rmse
        if need_typewell_interval_context:
            last_geo_context = typewell_interval_context_features(tw, np.array([last_known_tvt]), prefix='typewell_last_geo')
            for col, values in last_geo_context.items():
                out[col] = values[0]
            baseline_geo_context = typewell_interval_context_features(tw, slope_baseline_tvt_last200, prefix='typewell_baseline_last200_geo')
            for col, values in baseline_geo_context.items():
                out[col] = values
        anchor_offsets = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], dtype=float)
        if need_typewell_anchor_offsets:
            for anchor_offset in anchor_offsets:
                label = candidate_endpoint_label(float(anchor_offset))
                anchor_gr = typewell_gr_at_tvt(tw, np.array([last_known_tvt + float(anchor_offset)]))[0]
                out[f'typewell_anchor_gr_diff_{label}'] = tail_gr - anchor_gr
        if need_local_typewell:
            for col, values in local_raw.items():
                out[col] = values
            out['typewell_local_last200_gr_resid_best'] = tail_gr - out['typewell_local_last200_best_gr']
            out['typewell_local_last200_best_vs_zero_gain'] = out['typewell_local_last200_zero_abs_resid'] - out['typewell_local_last200_best_abs_resid']
        out['prefix_typewell_gr_affine_a'] = calib_a
        out['prefix_typewell_gr_affine_b'] = calib_b
        out['prefix_horizontal_vs_calibrated_typewell_gr_mae'] = prefix_calib_mae
        out['prefix_horizontal_vs_calibrated_typewell_gr_rmse'] = prefix_calib_rmse
        if need_calibrated_anchor_offsets:
            for anchor_offset in anchor_offsets:
                label = candidate_endpoint_label(float(anchor_offset))
                anchor_gr = typewell_gr_at_tvt(tw, np.array([last_known_tvt + float(anchor_offset)]))[0]
                anchor_gr_calibrated = apply_affine(np.array([anchor_gr]), calib_a, calib_b)[0]
                out[f'calibrated_typewell_anchor_gr_diff_{label}'] = tail_gr - anchor_gr_calibrated
        out['typewell_calibrated_gr_at_last_known_tvt'] = tw_last_known_gr_calibrated
        out['typewell_calibrated_gr_at_slope_baseline_all'] = tw_slope_all_gr_calibrated
        out['typewell_calibrated_gr_at_slope_baseline_last200'] = tw_slope_last200_gr_calibrated
        out['gr_minus_calibrated_typewell_last_known_tvt'] = tail_gr - tw_last_known_gr_calibrated
        out['gr_minus_calibrated_typewell_slope_baseline_all'] = tail_gr - tw_slope_all_gr_calibrated
        out['gr_minus_calibrated_typewell_slope_baseline_last200'] = tail_gr - tw_slope_last200_gr_calibrated
        out['calibrated_typewell_slope_last200_gr_prefix_z'] = (tw_slope_last200_gr_calibrated - prefix_gr_mean) / (prefix_gr_std + 1e-6) if np.isfinite(prefix_gr_std) else np.nan
        if need_local_calibrated_typewell:
            for col, values in local_calibrated.items():
                out[col] = values
            out['calibrated_typewell_local_last200_gr_resid_best'] = tail_gr - out['calibrated_typewell_local_last200_best_gr']
            out['calibrated_typewell_local_last200_best_vs_zero_gain'] = out['calibrated_typewell_local_last200_zero_abs_resid'] - out['calibrated_typewell_local_last200_best_abs_resid']
        if need_candidate_path:
            candidate_path_features = typewell_candidate_path_features(
                tw,
                last_known_tvt,
                tail_frac,
                tail_gr,
                endpoints=CANDIDATE_PATH_ENDPOINTS,
                prefix='tw_path',
            )
            candidate_path_ease_features = typewell_candidate_path_features(
                tw,
                last_known_tvt,
                np.power(np.clip(tail_frac, 0.0, None), 1.45),
                tail_gr,
                endpoints=CANDIDATE_PATH_ENDPOINTS,
                prefix='tw_path_ease',
            )
            for col, values in candidate_path_features.items():
                out[col] = values
            for col, values in candidate_path_ease_features.items():
                out[col] = values
        if need_tw_beam_anchor_offsets and 'tw_beam_conservative_delta' in out:
            beam_anchor_offsets = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], dtype=float)
            beam_anchor_tvt = last_known_tvt + np.asarray(out['tw_beam_conservative_delta'], dtype=float)
            for anchor_offset in beam_anchor_offsets:
                label = candidate_endpoint_label(float(anchor_offset))
                anchor_gr = typewell_gr_at_tvt(tw, beam_anchor_tvt + float(anchor_offset))
                out[f'tw_beam_cons_anchor_gr_diff_{label}'] = tail_gr - anchor_gr
        if need_tw_sc_anchor_offsets and 'sc_delta' in out:
            sc_anchor_offsets = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], dtype=float)
            sc_anchor_tvt = last_known_tvt + np.asarray(out['sc_delta'], dtype=float)
            for anchor_offset in sc_anchor_offsets:
                label = candidate_endpoint_label(float(anchor_offset))
                anchor_gr = typewell_gr_at_tvt(tw, sc_anchor_tvt + float(anchor_offset))
                out[f'tw_sc_anchor_gr_diff_{label}'] = tail_gr - anchor_gr
        if need_beam:
            if gr_full_filled is None:
                fallback_gr = prefix_gr_mean if np.isfinite(prefix_gr_mean) else 0.0
                gr_full_filled = gr_full_numeric.interpolate(limit_direction='both').fillna(fallback_gr)
            hidden_gr_for_beam = gr_full_filled.iloc[pred_idx].to_numpy(dtype=float)
            tw_tvt_values = tw_tvt.to_numpy(dtype=float)
            tw_gr_values = tw_gr.to_numpy(dtype=float)
            beam_configs = {
                'conservative': dict(beam_size=10, move_cost=20.0, emit_scale=144.0, radius=2),
                'loose': dict(beam_size=10, move_cost=8.0, emit_scale=64.0, radius=2),
                'vcons': dict(beam_size=8, move_cost=35.0, emit_scale=220.0, radius=1),
                'sm5': dict(beam_size=10, move_cost=14.0, emit_scale=90.0, radius=5),
                'vloose': dict(beam_size=20, move_cost=4.0, emit_scale=36.0, radius=3),
            }
            beam_paths = {
                name: beam_typewell_path(
                    hidden_gr_for_beam,
                    tw_tvt_values,
                    tw_gr_values,
                    last_known_tvt,
                    **config,
                )
                for name, config in beam_configs.items()
            }
            beam_cons = beam_paths['conservative']
            beam_loose = beam_paths['loose']
            beam_vcons = beam_paths['vcons']
            beam_sm5 = beam_paths['sm5']
            beam_vloose = beam_paths['vloose']
            beam_deltas = np.vstack([
                beam_cons - last_known_tvt,
                beam_loose - last_known_tvt,
                beam_vcons - last_known_tvt,
                beam_sm5 - last_known_tvt,
                beam_vloose - last_known_tvt,
            ])
            out['tw_beam_tight_delta'] = beam_vcons - last_known_tvt
            out['tw_beam_vcons_delta'] = beam_vcons - last_known_tvt
            out['tw_beam_conservative_delta'] = beam_cons - last_known_tvt
            out['tw_beam_sm5_delta'] = beam_sm5 - last_known_tvt
            out['tw_beam_loose_delta'] = beam_loose - last_known_tvt
            out['tw_beam_vloose_delta'] = beam_vloose - last_known_tvt
            out['tw_beam_gap'] = beam_loose - beam_cons
            out['tw_beam_spread'] = np.nanmax(beam_deltas, axis=0) - np.nanmin(beam_deltas, axis=0)
            out['tw_beam_mean_delta'] = np.nanmean(beam_deltas, axis=0)
            out['tw_beam_median_delta'] = np.nanmedian(beam_deltas, axis=0)
            out['tw_beam_std_delta'] = np.nanstd(beam_deltas, axis=0)
            out['tw_beam_tight_step'] = np.r_[np.nan, np.diff(beam_vcons)]
            out['tw_beam_vcons_step'] = np.r_[np.nan, np.diff(beam_vcons)]
            out['tw_beam_conservative_step'] = np.r_[np.nan, np.diff(beam_cons)]
            out['tw_beam_sm5_step'] = np.r_[np.nan, np.diff(beam_sm5)]
            out['tw_beam_loose_step'] = np.r_[np.nan, np.diff(beam_loose)]
            out['tw_beam_vloose_step'] = np.r_[np.nan, np.diff(beam_vloose)]
            out['tw_beam_gr_at_conservative'] = np.interp(beam_cons, tw_tvt_values, tw_gr_values, left=np.nan, right=np.nan)
            out['tw_beam_gr_at_loose'] = np.interp(beam_loose, tw_tvt_values, tw_gr_values, left=np.nan, right=np.nan)
            out['tw_beam_gr_minus_conservative'] = tail_gr - out['tw_beam_gr_at_conservative']
            out['tw_beam_gr_minus_loose'] = tail_gr - out['tw_beam_gr_at_loose']
            out['beam_tight_delta'] = out['tw_beam_tight_delta']
            out['beam_vcons_delta'] = out['tw_beam_vcons_delta']
            out['beam_cons_delta'] = out['tw_beam_conservative_delta']
            out['beam_sm5_delta'] = out['tw_beam_sm5_delta']
            out['beam_loose_delta'] = out['tw_beam_loose_delta']
            out['beam_vloose_delta'] = out['tw_beam_vloose_delta']
            out['beam_mean_delta'] = out['tw_beam_mean_delta']
            out['beam_med_delta'] = out['tw_beam_median_delta']
            out['beam_std_delta'] = out['tw_beam_std_delta']
            out['beam_spread'] = out['tw_beam_spread']
            out['beam_gap'] = out['tw_beam_gap']
            out['gr_minus_tw_beam_cons'] = out['tw_beam_gr_minus_conservative']
            out['gr_minus_tw_beam_loose'] = out['tw_beam_gr_minus_loose']
            if 'formation_plane_delta_formula' in out:
                out['beam_vs_spatial'] = np.asarray(out['tw_beam_conservative_delta'], dtype=float) - np.asarray(out['formation_plane_delta_formula'], dtype=float)
            if 'formation_row_delta_formula' in out:
                out['beam_vs_dense'] = np.asarray(out['tw_beam_conservative_delta'], dtype=float) - np.asarray(out['formation_row_delta_formula'], dtype=float)
            if 'pf_ancc_delta' in out:
                out['beam_vs_pf_ancc'] = np.asarray(out['tw_beam_conservative_delta'], dtype=float) - np.asarray(out['pf_ancc_delta'], dtype=float)
            if need_self_corr and 'sc_delta' in out:
                sc_trust = np.asarray(out.get('sc_trust', np.zeros(len(tail))), dtype=float)
                out['hyb_delta'] = sc_trust * np.asarray(out['sc_delta'], dtype=float) + (1.0 - sc_trust) * np.asarray(out['tw_beam_conservative_delta'], dtype=float)
            if need_tw_beam_anchor_offsets:
                beam_anchor_offsets = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], dtype=float)
                beam_anchor_tvt = last_known_tvt + np.asarray(out['tw_beam_conservative_delta'], dtype=float)
                for anchor_offset in beam_anchor_offsets:
                    label = candidate_endpoint_label(float(anchor_offset))
                    anchor_gr = typewell_gr_at_tvt(tw, beam_anchor_tvt + float(anchor_offset))
                    out[f'tw_beam_cons_anchor_gr_diff_{label}'] = tail_gr - anchor_gr
        if need_pf_tvt:
            pf_features = typewell_sequence_filter_features(
                tw,
                tail_gr,
                slope_baseline_tvt_last200,
                last_known_tvt,
                prefix='pf_tvt',
            )
            if 'tw_beam_conservative_delta' in out:
                pf_features['pf_tvt_beam_cons_diff'] = pf_features['pf_tvt_delta'] - np.asarray(out['tw_beam_conservative_delta'], dtype=float)
            if 'tw_beam_loose_delta' in out:
                pf_features['pf_tvt_beam_loose_diff'] = pf_features['pf_tvt_delta'] - np.asarray(out['tw_beam_loose_delta'], dtype=float)
            for col, values in pf_features.items():
                out[col] = values
            if 'pf_z_delta' in out:
                out['pf_z_vs_tvt'] = np.asarray(out['pf_z_delta'], dtype=float) - np.asarray(out['pf_tvt_delta'], dtype=float)
            if 'pf_z_delta' in out and 'tw_beam_conservative_delta' in out:
                out['pf_z_vs_beam'] = np.asarray(out['pf_z_delta'], dtype=float) - np.asarray(out['tw_beam_conservative_delta'], dtype=float)
    else:
        for col in [
            'typewell_tvt_min',
            'typewell_tvt_max',
            'typewell_tvt_range',
            'typewell_gr_mean',
            'typewell_gr_std',
            'typewell_gr_at_last_known_tvt',
            'typewell_gr_at_slope_baseline_all',
            'typewell_gr_at_slope_baseline_last200',
            'gr_minus_typewell_last_known_tvt',
            'gr_minus_typewell_slope_baseline_all',
            'gr_minus_typewell_slope_baseline_last200',
            'prefix_horizontal_vs_typewell_gr_corr',
            'prefix_horizontal_vs_typewell_gr_mae',
            'prefix_horizontal_vs_typewell_gr_rmse',
            'typewell_local_last200_best_delta',
            'typewell_local_last200_best_abs_resid',
            'typewell_local_last200_zero_abs_resid',
            'typewell_local_last200_top2_gap',
            'typewell_local_last200_soft_delta_mean',
            'typewell_local_last200_best_gr',
            'typewell_local_last200_gr_resid_best',
            'typewell_local_last200_best_vs_zero_gain',
            'prefix_typewell_gr_affine_a',
            'prefix_typewell_gr_affine_b',
            'prefix_horizontal_vs_calibrated_typewell_gr_mae',
    'prefix_horizontal_vs_typewell_gr_rmse',
    'prefix_tvt_step20',
    'typewell_last_geo_interval_phase',
    'typewell_baseline_last200_geo_boundary_proximity',
            'prefix_horizontal_vs_calibrated_typewell_gr_rmse',
            'typewell_calibrated_gr_at_last_known_tvt',
            'typewell_calibrated_gr_at_slope_baseline_all',
            'typewell_calibrated_gr_at_slope_baseline_last200',
            'gr_minus_calibrated_typewell_last_known_tvt',
            'gr_minus_calibrated_typewell_slope_baseline_all',
            'gr_minus_calibrated_typewell_slope_baseline_last200',
            'calibrated_typewell_slope_last200_gr_prefix_z',
            'calibrated_typewell_local_last200_best_delta',
            'calibrated_typewell_local_last200_best_abs_resid',
            'calibrated_typewell_local_last200_zero_abs_resid',
            'calibrated_typewell_local_last200_top2_gap',
            'calibrated_typewell_local_last200_soft_delta_mean',
            'calibrated_typewell_local_last200_best_gr',
            'calibrated_typewell_local_last200_gr_resid_best',
            'calibrated_typewell_local_last200_best_vs_zero_gain',
        ]:
            out[col] = np.nan
        for col in candidate_path_feature_names(prefix='tw_path'):
            out[col] = np.nan
        for col in candidate_path_feature_names(prefix='tw_path_ease'):
            out[col] = np.nan
        for col in typewell_interval_context_feature_names(prefix='typewell_last_geo'):
            out[col] = np.nan
        for col in typewell_interval_context_feature_names(prefix='typewell_baseline_last200_geo'):
            out[col] = np.nan
        for offset in [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]:
            label = candidate_endpoint_label(float(offset))
            out[f'typewell_anchor_gr_diff_{label}'] = np.nan
            out[f'calibrated_typewell_anchor_gr_diff_{label}'] = np.nan
        if need_beam:
            for col in offline_beam_feature_names(prefix='tw_beam'):
                out[col] = np.nan
        if need_pf_tvt:
            for col in sequence_filter_feature_names(prefix='pf_tvt'):
                out[col] = np.nan
    
    if include_target and 'TVT' in df.columns:
        out['target_tvt'] = tail['TVT'].to_numpy()
        out['target_delta_from_last_known'] = out['target_tvt'] - last_known_tvt
    return out


# Configure the residual model, feature sets, validation, and causal post-processing.

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Execution controls.
NOTEBOOK_RUN_VERSION = 'ROGII_EDA_v2_candidate_path_lgbm_2026_05_06'
# Local research runs keep grouped CV enabled. Kaggle submission runs skip CV by default
# and use the latest stored CV selection before fitting the all-row GPU LightGBM candidate.
QUICK_CHECK_MODE = False
KAGGLE_MEMORY_SAFE_MODE = True
ENABLE_KAGGLE_BEAM_FEATURES = False
RUN_HGB_DIAGNOSTIC_SUBMISSIONS = not KAGGLE_NOTEBOOK_RUN
RUN_GROUPED_CV = not KAGGLE_NOTEBOOK_RUN

MODEL_CONFIG = {
    # Resource-controlled settings for grouped CV and final training.
    'random_state': 42,
    'n_splits': 5,
    'cv_folds_to_run': 5,
    'train_rows_per_well': 350,
    'final_train_rows_per_well': 700,
    'feature_policy_for_selection': 'strict',
    'residual_shrinkage': 0.80,
    'alpha_bounds': (0.0, 1.5),
    'delta_clip_quantiles': (0.005, 0.995),
    'fade_in_tau_md_to_compare': [None, 25.0, 50.0, 100.0, 200.0],
    'slope_clip_quantiles_to_compare': [0.90, 0.95, 0.975, 0.99, 0.995],
    'postprocess_slope_quantile': 0.995,
    'feature_sets_to_compare': None,
    'select_best_feature_set_from_cv': True,
    'apply_slope_clip_if_cv_improves': True,
    'write_submission': True,
    'write_public_clean_submission': True,
}

if QUICK_CHECK_MODE:
    MODEL_CONFIG.update({
        'cv_folds_to_run': 1,
        'train_rows_per_well': 150,
        'final_train_rows_per_well': 150,
        'feature_sets_to_compare': [
            'calibrated_typewell_alignment',
            'offline_candidate_path_alignment',
        ],
    })

BASE_FEATURE_COLUMNS = [
    # Row position and trajectory after Prediction Start.
    'md_since_ps',
    'x_delta_ps',
    'y_delta_ps',
    'z_delta_ps',
    'xy_dist_ps',
    'MD',
    'X',
    'Y',
    'Z',
    # Prefix-only well context.
    'prefix_len',
    'prefix_azimuth_deg',
    'prefix_gr_missing_rate',
    # Row-level GR signal and trailing GR context.
    'GR',
    'GR_isna',
    'GR_prefix_z',
    'gr_diff_1',
    'gr_diff_5',
    'gr_slope_md_1',
    'md_step_1',
    'x_step_1',
    'y_step_1',
    'z_step_1',
    'trajectory_step_1',
    'z_slope_md_1',
    'prefix_gr_mean',
    'prefix_gr_std',
    'gr_roll_mean_25',
    'gr_roll_std_25',
    'gr_roll_min_25',
    'gr_roll_max_25',
    'gr_roll_range_25',
    'gr_roll_mean_100',
    'gr_roll_std_100',
    'gr_roll_range_100',
    'gr_roll_mean_300',
    'gr_roll_std_300',
    'gr_roll_range_300',
    # Known-prefix TVT trend. This can help, but shrinkage/clipping protects against blind extrapolation.
    'prefix_tvt_slope_md_all',
    'prefix_tvt_slope_md_last200',
]

PREFIX_GR_DETAIL_COLUMNS = [
    'prefix_gr_last5_mean',
    'prefix_gr_last5_std',
    'prefix_gr_last20_mean',
    'prefix_gr_last20_std',
    'prefix_gr_last100_mean',
    'prefix_gr_last100_std',
]

PREFIX_CONTEXT_COLUMNS = [
    'last_known_TVT',
    *PREFIX_GR_DETAIL_COLUMNS,
    'prefix_gr_min',
    'prefix_gr_max',
    'prefix_last_valid_gr',
    'rows_since_prefix_last_valid_gr',
    'md_since_prefix_last_valid_gr',
    'gr_minus_prefix_last_valid_gr',
    'gr_minus_prefix_gr_mean',
    'prefix_tvt_min',
    'prefix_tvt_max',
    'prefix_tvt_range',
    'prefix_tvt_mean',
    'prefix_tvt_std',
    'prefix_tvt_step20',
    'prefix_tvt_step100',
    'prefix_tvt_md_slope100',
    'prefix_tvt_z_slope100',
    'slope_baseline_delta_all',
    'slope_baseline_delta_last200',
    'slope_baseline_tvt_all',
    'slope_baseline_tvt_last200',
]

TYPEWELL_INTERVAL_CONTEXT_COLUMNS = (
    typewell_interval_context_feature_names(prefix='typewell_last_geo')
    + typewell_interval_context_feature_names(prefix='typewell_baseline_last200_geo')
)

TYPEWELL_ANCHOR_OFFSET_COLUMNS = [
    f'typewell_anchor_gr_diff_{candidate_endpoint_label(float(offset))}'
    for offset in [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
]

CALIBRATED_TYPEWELL_ANCHOR_OFFSET_COLUMNS = [
    f'calibrated_typewell_anchor_gr_diff_{candidate_endpoint_label(float(offset))}'
    for offset in [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
]

TYPEWELL_ALIGNMENT_COLUMNS = [
    'typewell_tvt_min',
    'typewell_tvt_max',
    'typewell_tvt_range',
    'typewell_gr_mean',
    'typewell_gr_std',
    'typewell_gr_at_last_known_tvt',
    'typewell_gr_at_slope_baseline_all',
    'typewell_gr_at_slope_baseline_last200',
    'gr_minus_typewell_last_known_tvt',
    'gr_minus_typewell_slope_baseline_all',
    'gr_minus_typewell_slope_baseline_last200',
    'prefix_horizontal_vs_typewell_gr_corr',
    'prefix_horizontal_vs_typewell_gr_mae',
    'prefix_horizontal_vs_typewell_gr_rmse',
    *TYPEWELL_INTERVAL_CONTEXT_COLUMNS,
    *TYPEWELL_ANCHOR_OFFSET_COLUMNS,
    'typewell_local_last200_best_delta',
    'typewell_local_last200_best_abs_resid',
    'typewell_local_last200_zero_abs_resid',
    'typewell_local_last200_top2_gap',
    'typewell_local_last200_soft_delta_mean',
    'typewell_local_last200_gr_resid_best',
    'typewell_local_last200_best_vs_zero_gain',
]

CALIBRATED_TYPEWELL_ALIGNMENT_COLUMNS = [
    'prefix_typewell_gr_affine_a',
    'prefix_typewell_gr_affine_b',
    'prefix_horizontal_vs_calibrated_typewell_gr_mae',
    'prefix_horizontal_vs_calibrated_typewell_gr_rmse',
    *CALIBRATED_TYPEWELL_ANCHOR_OFFSET_COLUMNS,
    'typewell_calibrated_gr_at_last_known_tvt',
    'typewell_calibrated_gr_at_slope_baseline_all',
    'typewell_calibrated_gr_at_slope_baseline_last200',
    'gr_minus_calibrated_typewell_last_known_tvt',
    'gr_minus_calibrated_typewell_slope_baseline_all',
    'gr_minus_calibrated_typewell_slope_baseline_last200',
    'calibrated_typewell_slope_last200_gr_prefix_z',
    'calibrated_typewell_local_last200_best_delta',
    'calibrated_typewell_local_last200_best_abs_resid',
    'calibrated_typewell_local_last200_zero_abs_resid',
    'calibrated_typewell_local_last200_top2_gap',
    'calibrated_typewell_local_last200_soft_delta_mean',
    'calibrated_typewell_local_last200_gr_resid_best',
    'calibrated_typewell_local_last200_best_vs_zero_gain',
]

FORMATION_PLANE_COLUMNS = formation_feature_names(include_row=False)
FORMATION_PHYSICS_DISAGREEMENT_COLUMNS = [
    'pf_vs_spatial',
    'pf_vs_dense',
    'beam_vs_spatial',
    'beam_vs_dense',
    'beam_vs_pf_ancc',
]
FORMATION_ROW_ANCC_COLUMNS = [
    col for col in formation_feature_names(include_row=True)
    if col not in set(FORMATION_PLANE_COLUMNS)
    and col not in set(FORMATION_PHYSICS_DISAGREEMENT_COLUMNS)
]

# Beam alignment is expensive during repeated local CV. Keep it off locally by default,
# but enable it for the final Kaggle GPU submission path.
ENABLE_OFFLINE_BEAM_FEATURES = bool(ENABLE_KAGGLE_BEAM_FEATURES and KAGGLE_NOTEBOOK_RUN and not KAGGLE_MEMORY_SAFE_MODE)
OFFLINE_CANDIDATE_PATH_COLUMNS = candidate_path_feature_names(prefix='tw_path')
OFFLINE_CANDIDATE_PATH_EASE_COLUMNS = candidate_path_feature_names(prefix='tw_path_ease')
OFFLINE_BEAM_FEATURE_COLUMNS = offline_beam_feature_names(prefix='tw_beam')
PF_TVT_FEATURE_COLUMNS = sequence_filter_feature_names(prefix='pf_tvt')
PF_Z_FEATURE_COLUMNS = z_filter_feature_names(prefix='pf_z')
PF_ANCC_FEATURE_COLUMNS = ancc_filter_feature_names(prefix='pf_ancc')
SELF_CORR_FEATURE_COLUMNS = self_corr_feature_names(prefix='sc')
TW_BEAM_ANCHOR_OFFSETS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], dtype=float)
TW_SC_ANCHOR_OFFSETS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], dtype=float)
TW_BEAM_ANCHOR_OFFSET_COLUMNS = typewell_offset_diff_feature_names('tw_beam_cons_anchor_gr_diff', TW_BEAM_ANCHOR_OFFSETS)
TW_SC_ANCHOR_OFFSET_COLUMNS = typewell_offset_diff_feature_names('tw_sc_anchor_gr_diff', TW_SC_ANCHOR_OFFSETS)

OFFLINE_GR_TEXTURE_EXTRA_COLUMNS = [
    'gr_center_roll_min_5',
    'gr_center_roll_max_5',
    'gr_center_roll_min_21',
    'gr_center_roll_max_21',
    'gr_center_roll_min_51',
    'gr_center_roll_max_51',
    'gr_center_roll_min_151',
    'gr_center_roll_max_151',
    'gr_center_grad_5',
    'gr_center_lag10',
    'gr_center_lead10',
    'gr_center_lag20',
    'gr_center_lead20',
    'gr_cumsum_anom_since_ps',
    'gr_center_highfreq_5',
    'gr_center_z_5',
    'gr_center_highfreq_21',
    'gr_center_z_21',
    'gr_center_highfreq_51',
    'gr_center_z_51',
]

OFFLINE_TRAJECTORY_EXTRA_COLUMNS = [
    'gr_diff_10',
    'gr_slope_md_5',
    'md_step_5',
    'x_step_5',
    'y_step_5',
    'z_step_5',
    'x_d2_1',
    'y_d2_1',
    'z_d2_1',
    'trajectory_step_5',
    'trajectory_curvature_1',
    'trajectory_azimuth_sin_5',
    'trajectory_azimuth_cos_5',
    'trajectory_inclination_sin_5',
    'z_slope_md_5',
]

OFFLINE_EXTRA_FEATURE_COLUMNS = [
    'tail_len',
    'tail_row_number',
    'tail_frac',
    'tail_frac2',
    'tail_frac3',
    'sqrt_tail_frac',
    'log1p_tail_row',
    'sin_tail_frac_pi',
    'sin_tail_frac_2pi',
    'cos_tail_frac_3pi',
    'n_rows',
    'row_frac',
    'md_tail_span',
    'md_tail_frac',
    'tail_gr_missing_rate',
    'gap_md_span',
    'gap_x_delta',
    'gap_y_delta',
    'gap_z_delta',
    'gap_xy_span',
    'gap_z_over_xy',
    'gap_gr_mean',
    'gap_gr_std',
    'gap_gr_min',
    'gap_gr_p05',
    'gap_gr_p25',
    'gap_gr_p50',
    'gap_gr_p75',
    'gap_gr_p95',
    'gap_gr_max',
    'dist_xyz_ps',
    'dx_per_md_since_ps',
    'dy_per_md_since_ps',
    'dz_per_md_since_ps',
    'gr_center_roll_mean_5',
    'gr_center_roll_std_5',
    'gr_center_roll_range_5',
    'gr_center_roll_mean_21',
    'gr_center_roll_std_21',
    'gr_center_roll_range_21',
    'gr_center_grad_1',
    'gr_center_lag1',
    'gr_center_lead1',
    'gr_center_lag5',
    'gr_center_lead5',
    'gr_cumsum_since_ps',
    'gr_center_roll_mean_51',
    'gr_center_roll_std_51',
    'gr_center_roll_range_51',
    'gr_center_roll_mean_151',
    'gr_center_roll_std_151',
    'gr_center_roll_range_151',
    'gr_center_roll_mean_301',
    'gr_center_roll_std_301',
    'gr_center_roll_range_301',
    *OFFLINE_GR_TEXTURE_EXTRA_COLUMNS,
    *OFFLINE_TRAJECTORY_EXTRA_COLUMNS,
]

STRICT_FEATURE_SETS = {
    'causal_base': BASE_FEATURE_COLUMNS,
    'prefix_context': BASE_FEATURE_COLUMNS + PREFIX_CONTEXT_COLUMNS,
    'typewell_alignment': BASE_FEATURE_COLUMNS + PREFIX_CONTEXT_COLUMNS + TYPEWELL_ALIGNMENT_COLUMNS,
    'calibrated_typewell_alignment': (
        BASE_FEATURE_COLUMNS
        + PREFIX_CONTEXT_COLUMNS
        + TYPEWELL_ALIGNMENT_COLUMNS
        + CALIBRATED_TYPEWELL_ALIGNMENT_COLUMNS
    ),
}

OFFLINE_FEATURE_SETS = {
    'offline_prefix_context': STRICT_FEATURE_SETS['prefix_context'] + OFFLINE_EXTRA_FEATURE_COLUMNS,
    'offline_typewell_alignment': STRICT_FEATURE_SETS['typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS,
    'offline_calibrated_typewell_alignment': STRICT_FEATURE_SETS['calibrated_typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS,
    'offline_candidate_path_alignment': STRICT_FEATURE_SETS['typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS + OFFLINE_CANDIDATE_PATH_COLUMNS + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS,
    'offline_candidate_path_calibrated_alignment': STRICT_FEATURE_SETS['calibrated_typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS + OFFLINE_CANDIDATE_PATH_COLUMNS + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS,
    'offline_formation_plane_alignment': STRICT_FEATURE_SETS['typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS + FORMATION_PLANE_COLUMNS,
    'offline_formation_top_alignment': STRICT_FEATURE_SETS['typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS + FORMATION_PLANE_COLUMNS + FORMATION_ROW_ANCC_COLUMNS,
    'offline_sequence_filter_alignment': STRICT_FEATURE_SETS['typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS + OFFLINE_CANDIDATE_PATH_COLUMNS + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS + PF_TVT_FEATURE_COLUMNS,
    'offline_full_physics_alignment': (
        STRICT_FEATURE_SETS['typewell_alignment']
        + OFFLINE_EXTRA_FEATURE_COLUMNS
        + OFFLINE_CANDIDATE_PATH_COLUMNS
        + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS
        + OFFLINE_BEAM_FEATURE_COLUMNS
        + FORMATION_PLANE_COLUMNS
        + FORMATION_ROW_ANCC_COLUMNS
        + FORMATION_PHYSICS_DISAGREEMENT_COLUMNS
        + PF_TVT_FEATURE_COLUMNS
        + PF_Z_FEATURE_COLUMNS
        + PF_ANCC_FEATURE_COLUMNS
        + SELF_CORR_FEATURE_COLUMNS
        + TW_BEAM_ANCHOR_OFFSET_COLUMNS
        + TW_SC_ANCHOR_OFFSET_COLUMNS
    ),
}
OFFLINE_FEATURE_SETS['offline_super_top3_alignment'] = OFFLINE_FEATURE_SETS['offline_full_physics_alignment']
if ENABLE_OFFLINE_BEAM_FEATURES:
    OFFLINE_FEATURE_SETS['offline_beam_candidate_path_alignment'] = (
        STRICT_FEATURE_SETS['typewell_alignment']
        + OFFLINE_EXTRA_FEATURE_COLUMNS
        + OFFLINE_CANDIDATE_PATH_COLUMNS
        + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS
        + OFFLINE_BEAM_FEATURE_COLUMNS
    )

FEATURE_SETS = {**STRICT_FEATURE_SETS, **OFFLINE_FEATURE_SETS}
FEATURE_SET_POLICY = {name: 'strict' for name in STRICT_FEATURE_SETS}
FEATURE_SET_POLICY.update({name: 'offline' for name in OFFLINE_FEATURE_SETS})

DEFAULT_FEATURE_SETS_TO_COMPARE = [
    name
    for name in FEATURE_SETS
    if name not in {
        # Row-level ANCC KNN is useful as a targeted experiment, but it is heavier than the
        # plane-fit formation features and is not needed for the memory-safe submission path.
        'offline_formation_top_alignment',
    }
]
if MODEL_CONFIG['feature_sets_to_compare'] is None:
    MODEL_CONFIG['feature_sets_to_compare'] = DEFAULT_FEATURE_SETS_TO_COMPARE

# Latest full grouped-CV selection saved in this notebook.
# These defaults make submission-only runs reproducible without rerunning CV.
SELECTED_FEATURE_SET = 'calibrated_typewell_alignment'
FEATURE_COLUMNS = FEATURE_SETS[SELECTED_FEATURE_SET]
SELECTED_POLICY_METRIC = 'global_alpha_0.812_fade_200_0_slope_q_0_9'
SELECTED_SHRINKAGE_ALPHA = 0.811832
SELECTED_FADE_IN_TAU_MD = 200.0
SELECTED_SLOPE_QUANTILE = 0.90
APPLY_SELECTED_SLOPE_CLIP = True
BEST_OVERALL_FEATURE_SET = 'offline_candidate_path_alignment'
BEST_OVERALL_SHRINKAGE_ALPHA = 0.941149
BEST_OVERALL_FADE_IN_TAU_MD = 200.0
BEST_OVERALL_SLOPE_QUANTILE = 0.90
APPLY_BEST_OVERALL_SLOPE_CLIP = True

PUBLIC_LGBM_STYLE_COLUMNS = [
    'row_index', 'last_known_TVT', 'prefix_len', 'tail_len', 'tail_row_number', 'tail_frac',
    'n_rows', 'row_frac', 'md_tail_frac', 'md_tail_span',
    'MD', 'Z', 'X', 'Y', 'GR', 'GR_isna',
    'gr_center_roll_mean_5', 'gr_center_roll_mean_21',
    'gr_center_grad_1', 'gr_center_roll_std_5', 'gr_center_roll_std_21',
    'gr_center_lag1', 'gr_center_lead1', 'gr_center_lag5', 'gr_center_lead5', 'gr_cumsum_since_ps',
    'md_since_ps', 'z_delta_ps', 'x_delta_ps', 'y_delta_ps',
    'dx_per_md_since_ps', 'dy_per_md_since_ps', 'dz_per_md_since_ps', 'xy_dist_ps', 'dist_xyz_ps',
    'prefix_tvt_step20', 'prefix_tvt_step100', 'prefix_tvt_md_slope100', 'prefix_tvt_z_slope100',
    'prefix_horizontal_vs_typewell_gr_rmse', 'prefix_horizontal_vs_typewell_gr_mae',
    'typewell_anchor_gr_diff_m80', 'typewell_anchor_gr_diff_m40', 'typewell_anchor_gr_diff_m20',
    'typewell_anchor_gr_diff_m10', 'typewell_anchor_gr_diff_m5', 'typewell_anchor_gr_diff_p0',
    'typewell_anchor_gr_diff_p5', 'typewell_anchor_gr_diff_p10', 'typewell_anchor_gr_diff_p20',
    'typewell_anchor_gr_diff_p40', 'typewell_anchor_gr_diff_p80',
]
PUBLIC_LGBM_ALWAYS_AVAILABLE_COLUMNS = ['row_index']
PUBLIC_LGBM_COMPACT_EXTRA_COLUMNS = [
    'tail_frac2',
    'tail_frac3',
    'sqrt_tail_frac',
    'log1p_tail_row',
    'sin_tail_frac_pi',
    'sin_tail_frac_2pi',
    'cos_tail_frac_3pi',
    'gap_md_span',
    'gap_x_delta',
    'gap_y_delta',
    'gap_z_delta',
    'gap_xy_span',
    'gap_z_over_xy',
    'gap_gr_mean',
    'gap_gr_std',
    'gap_gr_min',
    'gap_gr_p05',
    'gap_gr_p25',
    'gap_gr_p50',
    'gap_gr_p75',
    'gap_gr_p95',
    'gap_gr_max',
    'tail_gr_missing_rate',
    'gr_center_roll_range_5',
    'gr_center_roll_range_21',
    'gr_center_roll_mean_51',
    'gr_center_roll_std_51',
    'gr_center_roll_range_51',
    'gr_center_roll_mean_151',
    'gr_center_roll_std_151',
    'gr_center_roll_range_151',
    'prefix_gr_last5_mean',
    'prefix_gr_last5_std',
    'prefix_gr_last20_mean',
    'prefix_gr_last20_std',
    'prefix_gr_last100_mean',
    'prefix_gr_last100_std',
    'gr_diff_10',
    'gr_slope_md_5',
    'md_step_5',
    'trajectory_step_5',
    'trajectory_curvature_1',
    'trajectory_azimuth_sin_5',
    'trajectory_azimuth_cos_5',
    'trajectory_inclination_sin_5',
    'z_slope_md_5',
    'gr_center_roll_min_5',
    'gr_center_roll_max_5',
    'gr_center_roll_min_21',
    'gr_center_roll_max_21',
    'gr_center_grad_5',
    'gr_center_lag10',
    'gr_center_lead10',
    'gr_center_lag20',
    'gr_center_lead20',
    'gr_cumsum_anom_since_ps',
    'gr_center_highfreq_5',
    'gr_center_z_5',
    'gr_center_highfreq_21',
    'gr_center_z_21',
]
PUBLIC_LGBM_STYLE_COLUMNS = list(dict.fromkeys(PUBLIC_LGBM_STYLE_COLUMNS + PUBLIC_LGBM_COMPACT_EXTRA_COLUMNS))
PUBLIC_LGBM_STYLE_COLUMNS = [
    col
    for col in PUBLIC_LGBM_STYLE_COLUMNS
    if col in FEATURE_SETS['offline_typewell_alignment'] + OFFLINE_EXTRA_FEATURE_COLUMNS + PUBLIC_LGBM_ALWAYS_AVAILABLE_COLUMNS
]
FEATURE_SETS['offline_public_lgbm_style'] = list(dict.fromkeys(PUBLIC_LGBM_STYLE_COLUMNS))
FEATURE_SET_POLICY['offline_public_lgbm_style'] = 'offline'
PUBLIC_LGBM_FORMATION_STYLE_COLUMNS = list(dict.fromkeys(PUBLIC_LGBM_STYLE_COLUMNS + FORMATION_PLANE_COLUMNS))
FEATURE_SETS['offline_public_lgbm_formation_style'] = PUBLIC_LGBM_FORMATION_STYLE_COLUMNS
FEATURE_SET_POLICY['offline_public_lgbm_formation_style'] = 'offline'

SUPER_TOP3_COMPACT_BASE_COLUMNS = [
    'row_index', 'last_known_TVT', 'prefix_len', 'tail_len', 'tail_row_number', 'tail_frac',
    'n_rows', 'row_frac', 'md_tail_frac', 'md_tail_span',
    'MD', 'Z', 'X', 'Y', 'GR', 'GR_isna',
    'md_since_ps', 'z_delta_ps', 'x_delta_ps', 'y_delta_ps',
    'dx_per_md_since_ps', 'dy_per_md_since_ps', 'dz_per_md_since_ps', 'xy_dist_ps', 'dist_xyz_ps',
    'gr_center_roll_mean_5', 'gr_center_roll_mean_21', 'gr_center_roll_mean_51',
    'gr_center_roll_std_5', 'gr_center_roll_std_21', 'gr_center_roll_range_5', 'gr_center_roll_range_21',
    'gr_center_grad_1', 'gr_center_grad_5',
    'gr_center_lag1', 'gr_center_lead1', 'gr_center_lag5', 'gr_center_lead5', 'gr_center_lag10', 'gr_center_lead10',
    'gr_cumsum_since_ps', 'gr_cumsum_anom_since_ps',
    'prefix_gr_last5_mean', 'prefix_gr_last5_std', 'prefix_gr_last20_mean', 'prefix_gr_last20_std',
    'prefix_tvt_step20', 'prefix_tvt_step100', 'prefix_tvt_md_slope100', 'prefix_tvt_z_slope100',
    'prefix_horizontal_vs_typewell_gr_rmse', 'prefix_horizontal_vs_typewell_gr_mae',
    'typewell_anchor_gr_diff_m80', 'typewell_anchor_gr_diff_m40', 'typewell_anchor_gr_diff_m20',
    'typewell_anchor_gr_diff_m10', 'typewell_anchor_gr_diff_m5', 'typewell_anchor_gr_diff_p0',
    'typewell_anchor_gr_diff_p5', 'typewell_anchor_gr_diff_p10', 'typewell_anchor_gr_diff_p20',
    'typewell_anchor_gr_diff_p40', 'typewell_anchor_gr_diff_p80',
    'gap_md_span', 'gap_x_delta', 'gap_y_delta', 'gap_z_delta', 'gap_xy_span', 'gap_z_over_xy',
    'gap_gr_mean', 'gap_gr_std', 'gap_gr_p05', 'gap_gr_p50', 'gap_gr_p95', 'tail_gr_missing_rate',
    'trajectory_step_5', 'trajectory_curvature_1', 'trajectory_azimuth_sin_5', 'trajectory_azimuth_cos_5',
    'trajectory_inclination_sin_5', 'z_slope_md_5',
]

SUPER_TOP3_COMPACT_SIGNAL_COLUMNS = [
    'tw_beam_tight_delta', 'tw_beam_vcons_delta', 'tw_beam_conservative_delta', 'tw_beam_sm5_delta',
    'tw_beam_loose_delta', 'tw_beam_vloose_delta', 'tw_beam_gap', 'tw_beam_spread',
    'tw_beam_mean_delta', 'tw_beam_median_delta', 'tw_beam_std_delta',
    'tw_beam_gr_at_conservative', 'tw_beam_gr_at_loose', 'tw_beam_gr_minus_conservative', 'tw_beam_gr_minus_loose',
    'beam_tight_delta', 'beam_vcons_delta', 'beam_cons_delta', 'beam_sm5_delta', 'beam_loose_delta',
    'beam_vloose_delta', 'beam_mean_delta', 'beam_med_delta', 'beam_std_delta', 'beam_spread', 'beam_gap',
    'gr_minus_tw_beam_cons', 'gr_minus_tw_beam_loose',
    'pf_tvt_delta', 'pf_tvt_std', 'pf_tvt_p10_delta', 'pf_tvt_p90_delta', 'pf_tvt_width',
    'pf_tvt_confidence', 'pf_tvt_beam_cons_diff', 'pf_tvt_beam_loose_diff',
    'pf_z_delta', 'pf_z_std', 'pf_z_velocity_md', 'pf_z_velocity_z', 'pf_z_confidence',
    'pf_z_vs_tvt', 'pf_z_vs_ancc', 'pf_z_vs_beam',
    'pf_ancc_delta', 'pf_ancc_std', 'pf_ancc_plane_diff', 'pf_ancc_row_diff', 'pf_ancc_confidence',
    'pf_ancc_state_delta', 'pf_ancc_state_std', 'pf_ancc_state_vs_mean', 'pf_ancc_state_confidence',
    'sc_delta', 'sc_score', 'sc_trust', 'hyb_delta',
    'formation_plane_min_dist', 'formation_plane_anchor_b', 'formation_plane_anchor_b50',
    'formation_plane_prefix_rmse', 'formation_plane_prefix_mae', 'formation_plane_delta_formula',
    'formation_plane_formula_mean_delta', 'formation_plane_formula_std_delta',
    'formation_plane_formula_min_delta', 'formation_plane_formula_max_delta',
    'formation_row_ancc', 'formation_row_ancc_std', 'formation_row_min_dist',
    'formation_row_prefix_rmse', 'formation_row_prefix_mae', 'formation_row_delta_formula',
    'formation_formula_mean_delta', 'formation_formula_abs_gap',
    'dense_ancc', 'dense_std', 'dense_dist', 'dense_rmse', 'dense_bias', 'dense_nb_std',
    'tvt_dense_delta', 'tvt_dense50_delta', 'spatial_vs_dense', 'pf_vs_spatial', 'pf_vs_dense',
    'beam_vs_spatial', 'beam_vs_dense', 'beam_vs_pf_ancc',
]
for _label in FORMATION_LABELS:
    SUPER_TOP3_COMPACT_SIGNAL_COLUMNS.extend([
        f'formation_plane_{_label}',
        f'formation_plane_anchor_b_{_label}',
        f'formation_plane_anchor_b50_{_label}',
        f'formation_plane_prefix_rmse_{_label}',
        f'formation_plane_delta_formula_{_label}',
        f'formation_plane_delta_formula50_{_label}',
    ])
for _suffix in ['m20', 'm10', 'm5', 'p0', 'p5', 'p10', 'p20']:
    SUPER_TOP3_COMPACT_SIGNAL_COLUMNS.append(f'tw_beam_cons_anchor_gr_diff_{_suffix}')
for _suffix in ['m15', 'm8', 'm4', 'p0', 'p4', 'p8', 'p15']:
    SUPER_TOP3_COMPACT_SIGNAL_COLUMNS.append(f'tw_sc_anchor_gr_diff_{_suffix}')

SUPER_TOP3_COMPACT_COLUMNS = list(dict.fromkeys(SUPER_TOP3_COMPACT_BASE_COLUMNS + SUPER_TOP3_COMPACT_SIGNAL_COLUMNS))
FEATURE_SETS['offline_super_top3_compact_alignment'] = [
    col for col in SUPER_TOP3_COMPACT_COLUMNS
    if col in FEATURE_SETS['offline_super_top3_alignment']
]
FEATURE_SET_POLICY['offline_super_top3_compact_alignment'] = 'offline'
# The public-style feature set is available for targeted experiments but excluded from default CV
# unless explicitly listed in MODEL_CONFIG['feature_sets_to_compare'].

feature_set_report = pd.DataFrame({
    'feature_set': list(FEATURE_SETS.keys()),
    'feature_policy': [FEATURE_SET_POLICY[name] for name in FEATURE_SETS],
    'feature_count': [len(cols) for cols in FEATURE_SETS.values()],
})


# Build tail feature matrices and define model/post-processing helpers.

def feature_columns_require_beam(feature_columns) -> bool:
    if feature_columns is None:
        return bool(globals().get('ENABLE_OFFLINE_BEAM_FEATURES', False))
    return any(str(col).startswith('tw_beam_') or str(col).startswith('beam_') for col in feature_columns)


def feature_columns_require_formation(feature_columns) -> bool:
    if feature_columns is None:
        return False
    return any(str(col).startswith(('formation_', 'pf_ancc_', 'spatial_vs_', 'pf_vs_spatial', 'beam_vs_spatial')) for col in feature_columns)


def feature_columns_require_row_ancc(feature_columns) -> bool:
    if feature_columns is None:
        return False
    return any(str(col).startswith(('formation_row_', 'formation_formula_', 'dense_', 'tvt_dense_', 'spatial_vs_', 'pf_vs_dense', 'beam_vs_dense', 'pf_ancc_')) for col in feature_columns)


def compact_tail_feature_frame(
    frame: pd.DataFrame,
    keep_columns=None,
    include_target: bool = True,
    keep_metadata: bool = True,
) -> pd.DataFrame:
    if keep_columns is not None:
        required_columns = []
        if keep_metadata:
            required_columns += [
                'well_id',
                'row_index',
                'id',
                'MD',
                'last_known_TVT',
                'last_known_MD',
                'md_since_ps',
            ]
        if include_target:
            required_columns += ['target_tvt', 'target_delta_from_last_known']
        frame = frame.copy()
        for col in keep_columns:
            if col not in frame.columns:
                frame[col] = np.nan
        wanted = list(dict.fromkeys(required_columns + list(keep_columns)))
        frame = frame[[col for col in wanted if col in frame.columns]]
    else:
        frame = frame.copy()

    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], downcast='float')
        elif pd.api.types.is_integer_dtype(frame[col]) and col not in {'row_index'}:
            frame[col] = pd.to_numeric(frame[col], downcast='integer')
    return frame


def build_tail_feature_frame(
    well_ids,
    split_dir: Path,
    include_target: bool,
    rows_per_well: int | None = None,
    random_state: int = 42,
    use_beam_features: bool | None = None,
    keep_columns=None,
    formation_plane_imputer=None,
    row_ancc_imputer=None,
    imputer_well_ids=None,
    keep_metadata: bool = True,
    exclude_query_well_from_formation: bool = True,
) -> pd.DataFrame:
    if use_beam_features is None:
        use_beam_features = bool(globals().get('ENABLE_OFFLINE_BEAM_FEATURES', False))
    need_formation = feature_columns_require_formation(keep_columns)
    need_row_ancc = feature_columns_require_row_ancc(keep_columns)
    if need_formation and formation_plane_imputer is None:
        source_wells = imputer_well_ids if imputer_well_ids is not None else globals().get('all_train_wells', globals().get('train_h_ids', []))
        formation_plane_imputer, maybe_row = make_formation_imputers(
            source_wells,
            TRAIN_DIR,
            need_row_ancc=need_row_ancc and row_ancc_imputer is None,
            seed=random_state,
        )
        if row_ancc_imputer is None:
            row_ancc_imputer = maybe_row
    rng = np.random.default_rng(random_state)
    frames = []
    for well_id in sorted(well_ids):
        frame = make_tail_features_for_well(
            well_id,
            split_dir,
            include_target=include_target,
            use_beam_features=use_beam_features,
            required_feature_columns=keep_columns,
            formation_plane_imputer=formation_plane_imputer,
            row_ancc_imputer=row_ancc_imputer,
            exclude_query_well_from_formation=exclude_query_well_from_formation,
        )
        if frame.empty:
            continue
        if rows_per_well is not None and len(frame) > rows_per_well:
            sampled_idx = np.sort(rng.choice(frame.index.to_numpy(), size=rows_per_well, replace=False))
            frame = frame.loc[sampled_idx]
        frame = compact_tail_feature_frame(
            frame,
            keep_columns=keep_columns,
            include_target=include_target,
            keep_metadata=keep_metadata,
        )
        frames.append(frame)
        gc.collect()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, copy=False)


def model_ready_xy(frame: pd.DataFrame, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    X = frame[feature_columns].copy()
    y_delta = frame['target_delta_from_last_known'].to_numpy()
    y_tvt = frame['target_tvt'].to_numpy()
    return X, y_delta, y_tvt


def make_residual_model(random_state: int = 42):
    # Fixed-iteration HGB avoids row-random internal early stopping inside a grouped-CV fold.
    return HistGradientBoostingRegressor(
        loss='squared_error',
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=False,
        random_state=random_state,
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clip_delta_by_train_quantiles(delta_pred: np.ndarray, train_delta: np.ndarray, config=MODEL_CONFIG) -> np.ndarray:
    lo_q, hi_q = config['delta_clip_quantiles']
    lo, hi = np.nanquantile(train_delta, [lo_q, hi_q])
    return np.clip(delta_pred, lo, hi)


def shrink_delta(delta: np.ndarray, alpha: float | None = None, config=MODEL_CONFIG) -> np.ndarray:
    if alpha is None:
        alpha = config['residual_shrinkage']
    return float(alpha) * np.asarray(delta, dtype=float)


def fade_in_delta(frame: pd.DataFrame, delta: np.ndarray, tau_md: float | None) -> np.ndarray:
    """Dampen residuals close to Prediction Start; tau=None leaves them unchanged."""
    values = np.asarray(delta, dtype=float)
    if tau_md is None or not np.isfinite(tau_md) or tau_md <= 0:
        return values
    md_since = frame['md_since_ps'].to_numpy(dtype=float)
    rho = 1.0 - np.exp(-np.maximum(md_since, 0.0) / float(tau_md))
    return values * rho


def fade_in_tau_label(tau_md: float | None) -> str:
    if tau_md is None or (isinstance(tau_md, float) and not np.isfinite(tau_md)):
        return 'none'
    return str(float(tau_md)).replace('.', '_')


def fit_global_alpha_from_fold_parts(fold_parts, tau_md: float | None, config=MODEL_CONFIG) -> tuple[float, float]:
    """Fit one alpha over all validation folds for a fixed fade-in setting."""
    numerator = 0.0
    denominator = 0.0
    for part in fold_parts:
        frame = part['frame']
        base_delta = fade_in_delta(frame, part['clipped_delta'], tau_md)
        y_delta = frame['target_delta_from_last_known'].to_numpy(dtype=float)
        valid = np.isfinite(base_delta) & np.isfinite(y_delta)
        numerator += float(np.dot(y_delta[valid], base_delta[valid]))
        denominator += float(np.dot(base_delta[valid], base_delta[valid]))
    if denominator <= 1e-12:
        alpha = 0.0
    else:
        alpha = numerator / denominator
    lo, hi = config['alpha_bounds']
    alpha = float(np.clip(alpha, lo, hi))
    score = score_policy_from_fold_parts(fold_parts, alpha=alpha, tau_md=tau_md, slope_quantile=None)['rmse']
    return alpha, score


def score_policy_from_fold_parts(fold_parts, alpha: float, tau_md: float | None, slope_quantile: float | None) -> dict[str, float]:
    sse = 0.0
    n = 0
    for part in fold_parts:
        frame = part['frame']
        delta = shrink_delta(fade_in_delta(frame, part['clipped_delta'], tau_md), alpha=alpha)
        pred = frame['last_known_TVT'].to_numpy(dtype=float) + delta
        if slope_quantile is not None:
            pred = causal_slope_clip_by_well(frame, pred, part['slope_bounds'][float(slope_quantile)])
        y = frame['target_tvt'].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(pred)
        err = y[valid] - pred[valid]
        sse += float(np.dot(err, err))
        n += int(valid.sum())
    return {'rmse': float(np.sqrt(sse / n)), 'sse': sse, 'n': n}


def policy_metric_name(alpha: float, tau_md: float | None, slope_quantile: float | None) -> str:
    tau_label = fade_in_tau_label(tau_md)
    if slope_quantile is None:
        slope_label = 'none'
    else:
        slope_label = f'q_{slope_quantile_label(float(slope_quantile))}'
    return f'global_alpha_{alpha:.3f}_fade_{tau_label}_slope_{slope_label}'


def best_alpha_for_clipped_delta(
    last_known_tvt: np.ndarray,
    clipped_delta: np.ndarray,
    y_true: np.ndarray,
    alphas=None,
) -> tuple[float, float]:
    if alphas is None:
        alphas = np.linspace(0.0, 1.5, 61)
    best_score = np.inf
    best_alpha = np.nan
    for alpha in alphas:
        score = rmse(y_true, last_known_tvt + alpha * clipped_delta)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, float(best_score)


def estimate_abs_tvt_slope_quantiles(paths, quantiles, max_wells=None) -> dict[float, float]:
    slopes = []
    selected_paths = list(paths)[:max_wells] if max_wells is not None else list(paths)
    for path in selected_paths:
        df = pd.read_csv(path, usecols=['MD', 'TVT'])
        md = df['MD'].to_numpy(dtype=float)
        tvt = df['TVT'].to_numpy(dtype=float)
        dmd = np.diff(md)
        dtvt = np.diff(tvt)
        valid = np.isfinite(dmd) & np.isfinite(dtvt) & (np.abs(dmd) > 1e-9)
        if valid.any():
            slopes.append(np.abs(dtvt[valid] / dmd[valid]))
    if not slopes:
        return {float(q): np.nan for q in quantiles}
    values = np.concatenate(slopes)
    return {float(q): float(np.nanquantile(values, q)) for q in quantiles}


def estimate_abs_tvt_slope_quantile(paths, quantile=0.995, max_wells=None) -> float:
    return estimate_abs_tvt_slope_quantiles(paths, [quantile], max_wells=max_wells)[float(quantile)]


def slope_quantile_label(q: float) -> str:
    return str(float(q)).replace('.', '_')


def causal_slope_clip_by_well(frame: pd.DataFrame, pred_tvt: np.ndarray, max_abs_slope: float) -> np.ndarray:
    """Anchor-aware forward slope limiter. It never averages with future rows."""
    clipped = np.asarray(pred_tvt, dtype=float).copy()
    if not np.isfinite(max_abs_slope) or max_abs_slope <= 0:
        return clipped
    required = ['well_id', 'row_index', 'MD', 'last_known_TVT', 'last_known_MD']
    frame_ordered = frame[required].copy()
    frame_ordered['_pos'] = np.arange(len(frame_ordered))
    for _, g in frame_ordered.sort_values(['well_id', 'row_index']).groupby('well_id', sort=False):
        pos = g['_pos'].to_numpy(dtype=int)
        md = g['MD'].to_numpy(dtype=float)
        prev_tvt = float(g['last_known_TVT'].iloc[0])
        prev_md = float(g['last_known_MD'].iloc[0])
        for k in range(len(pos)):
            step_md = abs(md[k] - prev_md)
            limit = max_abs_slope * max(step_md, 1e-6)
            clipped[pos[k]] = np.clip(clipped[pos[k]], prev_tvt - limit, prev_tvt + limit)
            prev_tvt = clipped[pos[k]]
            prev_md = md[k]
    return clipped


MAX_ABS_TVT_SLOPE_BY_QUANTILE = estimate_abs_tvt_slope_quantiles(
    train_horizontal_files,
    quantiles=MODEL_CONFIG['slope_clip_quantiles_to_compare'],
)
MAX_ABS_TVT_SLOPE = MAX_ABS_TVT_SLOPE_BY_QUANTILE[float(MODEL_CONFIG['postprocess_slope_quantile'])]




# ---------------------------------------------------------------------------
# Artifact utilities
# ---------------------------------------------------------------------------

ARTIFACT_ROOT = OUTPUT_DIR / "rogii_artifacts"
ARTIFACT_SUBDIRS = [
    "splits",
    "metadata",
    "preds",
    "confidence",
    "stacking",
    "reports",
    "submissions",
]


def ensure_artifact_dirs(root: Path | None = None) -> Path:
    root = Path(root or ARTIFACT_ROOT)
    for name in ARTIFACT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        return fallback


def read_frame(path: Path) -> pd.DataFrame:
    path = Path(path)
    candidates = [path]
    if path.suffix != ".csv":
        candidates.append(path.with_suffix(".csv"))
    if path.suffix != ".parquet":
        candidates.append(path.with_suffix(".parquet"))
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".csv":
                return pd.read_csv(candidate)
            return pd.read_parquet(candidate)
    raise FileNotFoundError(f"No table found for {path}")


def artifact_path(root: Path, section: str, filename: str) -> Path:
    return Path(root) / section / filename


def build_tail_metadata(
    well_ids,
    split_dir: Path,
    include_target: bool = True,
) -> pd.DataFrame:
    rows = []
    usecols = ["MD", "TVT_input"]
    if include_target:
        usecols.append("TVT")
    for well_id in sorted(well_ids):
        path = Path(split_dir) / f"{well_id}__horizontal_well.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda col: col in set(usecols))
        mask = df["TVT_input"].isna().to_numpy()
        pred_idx = np.flatnonzero(mask)
        if len(pred_idx) == 0:
            continue
        ps = int(pred_idx[0])
        last_idx = max(ps - 1, 0)
        last_known_tvt = float(df["TVT_input"].iloc[last_idx])
        last_known_md = float(df["MD"].iloc[last_idx])
        for tail_row_number, row_idx in enumerate(pred_idx):
            item = {
                "id": f"{well_id}_{int(row_idx)}",
                "well_id": well_id,
                "row_index": int(row_idx),
                "tail_row_number": int(tail_row_number),
                "last_known_TVT": last_known_tvt,
                "last_known_MD": last_known_md,
                "MD": float(df["MD"].iloc[row_idx]),
                "md_since_ps": float(df["MD"].iloc[row_idx] - df["MD"].iloc[pred_idx[0]]),
            }
            if include_target and "TVT" in df.columns:
                item["target_tvt"] = float(df["TVT"].iloc[row_idx])
                item["target_delta"] = item["target_tvt"] - last_known_tvt
            rows.append(item)
    return pd.DataFrame(rows)


def ensure_offline_beam_feature_set() -> list[str]:
    if "offline_beam_candidate_path_alignment" not in FEATURE_SETS:
        FEATURE_SETS["offline_beam_candidate_path_alignment"] = list(dict.fromkeys(
            STRICT_FEATURE_SETS["typewell_alignment"]
            + OFFLINE_EXTRA_FEATURE_COLUMNS
            + OFFLINE_CANDIDATE_PATH_COLUMNS
            + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS
            + OFFLINE_BEAM_FEATURE_COLUMNS
        ))
        FEATURE_SET_POLICY["offline_beam_candidate_path_alignment"] = "offline"
    if "offline_full_physics_alignment" not in FEATURE_SETS:
        FEATURE_SETS["offline_full_physics_alignment"] = list(dict.fromkeys(
            STRICT_FEATURE_SETS["typewell_alignment"]
            + OFFLINE_EXTRA_FEATURE_COLUMNS
            + OFFLINE_CANDIDATE_PATH_COLUMNS
            + OFFLINE_CANDIDATE_PATH_EASE_COLUMNS
            + OFFLINE_BEAM_FEATURE_COLUMNS
            + FORMATION_PLANE_COLUMNS
            + FORMATION_ROW_ANCC_COLUMNS
            + PF_TVT_FEATURE_COLUMNS
            + PF_ANCC_FEATURE_COLUMNS
        ))
        FEATURE_SET_POLICY["offline_full_physics_alignment"] = "offline"
    return FEATURE_SETS["offline_full_physics_alignment"]


def make_lgb_residual_model(seed: int = 42):
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise RuntimeError(f"LightGBM is not available: {exc}") from exc
    memory_safe = bool(globals().get("KAGGLE_MEMORY_SAFE_MODE", False))
    params = dict(
        objective="regression_l2",
        metric="rmse",
        n_estimators=1200 if memory_safe else 1700,
        learning_rate=0.03 if memory_safe else 0.025,
        num_leaves=63 if memory_safe else 128,
        min_child_samples=100 if memory_safe else 80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70 if memory_safe else 0.75,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(lightgbm_accelerator_params())
    return LGBMRegressor(**params)


def make_lgb_goss_residual_model(seed: int = 42):
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise RuntimeError(f"LightGBM is not available: {exc}") from exc
    params = dict(
        boosting_type="goss",
        objective="regression_l2",
        metric="rmse",
        n_estimators=3500,
        learning_rate=0.025,
        num_leaves=70,
        min_child_samples=20,
        reg_alpha=50.0,
        reg_lambda=50.0,
        colsample_bytree=0.60,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(lightgbm_accelerator_params())
    return LGBMRegressor(**params)


def make_xgb_residual_model(seed: int = 42):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(f"XGBoost is not available: {exc}") from exc
    # XGBoost 2.x removed tree_method="gpu_hist". Use hist + device="cuda" on Kaggle GPUs.
    accelerator_params = {"tree_method": "hist"}
    if XGBOOST_USE_CUDA:
        accelerator_params["device"] = "cuda"
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1400,
        learning_rate=0.035,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        **accelerator_params,
    )


def make_catboost_residual_model(seed: int = 42):
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:
        raise RuntimeError(f"CatBoost is not available: {exc}") from exc
    task_type = CATBOOST_TASK_TYPE
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=1600,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=6.0,
        random_seed=seed,
        task_type=task_type,
        verbose=False,
        allow_writing_files=False,
    )


def make_model(model_name: str, seed: int = 42):
    if model_name in {"lgb_plain"} or re.fullmatch(r"lgb_seed\d+", model_name):
        return make_lgb_residual_model(seed)
    if model_name == "lgb_goss" or re.fullmatch(r"lgb_goss_seed\d+", model_name):
        return make_lgb_goss_residual_model(seed)
    if model_name == "xgb" or re.fullmatch(r"xgb_seed\d+", model_name):
        return make_xgb_residual_model(seed)
    if model_name == "catboost" or re.fullmatch(r"catboost_seed\d+", model_name):
        return make_catboost_residual_model(seed)
    raise ValueError(f"Unknown model_name: {model_name}")


def feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    # Return a compact float32 matrix rather than a pandas frame to reduce model-fit peak memory.
    return np.ascontiguousarray(frame[feature_columns].to_numpy(dtype=np.float32, copy=False))


def prediction_artifact_from_frame(
    frame: pd.DataFrame,
    pred_delta,
    branch_name: str,
    model_name: str,
    prediction_type: str,
    fold=None,
    include_target: bool = True,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    out_cols = ["id", "well_id", "row_index", "last_known_TVT"]
    if "MD" in frame.columns:
        out_cols.append("MD")
    out = frame[out_cols].copy()
    pred_delta = np.asarray(pred_delta, dtype=float)
    out["pred_delta"] = pred_delta
    out["pred_tvt"] = out["last_known_TVT"].to_numpy(dtype=float) + pred_delta
    out["branch_name"] = branch_name
    out["model_name"] = model_name
    out["prediction_type"] = prediction_type
    if fold is not None:
        out["fold"] = int(fold)
    if include_target and "target_tvt" in frame.columns:
        out["target_tvt"] = frame["target_tvt"].to_numpy(dtype=float)
        out["target_delta"] = frame["target_delta_from_last_known"].to_numpy(dtype=float)
    if extra_columns:
        for col in extra_columns:
            if col in frame.columns and col not in out.columns:
                out[col] = frame[col].to_numpy()
    return out


def average_test_predictions(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts, ignore_index=True, copy=False)
    meta_cols = ["id", "well_id", "row_index", "last_known_TVT", "branch_name", "model_name", "prediction_type"]
    if "MD" in data.columns:
        meta_cols.append("MD")
    meta = data.sort_values(["id"]).groupby("id", as_index=False)[meta_cols[1:]].first()
    out = meta.merge(data.groupby("id", as_index=False)["pred_delta"].mean(), on="id", how="left")
    standard = set(meta_cols + ["pred_delta", "pred_tvt", "fold", "target_tvt", "target_delta"])
    numeric_extra = [
        col for col in data.columns
        if col not in standard and pd.api.types.is_numeric_dtype(data[col])
    ]
    if numeric_extra:
        extra = data.groupby("id", as_index=False)[numeric_extra].mean()
        out = out.merge(extra, on="id", how="left")
    out["pred_tvt"] = out["last_known_TVT"].to_numpy(dtype=float) + out["pred_delta"].to_numpy(dtype=float)
    return out


def residual_rmse_from_artifact(frame: pd.DataFrame) -> float:
    return rmse(frame["target_tvt"].to_numpy(dtype=float), frame["pred_tvt"].to_numpy(dtype=float))


def align_submission_to_sample(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    submission = prediction_frame[["id", "tvt"]].copy()
    if SAMPLE_SUBMISSION.exists():
        sample = pd.read_csv(SAMPLE_SUBMISSION)
        submission = sample[["id"]].merge(submission, on="id", how="left")
    missing = int(submission["tvt"].isna().sum())
    if missing:
        missing_ids = submission.loc[submission["tvt"].isna(), "id"].head(10).tolist()
        raise RuntimeError(f"Submission has missing predictions, for example: {missing_ids}")
    return submission


def fit_constrained_residual_blend(y_delta: np.ndarray, pred_delta_matrix: np.ndarray) -> dict:
    y_delta = np.asarray(y_delta, dtype=float)
    pred_delta_matrix = np.asarray(pred_delta_matrix, dtype=float)
    valid = np.isfinite(y_delta) & np.all(np.isfinite(pred_delta_matrix), axis=1)
    y = y_delta[valid]
    X = pred_delta_matrix[valid]
    n_models = X.shape[1]
    if n_models == 0:
        return {"weights": [], "rmse": float(np.sqrt(np.mean(y ** 2))), "success": True}
    try:
        from scipy.optimize import minimize
        def objective(w):
            pred = X @ w
            return float(np.mean((y - pred) ** 2))
        constraints = ({"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)},)
        bounds = [(0.0, 1.0)] * n_models
        x0 = np.full(n_models, min(1.0 / max(n_models, 1), 0.95 / max(n_models, 1)))
        res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
        w = np.clip(res.x if res.success else x0, 0.0, 1.0)
    except Exception:
        ridge = 1e-4 * np.eye(n_models)
        w = np.linalg.solve(X.T @ X + ridge, X.T @ y)
        w = np.clip(w, 0.0, 1.0)
        if w.sum() > 1.0:
            w = w / w.sum()
    pred = X @ w
    return {
        "weights": [float(v) for v in w],
        "anchor_weight": float(max(0.0, 1.0 - float(np.sum(w)))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "success": True,
    }


def describe_environment() -> pd.Series:
    return pd.Series({
        "data_dir": DATA_DIR.as_posix() if DATA_DIR != LOCAL_DATA_DIR else "./",
        "output_dir": OUTPUT_DIR.as_posix() if OUTPUT_DIR != DATA_DIR else "./",
        "train_wells": len(train_h_ids),
        "test_wells": len(test_h_ids),
        "kaggle_run": KAGGLE_NOTEBOOK_RUN,
        "lightgbm_device_type": LIGHTGBM_DEVICE_TYPE,
        "xgboost_use_cuda": XGBOOST_USE_CUDA,
        "catboost_task_type": CATBOOST_TASK_TYPE,
        "feature_sets": len(FEATURE_SETS),
    })


# Notebook-local namespace adapter. The remaining cells use rg.* for readability,
# while all definitions above live inside this notebook.
class _RogiiNotebookNamespace:
    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        globals()[name] = value


rg = _RogiiNotebookNamespace()
ARTIFACT_ROOT = rg.ensure_artifact_dirs()
display(rg.describe_environment().to_frame('value'))
