from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HW_SUFFIX = "__horizontal_well"
TW_SUFFIX = "__typewell"


def list_wells(data_dir: Path, split: str) -> list[str]:
    files = (Path(data_dir) / split).glob(f"*{HW_SUFFIX}.csv")
    return sorted(p.name.split("__")[0] for p in files)


def read_well(data_dir: Path, wid: str, split: str = "train", sort_typewell: bool = False):
    base = Path(data_dir) / split
    hw = pd.read_csv(base / f"{wid}{HW_SUFFIX}.csv")
    tw = pd.read_csv(base / f"{wid}{TW_SUFFIX}.csv")
    return hw, tw.sort_values("TVT") if sort_typewell else tw


def read_sample(data_dir: Path) -> pd.DataFrame:
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv")
    sample["id"] = sample["id"].astype(str)
    return sample


def split_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("expected ids shaped like <well>_<row>")
    out["well"] = parts[0]
    out["row_idx"] = parts[1].astype(int)
    return out


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"id", "tvt"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame[["id", "tvt"]].copy()
    frame["id"] = frame["id"].astype(str)
    frame["tvt"] = frame["tvt"].astype(float)
    return frame


def write_submission(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df[["id", "tvt"]].copy()
    out["id"] = out["id"].astype(str)
    out["tvt"] = out["tvt"].astype(float)
    out.to_csv(path, index=False)
    return out


def align_to_sample(df: pd.DataFrame, sample: pd.DataFrame, label: str = "submission") -> pd.DataFrame:
    """Reindex a prediction frame to the sample ids, refusing gaps, duplicates and non-finite values."""
    frame = df[["id", "tvt"]].copy()
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"{label}: duplicate ids")
    aligned = sample[["id"]].merge(frame, on="id", how="left")
    aligned["tvt"] = pd.to_numeric(aligned["tvt"], errors="coerce")
    if not np.isfinite(aligned["tvt"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: missing or non-finite predictions after alignment")
    return aligned


def validate_submission(df: pd.DataFrame, sample: pd.DataFrame, label: str = "submission") -> pd.DataFrame:
    """Return a clean id,tvt frame, raising unless it matches the sample row for row."""
    out = df[["id", "tvt"]].copy()
    out["id"] = out["id"].astype(str)
    out["tvt"] = out["tvt"].astype(float)
    if len(out) != len(sample):
        raise ValueError(f"{label}: expected {len(sample)} rows, got {len(out)}")
    if not out["id"].equals(sample["id"].astype(str)):
        raise ValueError(f"{label}: id order does not match sample_submission.csv")
    if not np.isfinite(out["tvt"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite tvt values")
    return out
