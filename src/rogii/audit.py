from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_submission(sub_path: Path, sample_path: Path) -> dict:
    """Check the file against the sample (columns, rows, id order, finite values) and summarise it."""
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)
    if list(sub.columns) != ["id", "tvt"]:
        raise RuntimeError(f"unexpected submission columns: {list(sub.columns)}")
    if len(sub) != len(sample):
        raise RuntimeError(f"unexpected row count: submission={len(sub)} sample={len(sample)}")
    if not sub["id"].astype(str).equals(sample["id"].astype(str)):
        raise RuntimeError("submission id order does not match sample_submission.csv")
    tvt = sub["tvt"].to_numpy(dtype=float)
    if not np.isfinite(tvt).all():
        raise RuntimeError("submission contains non-finite tvt values")
    return {
        "rows": int(len(sub)),
        "columns": list(sub.columns),
        "id_order_matches_sample": True,
        "tvt_min": float(tvt.min()),
        "tvt_max": float(tvt.max()),
        "tvt_mean": float(tvt.mean()),
        "tvt_std": float(tvt.std()),
        "sha256_submission_csv": sha256_file(sub_path),
    }


def write_audit(out_dir: Path, sample_path: Path) -> dict:
    out_dir = Path(out_dir)
    submission = out_dir / "submission.csv"
    audit = audit_submission(submission, sample_path)
    text = json.dumps(audit, indent=2, sort_keys=True)
    (out_dir / "submission_audit.json").write_text(text, encoding="utf-8")
    frame = pd.read_csv(submission)
    frame.to_csv(out_dir / "submission_audit_copy.csv", index=False)
    frame.to_csv(out_dir / "latest_valid_submission.csv", index=False)
    (out_dir / "latest_valid_submission.json").write_text(text, encoding="utf-8")
    return audit
