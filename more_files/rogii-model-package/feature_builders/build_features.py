from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_core(data_dir: Path):
    data_dir = Path(data_dir)
    os.environ["ROGII_DATA_DIR"] = str(data_dir)
    core_path = Path(__file__).resolve().parent / "rogii_feature_core.py"
    spec = importlib.util.spec_from_file_location("rogii_feature_core_runtime", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load feature core: {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _feature_columns_for_manifest(package_root: Path, manifest: dict) -> list[str]:
    rel = manifest.get("feature_columns", "feature_builders/feature_columns.json")
    cfg = json.loads((package_root / rel).read_text())
    cols: list[str] = []
    for entry in manifest.get("models", []):
        if entry.get("model_type") == "direct_feature":
            source_col = entry.get("feature_column")
            if source_col:
                cols.append(str(source_col))
            continue
        feature_set = entry.get("feature_set")
        entry_cols = entry.get("feature_columns")
        if isinstance(entry_cols, list):
            cols.extend(map(str, entry_cols))
        elif isinstance(cfg, dict) and feature_set and isinstance(cfg.get(feature_set), list):
            cols.extend(map(str, cfg[feature_set]))
        elif isinstance(cfg, dict) and isinstance(cfg.get("columns"), list):
            cols.extend(map(str, cfg["columns"]))
        elif isinstance(cfg, list):
            cols.extend(map(str, cfg))
        else:
            raise RuntimeError(f"Could not resolve feature columns for {entry}")
    return list(dict.fromkeys(cols))


def build_features(
    data_dir=None,
    competition_root=None,
    sample_submission=None,
    sample=None,
    package_root=None,
    manifest=None,
    config=None,
):
    data_root = Path(data_dir or competition_root)
    package_root = Path(package_root or Path(__file__).resolve().parents[1])
    manifest = manifest or config or json.loads((package_root / "metadata" / "model_package_manifest.json").read_text())
    sample_df = sample_submission if sample_submission is not None else sample
    if sample_df is None:
        sample_df = pd.read_csv(data_root / "sample_submission.csv")
    sample_df = sample_df.copy()
    sample_df["id"] = sample_df["id"].astype(str)

    core = _load_core(data_root)
    feature_columns = _feature_columns_for_manifest(package_root, manifest)
    requires_beam = core.feature_columns_require_beam(feature_columns)
    requires_formation = core.feature_columns_require_formation(feature_columns)
    requires_row_ancc = core.feature_columns_require_row_ancc(feature_columns)
    plane_imputer = None
    row_imputer = None
    if requires_formation:
        train_wells = np.array(sorted(core.train_h_ids))
        plane_imputer, row_imputer = core.make_formation_imputers(
            train_wells,
            core.TRAIN_DIR,
            need_row_ancc=requires_row_ancc,
            seed=int(manifest.get("seed", 42)),
        )

    test_wells = np.array(sorted(core.test_h_ids))
    frame = core.build_tail_feature_frame(
        test_wells,
        core.TEST_DIR,
        include_target=False,
        rows_per_well=None,
        random_state=int(manifest.get("seed", 42)) + 1,
        use_beam_features=requires_beam,
        keep_columns=feature_columns,
        formation_plane_imputer=plane_imputer,
        row_ancc_imputer=row_imputer,
        imputer_well_ids=None,
        keep_metadata=True,
        exclude_query_well_from_formation=True,
    )
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    aligned = sample_df[["id"]].merge(frame, on="id", how="left")
    missing = aligned["last_known_TVT"].isna().sum() if "last_known_TVT" in aligned.columns else len(aligned)
    if int(missing):
        bad = aligned.loc[aligned["last_known_TVT"].isna(), "id"].head(10).tolist() if "last_known_TVT" in aligned.columns else sample_df["id"].head(10).tolist()
        raise RuntimeError(f"Feature builder missing {int(missing)} sample ids; examples={bad}")
    return aligned
