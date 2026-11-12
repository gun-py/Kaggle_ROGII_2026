"""Selector -> projection -> blend -> override -> prefix calibration -> hedge -> audit, end to end.

The ridge stack and the learned trajectory need downloaded artifacts, so both sides read the same stub files.
"""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rogii import pipeline
from rogii.io import list_wells, read_sample, split_ids
from tests.parity import legacy

PROFILES = ["vp_balanced_final", "bimodal_guarded", "contact_gated_anchor"]
IGNORED_JSON_KEYS = {
    "elapsed_sec",
    "chosen_path",
    "file",
    "sha256",
    "selected_sha256",
    "self_verified_anchor_sha256",
    "sha256_submission_csv",
}


def _equal(a, b, path="") -> None:
    """Recursive comparison of two parsed JSON documents, floats within 1e-8, hashes and paths ignored."""
    if isinstance(a, dict):
        keys = {k for k in a if k not in IGNORED_JSON_KEYS}
        assert keys == {k for k in b if k not in IGNORED_JSON_KEYS}, path
        for key in keys:
            _equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, float):
        assert a == pytest.approx(b, abs=1e-8), path
    else:
        assert a == b, path


@pytest.fixture(scope="module")
def stubs(data, tmp_path_factory):
    root = tmp_path_factory.mktemp("stubs")
    sample = split_ids(read_sample(data))
    last = {
        w: float(pd.read_csv(data / "test" / f"{w}__horizontal_well.csv")["TVT_input"].dropna().iloc[-1])
        for w in list_wells(data, "test")
    }
    carry = pd.DataFrame({"id": sample["id"], "tvt": sample["well"].map(last)})
    carry.to_csv(root / "ridge.csv", index=False)
    carry.assign(tvt=carry["tvt"] + 0.5).to_csv(root / "learned.csv", index=False)
    return root, carry


@pytest.mark.parametrize("profile", PROFILES)
def test_pipeline_matches_the_notebook(profile, data, stubs, tmp_path, monkeypatch):
    root, carry = stubs
    legacy_out, new_out = tmp_path / "legacy", tmp_path / "new"
    subprocess.run(
        [sys.executable, "-m", "tests.parity.legacy_chain", str(data), str(legacy_out), profile, str(root / "ridge.csv"), str(root / "learned.csv")],
        cwd=legacy.REPO, check=True, capture_output=True,
    )  # fmt: skip
    monkeypatch.setattr(pipeline, "_ridge", lambda run: carry.copy())
    monkeypatch.setattr(pipeline, "_learned", lambda run, fast: carry.assign(tvt=carry["tvt"] + 0.5))
    pipeline.run(profile, data_dir=data, out_dir=new_out, artifacts_dir=tmp_path / "none", fast=True)

    compared = 0
    for old in sorted(legacy_out.iterdir()):
        new = new_out / old.name
        assert new.exists(), f"{old.name} is not written by the new pipeline"
        if old.suffix == ".json":
            _equal(json.loads(old.read_text()), json.loads(new.read_text()), old.name)
            continue
        a, b = pd.read_csv(old), pd.read_csv(new)
        assert list(a.columns) == list(b.columns) and a.shape == b.shape, old.name
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]):
                assert np.allclose(
                    a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=1e-8, equal_nan=True
                ), (old.name, col)
            else:
                assert a[col].astype(str).tolist() == b[col].astype(str).tolist(), (old.name, col)
        compared += 1
    assert compared >= 20
