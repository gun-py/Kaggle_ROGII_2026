from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.parity import legacy


@pytest.fixture(scope="session")
def data() -> Path:
    root = Path(os.environ.get("ROGII_DATA", legacy.REPO / "data" / "raw"))
    if not (root / "train").is_dir() or not (root / "sample_submission.csv").exists():
        pytest.skip("competition data not available (run scripts/fetch_data.sh)")
    return root


@pytest.fixture(scope="session", autouse=True)
def legacy_tag():
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", legacy.TAG], cwd=legacy.REPO, capture_output=True
    )
    if probe.returncode != 0:
        pytest.skip(f"git tag {legacy.TAG} not found")


def pytest_collection_modifyitems(items):
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.parity)
