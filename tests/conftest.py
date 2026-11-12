from __future__ import annotations

import pytest

from tests import synthetic


@pytest.fixture(scope="session")
def competition(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "comp"
    return synthetic.write_competition(root)


@pytest.fixture(scope="session")
def well():
    return synthetic.make_well("w0", 1500.0, 900.0, n_rows=1400, known_frac=0.45, seed=3)
