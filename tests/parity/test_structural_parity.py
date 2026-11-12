"""Every structural variant against the appended notebook cell it replaces."""

import numpy as np
import pytest

from rogii.io import list_wells
from rogii.structural.field import Field
from rogii.structural.variants import load_variant
from rogii.structural.wells import load_wells
from tests.parity import legacy

NOTEBOOKS = {
    "v2": "tvt-v2-blended.ipynb",
    "v3": "tvt-v3-threeway.ipynb",
    "v5": "tvt-v5-threeway.ipynb",
    "cv1": "CV1.ipynb",
    "cv2": "CV2.ipynb",
    "cv3": "CV3.ipynb",
    "cv4": "CV4.ipynb",
}


@pytest.fixture(scope="module")
def new_field(data):
    return Field(load_wells(data, "train"))


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_variant_matches_the_notebook(name, data, new_field):
    if name == "cv3":
        pytest.importorskip("torch")
    patches = [('Path(".")]', f'Path("{data}")]'), ('OUT = Path("/kaggle/working")', 'OUT = Path(".")')]
    ns = legacy.run_until(NOTEBOOKS[name], 58, "existing = pd.read_csv", patches=patches)
    old_field = ns["Field"](ns["train_ids"])
    variant = load_variant(name)
    for wid in list_wells(data, "test"):
        old = ns["predict"](old_field, ns["load_well"](wid, "test"), {wid})
        old = old[0] if isinstance(old, tuple) else old
        new, _ = variant.predict(new_field, load_wells(data, "test", [wid])[wid], {wid})
        assert np.nanmax(np.abs(old - new)) < 1e-8
