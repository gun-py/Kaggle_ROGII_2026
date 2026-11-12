"""Load definitions from the pre-refactor notebooks so new code can be compared against them."""

from __future__ import annotations

import ast
import json
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
TAG = "legacy-notebooks"
BASE_NOTEBOOK = "high-scoring-reproducible.ipynb"


@cache
def cells(notebook: str = BASE_NOTEBOOK) -> tuple[str, ...]:
    raw = subprocess.run(
        ["git", "show", f"{TAG}:{notebook}"], cwd=REPO, capture_output=True, check=True
    ).stdout
    return tuple("".join(c["source"]) for c in json.loads(raw)["cells"])


def _is_definition(node: ast.stmt, constants: set[str]) -> bool:
    if isinstance(node, ast.FunctionDef | ast.ClassDef | ast.Import | ast.ImportFrom):
        return True
    if isinstance(node, ast.Assign):
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        return bool(targets & constants)
    return False


# The notebooks mutate `.to_numpy()` views in place, which only works on pandas < 3.
WRITABLE_VIEWS = [
    ("hw['TVT_input'].isna().to_numpy()[:n]", "hw['TVT_input'].isna().to_numpy().copy()[:n]"),
    ("hw['TVT_input'].notna().to_numpy()[:n]", "hw['TVT_input'].notna().to_numpy().copy()[:n]"),
    ("_mask = (_well == str(_wid)).to_numpy()", "_mask = (_well == str(_wid)).to_numpy().copy()"),
    (
        "_tvt = _bh_pd.to_numeric(_sub['tvt'], errors='coerce').to_numpy(dtype=float)",
        "_tvt = _bh_pd.to_numeric(_sub['tvt'], errors='coerce').to_numpy(dtype=float).copy()",
    ),
]


def load(
    index: int, constants=(), notebook: str = BASE_NOTEBOOK, namespace: dict | None = None, patches=()
) -> dict:
    """Execute only the defs (and the named constants) of one notebook cell."""
    ns = {} if namespace is None else namespace
    ns.update(np=np, pd=pd, njit=njit, savgol_filter=savgol_filter, cKDTree=cKDTree, Path=Path)
    source = cells(notebook)[index]
    for old, new in [*WRITABLE_VIEWS, *patches]:
        source = source.replace(old, new)
    source = source.replace("@njit(cache=True, nogil=True)", "@njit(nogil=True)").replace(
        "@njit(cache=True)", "@njit"
    )
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if _is_definition(n, set(constants))]
    exec(compile(tree, f"<legacy cell {index}>", "exec"), ns)
    return ns


def seed_numba(seed: int = 0) -> None:
    @njit
    def _seed(s):
        np.random.seed(s)

    _seed(seed)


def run_until(notebook: str, index: int, marker: str, patches=()) -> dict:
    """Execute a notebook cell up to (not including) the line that starts with `marker`."""
    source = cells(notebook)[index]
    for old, new in patches:
        source = source.replace(old, new)
    head = source.split(marker, 1)[0]
    ns: dict = {}
    exec(compile(head, f"<legacy {notebook}:{index}>", "exec"), ns)
    return ns
