"""Run the notebook's post-ridge cells (selector -> audit) in a scratch folder.

usage: python -m tests.parity.legacy_chain DATA OUT PROFILE RIDGE_CSV LEARNED_CSV
The two stubbed stages (ridge stack, learned trajectory) are replaced by the given CSVs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from tests.parity import legacy

PRELUDE = """
import numpy as np, pandas as pd, os, glob, time, warnings, multiprocessing, joblib
from pathlib import Path
from numba import njit
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
"""
FAST = {
    "SP45_SELECTOR_N_PARTICLES = 500": "SP45_SELECTOR_N_PARTICLES = 100",
    "SP45_SELECTOR_N_SEEDS = 128": "SP45_SELECTOR_N_SEEDS = 8",
    "VISIBLE_PREFIX_CAL_SEEDS = 24": "VISIBLE_PREFIX_CAL_SEEDS = 4",
    "VISIBLE_PREFIX_FINAL_SEEDS = 48": "VISIBLE_PREFIX_FINAL_SEEDS = 8",
    "VISIBLE_PREFIX_PARTICLES = 350": "VISIBLE_PREFIX_PARTICLES = 100",
}


def main(data: str, out: str, profile: str, ridge_csv: str, learned_csv: str) -> None:
    data_dir, out_dir = Path(data).resolve(), Path(out).resolve()
    ridge_csv, learned_csv = str(Path(ridge_csv).resolve()), Path(learned_csv).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    cells = legacy.cells()

    def source(i: int) -> str:
        src = cells[i]
        if i == 6:
            src = src.replace(
                "'/kaggle/input/competitions/rogii-wellbore-geology-prediction'", repr(str(data_dir))
            )
            src = src.replace(
                "SUBMISSION_PROFILE = 'vp_balanced_modelpkg_005'", f"SUBMISSION_PROFILE = '{profile}'"
            )
            for old, new in FAST.items():
                src = src.replace(old, new)
        if i == 48:
            src = src.replace(
                "_OvPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction')",
                f"_OvPath({str(data_dir)!r})",
            )
        for old, new in legacy.WRITABLE_VIEWS:
            src = src.replace(old, new)
        return src.replace("@njit(cache=True, nogil=True)", "@njit(nogil=True)").replace(
            "@njit(cache=True)", "@njit"
        )

    ns = {"__name__": "__main__", "display": print, "OUTPUT_DIR": Path(".")}
    exec(PRELUDE, ns)
    for i in (6, 7, 10, 11, 14):
        exec(compile(source(i), f"<cell {i}>", "exec"), ns)
    ns["sub_1"] = pd.read_csv(ridge_csv)
    for i in (26, 27, 28, 30, 32):
        exec(compile(source(i), f"<cell {i}>", "exec"), ns)
    pd.read_csv(learned_csv).to_csv(
        "submission.csv", index=False
    )  # stands in for the learned-trajectory cells
    for i in (45, 47, 48, 50, 51, 52, 53):
        exec(compile(source(i), f"<cell {i}>", "exec"), ns)


if __name__ == "__main__":
    sys.path.insert(0, str(legacy.REPO))
    main(*sys.argv[1:6])
