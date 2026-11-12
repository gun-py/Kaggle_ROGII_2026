from __future__ import annotations

import os
from pathlib import Path

SLUG = "rogii-wellbore-geology-prediction"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _is_competition_dir(path: Path) -> bool:
    return (
        (path / "train").is_dir() and (path / "test").is_dir() and (path / "sample_submission.csv").exists()
    )


def competition_dir(explicit: str | Path | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("ROGII_DATA"),
        KAGGLE_INPUT / "competitions" / SLUG,
        KAGGLE_INPUT / SLUG,
        Path("data/raw"),
        Path("."),
    ]
    for candidate in candidates:
        if candidate and _is_competition_dir(Path(candidate)):
            return Path(candidate)
    if KAGGLE_INPUT.exists():
        for hit in sorted(KAGGLE_INPUT.glob("**/sample_submission.csv")):
            if _is_competition_dir(hit.parent):
                return hit.parent
    raise FileNotFoundError(
        "competition data not found; pass --data, set ROGII_DATA or run scripts/fetch_data.sh"
    )


def work_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        out = Path(explicit)
    elif os.environ.get("ROGII_OUT"):
        out = Path(os.environ["ROGII_OUT"])
    elif KAGGLE_WORKING.exists():
        out = KAGGLE_WORKING
    else:
        out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifact_roots(name: str) -> list[Path]:
    """Candidate mount points for a named artifact dataset, most specific first."""
    env = os.environ.get(f"ROGII_{name.upper().replace('-', '_')}")
    roots = [Path(env)] if env else []
    roots += [
        Path("artifacts") / name,
        KAGGLE_INPUT / name,
        *sorted(KAGGLE_INPUT.glob(f"datasets/*/{name}")),
    ]
    return roots


def first_existing(roots, marker: str) -> Path | None:
    """First root that contains `marker` (a relative file or folder)."""
    return next((r for r in map(Path, roots) if (r / marker).exists()), None)
