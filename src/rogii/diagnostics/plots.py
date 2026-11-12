from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.io import list_wells, read_well
from rogii.tracking.trackers import lik_pf, run_pf_ancc


def _plt():
    import matplotlib.pyplot as plt

    return plt


def well_overview(data_dir: Path, wid: str, split: str = "train"):
    plt = _plt()
    hw, _ = read_well(data_dir, wid, split)
    known, unknown = hw[hw.TVT_input.notna()], hw[hw.TVT_input.isna()]
    stop = known.MD.iloc[-1]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].plot(hw.MD, hw.Z, lw=1.2, color="#333")
    ax[0].axvline(stop, color="crimson", ls="--", label="last known")
    ax[0].set_ylabel("Z / TVD (ft)")
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"Well {wid}: trajectory, gamma ray, TVT")
    ax[1].plot(known.MD, known.GR, lw=0.7, color="steelblue", label="GR known")
    ax[1].plot(unknown.MD, unknown.GR, lw=0.7, color="darkorange", label="GR to predict")
    ax[1].axvline(stop, color="crimson", ls="--")
    ax[1].set_ylabel("GR (API)")
    ax[1].legend(loc="upper right")
    ax[2].plot(known.MD, known.TVT_input, lw=1.6, color="seagreen", label="TVT known")
    if "TVT" in hw.columns:
        ax[2].plot(unknown.MD, unknown.TVT, lw=1.6, color="crimson", label="TVT to predict")
    ax[2].axvline(stop, color="crimson", ls="--")
    ax[2].set_ylabel("TVT (ft)")
    ax[2].set_xlabel("MD (ft)")
    ax[2].invert_yaxis()
    ax[2].legend(loc="upper right")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def typewell_signature(data_dir: Path, wid: str):
    plt = _plt()
    hw, tw = read_well(data_dir, wid, "train", sort_typewell=True)
    unknown = hw[hw.TVT_input.isna()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    ax[0].plot(tw.GR, tw.TVT, lw=1.0, color="black")
    ax[0].set(xlabel="GR (API)", ylabel="TVT (ft)", title="Typewell signature")
    ax[0].invert_yaxis()
    points = ax[1].scatter(unknown.GR, unknown.TVT, s=4, c=unknown.MD, cmap="viridis")
    ax[1].set(xlabel="GR (API)", ylabel="TVT (ft)", title="Horizontal GR at its true TVT")
    ax[1].invert_yaxis()
    fig.colorbar(points, ax=ax[1], label="MD (ft)")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def drift_tail(data_dir: Path, n_wells: int = 250, seed: int = 1):
    """Per-well error of the last-known baseline: a handful of drifting wells dominate the pooled metric."""
    plt = _plt()
    wells = list_wells(data_dir, "train")
    sample = sorted(
        np.random.default_rng(seed).choice(wells, min(n_wells, len(wells)), replace=False).tolist()
    )
    per = []
    for wid in sample:
        try:
            hw = pd.read_csv(
                Path(data_dir) / "train" / f"{wid}__horizontal_well.csv", usecols=["TVT_input", "TVT"]
            )
        except Exception:
            continue
        unknown, known = hw[hw.TVT_input.isna()], hw[hw.TVT_input.notna()]
        if len(unknown) == 0 or len(known) < 10 or unknown.TVT.isna().any():
            continue
        per.append(np.sqrt(np.mean((unknown.TVT.values - known.TVT_input.iloc[-1]) ** 2)))
    per = np.array(per)
    ranked = np.sort(per)[::-1]
    cumulative = np.cumsum(ranked**2) / np.sum(ranked**2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(per, bins=40, color="indianred", alpha=0.85)
    ax[0].axvline(np.median(per), color="k", ls="--", label=f"median={np.median(per):.1f}")
    ax[0].axvline(per.mean(), color="b", ls="--", label=f"mean={per.mean():.1f}")
    ax[0].set(xlabel="per-well last-known RMSE (ft)", ylabel="wells", title="Per-well error is right-skewed")
    ax[0].legend()
    ax[1].plot(np.arange(1, len(ranked) + 1) / len(ranked) * 100, cumulative * 100, color="purple")
    ax[1].axhline(80, color="gray", ls=":")
    ax[1].set(xlabel="% of wells (worst first)", ylabel="% of pooled squared error")
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def trackers_vs_truth(data_dir: Path, wid: str):
    plt = _plt()
    hw, tw = read_well(data_dir, wid, "train", sort_typewell=True)
    known, unknown = hw[hw.TVT_input.notna()], hw[hw.TVT_input.isna()]
    single, _ = run_pf_ancc(hw, tw.TVT.to_numpy(np.float32), tw.GR.to_numpy(np.float32))
    ensemble, _, _ = lik_pf(hw, tw, scales=(3.0,))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(unknown.MD, unknown.TVT, lw=2.2, color="black", label="true TVT", zorder=5)
    ax.plot(
        unknown.MD,
        np.full(len(unknown), float(known.TVT_input.iloc[-1])),
        lw=1.1,
        color="gray",
        ls=":",
        label="last known",
    )
    ax.plot(unknown.MD, single, lw=1.0, color="tab:blue", alpha=0.8, label="single particle filter")
    ax.plot(
        unknown.MD,
        ensemble["pf_scale_3"],
        lw=1.5,
        color="crimson",
        alpha=0.9,
        label="likelihood-weighted PF ensemble",
    )
    ax.set(xlabel="MD (ft)", ylabel="TVT (ft)", title=f"Well {wid}: trackers vs ground truth")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
