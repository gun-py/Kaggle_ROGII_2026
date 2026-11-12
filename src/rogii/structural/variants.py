from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from .components import components
from .field import Field
from .wells import anchor_index

NEIGHBOUR_CAP = 2000.0


@dataclass(frozen=True)
class BlendRule:
    """How the structural prediction is mixed into the existing submission."""

    mode: str = "fixed"
    stack: float = 0.65
    struct: float = 0.35
    package: float = 0.0
    schedule: tuple[tuple[float, float], ...] = ()
    beyond: float = 0.30
    no_neighbours: float = 0.30


@dataclass(frozen=True)
class DLRule:
    weight: float = 0.0
    folds: int = 5
    seeds: int = 2
    epochs: int = 60
    search: bool = True
    guard_oof_ratio: float = 0.97
    guard_mean_ft: float = 15.0
    guard_max_nan: float = 0.02


@dataclass(frozen=True)
class Variant:
    name: str
    combiner: str
    sanitize: bool
    blend: BlendRule
    dl: DLRule | None = None
    description: str = ""
    tables: dict = field(default_factory=dict, compare=False, repr=False)

    def predict(self, fld: Field, d: dict, drop) -> tuple[np.ndarray, float]:
        """TVT path for one well and the distance to its nearest supporting neighbour."""
        P, nearest = components(fld, d, drop, sanitize=self.sanitize)
        ia = anchor_index(d)
        ta, za = d["tvt"][ia], d["z"][ia]
        t = self.tables

        if self.combiner == "distance_bins":
            return P @ _bin_row(t["bins"], nearest), nearest
        if self.combiner == "md_bands":
            band = np.digitize(d["md"] - d["md"][ia], t["bands"]["md_edges"])
            b = int(np.digitize(nearest, t["bands"]["nb_edges"]))
            return np.einsum("ij,ij->i", P, np.asarray(t["bands"]["weights"])[b][band]), nearest

        drift = P - ta
        capped = float(min(nearest, NEIGHBOUR_CAP)) if np.isfinite(nearest) else NEIGHBOUR_CAP
        log_nd = np.log1p(capped)
        md_since = d["md"] - d["md"][ia]
        mdk = md_since / 1000.0
        dz = d["z"] - za
        feat = np.column_stack([drift, np.full(len(drift), log_nd), mdk, dz])
        feat_i = np.column_stack([feat, drift * log_nd, drift * mdk[:, None]])
        e = t["ensemble"]
        refit = drift @ np.asarray(e["refit"])[int(np.digitize(capped, e["nb_edges"]))]
        enet = feat_i @ np.asarray(e["en"]) + e["en_bias"]
        pls = feat @ np.asarray(e["pls"]) + e["pls_bias"]
        members = [refit, enet, pls]

        unknown = np.where(~d["known"])[0]
        if self.combiner == "ensemble_seq" and len(unknown):
            from . import sequence

            base = P @ _bin_row(t["bins"], nearest) - ta
            members += list(sequence.member_paths(P, base, nearest, md_since, dz, ta, unknown))
        return ta + sum(members) / len(members), nearest


def _bin_row(bins: dict, nearest: float) -> np.ndarray:
    return np.asarray(bins["weights"])[int(np.digitize(nearest, bins["edges"]))]


def _read(name: str) -> dict:
    path = resources.files("rogii") / "configs" / "structural" / f"{name}.toml"
    return tomllib.loads(path.read_text())


def variant_names() -> list[str]:
    root = resources.files("rogii") / "configs" / "structural"
    return sorted(
        p.name[:-5] for p in root.iterdir() if p.name.endswith(".toml") and not p.name.startswith("_")
    )


def load_variant(name: str) -> Variant:
    if name not in variant_names():
        raise ValueError(f"unknown structural variant {name!r}; choose from {variant_names()}")
    raw = _read(name)
    combiner = raw["combiner"]
    tables = {}
    if combiner in {"distance_bins", "ensemble_seq"}:
        tables["bins"] = _read("_distance_bins")
    if combiner == "md_bands":
        tables["bands"] = _read("_distance_md_bands")
    if combiner.startswith("ensemble"):
        tables["ensemble"] = _read("_ensemble")

    blend = raw.get("blend", {})
    blend = BlendRule(**{**blend, "schedule": tuple(tuple(x) for x in blend.get("schedule", ()))})
    dl = DLRule(**raw["dl"]) if "dl" in raw else None
    return Variant(
        name, combiner, bool(raw.get("sanitize", False)), blend, dl, raw.get("description", ""), tables
    )
