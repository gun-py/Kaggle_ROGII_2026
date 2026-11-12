from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from importlib import resources

CONTACT_REFS = ("EGFDU", "ASTNU", "ANCC", "ASTNL", "EGFDL", "BUDA")


@dataclass(frozen=True)
class HeelConfig:
    enabled: bool = True
    fft_denoise: bool = False
    min_rows: int = 40
    alpha_min: float = 0.25
    alpha_max: float = 4.0
    beta_abs_max: float = 500.0


@dataclass(frozen=True)
class BimodalConfig:
    enabled: bool = False
    dz_range: float = 20.0
    dz_step: float = 0.5
    bundle_min: float = 10.0
    bundle_max: float = 20.0
    j_ratio_eps: float = 0.15
    scan_min_sep: float = 8.0
    temp: float = 0.75
    adaptive_temp: bool = True
    temp_min: float = 0.25
    temp_max: float = 4.0
    prefix_trust_gate: bool = True
    trust_min_prefix_rows: int = 60
    trust_fallback: float = 0.0
    trust_kappa: float = 1.0
    force_midpoint: bool = False
    trigger_min_median_diff: float = 12.0
    trigger_min_p90_diff: float = 18.0
    trigger_min_big_diff_frac: float = 0.18
    trigger_big_diff: float = 15.0
    min_valid_gr_rows: int = 80


@dataclass(frozen=True)
class SelectorConfig:
    n_seeds: int = 128
    n_particles: int = 500
    scales: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0)
    return_std: bool = False
    heel: HeelConfig = field(default_factory=HeelConfig)
    bimodal: BimodalConfig = field(default_factory=BimodalConfig)


@dataclass(frozen=True)
class OverrideConfig:
    enabled: bool = True
    primary_ref: str = "EGFDU"
    ref_cols: tuple[str, ...] = CONTACT_REFS
    min_valid_rows: int = 100
    min_known_rows: int = 50
    prefix_rmse_limit: float = 1.0


@dataclass(frozen=True)
class VisiblePrefixConfig:
    enabled: bool = True
    profile: str = "balanced"
    final_selection: str = "self_verified_anchor"
    include_pf: bool = True
    cal_seeds: int = 24
    final_seeds: int = 48
    particles: int = 350
    cut_fracs: tuple[float, ...] = (0.50, 0.65, 0.75)
    max_wells: int = 1_000_000
    bimodal_guard: bool = False
    guard_requires_low_trust: bool = False
    low_trust_threshold: float = 0.25


@dataclass(frozen=True)
class PackageConfig:
    enabled: bool = False
    max_weight: float = 0.01
    scale: float = 6.0
    candidates: tuple[float, ...] = (0.005, 0.010, 0.0125, 0.015, 0.020)
    diff_p95_disable: float | None = 25.0
    require: bool = False
    clip_min: float | None = None
    clip_max: float | None = None


@dataclass(frozen=True)
class HedgeConfig:
    strength: float = 0.60
    min_mass: float = 0.25
    sep_low: float = 4.0
    sep_high: float = 40.0
    cap: float = 2.0
    skip_existing: bool = False


@dataclass(frozen=True)
class PostprocessConfig:
    alpha: float = 1.0
    tau: float = 85.0
    pf_weight: float = 0.0
    learned_weight: float = 0.60
    lik_scale: str = "scale_5"
    sg_window: int = 61
    sg_poly: int = 3


@dataclass(frozen=True)
class PipelineConfig:
    profile: str = "default"
    ridge_weight: float = 0.30
    selector_weight: float = 0.70
    projection_degree: int = 3
    projection_blend: float = 0.75
    sp45_weight: float = 0.60
    sp45_candidates: tuple[float, ...] = (0.50, 0.52, 0.55, 0.58, 0.60)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    override: OverrideConfig = field(default_factory=OverrideConfig)
    visible_prefix: VisiblePrefixConfig = field(default_factory=VisiblePrefixConfig)
    package: PackageConfig = field(default_factory=PackageConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    learned: PostprocessConfig = field(default_factory=PostprocessConfig)


def _tuple(value):
    return tuple(value) if isinstance(value, list) else value


def _override(obj, table: dict):
    names = {f.name for f in fields(obj)}
    unknown = set(table) - names
    if unknown:
        raise KeyError(f"unknown {type(obj).__name__} keys: {sorted(unknown)}")
    updates = {}
    for key, value in table.items():
        current = getattr(obj, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            updates[key] = _override(current, value)
        else:
            updates[key] = _tuple(value)
    return replace(obj, **updates)


def profile_names() -> list[str]:
    root = resources.files("rogii") / "configs" / "profiles"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".toml"))


def load_profile(name: str) -> PipelineConfig:
    path = resources.files("rogii") / "configs" / "profiles" / f"{name}.toml"
    if not path.is_file():
        raise ValueError(f"unknown profile {name!r}; choose from {profile_names()}")
    table = tomllib.loads(path.read_text())
    cfg = _override(PipelineConfig(profile=name), table)
    return replace(cfg, sp45_candidates=tuple(sorted({*cfg.sp45_candidates, cfg.sp45_weight})))
