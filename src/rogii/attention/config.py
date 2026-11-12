from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from importlib import resources

from rogii.config import _override


@dataclass(frozen=True)
class TrailConfig:
    d_model: int = 96
    heads: int = 4
    encoder_layers: int = 3
    typewell_layers: int = 0
    update_layers: int = 2
    iterations: int = 3
    window: int = 48
    dropout: float = 0.1
    max_visible_tokens: int = 48
    use_typewell: bool = True
    use_neighbours: bool = True
    offset_bias: bool = True
    integrate: bool = False

    lr: float = 3e-4
    weight_decay: float = 0.01
    token_budget: int = 6000
    gamma: float = 0.8
    ce_weight: float = 0.1
    smooth_weight: float = 1e-3
    pretrain_steps: int = 900
    finetune_epochs: int = 15
    patience: int = 4
    real_anchor_prob: float = 0.5
    val_fraction: float = 0.1
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


ABLATIONS = {
    "full": {},
    "no_neighbours": {"use_neighbours": False},
    "no_typewell": {"use_typewell": False},
    "single_pass": {"iterations": 1},
    "no_offset_bias": {"offset_bias": False},
    "integrated": {"integrate": True},
}


def load_config(name: str = "trail", **overrides) -> TrailConfig:
    path = resources.files("rogii") / "configs" / "attention" / f"{name}.toml"
    if not path.is_file():
        raise ValueError(f"unknown attention config {name!r}")
    cfg = _override(TrailConfig(), tomllib.loads(path.read_text()))
    return replace(cfg, **overrides) if overrides else cfg


def ablated(cfg: TrailConfig, ablation: str) -> TrailConfig:
    if ablation == "no_pretrain":
        return cfg
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}; choose from {sorted([*ABLATIONS, 'no_pretrain'])}")
    return replace(cfg, **ABLATIONS[ablation])
