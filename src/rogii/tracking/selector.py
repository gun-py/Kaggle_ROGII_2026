from __future__ import annotations

import numpy as np

from rogii.config import SelectorConfig

from .bimodal import bimodal_hedge

N_EVAL_THRESHOLD = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
GLOBAL_VARIANT = "pf_scale_8_hold_0.2"


def well_code(hw):
    """Bucket a well by eval length and depth span; returns (code, variant name, n_eval, z_span)."""
    unknown = hw["TVT_input"].isna().to_numpy()
    n_eval = float(unknown.sum())
    z = hw.loc[unknown, "Z"].to_numpy(dtype=float)
    z_span = float(np.nanmax(z) - np.nanmin(z)) if len(z) else 0.0
    code = int(n_eval > N_EVAL_THRESHOLD) + 2 * int(np.searchsorted(Z_SPAN_THRESHOLDS, z_span, side="right"))
    return code, BIN_VARIANTS.get(code, GLOBAL_VARIANT), n_eval, z_span


def parse_variant(name: str):
    parts = name.split("_")
    scale = float(parts[2])
    beam = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return scale, beam, hold


def variant_grid() -> list[str]:
    """Every selector variant the visible-prefix calibration is allowed to try."""
    variants = {*BIN_VARIANTS.values(), GLOBAL_VARIANT}
    for scale in (3, 5, 8, 12):
        for hold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            variants.add(f"pf_scale_{scale:g}_hold_{hold:g}")
        for beam in (0.05, 0.10, 0.20, 0.30):
            for hold in (0.0, 0.05, 0.10, 0.15, 0.20):
                variants.add(f"pf_scale_{scale:g}_beam_{beam:g}_hold_{hold:g}")
    return sorted(variants)


def apply_variant(
    name, pf_by_scale, beam_path, last_known_tvt, hw, tw, cfg: SelectorConfig, return_info=False
):
    scale, beam_weight, hold_weight = parse_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        base = pf_by_scale[GLOBAL_VARIANT.split("_beam_")[0].split("_hold_")[0]]
    base = np.asarray(base, dtype=float)
    beam_path = np.asarray(beam_path, dtype=float)

    info = {
        "bimodal_active": False,
        "base_beam_weight": float(beam_weight),
        "effective_beam_weight": float(beam_weight),
        "delta_star": 0.0,
        "p_base": np.nan,
        "p_eff": np.nan,
        "prefix_trust": np.nan,
        "temperature": np.nan,
        "rho1": np.nan,
        "n_eff": np.nan,
        "heel_calibrated": False,
        "heel_rows": 0,
        "heel_alpha": np.nan,
        "heel_beta": np.nan,
        "heel_rmse_raw": np.nan,
        "heel_rmse_calibrated": np.nan,
        "heel_denoised": False,
    }

    hedge = bimodal_hedge(base, beam_path, hw, tw, cfg) if cfg.bimodal.enabled else None
    if hedge is not None:
        info.update(hedge)
        info["bimodal_active"] = True
        pred = base + float(hedge["delta_star"])
    else:
        pred = (1.0 - beam_weight) * base + beam_weight * beam_path
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return (pred, info) if return_info else pred
