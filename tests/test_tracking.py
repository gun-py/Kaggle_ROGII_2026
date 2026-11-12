import numpy as np
import pandas as pd
import pytest

from rogii.config import BimodalConfig, SelectorConfig
from rogii.tracking import beam, bimodal, selector, trackers
from rogii.tracking.particle_filter import ensemble_by_scale, run_particle_filter, seed_branches


def _rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def _hidden(hw):
    return hw["TVT_input"].isna().to_numpy()


def test_particle_filter_is_seeded_and_beats_carry(well):
    hw, tw = well
    a, ll_a = run_particle_filter(hw, tw, n_particles=150, seed=1)
    b, ll_b = run_particle_filter(hw, tw, n_particles=150, seed=1)
    c, _ = run_particle_filter(hw, tw, n_particles=150, seed=2)
    assert np.array_equal(a, b) and ll_a == ll_b and not np.array_equal(a, c)
    mask = _hidden(hw)
    carry = np.full(mask.sum(), hw["TVT_input"].dropna().iloc[-1])
    assert _rmse(a[mask], hw["TVT"][mask]) < _rmse(carry, hw["TVT"][mask])
    assert np.array_equal(a[~mask], hw["TVT_input"].to_numpy()[~mask])


def test_particle_filter_without_hidden_rows(well):
    hw, tw = well
    full = hw.assign(TVT_input=hw["TVT"])
    out, ll = run_particle_filter(full, tw, n_particles=20)
    assert ll == 0.0 and np.array_equal(out, full["TVT"].to_numpy())


def test_ensemble_outputs_every_scale(well):
    hw, tw = well
    stats = {}
    out = ensemble_by_scale(hw, tw, n_particles=60, n_seeds=5, branch_stats=stats, return_std=True)
    assert set(out) == {"pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12", "pf_mean", "pf_seed_std"}
    assert all(len(v) == len(hw) for v in out.values())
    assert {"center_low", "center_high", "mass_low", "mass_high", "eval_rows"} <= set(stats)
    assert stats["mass_low"] + stats["mass_high"] == pytest.approx(1.0)


def test_seed_branches_splits_two_clusters():
    rng = np.random.default_rng(0)
    levels = np.r_[rng.normal(0, 0.2, 12), rng.normal(30, 0.2, 12)]
    preds = np.repeat(levels[:, None], 40, 1)
    found = seed_branches(preds, np.zeros(24), np.ones(40, dtype=bool))
    assert found["center_high"] - found["center_low"] == pytest.approx(30, abs=1.0)
    assert found["mass_low"] == pytest.approx(0.5, abs=0.05)
    assert seed_branches(preds, np.zeros(24), np.zeros(40, dtype=bool)) is None


def test_beam_ensemble_tracks_the_truth(well):
    hw, tw = well
    out = beam.beam_ensemble(hw, tw)
    mask = _hidden(hw)
    carry = np.full(mask.sum(), hw["TVT_input"].dropna().iloc[-1])
    assert np.isfinite(out).all() and np.array_equal(out[~mask], hw["TVT_input"].to_numpy()[~mask])
    assert _rmse(out[mask], hw["TVT"][mask]) < _rmse(carry, hw["TVT"][mask])


def test_jit_beam_returns_typewell_values(well):
    hw, tw = well
    tw = tw.sort_values("TVT")
    path = beam.jit_beam_search(
        hw["GR"].to_numpy()[700:900], tw.TVT.to_numpy(), tw.GR.to_numpy(), 11600.0, 10, 20.0, 144.0, 2
    )
    assert path.dtype == np.float32 and len(path) == 200 and np.isin(path, tw.TVT.to_numpy(np.float32)).all()


def test_selector_variant_parsing_and_grid():
    assert selector.parse_variant("pf_scale_5_hold_0.2") == (5.0, 0.0, 0.2)
    assert selector.parse_variant("pf_scale_12_beam_0.2_hold_0.15") == (12.0, 0.2, 0.15)
    grid = selector.variant_grid()
    assert len(grid) == len(set(grid)) > 100
    assert set(selector.BIN_VARIANTS.values()) | {selector.GLOBAL_VARIANT} <= set(grid)


def test_well_code_buckets(well):
    hw, _ = well
    code, name, n_eval, z_span = selector.well_code(hw)
    assert name == selector.BIN_VARIANTS[code] and n_eval == _hidden(hw).sum() and z_span > 0
    long_flat = hw.assign(Z=0.0)
    assert selector.well_code(long_flat)[0] in (0, 1)


def test_apply_variant_mixes_beam_and_hold(well):
    hw, tw = well
    n = len(hw)
    pf = {f"pf_scale_{s:g}": np.full(n, 100.0) for s in (3, 5, 8, 12)}
    beam_path = np.full(n, 200.0)
    cfg = SelectorConfig()
    out = selector.apply_variant("pf_scale_5_beam_0.5_hold_0.25", pf, beam_path, 400.0, hw, tw, cfg)
    assert np.allclose(out, 0.75 * (0.5 * 100 + 0.5 * 200) + 0.25 * 400)
    plain, info = selector.apply_variant(
        "pf_scale_5_hold_0.0", pf, beam_path, 400.0, hw, tw, cfg, return_info=True
    )
    assert np.allclose(plain, 100.0) and info["bimodal_active"] is False


def test_bimodal_scan_finds_planted_shift(well):
    hw, tw = well
    shifted = hw["TVT"].to_numpy() + 8.0
    scan = bimodal.bimodal_scan(hw, tw, shifted, SelectorConfig(), mask=_hidden(hw))
    assert abs(scan["delta_a"] + 8.0) <= 1.5
    assert 0.0 <= scan["p_eff"] <= 1.0 and scan["valid_gr_rows"] > 80


def test_bimodal_hedge_needs_disagreement(well):
    hw, tw = well
    cfg = SelectorConfig(bimodal=BimodalConfig(enabled=True))
    truth = hw["TVT"].to_numpy()
    assert bimodal.bimodal_hedge(truth, truth + 1.0, hw, tw, cfg) is None
    assert bimodal.bimodal_hedge(truth[:0], truth[:0], hw, tw, cfg) is None


def test_prefix_trust_and_misfit(well):
    hw, tw = well
    cfg = SelectorConfig()
    truth = hw["TVT"].to_numpy()
    prefix = hw["TVT_input"].notna().to_numpy()
    good, rows = bimodal.gr_misfit(hw, tw, truth, cfg, mask=prefix)
    bad, _ = bimodal.gr_misfit(hw, tw, truth + 25.0, cfg, mask=prefix)
    assert rows > 80 and good < bad
    trust, j0, j1, margin, n = bimodal.prefix_trust(hw, tw, truth, 25.0, cfg)
    assert trust == 1.0 and j1 > j0 and margin > 0 and n == rows
    off = SelectorConfig(bimodal=BimodalConfig(prefix_trust_gate=False))
    assert bimodal.prefix_trust(hw, tw, truth, 25.0, off)[0] == 1.0


def test_lag1_autocorr_and_temperature():
    ramp = np.cumsum(np.random.default_rng(0).normal(size=500))
    assert bimodal.lag1_autocorr(ramp) > 0.9
    assert bimodal.lag1_autocorr(np.array([1.0, 2.0])) == 0.0
    cfg = SelectorConfig()
    temp, rho, n_eff = bimodal.softmax_temperature(ramp, 500, cfg)
    assert cfg.bimodal.temp_min <= temp <= cfg.bimodal.temp_max and rho > 0.9 and n_eff < 100


def test_compiled_trackers_follow_the_well(well):
    hw, tw = well
    tw = tw.sort_values("TVT")
    tvt, gr = tw.TVT.to_numpy(np.float32), tw.GR.to_numpy(np.float32)
    mask = _hidden(hw)
    carry = _rmse(np.full(mask.sum(), hw["TVT_input"].dropna().iloc[-1]), hw["TVT"][mask])
    anchored, spread = trackers.run_pf_ancc(hw, tvt, gr)
    assert len(anchored) == mask.sum() and (spread >= 0).all() and _rmse(anchored, hw["TVT"][mask]) < carry
    z_path, _ = trackers.run_pf_z(hw, tvt, gr)
    assert len(z_path) == mask.sum() and np.isfinite(z_path).all()
    out, idx, quality = trackers.lik_pf(hw, tw, n_particles=80, n_seeds=6, with_quality=True)
    assert list(idx) == list(np.where(mask)[0]) and set(quality) == {
        "pf_best_ll",
        "pf_ll_spread",
        "pf_pt_std",
        "pf_gr_sig",
    }
    again, _, _ = trackers.lik_pf(hw, tw, n_particles=80, n_seeds=6)
    assert all(np.array_equal(out[k], again[k]) for k in out)


def test_trackers_handle_fully_known_well(well):
    hw, tw = well
    tw = tw.sort_values("TVT")
    full = hw.assign(TVT_input=hw["TVT"])
    assert len(trackers.run_pf_ancc(full, tw.TVT.to_numpy(np.float32), tw.GR.to_numpy(np.float32))[0]) == 0
    assert trackers.lik_pf(full, tw)[0] == {}
    assert pd.notna(trackers.gr_sigma(full, tw.TVT.to_numpy(), tw.GR.to_numpy()))
