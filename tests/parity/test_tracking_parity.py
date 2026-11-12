"""Tracking, selector and feature code against the notebook definitions, bit for bit."""

import numpy as np
import pytest

from rogii.config import BimodalConfig, SelectorConfig
from rogii.features import builder
from rogii.features.imputers import Imputers
from rogii.features.ncc import multi_scale_ncc
from rogii.io import list_wells, read_well
from rogii.tracking import beam, bimodal, selector, trackers
from rogii.tracking import particle_filter as pf
from tests.parity import legacy

WELLS = ["00bbac68", "000d7d20", "00e12e8b"]
SELECTOR_CONSTANTS = {
    "SELECTOR_N_EVAL_THRESHOLD", "SELECTOR_Z_SPAN_THRESHOLDS", "SELECTOR_BIN_VARIANTS", "SELECTOR_GLOBAL_VARIANT",
    "SELECTOR_SCALES", "FORMATION_COLS", "BEAM_CONFIGS",
}  # fmt: skip
SELECTOR_FLAGS = dict(
    RUN_GR_FFT_DENOISE=False, RUN_HEEL_CALIBRATION=True, HEEL_MIN_ROWS=40, HEEL_ALPHA_MIN=0.25, HEEL_ALPHA_MAX=4.0,
    HEEL_BETA_ABS_MAX=500.0, BIMODAL_MIN_VALID_GR_ROWS=80, BIMODAL_DZ_RANGE=20.0, BIMODAL_DZ_STEP=0.5, BUNDLE_MIN=10.0,
    BUNDLE_MAX=20.0, BIMODAL_J_RATIO_EPS=0.15, SCAN_MIN_SEP=8.0, BIMODAL_TEMP=0.75, RUN_ADAPTIVE_TEMP=True, T_MIN=0.25,
    T_MAX=4.0, RUN_PREFIX_TRUST_GATE=True, TRUST_MIN_PREFIX_ROWS=60, TRUST_FALLBACK=0.0, TRUST_KAPPA=1.0,
    BIMODAL_FORCE_MIDPOINT=False, BIMODAL_TRIGGER_MIN_MEDIAN_DIFF=12.0, BIMODAL_TRIGGER_MIN_P90_DIFF=18.0,
    BIMODAL_TRIGGER_MIN_BIG_DIFF_FRAC=0.18, BIMODAL_TRIGGER_BIG_DIFF_THRESHOLD=15.0, SELECTOR_PF_RETURN_STD=False,
)  # fmt: skip
FORMATION_AND_FEATURE_CONSTANTS = {
    "FORMATIONS", "PLANE_K", "DENSE_SPW", "DENSE_K", "N_SPLITS", "BEAMS", "PF_N", "ANCC_N", "PF_MOM", "PF_VN", "PF_PN",
    "PF_GR_SIG_MIN", "PF_GR_SIG_MAX", "PF_GR_SIG_DEF",
}  # fmt: skip


def same(a, b):
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


@pytest.fixture(scope="module")
def selector_ns():
    """Cell 11 followed by cell 14: the notebook's compiled beam_search shadows the python one before first use."""
    ns = {}
    legacy.load(11, constants=SELECTOR_CONSTANTS, namespace=ns)
    legacy.load(14, constants=FORMATION_AND_FEATURE_CONSTANTS, namespace=ns)
    ns.update(SELECTOR_FLAGS, RUN_BIMODAL_DETECTOR=False)
    return ns


@pytest.mark.parametrize("wid", WELLS)
def test_particle_filter_and_ensemble(selector_ns, data, wid):
    hw, tw = read_well(data, wid, "test")
    a, la = selector_ns["run_particle_filter"](hw, tw, 100, 3)
    b, lb = pf.run_particle_filter(hw, tw, 100, 3)
    assert np.array_equal(a, b) and la == lb

    old_stats, new_stats = {}, {}
    old = selector_ns["run_pf_lik_ensemble_scales"](hw, tw, n_particles=60, n_seeds=6, branch_stats=old_stats)
    new = pf.ensemble_by_scale(hw, tw, n_particles=60, n_seeds=6, branch_stats=new_stats)
    assert set(old) == set(new) and all(np.array_equal(old[k], new[k]) for k in old)
    assert old_stats == new_stats


@pytest.mark.parametrize("wid", WELLS)
def test_beam_ensemble_uses_the_live_definition(selector_ns, data, wid):
    hw, tw = read_well(data, wid, "test")
    assert np.array_equal(selector_ns["run_beam_ensemble"](hw, tw), beam.beam_ensemble(hw, tw))


@pytest.mark.parametrize("bimodal_on", [False, True])
@pytest.mark.parametrize("wid", WELLS)
def test_selector_variants(selector_ns, data, wid, bimodal_on):
    selector_ns["RUN_BIMODAL_DETECTOR"] = bimodal_on
    cfg = SelectorConfig(bimodal=BimodalConfig(enabled=bimodal_on))
    hw, tw = read_well(data, wid, "test")
    pf_by_scale = pf.ensemble_by_scale(hw, tw, n_particles=60, n_seeds=6)
    beam_path = beam.beam_ensemble(hw, tw)
    last = float(hw.TVT_input.dropna().iloc[-1])
    assert selector.well_code(hw) == selector_ns["selector_well_code"](hw)
    for name in (
        selector.well_code(hw)[1],
        "pf_scale_12_beam_0.2_hold_0.15",
        "pf_scale_5_beam_0.05_hold_0.05",
    ):
        for shift in (0.0, 18.0):
            old, old_info = selector_ns["apply_selector_variant"](
                name, pf_by_scale, beam_path + shift, last, hw=hw, tw=tw, return_info=True
            )
            new, new_info = selector.apply_variant(
                name, pf_by_scale, beam_path + shift, last, hw, tw, cfg, return_info=True
            )
            assert np.array_equal(old, new) and old_info.keys() == new_info.keys()
            assert all(same(old_info[k], new_info[k]) for k in old_info)


@pytest.mark.parametrize("shift", [0.0, 12.0, -9.0])
@pytest.mark.parametrize("wid", WELLS)
def test_bimodal_scan(selector_ns, data, wid, shift):
    hw, tw = read_well(data, wid, "test")
    base = beam.beam_ensemble(hw, tw) + shift
    old = selector_ns["selector_bimodal_scan"](hw, tw, base)
    new = bimodal.bimodal_scan(hw, tw, base, SelectorConfig())
    assert old.keys() == new.keys() and all(same(old[k], new[k]) for k in old)


@pytest.fixture(scope="module")
def feature_ns(data):
    ns = {"FORMATIONS": ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]}
    legacy.load(36, constants={"PF_N", "ANCC_N", "PF_MOM", "PF_VN", "PF_PN", "PF_GR_SIG_MIN", "PF_GR_SIG_MAX", "PF_GR_SIG_DEF", "PF_GR_WIN",
                               "PF_GR_WT", "PF_RESAMP", "PF_ROUGH_P", "PF_ROUGH_V", "ANCC_ALPHA", "ANCC_RN", "ANCC_PN", "ANCC_IS", "ANCC_RP",
                               "ANCC_RR", "BEAMS"}, namespace=ns)  # fmt: skip
    ns["CFG"] = type(
        "CFG", (), dict(DATA=data, PF_PARTICLES=500, PF_SEEDS=128, PF_SCALES=(3.0, 5.0, 8.0, 12.0))
    )
    legacy.load(37, namespace=ns)
    legacy.load(
        39,
        constants={"PLANE_K", "DENSE_SPW", "DENSE_K", "ANCH_OFFS", "BEAM_OFFS", "SC_OFFS", "PF_OFFS"},
        namespace=ns,
    )
    legacy.load(40, namespace=ns)
    return ns


@pytest.mark.parametrize("wid, split", [("00bbac68", "test"), ("000d7d20", "train")])
def test_compiled_trackers(feature_ns, data, wid, split):
    hw, tw = read_well(data, wid, split, sort_typewell=True)
    tvt, gr = tw.TVT.to_numpy(np.float32), tw.GR.to_numpy(np.float32)
    legacy.seed_numba(5)
    old, old_spread = feature_ns["run_pf_ancc"](hw, tvt, gr)
    legacy.seed_numba(5)
    new, new_spread = trackers.run_pf_ancc(hw, tvt, gr)
    assert np.array_equal(old, new) and np.array_equal(old_spread, new_spread)
    legacy.seed_numba(5)
    old_z, _ = feature_ns["run_pf_z"](hw, tvt, gr)
    legacy.seed_numba(5)
    new_z, _ = trackers.run_pf_z(hw, tvt, gr)
    assert np.array_equal(old_z, new_z)
    old_out, old_idx, old_q = feature_ns["lik_pf"](hw, tw, n_seeds=8, n_particles=100, with_quality=True)
    new_out, new_idx, new_q = trackers.lik_pf(hw, tw, n_seeds=8, n_particles=100, with_quality=True)
    assert all(np.array_equal(old_out[k], new_out[k]) for k in old_out) and np.array_equal(old_idx, new_idx)
    assert old_q["pf_best_ll"] == new_q["pf_best_ll"]
    g = hw.GR.fillna(40).to_numpy(np.float32)
    t = hw.TVT_input.fillna(0).to_numpy(np.float32)
    old_ncc = feature_ns["multi_scale_ncc"](g[:600], t[:600], g[600:1200])
    new_ncc = multi_scale_ncc(g[:600], t[:600], g[600:1200])
    assert np.array_equal(old_ncc[1], new_ncc[1])
    assert all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(old_ncc[0], new_ncc[0])
    )


@pytest.mark.parametrize("wid, split, is_train", [("00bbac68", "test", False), ("000d7d20", "train", True)])
def test_feature_builder(feature_ns, data, wid, split, is_train):
    train_ids = sorted(set(list_wells(data, "train")[:60] + ["000d7d20", "00bbac68", "00e12e8b"]))
    imputers = Imputers(train_ids, data / "train")
    feature_ns["init_imputers"](train_ids)
    hw_path, tw_path = data / split / f"{wid}__horizontal_well.csv", data / split / f"{wid}__typewell.csv"
    legacy.seed_numba(11)
    old = feature_ns["build_well"](str(hw_path), str(tw_path), is_train)
    legacy.seed_numba(11)
    new = builder.build_well(str(hw_path), str(tw_path), is_train, imputers)
    assert list(old.columns) == list(new.columns) and old.shape == new.shape
    for col in old.columns:
        assert old[col].dtype == new[col].dtype, col
        if old[col].dtype.kind == "f":
            assert np.array_equal(old[col].to_numpy(), new[col].to_numpy(), equal_nan=True), col
        else:
            assert (old[col] == new[col]).all(), col
