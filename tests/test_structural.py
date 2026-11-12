import numpy as np
import pandas as pd
import pytest

from rogii.structural import blend as sblend
from rogii.structural import components as comp
from rogii.structural import field as sfield
from rogii.structural import hmm, variants
from rogii.structural.wells import load_wells
from tests import synthetic


@pytest.fixture(scope="module")
def pool(competition):
    return load_wells(competition, "train")


@pytest.fixture(scope="module")
def fld(pool):
    return sfield.Field(pool)


@pytest.fixture(scope="module")
def test_well(competition):
    return load_wells(competition, "test", ["te000000"])["te000000"]


def _truth(competition, wid="te000000"):
    return pd.read_csv(competition.parent / f"{competition.name}_truth_{wid}.csv")["TVT"].to_numpy()


def test_delta_s_matches_the_structural_plane(fld, test_well, competition):
    truth = _truth(competition)
    ia = int(np.where(test_well["known"])[0][-1])
    med, nearest = fld.delta_s(np.c_[test_well["x"], test_well["y"]], ia, drop={"te000000"})
    unknown = ~test_well["known"]
    implied = test_well["tvt"][ia] + (med - (test_well["z"] - test_well["z"][ia]))
    err_neighbour = np.sqrt(np.nanmean((implied[unknown] - truth[unknown]) ** 2))
    carry = np.sqrt(np.mean((test_well["tvt"][ia] - truth[unknown]) ** 2))
    assert nearest < sfield.NB_CUTOFF and err_neighbour < carry


def test_delta_s_without_neighbours(fld, test_well):
    far = np.c_[test_well["x"] + 1e7, test_well["y"] + 1e7]
    med, nearest = fld.delta_s(far, 10, drop=set())
    assert np.isnan(med).all() and nearest == np.inf


def test_weighted_delta_s_and_plane_are_finite(fld, test_well):
    xy = np.c_[test_well["x"], test_well["y"]]
    ia = int(np.where(test_well["known"])[0][-1])
    weighted = fld.delta_s_weighted(xy, ia, drop={"te000000"})
    plane = sfield.plane_delta_s(fld, test_well, drop={"te000000"})
    assert np.isfinite(weighted).any() and np.isfinite(plane).all()
    assert abs(plane[ia]) < 1e-6 and np.abs(plane).max() <= 400.0


def test_hmm_recovers_planted_drift(test_well, competition):
    truth = _truth(competition)
    unknown = np.where(~test_well["known"])[0]
    ia = unknown[0] - 1
    prior = np.full(len(truth), np.nan)
    prior[unknown] = (truth[unknown] + test_well["z"][unknown]) - (truth[ia] + test_well["z"][ia])
    path = hmm.hmm_align(test_well, prior, sigma_gr=6.0, step_sd=0.15, prior_sd=25.0)
    carry = np.sqrt(np.mean((truth[ia] - truth[unknown]) ** 2))
    assert np.sqrt(np.mean((path[unknown] - truth[unknown]) ** 2)) < 0.6 * carry


def test_gr_calibration_defaults_without_data():
    assert hmm.gr_calibration(np.ones(10), np.ones(10), np.arange(5.0), np.arange(5.0)) == (1.0, 0.0)


def test_components_shape_and_sanitize(fld, test_well):
    P, nearest = comp.components(fld, test_well, {"te000000"}, sanitize=True)
    assert P.shape == (len(test_well["md"]), comp.N_COMPONENTS) and np.isfinite(P).all()
    ia = int(np.where(test_well["known"])[0][-1])
    assert np.allclose(P[:, -1], test_well["tvt"][ia])
    assert np.isfinite(nearest)


def test_every_variant_loads_and_predicts(fld, test_well, competition):
    truth = _truth(competition)
    unknown = ~test_well["known"]
    ia = int(np.where(test_well["known"])[0][-1])
    carry = np.sqrt(np.mean((test_well["tvt"][ia] - truth[unknown]) ** 2))
    for name in variants.variant_names():
        if name == "cv3":
            pytest.importorskip("torch")
        variant = variants.load_variant(name)
        path, nearest = variant.predict(fld, test_well, {"te000000"})
        assert path.shape == test_well["md"].shape and np.isfinite(path[unknown]).all(), name
        assert np.isfinite(nearest)
        assert np.sqrt(np.mean((path[unknown] - truth[unknown]) ** 2)) < 3 * carry, name


def test_variant_tables_are_consistent():
    bins = variants.load_variant("v2").tables["bins"]
    assert np.asarray(bins["weights"]).shape == (3, 11) and np.allclose(
        np.sum(bins["weights"], axis=1), 1.0, atol=1e-3
    )
    bands = variants.load_variant("v5").tables["bands"]
    assert np.asarray(bands["weights"]).shape == (3, 4, 11)
    assert np.allclose(np.sum(bands["weights"], axis=2), 1.0, atol=1e-3)
    ens = variants.load_variant("cv2").tables["ensemble"]
    assert len(ens["en"]) == 36 and len(ens["pls"]) == 14 and np.asarray(ens["refit"]).shape == (3, 11)
    with pytest.raises(ValueError):
        variants.load_variant("nope")


def test_variant_blend_rules():
    assert variants.load_variant("v5").blend == variants.BlendRule(
        mode="fixed", stack=0.57, struct=0.43, package=0.0
    )
    assert variants.load_variant("cv4").blend.mode == "gated"
    assert variants.load_variant("v4").blend.package == 0.5
    assert variants.load_variant("v2_dl").dl.weight == 0.0


def test_gated_weight_schedule():
    rule = variants.load_variant("cv4").blend
    assert [sblend.gated_weight(rule, nd) for nd in (100.0, 500.0, 900.0, 5000.0, np.inf)] == [
        0.40,
        0.45,
        0.30,
        0.30,
        0.30,
    ]


def test_blend_normalises_and_falls_back():
    index = pd.Index(["a", "b", "c"])
    existing = pd.Series([10.0, 10.0, 10.0], index=index)
    struct = pd.Series([20.0, np.nan, 20.0], index=index)
    rule = variants.BlendRule(stack=0.6, struct=0.4)
    out = sblend.blend(existing, struct, pd.Series(np.nan, index=index), rule)
    assert out.tolist() == pytest.approx([14.0, 10.0, 14.0])
    package = pd.Series([30.0, 30.0, 30.0], index=index)
    three = sblend.blend(
        existing, struct, None, variants.BlendRule(stack=0.3, struct=0.2, package=0.5), package
    )
    assert three.tolist() == pytest.approx([22.0, 10.0, 22.0])
    no_package = sblend.blend(
        existing, struct, None, variants.BlendRule(stack=0.3, struct=0.2, package=0.5), None
    )
    assert no_package.tolist() == pytest.approx([0.6 * 10 + 0.4 * 20, 10.0, 14.0])
    gated = sblend.blend(
        existing, struct, pd.Series([0.5, 0.5, np.nan], index=index), variants.load_variant("cv4").blend
    )
    assert gated.tolist() == pytest.approx([15.0, 10.0, 10.0])


def test_struct_predictions_cover_every_unknown_row(competition):
    sample = pd.read_csv(competition / "sample_submission.csv")
    struct, weights = sblend.struct_predictions(
        variants.load_variant("v5"), competition, pd.Index(sample["id"])
    )
    assert struct.notna().all() and weights.isna().all()
    gated_struct, gated_weights = sblend.struct_predictions(
        variants.load_variant("cv4"), competition, pd.Index(sample["id"])
    )
    assert gated_weights.between(0.3, 0.45).all() and np.allclose(
        gated_struct.to_numpy(), gated_struct.to_numpy()
    )


def test_synthetic_neighbours_are_close_enough():
    hw, _ = synthetic.make_well("x", 1000, 500)
    assert len(hw) == 1400
