import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

from rogii.features import builder, imputers, ncc, surfaces
from rogii.io import list_wells
from tests import synthetic


@pytest.fixture(scope="module")
def imps(competition):
    return imputers.Imputers(list_wells(competition, "train"), competition / "train")


def test_robust_slope_and_affine_cal():
    x = np.arange(100.0)
    assert surfaces.robust_slope(x, 2 * x + 1) == pytest.approx(2.0)
    assert surfaces.robust_slope(np.ones(10), np.arange(10.0)) == 0.0
    ref = np.sin(np.arange(200) / 10.0)
    a, b = surfaces.affine_cal(1.5 * ref + 4.0, ref)
    assert (a, b) == pytest.approx((1.5, 4.0))
    assert surfaces.affine_cal(np.ones(5), np.ones(5)) == (1.0, 0.0)


def test_segment_offsets_on_constant_offset():
    tvt, z = np.linspace(100, 200, 90), np.linspace(-50, -80, 90)
    surface = tvt + z - 7.0
    assert surfaces.segment_offsets(tvt, z, surface) == pytest.approx((7.0,) * 5)


def test_multi_scale_ncc_locates_a_repeated_pattern():
    t = np.arange(800, dtype=np.float32)
    pattern = gaussian_filter1d(np.random.default_rng(0).normal(size=800), 4).astype(np.float32)
    per_scale, ensemble = ncc.multi_scale_ncc(pattern, t, pattern[300:500])
    assert len(per_scale) == 3 and ensemble.shape == (200,)
    expected = 300 + np.arange(200)
    for tvt, score in per_scale[1:]:  # the 17-row window is too short to be unique
        assert np.median(np.abs(tvt - expected)) <= 2 and score.max() <= 1.0 + 1e-5
    short_tvt, short_scores = ncc.multi_scale_ncc(
        np.ones(5, np.float32), np.arange(5, dtype=np.float32), np.ones(3, np.float32)
    )[0][0]
    assert (short_scores == 0).all() and (short_tvt == 4).all()


def test_formation_plane_recovers_planar_surfaces(competition, imps):
    hw = pd.read_csv(competition / "train" / "tr000005__horizontal_well.csv")
    xy = hw[["X", "Y"]].to_numpy(float)[::100]
    pred, nearest = imps.formation.impute(xy, self_wid="tr000005")
    assert pred.shape == (len(xy), 6) and (nearest > 0).all()
    truth = hw[synthetic.FORMATIONS].to_numpy()[::100]
    # datum offsets differ per well, so only the shape (differences between formations) is comparable
    assert np.allclose(np.diff(pred, axis=1), np.diff(truth, axis=1), atol=1e-2)


def test_dense_imputer_excludes_the_query_well(competition, imps):
    hw = pd.read_csv(competition / "train" / "tr000001__horizontal_well.csv")
    xy = hw[["X", "Y"]].to_numpy(float)[::200]
    own, _, d_own = imps.dense.impute(xy, self_wid=None)
    other, _, d_other = imps.dense.impute(xy, self_wid="tr000001")
    assert (d_own <= d_other).all() and not np.array_equal(own, other)


def test_build_well_columns_and_target(competition, imps):
    train_path = competition / "train" / "tr000002__horizontal_well.csv"
    frame = builder.build_well(train_path, str(train_path).replace("horizontal_well", "typewell"), True, imps)
    n_hidden = int(pd.read_csv(train_path)["TVT_input"].isna().sum())
    assert len(frame) == n_hidden and "target" in frame.columns and frame["well"].eq("tr000002").all()
    assert frame["id"].iloc[0] == "tr000002_700"
    numeric = frame.drop(columns=["well", "id"])
    assert np.isfinite(numeric.to_numpy(float)[:, :-1]).all() or numeric.isna().mean().max() < 0.05
    assert {
        "pf_ancc",
        "beam_cons_d",
        "sc_ens_d",
        "tvtF_ANCC",
        "dense_ancc",
        "tdpf0",
        "grm101",
        "glead30",
    } <= set(frame.columns)
    assert frame.shape[1] == 198


def test_build_well_for_test_split_has_no_target(competition, imps):
    path = competition / "test" / "te000000__horizontal_well.csv"
    frame = builder.build_well(path, str(path).replace("horizontal_well", "typewell"), False, imps)
    assert frame is not None and "target" not in frame.columns and frame.shape[1] == 197


def test_build_well_rejects_short_or_complete_wells(competition, imps, tmp_path):
    hw = pd.read_csv(competition / "test" / "te000000__horizontal_well.csv")
    tw = pd.read_csv(competition / "test" / "te000000__typewell.csv")
    tw.to_csv(tmp_path / "w__typewell.csv", index=False)
    hw.assign(TVT_input=hw["MD"]).to_csv(tmp_path / "w__horizontal_well.csv", index=False)
    assert (
        builder.build_well(tmp_path / "w__horizontal_well.csv", tmp_path / "w__typewell.csv", False, imps)
        is None
    )
    hw.assign(TVT_input=np.nan).to_csv(tmp_path / "w__horizontal_well.csv", index=False)
    assert (
        builder.build_well(tmp_path / "w__horizontal_well.csv", tmp_path / "w__typewell.csv", False, imps)
        is None
    )
    assert builder.build_well(tmp_path / "missing.csv", tmp_path / "w__typewell.csv", False, imps) is None


def test_build_features_and_likpf_merge(competition, imps):
    wells = list_wells(competition, "test")
    frame = builder.build_features(competition, wells, "test", imps, n_jobs=2)
    assert set(frame["well"]) == set(wells)
    likpf = builder.build_likpf(competition, wells, "test", n_particles=60, n_seeds=4, n_jobs=2)
    merged = builder.add_likpf_features(frame, likpf)
    assert {"likpf_scale_5", "likpf_scale_5_d", "likpf_mean_d"} <= set(merged.columns) and len(merged) == len(
        frame
    )
    assert merged["likpf_scale_5"].notna().all()
