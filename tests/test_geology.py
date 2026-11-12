import numpy as np
import pandas as pd

from rogii.config import HeelConfig
from rogii.geology import calibration, contacts
from tests import synthetic


def test_contact_path_reproduces_train_tvt(well):
    hw, tw = well
    for ref in synthetic.FORMATIONS:
        path = contacts.contact_tvt(hw, tw, ref)
        assert np.allclose(path, hw["TVT"], atol=1e-6)
    assert np.allclose(contacts.tvt_from_contacts(hw, tw).to_numpy(), hw["TVT"], atol=1e-6)


def test_contact_path_missing_reference(well):
    hw, tw = well
    assert contacts.contact_tvt(hw, tw, "NOPE") is None
    assert contacts.contact_tvt(hw.drop(columns=["EGFDU"]), tw, "EGFDU") is None
    assert contacts.contact_tvt(hw, tw.drop(columns=["Geology"]), "EGFDU") is None


def test_md_lookup_sorts_and_filters():
    md = np.array([3.0, 1.0, 2.0, np.nan])
    tvt = np.array([30.0, 10.0, 20.0, 40.0])
    out = contacts.md_lookup(md, tvt, 3)
    assert out is not None and out[0].tolist() == [1.0, 2.0, 3.0] and out[1].tolist() == [10.0, 20.0, 30.0]
    assert contacts.md_lookup(md, tvt, 4) is None


def test_typewell_arrays_sorted_and_finite(well):
    _, tw = well
    shuffled = tw.sample(frac=1.0, random_state=0)
    tvt, gr = calibration.typewell_arrays(shuffled)
    assert np.all(np.diff(tvt) > 0) and np.isfinite(gr).all()
    assert calibration.typewell_arrays(tw.head(5)) == (None, None)
    assert calibration.typewell_arrays(None) == (None, None)


def test_heel_calibration_recovers_gain_and_offset(well):
    hw, tw = well
    tw_tvt, tw_gr = calibration.typewell_arrays(tw)
    gr = hw["GR"].to_numpy()  # generated as 1.1 * typewell + 3
    calibrated, info = calibration.heel_calibrate(hw, tw_tvt, tw_gr, gr, HeelConfig())
    assert info["heel_calibrated"]
    assert info["heel_alpha"] == np.float64(info["heel_alpha"]) and abs(info["heel_alpha"] - 1.1) < 0.02
    assert abs(info["heel_beta"] - 3.0) < 1.5
    assert info["heel_rmse_calibrated"] < info["heel_rmse_raw"]
    assert np.nanmean(np.abs(calibrated - np.interp(hw["TVT"], tw_tvt, tw_gr))) < 3.0


def test_heel_calibration_disabled_is_identity(well):
    hw, tw = well
    tw_tvt, tw_gr = calibration.typewell_arrays(tw)
    out, info = calibration.heel_calibrate(hw, tw_tvt, tw_gr, hw["GR"].to_numpy(), HeelConfig(enabled=False))
    assert np.array_equal(out, hw["GR"].to_numpy()) and not info["heel_calibrated"]


def test_heel_calibration_needs_a_prefix(well):
    hw, tw = well
    tw_tvt, tw_gr = calibration.typewell_arrays(tw)
    short = hw.assign(TVT_input=np.nan)
    _, info = calibration.heel_calibrate(short, tw_tvt, tw_gr, short["GR"].to_numpy(), HeelConfig())
    assert not info["heel_calibrated"] and info["heel_rows"] == 0


def test_fft_notch_removes_dominant_tone():
    x = np.sin(np.arange(512) * 2 * np.pi * 16 / 512) * 10 + np.random.default_rng(0).normal(0, 0.1, 512) + 50
    out, changed = calibration.notch_dominant_tone(x)
    assert changed and np.std(out) < 0.3 * np.std(x)
    assert calibration.notch_dominant_tone(x[:10])[1] is False


def test_gr_scale_prefers_the_prefix(well):
    hw, tw = well
    tw_tvt, tw_gr = calibration.typewell_arrays(tw)
    scale = calibration.gr_scale(hw, tw_tvt, tw_gr, hw["GR"].to_numpy())
    assert scale >= 8.0
    assert calibration.gr_scale(None, tw_tvt, tw_gr, np.array([1.0])) == 20.0
    assert pd.notna(scale)
