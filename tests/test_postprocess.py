import numpy as np
import pandas as pd

from rogii.config import HedgeConfig, OverrideConfig
from rogii.io import read_sample
from rogii.postprocess import branch_hedge, contact_override, projection, smoothing, visible_prefix
from tests import synthetic


def test_robust_fit_ignores_outliers():
    s = np.linspace(0, 1, 200)
    y = 3 * s**2 - s
    y_noisy = y.copy()
    y_noisy[::17] += 80.0
    fit = projection.robust_fit(s, y_noisy, 3)
    assert np.abs(fit - y).max() < 2.0
    assert np.array_equal(projection.robust_fit(s[:3], y[:3], 5), y[:3])


def test_projection_shrinks_jitter(competition):
    sample = read_sample(competition)
    truth = pd.read_csv(competition.parent / f"{competition.name}_truth_te000000.csv")["TVT"].to_numpy()
    rows = sample["id"].str.rsplit("_", n=1).str[1].astype(int)
    wid = sample["id"].str.rsplit("_", n=1).str[0]
    pick = wid == "te000000"
    noisy = pd.DataFrame(
        {
            "id": sample["id"][pick],
            "tvt": truth[rows[pick]] + np.random.default_rng(0).normal(0, 3, pick.sum()),
        }
    )
    smooth = projection.project(noisy, competition, degree=3, weight=1.0)
    err_before = np.sqrt(np.mean((noisy["tvt"] - truth[rows[pick]]) ** 2))
    err_after = np.sqrt(np.mean((smooth["tvt"].to_numpy() - truth[rows[pick]]) ** 2))
    assert err_after < 0.7 * err_before and list(smooth["id"]) == list(noisy["id"])
    untouched = projection.project(noisy, competition, degree=3, weight=0.0)
    assert np.allclose(untouched["tvt"], noisy["tvt"])


def test_fade_in_and_group_smoothing():
    assert smoothing.fade_in(np.array([-5.0, 0.0, 85.0, 1e6]), 85.0).round(3).tolist() == [
        0.0,
        0.0,
        0.632,
        1.0,
    ]
    assert smoothing.fade_in(np.array([1.0]), 0.0) == 1.0
    wells = pd.Series(["a"] * 100 + ["b"] * 100)
    rng = np.random.default_rng(0)
    pred = np.r_[np.linspace(0, 10, 100), np.linspace(50, 60, 100)] + rng.normal(0, 0.5, 200)
    out = smoothing.smooth_groups(pred, wells, window=21, poly=3)
    assert np.std(np.diff(out[:100])) < np.std(np.diff(pred[:100])) and abs(out[100] - 50) < 3
    df = pd.DataFrame({"well": wells, "v": pred})
    assert np.allclose(smoothing.savgol_by_well(df, "v", 21, 3)["v"].to_numpy(), out, atol=1.0)


def test_best_reconstruction_accepts_matching_prefix(well):
    hw, tw = well
    test_hw = hw[["MD", "X", "Y", "Z", "GR", "TVT_input"]]
    best, reason = contact_override.best_reconstruction(test_hw, hw, tw, OverrideConfig())
    assert reason == "" and best["ref_col"] == "EGFDU" and best["rmse"] < 1e-6
    assert best["comparable_known_rows"] > 50


def test_best_reconstruction_rejects_shifted_prefix(well):
    hw, tw = well
    shifted = hw[["MD", "X", "Y", "Z", "GR", "TVT_input"]].assign(TVT_input=hw["TVT_input"] + 5.0)
    best, reason = contact_override.best_reconstruction(shifted, hw, tw, OverrideConfig())
    assert best is None and "rmse" in reason
    none, why = contact_override.best_reconstruction(hw.assign(TVT_input=np.nan), hw, tw, OverrideConfig())
    assert none is None and "known_rows" in why
    missing, why = contact_override.best_reconstruction(
        hw, hw, tw.drop(columns=["Geology"]), OverrideConfig()
    )
    assert missing is None and "no_ref" in why


def test_reference_priority_dedupes_and_inserts_a_missing_primary():
    assert contact_override.reference_priority(OverrideConfig())[0] == "EGFDU"
    assert contact_override.reference_priority(OverrideConfig(ref_cols=("A", "B"), primary_ref="C")) == (
        "C",
        "A",
        "B",
    )
    assert contact_override.reference_priority(OverrideConfig(ref_cols=("A", "A", "B"), primary_ref="A")) == (
        "A",
        "B",
    )


def test_apply_override_replaces_wells_that_exist_in_train(tmp_path, competition):
    data = synthetic.write_competition(tmp_path / "c", n_train=3, n_test=1)
    # make the test well a copy of train well tr000000
    for kind in ("horizontal_well", "typewell"):
        src = data / "train" / f"tr000000__{kind}.csv"
        frame = pd.read_csv(src)
        if kind == "horizontal_well":
            frame = frame[["MD", "X", "Y", "Z", "GR", "TVT_input"]]
        else:
            frame = frame[["TVT", "GR"]]
        frame.to_csv(data / "test" / f"tr000000__{kind}.csv", index=False)
    hw = pd.read_csv(data / "train" / "tr000000__horizontal_well.csv")
    unknown = np.where(hw["TVT_input"].isna())[0]
    sub = pd.DataFrame({"id": [f"tr000000_{i}" for i in unknown], "tvt": 0.0})
    report = tmp_path / "report.csv"
    out = contact_override.apply_override(sub, data, OverrideConfig(), report)
    assert np.allclose(out["tvt"], hw["TVT"].to_numpy()[unknown], atol=1e-6)
    assert pd.read_csv(report)["status"].tolist() == ["override"]
    other = pd.DataFrame({"id": ["te000000_900", "te000000_901"], "tvt": [1.0, 2.0]})
    assert contact_override.apply_override(other, data, OverrideConfig())["tvt"].tolist() == [1.0, 2.0]


def test_overlap_probe(tmp_path, competition):
    sub = pd.DataFrame({"id": ["te000000_900", "tr000001_5"], "tvt": [1.0, 2.0]})
    probe = contact_override.probe_overlap(sub, competition, tmp_path / "probe.csv")
    assert probe.set_index("well").loc["te000000", "known_prefix_rows"] > 0
    assert probe.set_index("well").loc["tr000001", "has_train_copy"]


def test_branch_hedge_moves_towards_the_midpoint_and_caps(tmp_path):
    sub = pd.DataFrame({"id": [f"w1_{i}" for i in range(6)] + ["w2_0"], "tvt": [100.0] * 6 + [100.0]})
    stats = {
        "w1": dict(
            center_low=90.0,
            center_high=130.0,
            mass_low=0.5,
            mass_high=0.5,
            weighted_center=94.0,
            eval_rows=[1, 2, 3],
        ),
        "w2": dict(
            center_low=90.0,
            center_high=91.0,
            mass_low=0.5,
            mass_high=0.5,
            weighted_center=90.0,
            eval_rows=[0],
        ),
    }
    out = branch_hedge.apply_hedge(sub, stats, HedgeConfig(), tmp_path)
    assert out["tvt"].tolist() == [
        100.0,
        102.0,
        102.0,
        102.0,
        100.0,
        100.0,
        100.0,
    ]  # 0.6 * (110 - 94) capped at 2
    report = pd.read_csv(tmp_path / "pf_seed_branch_hedge_report.csv").set_index("well")
    assert report.loc["w1", "reason"] == "applied" and report.loc["w2", "reason"] == "skip_separation"
    assert (tmp_path / "submission_before_branch_hedge.csv").exists()


def test_branch_hedge_skips_minor_mass_and_bad_stats(tmp_path):
    sub = pd.DataFrame({"id": ["w1_0", "w3_0"], "tvt": [1.0, 1.0]})
    stats = {
        "w1": dict(
            center_low=0.0, center_high=20.0, mass_low=0.1, mass_high=0.9, weighted_center=1.0, eval_rows=[0]
        ),
        "w3": {"error": "boom"},
    }
    out = branch_hedge.apply_hedge(sub, stats, HedgeConfig(), tmp_path)
    assert out["tvt"].tolist() == [1.0, 1.0]
    report = pd.read_csv(tmp_path / "pf_seed_branch_hedge_report.csv").set_index("well")
    assert report.loc["w1", "reason"] == "skip_minor_mass" and report.loc["w3", "reason"] == "error"


def test_branch_hedge_can_skip_overridden_wells(tmp_path):
    pd.DataFrame({"well": ["w1"], "status": ["override"]}).to_csv(
        tmp_path / "gold_contact_override_report.csv", index=False
    )
    sub = pd.DataFrame({"id": ["w1_0"], "tvt": [1.0]})
    stats = {
        "w1": dict(
            center_low=0.0, center_high=20.0, mass_low=0.5, mass_high=0.5, weighted_center=1.0, eval_rows=[0]
        )
    }
    out = branch_hedge.apply_hedge(sub, stats, HedgeConfig(skip_existing=True), tmp_path)
    assert out["tvt"].tolist() == [1.0]
    assert pd.read_csv(tmp_path / "pf_seed_branch_hedge_report.csv")["reason"].tolist() == [
        "skip_existing_route"
    ]


def test_robust_poly_predict_and_candidates(well):
    hw, _ = well
    md = hw["MD"].to_numpy()
    pred = visible_prefix.robust_poly_predict(md[:200], 2 * md[:200] + 1, md, 1)
    assert np.allclose(pred, 2 * md + 1, rtol=1e-6)
    masked = hw.copy()
    masked.loc[masked.index > 600, "TVT_input"] = np.nan
    cands = visible_prefix.poly_candidates(masked)
    assert {"poly_u_deg1_all", "poly_u_deg3_tail80"} <= set(cands)
    assert all(len(v) == len(hw) for v in cands.values())
    assert visible_prefix.poly_candidates(masked.head(20)) == {}


def test_move_weight_gates():
    good = dict(
        status="ok", gain=3.0, best_score=2.0, rank_margin=2.0, consistency=1.0, best_name="surface_X"
    )
    assert visible_prefix.move_weight(good, 5.0, 10.0, "balanced") > 0.1
    assert visible_prefix.move_weight({**good, "status": "skip"}, 5.0, 10.0, "balanced") == 0.0
    assert visible_prefix.move_weight({**good, "gain": 0.5}, 5.0, 10.0, "balanced") == 0.0
    assert visible_prefix.move_weight({**good, "consistency": 0.5}, 5.0, 10.0, "balanced") == 0.0
    assert visible_prefix.move_weight(good, 5.0, 500.0, "aggressive") == 0.0
    assert (
        visible_prefix.move_weight(good, 5.0, 10.0, "aggressive")
        <= visible_prefix.PROFILES["aggressive"]["cap"]
    )
    assert visible_prefix.move_weight({**good, "gain": 0.5}, 5.0, 10.0, "aggressive") > 0.0


def test_profile_output_ramps_and_clips():
    base = pd.DataFrame(
        {"id": [f"w_{i}" for i in range(200)], "tvt": 0.0, "well": "w", "row_idx": np.arange(200)}
    )
    cand = {f"w_{i}": 20.0 for i in range(200)}
    report = {
        "w": dict(
            well="w",
            status="ok",
            gain=4.0,
            best_score=2.0,
            rank_margin=2.0,
            consistency=1.0,
            best_name="pf|x",
        )
    }
    out, moves = visible_prefix.profile_output(base, cand, report, "balanced")
    move = out["tvt"].to_numpy()
    assert (
        move[0] == 0.0 and np.all(np.diff(move) >= -1e-9) and move.max() <= moves[0]["max_move_clip"] + 1e-9
    )
    assert moves[0]["apply_status"] == "applied" and 0 < move.max() < 20.0
    far, far_moves = visible_prefix.profile_output(base, {k: 500.0 for k in cand}, report, "balanced")
    assert far["tvt"].eq(0.0).all() and far_moves[0]["apply_status"] == "kept_base"
    skipped, rows = visible_prefix.profile_output(base, {}, report, "balanced")
    assert skipped["tvt"].eq(0.0).all() and rows[0]["apply_status"] == "skip_nonfinite_candidate"


def test_bimodal_guard_wells(tmp_path):
    pd.DataFrame({"well": ["a", "b"], "bimodal_active": [True, False], "prefix_trust": [0.1, 0.9]}).to_csv(
        tmp_path / "bimodal_selector_report.csv", index=False
    )
    wells, report = visible_prefix.bimodal_guard_wells(tmp_path, False, 0.25)
    assert wells == {"a"} and report["visible_prefix_action"].eq("keep_bimodal_hedge").all()
    assert visible_prefix.bimodal_guard_wells(tmp_path, True, 0.05)[0] == set()
    empty_wells, empty_report = visible_prefix.bimodal_guard_wells(tmp_path / "empty", False, 0.25)
    assert empty_wells == set() and empty_report.empty
