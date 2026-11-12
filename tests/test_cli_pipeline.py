import json

import numpy as np
import pandas as pd
import pytest

from rogii import cli, pipeline
from rogii.config import load_profile
from rogii.diagnostics import cv
from rogii.io import read_sample


def _truth(competition, wid):
    return pd.read_csv(competition.parent / f"{competition.name}_truth_{wid}.csv")["TVT"].to_numpy()


def test_cli_list_and_version(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "vp_balanced_modelpkg_005" in out and "cv4" in out
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    with pytest.raises(SystemExit):
        cli.main(["predict", "--bogus"])


def test_cli_blend_and_audit(tmp_path, competition, capsys):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    sample = read_sample(competition)
    pd.DataFrame({"id": sample["id"], "tvt": 10.0}).to_csv(a, index=False)
    pd.DataFrame({"id": sample["id"], "tvt": 20.0}).to_csv(b, index=False)
    out = tmp_path / "mix.csv"
    cli.main(["blend", str(a), str(b), "-w", "0.25", "-o", str(out)])
    assert pd.read_csv(out)["tvt"].eq(17.5).all()
    out.replace(tmp_path / "submission.csv")
    cli.main(["audit", "--data", str(competition), "--out", str(tmp_path)])
    assert json.loads((tmp_path / "submission_audit.json").read_text())["rows"] == len(sample)


def test_cli_cv_commands(tmp_path, competition):
    cli.main(["cv", "oracle", "--data", str(competition), "--out", str(tmp_path), "--wells", "6"])
    summary = pd.read_csv(tmp_path / "oracle_ladder_summary.csv").set_index("oracle")
    assert summary.loc["smooth", "pooled_rmse"] < summary.loc["flat", "pooled_rmse"]
    cli.main(["cv", "heel", "--data", str(competition), "--out", str(tmp_path), "--wells", "4"])
    assert (tmp_path / "heel_calibration_localization_summary.csv").exists()
    cli.main(["cv", "struct", "--data", str(competition), "--out", str(tmp_path), "--wells", "4"])
    scores = pd.read_csv(tmp_path / "struct_cv_summary.csv", index_col=0).iloc[:, 0]
    assert scores["v5"] < scores["carry"]


def test_selector_cv_report(tmp_path, competition, monkeypatch):
    from rogii.config import SelectorConfig

    monkeypatch.setattr(cv, "CV_PF_SEEDS", 2)
    summary = cv.selector_cv(competition, tmp_path, 3, 0, SelectorConfig(n_particles=40))
    assert summary["wells"] == 3 and np.isfinite(summary["pooled_rmse"])
    assert (tmp_path / "selector_cv_by_well.csv").exists()


def test_quick_profile_shrinks_budgets():
    cfg = pipeline.quick(load_profile("vp_balanced_modelpkg_005"))
    assert (
        cfg.selector.n_seeds == 8
        and cfg.visible_prefix.cal_seeds == 4
        and cfg.visible_prefix.particles == 100
    )
    assert cfg.package.max_weight == 0.00425


def test_artifact_lookup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "given" / pipeline.RIDGE_DATASET / "models").mkdir(parents=True)
    assert pipeline.Artifacts.locate(tmp_path / "given").ridge == tmp_path / "given" / pipeline.RIDGE_DATASET
    assert pipeline.Artifacts.locate(tmp_path / "empty").ridge is None
    assert pipeline.Artifacts.locate().learned[0].name == pipeline.LEARNED_DATASET


def test_pipeline_refuses_to_train_by_surprise(tmp_path, competition, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_artifacts"):
        pipeline.run(
            "vp_balanced_final",
            data_dir=competition,
            out_dir=tmp_path,
            artifacts_dir=tmp_path / "none",
            fast=True,
        )


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory, competition):
    """End-to-end run on the synthetic competition with the two artifact-backed stages stubbed out."""
    sample = read_sample(competition)
    rows = sample["id"].str.rsplit("_", n=1, expand=True)
    last_known = {}
    for wid in rows[0].unique():
        hw = pd.read_csv(competition / "test" / f"{wid}__horizontal_well.csv")
        last_known[wid] = float(hw["TVT_input"].dropna().iloc[-1])
    carry = pd.DataFrame({"id": sample["id"], "tvt": rows[0].map(last_known)})

    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "_ridge", lambda run: carry.copy())
    mp.setattr(pipeline, "_learned", lambda run, fast: carry.assign(tvt=carry["tvt"] - 1.0))
    out = tmp_path_factory.mktemp("out")
    sub = pipeline.run(
        "vp_balanced_final", structural="v5", data_dir=competition, out_dir=out, fast=True, n_jobs=2
    )
    mp.undo()
    return sub, out, carry


def test_smoke_submission_is_valid(smoke_run, competition):
    sub, out, carry = smoke_run
    sample = read_sample(competition)
    assert list(sub["id"]) == list(sample["id"]) and np.isfinite(sub["tvt"]).all()
    audit = json.loads((out / "submission_audit.json").read_text())
    assert audit["rows"] == len(sample) and audit["id_order_matches_sample"]
    on_disk = pd.read_csv(out / "submission.csv")
    assert np.allclose(on_disk["tvt"], sub["tvt"])


def test_smoke_stage_reports_exist(smoke_run):
    _, out, _ = smoke_run
    expected = [
        "bimodal_selector_report.csv", "sp45_projection_submission.csv", "sp45_learned_blend_report.csv",
        "overlap_layer_probe.csv", "gold_prefix_calibration_report.csv", "gold_prefix_moves_balanced.csv",
        "gold_prefix_submission_audit.json", "pf_seed_branch_hedge_report.csv", "submission_audit.json",
        "submission_self_verified_anchor.csv", "submission_gold_prefix_balanced.csv",
    ]  # fmt: skip
    missing = [name for name in expected if not (out / name).exists()]
    assert not missing
    report = pd.read_csv(out / "gold_prefix_calibration_report.csv")
    assert set(report["status"]) <= {
        "ok",
        "skip_no_final_candidate",
        "skip_short_prefix",
        "skip_bimodal_hedge",
    }


def test_smoke_prediction_beats_carry_on_the_drifting_wells(smoke_run, competition):
    sub, _, carry = smoke_run
    sample = read_sample(competition)
    rows = sample["id"].str.rsplit("_", n=1, expand=True)
    truth = np.concatenate(
        [_truth(competition, w)[rows[1][rows[0] == w].astype(int)] for w in rows[0].unique()]
    )
    order = np.concatenate([np.where(rows[0] == w)[0] for w in rows[0].unique()])
    err_sub = np.sqrt(np.mean((sub["tvt"].to_numpy()[order] - truth) ** 2))
    err_carry = np.sqrt(np.mean((carry["tvt"].to_numpy()[order] - truth) ** 2))
    assert err_sub < err_carry * 1.25
