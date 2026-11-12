import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from rogii.config import PackageConfig, PostprocessConfig
from rogii.models import learned, package, ridge_stack, trainer


def test_trainer_cross_validates_by_group(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = X["a"] * 2 + X["b"] - X["c"] + rng.normal(0, 0.01, 300)
    groups = np.repeat(np.arange(30), 10)
    t = trainer.Trainer(
        Ridge(alpha=1e-3), cv=GroupKFold(3), cv_args={"groups": groups}, save=True, save_path=tmp_path
    )
    t.fit(X, y)
    assert t.overall_score < 0.05 and len(t.models) == 3 and len(t.fold_scores) == 3
    assert np.isfinite(t.oof_preds).all() and np.allclose(t.predict(X), y, atol=0.1)
    assert (tmp_path / "trainer.pkl").exists()
    with pytest.raises(RuntimeError):
        trainer.Trainer(Ridge()).predict(X)


def test_trainer_default_splits_and_extra_fit_kwargs():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(80, 2)), rng.normal(size=80)
    t = trainer.Trainer(Ridge()).fit(X, y, fit_args={"callbacks": [], "eval_metric": "rmse"})
    assert len(t.models) == 5


def test_koolbox_fallback_registers_a_loadable_module(tmp_path, monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "koolbox", raising=False)
    assert trainer.ensure_koolbox([tmp_path / "missing"]) in {"fallback", "external"}
    import koolbox

    assert hasattr(koolbox, "Trainer")
    assert trainer.ensure_koolbox([]) == "loaded"
    fitted = koolbox.Trainer(Ridge()).fit(np.random.default_rng(0).normal(size=(60, 2)), np.arange(60.0))
    joblib.dump(fitted, tmp_path / "t.pkl")
    assert joblib.load(tmp_path / "t.pkl").overall_score == pytest.approx(fitted.overall_score)


@pytest.mark.parametrize(
    "name, ok",
    [
        ("pkg-1.0-py3-none-any.whl", True),
        ("pkg-1.0-py2.py3-none-any.whl", True),
        ("pkg-1.0-cp27-cp27-linux.whl", False),
        ("pkg-1.0-py3-none-any (1).whl", False),
        ("notes.txt", False),
        ("short-1.whl", False),
    ],
)
def test_wheel_runtime_matching(name, ok):
    assert trainer._wheel_matches_runtime(name) is ok


def test_cpu_params_drop_gpu_options():
    assert "device" not in ridge_stack.cpu_params(ridge_stack.LGB_PARAMS[1])
    assert "device_type" not in ridge_stack.cpu_params(ridge_stack.LGB_PARAMS[0])
    cb = ridge_stack.cpu_params(ridge_stack.CB_PARAMS[0])
    assert cb["task_type"] == "CPU" and "devices" not in cb
    assert ridge_stack.LGB_PARAMS[0]["device_type"] == "gpu"


def test_ridge_postprocess_fades_and_scales():
    df = pd.DataFrame({"md_since": [0.0, 85.0, 1e6]})
    out = ridge_stack.apply_postprocess(
        df, np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 0.0]), alpha=1.0, tau=85, w_pf=0.1
    )
    assert out == pytest.approx([0.0, 9.0 * 0.6321, 9.0], rel=1e-3)


def test_learned_configs_follow_device_and_fast():
    cpu, gpu = learned.lgb_configs("cpu"), learned.lgb_configs("gpu", fast=True)
    assert "device_type" not in cpu[0] and gpu[0]["device_type"] == "gpu" and gpu[0]["n_estimators"] == 600
    assert (
        learned.cb_configs("cpu")[0]["task_type"] == "CPU"
        and learned.cb_configs("gpu", fast=True)[0]["iterations"] == 800
    )
    assert learned.detect_device("cpu") == "cpu" and learned.detect_device("gpu") == "gpu"


def test_make_prediction_blends_and_fades():
    n = 120
    df = pd.DataFrame({
        "well": ["a"] * n, "last_known_tvt": 1000.0, "pf_ancc": 1000.0, "md_since": np.arange(n) * 100.0,
        "likpf_scale_5": 1010.0,
    })  # fmt: skip
    cfg = PostprocessConfig(learned_weight=0.6, sg_window=1, sg_poly=3)  # window 1 disables smoothing
    pred = learned.make_prediction(df, np.full(n, 50.0), cfg)
    assert pred[0] == pytest.approx(1000.0 + 0.4 * 10.0)
    assert pred[-1] == pytest.approx(1000.0 + 0.6 * 50.0 + 0.4 * 10.0, abs=0.01)
    smoothed = learned.make_prediction(
        df, np.random.default_rng(0).normal(50, 5, n), PostprocessConfig(sg_window=21)
    )
    assert np.std(np.diff(smoothed)) < np.std(
        np.diff(learned.make_prediction(df, np.random.default_rng(0).normal(50, 5, n), cfg))
    )


def test_find_models_and_precomputed(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    assert learned.find_models([models]) is None
    (models / "features.json").write_text("[]")
    (models / "lgb0.pkl").write_bytes(b"x")
    assert learned.find_models([tmp_path / "none", models]) == models

    ids = ["a_1", "a_2"]
    pd.DataFrame({"id": ids, "tvt": [1.0, 2.0]}).to_csv(models / "submission.csv", index=False)
    path, frame = learned.find_precomputed([models], ids)
    assert path.name == "submission.csv" and frame["tvt"].tolist() == [1.0, 2.0]
    assert learned.find_precomputed([models], ["a_1", "a_9"]) == (None, None)


def test_package_weights_and_columns():
    assert package._weights({"weights": {"a": 1}}) == {"a": 1.0}
    assert package._weights({"stacker": {"result_keys": ["a", "b"], "coef": [0.5, 0.5]}}) == {
        "a": 0.5,
        "b": 0.5,
    }
    assert package._weights({"models": [{"prediction_column": "a", "weight": 2}]}) == {"a": 2.0}
    with pytest.raises(RuntimeError):
        package._weights({"stacker": {"result_keys": ["a"], "coef": [1, 2]}})
    with pytest.raises(RuntimeError):
        package._weights({})
    assert package._intercept({"stacker": {"intercept": 1.5}}) == 1.5 and package._intercept({}) == 0.0
    assert package._columns_for({"set": ["x", "y"]}, {"feature_set": "set"}) == ["x", "y"]
    assert package._columns_for(["q"], {}) == ["q"] and package._columns_for(
        {}, {"feature_columns": ["z"]}
    ) == ["z"]
    with pytest.raises(RuntimeError):
        package._columns_for({}, {})
    assert package._prediction_column({"branch_name": "b", "model_name": "m"}) == "pred_delta_b_m"


def test_package_feature_matrix_policies():
    frame = pd.DataFrame({"a": [1.0, np.inf, np.nan]})
    native = package._feature_matrix(frame, ["a"], {}, {})
    assert native["a"].isna().sum() == 2
    assert package._feature_matrix(frame, ["a"], {"fillna": -1}, {})["a"].tolist() == [1.0, -1.0, -1.0]
    assert package._feature_matrix(frame, ["a"], {"missing_value_policy": "zero"}, {})["a"].tolist() == [
        1.0,
        0.0,
        0.0,
    ]
    with pytest.raises(RuntimeError):
        package._feature_matrix(frame, ["b"], {}, {})
    with pytest.raises(RuntimeError):
        package._feature_matrix(frame, ["a"], {"missing_value_policy": "weird"}, {})


def test_package_postprocess_steps():
    frame = pd.DataFrame(
        {"md_since": [0.0, 1000.0], "well": ["a", "a"], "row_index": [0, 1], "id": ["a_0", "a_1"]}
    )
    faded = package._fade_and_scale(
        np.array([10.0, 10.0]), {"postprocess": {"fade_tau_md": 120.0, "alpha": 2.0}}, frame
    )
    assert faded[0] == 0.0 and faded[1] == pytest.approx(20.0, rel=1e-3)
    with pytest.raises(RuntimeError):
        package._fade_and_scale(
            np.array([1.0]), {"postprocess": {"fade_tau_md": 1.0}}, pd.DataFrame({"x": [1]})
        )
    n = 60
    wiggly = pd.DataFrame({"id": [f"a_{i}" for i in range(n)], "well": "a", "row_index": np.arange(n)})
    tvt = np.sin(np.arange(n)) + np.arange(n)
    smooth = package._savgol_groups(tvt, {"postprocess": {"savgol_window": 25, "savgol_poly": 2}}, wiggly)
    assert np.std(np.diff(smooth)) < np.std(np.diff(tvt))
    assert package._savgol_groups(tvt, {"postprocess": {"savgol_window": 0}}, wiggly) is tvt


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_package_gated_blend_selects_and_guards(tmp_path):
    ids = [f"w_{i}" for i in range(100)]
    base = pd.DataFrame({"id": ids, "tvt": 100.0})
    close = pd.DataFrame({"id": ids, "tvt": 101.0})
    cfg = PackageConfig(enabled=True, max_weight=0.5, scale=6.0, candidates=(0.25, 0.5))
    out = package.gated_blend(base, close, cfg, tmp_path, {"info": "x"})
    gate = 0.5 / (1 + (1 / 6) ** 2)
    assert out["tvt"].iloc[0] == pytest.approx(100.0 + gate)
    report = pd.read_csv(tmp_path / "model_package_correction_report.csv")
    assert (
        report["selected_for_submission_csv"].tolist() == [False, True]
        and (tmp_path / "submission_model_package_gated_500.csv").exists()
    )

    far = pd.DataFrame({"id": ids, "tvt": 160.0})
    guarded = package.gated_blend(base, far, cfg, tmp_path, {})
    assert guarded["tvt"].eq(100.0).all()
    assert pd.read_csv(tmp_path / "model_package_correction_report.csv")["disabled_by_diff_guard"].all()
    with pytest.raises(RuntimeError, match="mismatch"):
        package.gated_blend(base, close.iloc[:50], cfg, tmp_path, {})


def test_package_correction_without_a_package(tmp_path):
    sample = pd.DataFrame({"id": ["a_1", "a_2"]})
    base = pd.DataFrame({"id": ["a_1", "a_2"], "tvt": [1.0, 2.0]})
    out = package.apply_correction(
        base, tmp_path, sample, [tmp_path / "none"], PackageConfig(enabled=True), tmp_path
    )
    assert out["tvt"].tolist() == [1.0, 2.0] and (tmp_path / "submission_before_model_package.csv").exists()
    assert not pd.read_csv(tmp_path / "model_package_correction_summary.csv", index_col=0).iloc[0, 0]
    with pytest.raises(RuntimeError):
        package.apply_correction(
            base, tmp_path, sample, [], PackageConfig(enabled=True, require=True), tmp_path
        )


def test_package_runs_a_tiny_package(tmp_path):
    """A hand-made package: one direct feature model, delta space, no postprocess."""
    root = tmp_path / "pkg"
    (root / "metadata").mkdir(parents=True)
    (root / "feature_builders").mkdir()
    (root / "stacking").mkdir()
    manifest = {
        "target_space": "delta",
        "feature_columns": "feature_builders/feature_columns.json",
        "blend_config": "stacking/blend_config.json",
        "models": [
            {"model_type": "direct_feature", "feature_column": "signal", "prediction_column": "p_signal"}
        ],
    }
    (root / "metadata" / "model_package_manifest.json").write_text(json.dumps(manifest))
    (root / "feature_builders" / "feature_columns.json").write_text(json.dumps({"columns": []}))
    (root / "stacking" / "blend_config.json").write_text(
        json.dumps({"weights": {"p_signal": 2.0}, "postprocess": {"alpha": 1.0}})
    )
    (root / "feature_builders" / "build_features.py").write_text(
        "import pandas as pd\n"
        "def build_features(sample):\n"
        "    out = sample[['id']].copy()\n"
        "    out['last_known_TVT'] = 100.0\n"
        "    out['signal'] = 3.0\n"
        "    return out\n"
    )
    sample = pd.DataFrame({"id": ["w_1", "w_2"]})
    sub, report, weights, info = package.package_submission(root, tmp_path, sample, PackageConfig())
    assert sub["tvt"].tolist() == [106.0, 106.0] and weights["weight"].tolist() == [2.0]
    assert report["model_type"].tolist() == ["direct_feature"] and info["target_space"] == "delta"
    assert package.find_package_root([tmp_path, root]) == root
