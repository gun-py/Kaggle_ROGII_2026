import pytest

from rogii import config


def test_every_profile_loads():
    names = config.profile_names()
    assert len(names) == 9
    for name in names:
        cfg = config.load_profile(name)
        assert cfg.profile == name
        assert cfg.sp45_weight in cfg.sp45_candidates
        assert cfg.visible_prefix.profile in {"conservative", "balanced", "aggressive"}


def test_default_submission_profile():
    cfg = config.load_profile("vp_balanced_modelpkg_005")
    assert cfg.package.enabled and cfg.package.max_weight == pytest.approx(0.00425)
    assert cfg.visible_prefix.final_selection == "profile"
    assert cfg.visible_prefix.cut_fracs == (0.50, 0.65, 0.75)
    assert cfg.selector.n_seeds == 128 and cfg.selector.heel.enabled


def test_profile_overrides_reach_nested_fields():
    cfg = config.load_profile("bimodal_guarded")
    assert cfg.selector.bimodal.enabled and cfg.visible_prefix.bimodal_guard
    assert config.load_profile("vp_balanced_cut557084").visible_prefix.cut_fracs == (0.55, 0.70, 0.84)
    assert not config.load_profile("contact_gated_anchor").package.enabled


def test_candidate_weights_include_selected():
    cfg = config.load_profile("contact_gated_anchor_w058")
    assert cfg.sp45_weight == 0.58 and cfg.sp45_candidates == (0.50, 0.52, 0.55, 0.58, 0.60)


def test_unknown_profile_and_keys():
    with pytest.raises(ValueError, match="unknown profile"):
        config.load_profile("nope")
    with pytest.raises(KeyError):
        config._override(config.PipelineConfig(), {"not_a_field": 1})
