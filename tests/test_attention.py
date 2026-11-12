import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rogii.attention import evaluate  # noqa: E402
from rogii.attention.config import ABLATIONS, ablated, load_config  # noqa: E402
from rogii.attention.data import STRIDE, WellSet, rows_from_tokens, token_centres, typewell_grid  # noqa: E402
from rogii.attention.model import Trail  # noqa: E402
from rogii.attention.synthetic import SyntheticWells, synthetic_gr  # noqa: E402
from rogii.attention.train import collate, finetune, predict  # noqa: E402
from rogii.cli import main  # noqa: E402
from rogii.structural.wells import load_wells  # noqa: E402

TINY = dict(d_model=32, heads=2, encoder_layers=1, typewell_layers=1, update_layers=1, iterations=2,
            window=16, dropout=0.0, token_budget=4000, finetune_epochs=2, patience=2)  # fmt: skip


@pytest.fixture(scope="module")
def ws(competition):
    return WellSet(load_wells(competition, "train"), load_wells(competition, "test"))


def _shifted(d, c):
    out = dict(d)
    out["tvt"] = d["tvt"] + c
    out["tw_tvt"] = d["tw_tvt"] + c
    return out


def test_token_centres_hit_anchor_and_last_row():
    c = token_centres(1000, 333, 10)
    assert 333 in c and c[-1] == 999 and c[0] == 333 - 10 * STRIDE
    assert np.all(np.diff(c) <= STRIDE)


def test_typewell_grid_starts_on_whole_foot():
    tw = typewell_grid(np.arange(100.25, 300.0, 0.5), np.sin(np.arange(400) / 9.0))
    assert tw.lo == np.floor(100.25) - 64 and tw.valid.sum() > 190
    assert np.isclose(tw.gr[tw.valid].mean(), 0.0, atol=0.1)


def test_sample_is_anchor_relative(ws):
    wid = ws.train_ids[0]
    s = ws.sample(wid, frozenset(ws.train_ids) - {wid})
    a = int(np.flatnonzero(s.pos == 0)[0])
    assert s.centres[a] == s.ia and not s.hidden[a] and s.drift[a] == 0.0
    assert s.hidden[a + 1 :].all() and not s.hidden[:a].any()
    assert np.isfinite(s.feats).all()


def test_neighbours_respect_pool(ws):
    wid = ws.train_ids[0]
    s = ws.sample(wid, frozenset())
    assert not s.nb_mask.any()
    s = ws.sample(wid, frozenset(ws.train_ids) - {wid})
    assert s.nb_mask.any()


def test_model_shapes_and_anchor_continuity(ws):
    cfg = load_config(**TINY)
    model = Trail(cfg)
    samples = [ws.sample(w, frozenset(ws.train_ids) - {w}) for w in ws.train_ids[:3]]
    paths, info = model(collate(samples, "cpu"))
    assert len(paths) == cfg.iterations and paths[-1].shape == (3, max(len(s) for s in samples))
    for row, s in enumerate(samples):
        visible = ~s.hidden
        assert torch.equal(
            paths[-1][row, : len(s)][torch.from_numpy(visible)], torch.from_numpy(s.drift[visible])
        )
    assert info["looks"][0]["p"].shape[-1] >= 2 * cfg.window + 1


def test_prediction_is_datum_invariant(ws):
    torch.manual_seed(0)
    model = Trail(load_config(**TINY)).eval()
    wid = ws.train_ids[1]
    d = ws.wells[wid]
    base = ws.sample(wid, frozenset())
    moved = WellSet.__new__(WellSet)
    moved.wells = {wid: _shifted(d, 37.0)}
    moved.typewells = {wid: typewell_grid(d["tw_tvt"] + 37.0, d["tw_gr"])}
    moved.index = ws.index
    shifted = moved.sample(wid, frozenset())
    a = predict(model, [base], "cpu", 4000)[0]
    b = predict(model, [shifted], "cpu", 4000)[0]
    np.testing.assert_allclose(a, b, atol=1e-4)
    rows_a = rows_from_tokens(base, a, d["z"])
    rows_b = rows_from_tokens(shifted, b, d["z"])
    np.testing.assert_allclose(rows_b - rows_a, 37.0, atol=1e-4)


def test_synthetic_gr_without_noise_reads_the_typewell(ws):
    tw = next(iter(ws.typewells.values()))
    tvt = np.linspace(tw.tvt[0] + 10, tw.tvt[-1] - 10, 500)
    np.testing.assert_allclose(synthetic_gr(tvt, tw, np.random.default_rng(0), noise=False),
                               np.interp(tvt, tw.tvt, tw.raw_gr))  # fmt: skip


def test_synthetic_samples_have_targets(ws):
    synth = SyntheticWells.from_wellset(ws)
    s = synth.sample(np.random.default_rng(1), 64)
    assert s.hidden.any() and np.isfinite(s.target[s.hidden]).all() and not s.nb_mask.any()


def test_tiny_model_beats_nothing_on_a_synthetic_well(ws):
    cfg = load_config(**{**TINY, "lr": 3e-3})
    synth = SyntheticWells.from_wellset(ws)
    s = synth.sample(np.random.default_rng(4), 64)
    model = Trail(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = collate([s], "cpu")
    from rogii.attention.model import trail_loss

    first = None
    for _ in range(60):
        paths, info = model(batch)
        loss, mse = trail_loss(paths, info, batch, cfg)
        first = float(mse) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(mse) < 0.5 * first


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_every_ablation_builds(name):
    cfg = ablated(load_config(**TINY), name)
    assert Trail(cfg).cfg == cfg


def test_finetune_and_score_rows(ws):
    cfg = load_config(**TINY)
    train, val = ws.train_ids[:8], ws.train_ids[8:10]
    model, history = finetune(cfg, ws, train, val, "cpu", epochs=2)
    assert len(history) >= 1 and np.isfinite(history[-1]["val_rmse"])
    scores, rows, _ = evaluate.evaluate_wells(model, ws, ws.train_ids[10:], cfg, "cpu")
    assert set(scores) == set(ws.train_ids[10:]) and all(n > 0 for _, n in scores.values())


def test_bootstrap_reports_paired_differences():
    frame = pd.DataFrame({"well": list("abcd") * 2, "method": ["x"] * 4 + ["y"] * 4,
                          "sse": [1, 4, 9, 16, 2, 8, 18, 32], "n": [1] * 8})  # fmt: skip
    out = evaluate.bootstrap(frame, base="x", resamples=200).set_index("method")
    assert out.loc["x", "diff"] == 0 and out.loc["y", "rmse"] > out.loc["x", "rmse"]


def test_cli_holdout_and_predict(competition, tmp_path, monkeypatch):
    import rogii.attention.config as config

    monkeypatch.setattr(
        config, "load_config", lambda name="trail", **kw: config.TrailConfig(**{**TINY, **kw})
    )
    out = tmp_path / "attn"
    assert (
        main(
            [
                "attn",
                "holdout",
                "--data",
                str(competition),
                "--out",
                str(out),
                "--wells",
                "3",
                "--epochs",
                "1",
            ]
        )
        == 0
    )
    summary = pd.read_json(out / "holdout" / "trail_full_summary.json", typ="series")
    assert summary["wells"] == 3 and np.isfinite(summary["rmse"])
    assert main(["attn", "predict", "--data", str(competition), "--out", str(out),
                 "--checkpoint", str(out / "holdout" / "trail_full.pt"),
                 "--submission-dir", str(tmp_path / "sub")]) == 0  # fmt: skip
    sub = pd.read_csv(tmp_path / "sub" / "submission.csv")
    assert len(sub) == len(pd.read_csv(competition / "sample_submission.csv")) and sub.tvt.notna().all()


def test_test_wells_never_shadow_train_copies(competition, ws):
    from rogii.attention.data import TEST_PREFIX

    assert all(w.startswith(TEST_PREFIX) for w in ws.test_ids)
    assert all(np.isfinite(ws.wells[w]["tvt"]).all() for w in ws.train_ids)
