import numpy as np
import pandas as pd
import pytest

from rogii.structural import dl
from rogii.structural.field import Field
from rogii.structural.variants import DLRule
from rogii.structural.wells import load_wells


@pytest.fixture(scope="module")
def samples(competition):
    train = load_wells(competition, "train")
    fld = Field(train)
    out = {}
    for wid, d in train.items():
        s = dl.sample_features(wid, d, fld, {wid})
        if s is not None:
            out[wid] = s
    return out


def test_sample_features_layout(samples):
    s = next(iter(samples.values()))
    n = len(s["idx"])
    assert s["feats"].shape == (n, 15) and s["target"].shape == (n,) and s["evalmask"].shape == (n,)
    assert s["idx"][-1] == s["n"] - 1 and s["evalmask"].any() and not s["evalmask"].all()
    assert np.isfinite(s["feats"]).all()


def test_sample_features_skips_fully_known_wells(competition):
    d = load_wells(competition, "train", ["tr000000"])["tr000000"]
    d["known"] = np.ones_like(d["known"])
    assert dl.sample_features("tr000000", d, Field({"tr000000": d}), set()) is None


def test_reconstruct_and_eval_rmse(samples):
    zero = {w: np.zeros_like(s["target"]) for w, s in samples.items()}
    assert dl.eval_rmse(samples, zero) > 0
    perfect = {w: s["target"] for w, s in samples.items()}
    assert dl.eval_rmse(samples, perfect) == 0.0
    series = dl.reconstruct(samples, perfect)
    first = next(iter(samples))
    assert series.index[0].startswith(first) and series.notna().all()
    assert series.index.is_unique


def test_hgb_fallback_trains_and_predicts(samples):
    oof, predict = dl.train_hgbr(samples, folds=2, verbose=False)
    assert set(oof) == set(samples)
    out = predict({k: samples[k] for k in list(samples)[:2]})
    assert all(len(v) == len(samples[k]["feats"]) for k, v in out.items())


def test_torch_sequence_model_trains_one_epoch(samples):
    torch = pytest.importorskip("torch")
    from rogii.structural import dl_torch

    torch.set_num_threads(1)
    cfg = dict(hid=8, layers=1, drop=0.0, lr=1e-3)
    oof, predict = dl_torch.train(samples, cfg, folds=2, seeds=1, epochs=1, verbose=False)
    assert set(oof) == set(samples) and all(np.isfinite(v).all() for v in oof.values())
    assert predict({k: samples[k] for k in list(samples)[:1]})


def test_guarded_blend_gates():
    index = pd.Index([f"a_{i}" for i in range(100)])
    existing = pd.Series(100.0, index=index)
    rule = DLRule(weight=0.5)
    good = dl.DLResult(pd.Series(110.0, index=index), oof_rmse=5.0, carry_rmse=10.0, backend="hgbr")
    assert dl.guarded_blend(existing, good, rule).eq(105.0).all()
    assert dl.guarded_blend(existing, dl.DLResult(good.test, 9.9, 10.0, "hgbr"), rule).eq(100.0).all()
    gap = dl.DLResult(pd.Series(200.0, index=index), 5.0, 10.0, "hgbr")
    assert dl.guarded_blend(existing, gap, rule).eq(100.0).all()
    sparse = dl.DLResult(pd.Series(110.0, index=index[:90]), 5.0, 10.0, "hgbr")
    assert dl.guarded_blend(existing, sparse, rule).eq(100.0).all()
