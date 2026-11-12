import numpy as np
import pandas as pd
import pytest

from rogii import io, paths


def test_list_wells_and_read(competition):
    train = io.list_wells(competition, "train")
    assert len(train) == 12 and train == sorted(train)
    hw, tw = io.read_well(competition, train[0], "train")
    assert {"MD", "TVT", "TVT_input", "GR"} <= set(hw.columns)
    _, tw_sorted = io.read_well(competition, train[0], "train", sort_typewell=True)
    assert tw_sorted["TVT"].is_monotonic_increasing


def test_split_ids():
    out = io.split_ids(pd.DataFrame({"id": ["abc_12", "a_b_3"], "tvt": [1.0, 2.0]}))
    assert out["well"].tolist() == ["abc", "a_b"]
    assert out["row_idx"].tolist() == [12, 3]


def test_split_ids_rejects_bad_format():
    with pytest.raises(ValueError):
        io.split_ids(pd.DataFrame({"id": ["nodelimiter"], "tvt": [1.0]}))


def test_validate_submission_accepts_and_rejects():
    sample = pd.DataFrame({"id": ["a_1", "a_2"], "tvt": [0.0, 0.0]})
    good = pd.DataFrame({"id": ["a_1", "a_2"], "tvt": [1.0, 2.0]})
    assert io.validate_submission(good, sample)["tvt"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="order"):
        io.validate_submission(good.iloc[::-1], sample)
    with pytest.raises(ValueError, match="rows"):
        io.validate_submission(good.iloc[:1], sample)
    with pytest.raises(ValueError, match="non-finite"):
        io.validate_submission(good.assign(tvt=[1.0, np.nan]), sample)


def test_align_to_sample_reorders_but_refuses_gaps():
    sample = pd.DataFrame({"id": ["a_1", "a_2"]})
    frame = pd.DataFrame({"id": ["a_2", "a_1"], "tvt": [2.0, 1.0]})
    assert io.align_to_sample(frame, sample)["tvt"].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        io.align_to_sample(frame.iloc[:1], sample)
    with pytest.raises(ValueError):
        io.align_to_sample(pd.concat([frame, frame]), sample)


def test_submission_roundtrip(tmp_path):
    path = tmp_path / "s.csv"
    io.write_submission(pd.DataFrame({"id": [1, 2], "tvt": [1, 2], "junk": [0, 0]}), path)
    back = io.read_submission(path)
    assert list(back.columns) == ["id", "tvt"] and back["id"].tolist() == ["1", "2"]
    (tmp_path / "bad.csv").write_text("id,value\n1,2\n")
    with pytest.raises(ValueError):
        io.read_submission(tmp_path / "bad.csv")


def test_competition_dir_resolution(competition, monkeypatch, tmp_path):
    assert paths.competition_dir(competition) == competition
    monkeypatch.setenv("ROGII_DATA", str(competition))
    assert paths.competition_dir() == competition
    monkeypatch.delenv("ROGII_DATA")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        paths.competition_dir()


def test_work_dir_prefers_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGII_OUT", str(tmp_path / "env"))
    assert paths.work_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert paths.work_dir() == tmp_path / "env"
    assert (tmp_path / "env").is_dir()


def test_first_existing(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "models").mkdir()
    assert paths.first_existing([tmp_path / "a", tmp_path / "b"], "models") == tmp_path / "b"
    assert paths.first_existing([tmp_path / "a"], "models") is None


def test_artifact_roots_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ROGII_ROGII_MODEL_PACKAGE", str(tmp_path))
    assert paths.artifact_roots("rogii-model-package")[0] == tmp_path
