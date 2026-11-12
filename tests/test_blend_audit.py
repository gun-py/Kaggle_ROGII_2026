import json

import numpy as np
import pandas as pd
import pytest

from rogii import audit, blend


def _sub(values, ids=("a_1", "a_2", "a_3")):
    return pd.DataFrame({"id": list(ids), "tvt": values})


def test_weighted_blend_and_mismatch():
    out = blend.weighted(_sub([10.0, 20.0, 30.0]), _sub([0.0, 0.0, 0.0]), 0.25)
    assert out["tvt"].tolist() == [2.5, 5.0, 7.5]
    explicit = blend.weighted(_sub([1.0, 1.0, 1.0]), _sub([1.0, 1.0, 1.0]), 0.3, 0.7)
    assert explicit["tvt"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="mismatch"):
        blend.weighted(_sub([1.0, 2.0, 3.0]), _sub([1.0, 2.0], ids=("a_1", "a_2")), 0.5)


def test_sp45_learned_writes_every_candidate(tmp_path):
    sp45, learned = _sub([10.0, 10.0, 10.0]), _sub([20.0, 20.0, 20.0])
    chosen = blend.sp45_learned(sp45, learned, 0.6, (0.5, 0.6), tmp_path)
    assert chosen["tvt"].tolist() == pytest.approx([14.0] * 3)
    report = pd.read_csv(tmp_path / "sp45_learned_blend_report.csv")
    assert report["w_sp45"].tolist() == [0.5, 0.6] and report["rmse_vs_sp45"].iloc[1] == pytest.approx(4.0)
    assert (tmp_path / "submission_sp45_learned_w0.60.csv").exists()
    assert (tmp_path / "learned_trajectory_submission.csv").exists()


def test_sp45_learned_rejects_non_finite(tmp_path):
    with pytest.raises(RuntimeError, match="non-finite"):
        blend.sp45_learned(_sub([1.0, 1.0, 1.0]), _sub([1.0, np.nan, 1.0]), 0.5, (0.5,), tmp_path)


def _write(tmp_path, sub, sample=None):
    sample = sample if sample is not None else _sub([0.0, 0.0, 0.0])
    sub.to_csv(tmp_path / "submission.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    return tmp_path / "sample_submission.csv"


def test_audit_accepts_a_valid_file(tmp_path):
    sample_path = _write(tmp_path, _sub([1.0, 2.0, 3.0]))
    result = audit.write_audit(tmp_path, sample_path)
    assert result["rows"] == 3 and result["tvt_mean"] == 2.0 and len(result["sha256_submission_csv"]) == 64
    assert json.loads((tmp_path / "submission_audit.json").read_text()) == result
    for name in ("latest_valid_submission.csv", "latest_valid_submission.json", "submission_audit_copy.csv"):
        assert (tmp_path / name).exists()


@pytest.mark.parametrize(
    "sub, message",
    [
        (_sub([1.0, 2.0, 3.0]).rename(columns={"tvt": "value"}), "columns"),
        (_sub([1.0, 2.0], ids=("a_1", "a_2")), "row count"),
        (_sub([1.0, 2.0, 3.0]).iloc[::-1], "order"),
        (_sub([1.0, np.nan, 3.0]), "non-finite"),
    ],
)
def test_audit_rejects_bad_files(tmp_path, sub, message):
    sample_path = _write(tmp_path, sub)
    with pytest.raises(RuntimeError, match=message):
        audit.audit_submission(tmp_path / "submission.csv", sample_path)


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("hello")
    assert audit.sha256_file(path) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
