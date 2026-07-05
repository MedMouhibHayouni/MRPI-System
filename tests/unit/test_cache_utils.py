import pytest

from src.cache_utils import (
    compute_content_fingerprint,
    compute_source_fingerprint,
    is_cache_valid,
)


def test_fingerprint_stable_for_same_content(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")

    fp1 = compute_source_fingerprint([str(f)])
    fp2 = compute_source_fingerprint([str(f)])

    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) == 64  # sha256 hex digest


def test_fingerprint_changes_when_content_changes(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2\n")
    fp_before = compute_source_fingerprint([str(f)])

    f.write_text("a,b\n1,3\n")
    fp_after = compute_source_fingerprint([str(f)])

    assert fp_before != fp_after


def test_fingerprint_handles_missing_file_without_raising(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    fp = compute_source_fingerprint([str(missing)])
    assert isinstance(fp, str)


def test_fingerprint_order_independent(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("x")
    f2.write_text("y")

    fp_ab = compute_source_fingerprint([str(f1), str(f2)])
    fp_ba = compute_source_fingerprint([str(f2), str(f1)])

    assert fp_ab == fp_ba  # sorted(paths) à l'intérieur


def test_is_cache_valid_true_when_fingerprint_matches(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("hello")
    model_data = {"source_fingerprint": compute_source_fingerprint([str(f)])}

    assert is_cache_valid(model_data, [str(f)]) is True


def test_is_cache_valid_false_when_source_changed(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("hello")
    model_data = {"source_fingerprint": compute_source_fingerprint([str(f)])}

    f.write_text("changed")
    assert is_cache_valid(model_data, [str(f)]) is False


def test_is_cache_valid_false_when_key_missing():
    assert is_cache_valid({}, ["/nonexistent"]) is False


def test_content_fingerprint_deterministic():
    fp1 = compute_content_fingerprint("explanation", {"a": 1, "b": 2}, ["x", "y"])
    fp2 = compute_content_fingerprint("explanation", {"b": 2, "a": 1}, ["x", "y"])
    assert fp1 == fp2  # sort_keys=True -> insensible à l'ordre des clés


def test_content_fingerprint_differs_on_different_args():
    fp1 = compute_content_fingerprint("wording", {"resource_id": "RES-001"})
    fp2 = compute_content_fingerprint("wording", {"resource_id": "RES-002"})
    assert fp1 != fp2
