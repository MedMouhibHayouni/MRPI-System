import numpy as np
import pytest

from src.metrics import (
    average_ignoring_none,
    coverage,
    get_relevant_resources,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ── get_relevant_resources ────────────────────────────────────────────


def test_get_relevant_resources_exact_match(sample_resources_df):
    profile = {
        "subject": "math",
        "weak_concept": "fractions",
        "academic_level": "beginner",
        "past_interactions": [],
    }
    relevant = get_relevant_resources(profile, sample_resources_df)
    assert relevant == {"RES-001", "RES-002"}


def test_get_relevant_resources_excludes_unmet_prerequisites(sample_resources_df):
    profile = {
        "subject": "math",
        "weak_concept": "fractions",
        "academic_level": "advanced",
        "past_interactions": [],
    }
    # RES-003 nécessite RES-001, non complété -> exclu
    relevant = get_relevant_resources(profile, sample_resources_df)
    assert relevant == set()


def test_get_relevant_resources_includes_when_prereq_met(sample_resources_df):
    profile = {
        "subject": "math",
        "weak_concept": "fractions",
        "academic_level": "advanced",
        "past_interactions": ["RES-001"],
    }
    relevant = get_relevant_resources(profile, sample_resources_df)
    assert relevant == {"RES-003"}


def test_get_relevant_resources_empty_when_no_candidates(sample_resources_df):
    profile = {
        "subject": "physics",
        "weak_concept": "mechanics",
        "academic_level": "beginner",
        "past_interactions": [],
    }
    assert get_relevant_resources(profile, sample_resources_df) == set()


# ── precision_at_k / recall_at_k / ndcg_at_k ──────────────────────────


def test_precision_at_k_basic():
    recommended = ["A", "B", "C", "D"]
    relevant = {"B", "D", "Z"}
    assert precision_at_k(recommended, relevant, k=4) == 0.5


def test_precision_at_k_empty_recommendation_list():
    assert precision_at_k([], {"A"}, k=5) == 0.0


def test_recall_at_k_basic():
    recommended = ["A", "B", "C"]
    relevant = {"B", "C", "D"}
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_none_when_no_relevant():
    assert recall_at_k(["A"], set(), k=5) is None


def test_ndcg_at_k_perfect_ranking_is_one():
    recommended = ["A", "B", "C"]
    relevant = {"A", "B"}
    assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worse_ranking_is_less_than_one():
    perfect = ndcg_at_k(["A", "B", "C"], {"A", "B"}, k=3)
    worse = ndcg_at_k(["C", "A", "B"], {"A", "B"}, k=3)
    assert worse < perfect


def test_ndcg_at_k_none_when_no_relevant():
    assert ndcg_at_k(["A", "B"], set(), k=2) is None


# ── intra_list_diversity ──────────────────────────────────────────────


def test_intra_list_diversity_none_for_single_item(sample_resources_df):
    encoder = None
    resource_matrix = np.eye(len(sample_resources_df))
    result = intra_list_diversity(["RES-001"], sample_resources_df, encoder, resource_matrix)
    assert result is None


def test_intra_list_diversity_zero_for_identical_vectors(sample_resources_df):
    resource_matrix = np.ones((len(sample_resources_df), 3))
    result = intra_list_diversity(
        ["RES-001", "RES-002"], sample_resources_df, None, resource_matrix
    )
    assert result == pytest.approx(0.0, abs=1e-9)


def test_intra_list_diversity_positive_for_orthogonal_vectors(sample_resources_df):
    resource_matrix = np.eye(len(sample_resources_df))
    result = intra_list_diversity(
        ["RES-001", "RES-002", "RES-003"], sample_resources_df, None, resource_matrix
    )
    assert result == pytest.approx(1.0)  # cosine sim = 0 entre vecteurs orthogonaux


# ── coverage ────────────────────────────────────────────────────────


def test_coverage_full_catalog_covered(sample_resources_df):
    all_ids = sample_resources_df["resource_id"].tolist()
    assert coverage(all_ids, sample_resources_df) == pytest.approx(1.0)


def test_coverage_partial(sample_resources_df):
    result = coverage(["RES-001"], sample_resources_df)
    assert result == pytest.approx(1 / len(sample_resources_df))


def test_coverage_empty_catalog_returns_zero():
    import pandas as pd

    empty_df = pd.DataFrame(columns=["resource_id"])
    assert coverage(["RES-001"], empty_df) == 0.0


def test_coverage_deduplicates_ids(sample_resources_df):
    ids_with_dupes = ["RES-001", "RES-001", "RES-002", "RES-002"]
    result = coverage(ids_with_dupes, sample_resources_df)
    assert result == pytest.approx(2 / len(sample_resources_df))


# ── average_ignoring_none ─────────────────────────────────────────────


def test_average_ignoring_none_basic():
    assert average_ignoring_none([1.0, 2.0, 3.0]) == 2.0


def test_average_ignoring_none_skips_none():
    assert average_ignoring_none([1.0, None, 3.0]) == 2.0


def test_average_ignoring_none_all_none_returns_zero():
    assert average_ignoring_none([None, None]) == 0.0


def test_average_ignoring_none_empty_list_returns_zero():
    assert average_ignoring_none([]) == 0.0
