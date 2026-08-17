import pandas as pd
import pytest

from src.hybrid import (
    apply_fallback_penalty,
    get_resource_type,
    hybrid_fusion,
    normalize_scores,
)


@pytest.fixture
def resources_df():
    return pd.DataFrame(
        [
            {
                "resource_id": "RES-001",
                "title": "Vid 1",
                "type": "video",
                "difficulty": "beginner",
                "subject": "math",
                "estimated_time_min": 15,
                "description": "desc 1",
            },
            {
                "resource_id": "RES-002",
                "title": "Ex 2",
                "type": "exercise",
                "difficulty": "beginner",
                "subject": "math",
                "estimated_time_min": 20,
                "description": "desc 2",
            },
            {
                "resource_id": "RES-003",
                "title": "Vid 3",
                "type": "video",
                "difficulty": "intermediate",
                "subject": "french",
                "estimated_time_min": 25,
                "description": "desc 3",
            },
        ]
    )


# ── normalize_scores ────────────────────────────────────────────────────


def test_normalize_scores_empty_list():
    assert normalize_scores([]) == []


def test_normalize_scores_below_min_candidates_returns_raw():
    assert normalize_scores([0.2, 0.4]) == [0.2, 0.4]


def test_normalize_scores_minmax():
    result = normalize_scores([1.0, 2.0, 3.0, 4.0])
    assert result[0] == 0.0
    assert result[-1] == 1.0


def test_normalize_scores_all_equal_returns_ones():
    result = normalize_scores([2.0, 2.0, 2.0])
    assert result == [1.0, 1.0, 1.0]


# ── apply_fallback_penalty / get_resource_type ─────────────────────────


def test_apply_fallback_penalty_applies_when_true():
    assert apply_fallback_penalty(1.0, True) == pytest.approx(0.8)


def test_apply_fallback_penalty_untouched_when_false():
    assert apply_fallback_penalty(1.0, False) == 1.0


def test_get_resource_type_valid():
    from src.schemas.response import ResourceType

    assert get_resource_type("video") == ResourceType.video


def test_get_resource_type_unknown_defaults_to_video():
    from src.schemas.response import ResourceType

    assert get_resource_type("unknown-type") == ResourceType.video


# ── hybrid_fusion ───────────────────────────────────────────────────────


def test_hybrid_fusion_both_empty_returns_empty_list(resources_df):
    result = hybrid_fusion([], pd.DataFrame(), resources_df)
    assert result == []


def test_hybrid_fusion_cf_only(resources_df):
    cf_results = [
        {"resource_id": "RES-001", "predicted_score": 0.9},
        {"resource_id": "RES-002", "predicted_score": 0.5},
        {"resource_id": "RES-003", "predicted_score": 0.1},
    ]
    result = hybrid_fusion(cf_results, pd.DataFrame(), resources_df, request_subject="math")
    ids = [r.resource_id for r in result]
    assert "RES-001" in ids
    # Le garde-fou matière filtre RES-003 (french) quand request_subject="math"
    assert "RES-003" not in ids


def test_hybrid_fusion_cf_only_falls_back_to_unfiltered_when_no_subject_match(resources_df):
    cf_results = [{"resource_id": "RES-003", "predicted_score": 0.5}]
    result = hybrid_fusion(cf_results, pd.DataFrame(), resources_df, request_subject="math")
    # Aucun résultat CF ne correspond à "math" -> fallback sur pool non filtré
    assert len(result) == 1
    assert result[0].resource_id == "RES-003"


def test_hybrid_fusion_cbf_only(resources_df):
    cbf_df = pd.DataFrame(
        [
            {
                "resource_id": "RES-001",
                "title": "Vid 1",
                "type": "video",
                "difficulty": "beginner",
                "cbf_score": 0.8,
                "fallback_used": False,
                "estimated_time_min": 15,
            },
            {
                "resource_id": "RES-002",
                "title": "Ex 2",
                "type": "exercise",
                "difficulty": "beginner",
                "cbf_score": 0.3,
                "fallback_used": True,
                "estimated_time_min": 20,
            },
        ]
    )
    result = hybrid_fusion([], cbf_df, resources_df)
    assert len(result) == 2
    assert result[0].resource_id == "RES-001"  # score le plus haut en premier


def test_hybrid_fusion_merges_cf_and_cbf_scores(resources_df):
    cf_results = [
        {"resource_id": "RES-001", "predicted_score": 0.5},
        {"resource_id": "RES-002", "predicted_score": 0.9},
        {"resource_id": "RES-003", "predicted_score": 0.1},
    ]
    cbf_df = pd.DataFrame(
        [
            {
                "resource_id": "RES-001",
                "title": "Vid 1",
                "type": "video",
                "difficulty": "beginner",
                "cbf_score": 0.9,
                "fallback_used": False,
                "estimated_time_min": 15,
            },
        ]
    )
    result = hybrid_fusion(cf_results, cbf_df, resources_df, request_subject="math")
    ids = {r.resource_id for r in result}
    # RES-001 présent dans les deux sources -> score hybride, doit apparaître
    assert "RES-001" in ids


def test_hybrid_fusion_applies_fallback_penalty_to_merged_item(resources_df):
    cf_results = [{"resource_id": "RES-001", "predicted_score": 0.5}]
    cbf_df = pd.DataFrame(
        [
            {
                "resource_id": "RES-001",
                "title": "Vid 1",
                "type": "video",
                "difficulty": "beginner",
                "cbf_score": 0.9,
                "fallback_used": True,
                "estimated_time_min": 15,
            },
        ]
    )
    result = hybrid_fusion(cf_results, cbf_df, resources_df)
    assert len(result) == 1
    # avec fallback_used=True, le poids CBF effectif est réduit -> score < cbf_score brut
    assert result[0].relevance_score < 0.9


def test_hybrid_fusion_respects_max_recommendations(resources_df):
    cbf_df = pd.DataFrame(
        [
            {
                "resource_id": rid,
                "title": rid,
                "type": "video",
                "difficulty": "beginner",
                "cbf_score": score,
                "fallback_used": False,
                "estimated_time_min": 15,
            }
            for rid, score in [("RES-001", 0.9), ("RES-002", 0.5), ("RES-003", 0.1)]
        ]
    )
    result = hybrid_fusion([], cbf_df, resources_df, max_recommendations=2)
    assert len(result) == 2


def test_hybrid_fusion_missing_fallback_column_defaults_to_false(resources_df):
    cbf_df = pd.DataFrame(
        [
            {
                "resource_id": "RES-001",
                "title": "Vid 1",
                "type": "video",
                "difficulty": "beginner",
                "cbf_score": 0.7,
                "estimated_time_min": 15,
            },
        ]
    )
    result = hybrid_fusion([], cbf_df, resources_df)
    assert len(result) == 1  # ne plante pas malgré la colonne absente
