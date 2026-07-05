import os

import numpy as np
import pandas as pd
import pytest

from src import CBF
from src.CBF import (
    build_vectorizer,
    get_or_build_vectorizer,
    load_vectorizer,
    prerequisites_met,
    recommend_cbf,
)


# ── prerequisites_met ──────────────────────────────────────────────────


def test_prerequisites_met_true_when_all_completed():
    assert prerequisites_met(["RES-001", "RES-002"], {"RES-001", "RES-002", "RES-003"})


def test_prerequisites_met_false_when_missing_one():
    assert not prerequisites_met(["RES-001", "RES-002"], {"RES-001"})


def test_prerequisites_met_handles_csv_string_values():
    assert prerequisites_met("RES-001", {"RES-001"})
    assert not prerequisites_met("RES-001", set())


def test_prerequisites_met_handles_delimited_string_values():
    assert prerequisites_met("RES-001;RES-002", {"RES-001", "RES-002"})
    assert not prerequisites_met("RES-001,RES-002", {"RES-001"})


def test_prerequisites_met_true_when_not_a_list():
    # NaN / None / autre type non-liste => pas de prérequis => accessible
    assert prerequisites_met(float("nan"), {"RES-001"})
    assert prerequisites_met(None, set())


def test_prerequisites_met_empty_list_is_always_true():
    assert prerequisites_met([], set())


# ── build_vectorizer / load_vectorizer / get_or_build_vectorizer ─────


def test_build_vectorizer_persists_and_weights_columns(sample_resources_df, tmp_path):
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, df, weight_vector = build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )

    assert save_path.exists()
    assert matrix.shape[0] == len(sample_resources_df)
    assert weight_vector.shape[0] == matrix.shape[1]
    # Le vecteur de poids ne doit contenir que les 4 valeurs définies dans
    # FEATURE_WEIGHTS (2.0, 3.0, 2.0, 0.5)
    assert set(np.unique(weight_vector)) <= {2.0, 3.0, 0.5}


def test_load_vectorizer_roundtrip(sample_resources_df, tmp_path):
    save_path = tmp_path / "encoder.pkl"
    build_vectorizer(sample_resources_df, save_path=str(save_path))

    encoder, matrix, df, weight_vector = load_vectorizer(str(save_path))
    assert matrix.shape[0] == len(sample_resources_df)
    assert isinstance(df, pd.DataFrame)


def test_load_vectorizer_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vectorizer(str(tmp_path / "nope.pkl"))


def test_get_or_build_vectorizer_builds_when_no_cache(sample_resources_df, tmp_path):
    save_path = tmp_path / "encoder.pkl"
    assert not save_path.exists()
    encoder, matrix, df, weight_vector = get_or_build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )
    assert save_path.exists()


def test_get_or_build_vectorizer_reuses_valid_cache(
    sample_resources_df, tmp_path, monkeypatch
):
    save_path = tmp_path / "encoder.pkl"
    # Les fingerprints sont calculés sur CBF.SOURCE_FILES (resources.csv du
    # projet), pas sur sample_resources_df -> ils restent stables entre
    # les deux appels ici, donc le cache doit être jugé valide.
    get_or_build_vectorizer(sample_resources_df, save_path=str(save_path))
    mtime_1 = save_path.stat().st_mtime_ns

    get_or_build_vectorizer(sample_resources_df, save_path=str(save_path))
    mtime_2 = save_path.stat().st_mtime_ns

    assert mtime_1 == mtime_2  # pas réécrit -> cache réutilisé


def test_get_or_build_vectorizer_rebuilds_when_cache_invalid(
    sample_resources_df, tmp_path, monkeypatch
):
    save_path = tmp_path / "encoder.pkl"
    get_or_build_vectorizer(sample_resources_df, save_path=str(save_path))

    # Force la fonction de validation à renvoyer False pour simuler un
    # cache périmé (source_fingerprint ne correspond plus).
    monkeypatch.setattr(CBF, "is_cache_valid", lambda model_data, files: False)
    mtime_1 = save_path.stat().st_mtime_ns

    get_or_build_vectorizer(sample_resources_df, save_path=str(save_path))
    mtime_2 = save_path.stat().st_mtime_ns

    assert mtime_2 != mtime_1  # reconstruit -> fichier réécrit


# ── recommend_cbf ──────────────────────────────────────────────────────


def _make_request(**overrides):
    from src.schemas.request import AcademicLevel, LearningStyle, RecommendationRequest

    defaults = dict(
        student_id="STU-2026-0001",
        academic_level=AcademicLevel.beginner,
        learning_style=LearningStyle.visual,
        subject="math",
        weak_concept="fractions",
        past_interactions=[],
    )
    defaults.update(overrides)
    return RecommendationRequest(**defaults)


def test_recommend_cbf_returns_matching_profile_first(sample_resources_df, tmp_path):
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, df, weight_vector = build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )
    request = _make_request()  # math / fractions / beginner / visual->video

    result = recommend_cbf(
        request, sample_resources_df, encoder, matrix, weight_vector, top_n=5
    )

    assert not result.empty
    # RES-001 est un match exact (video/beginner/fractions/math)
    assert result.iloc[0]["resource_id"] == "RES-001"
    assert result.iloc[0]["fallback_used"] == False


def test_recommend_cbf_excludes_completed_resources(sample_resources_df, tmp_path):
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, df, weight_vector = build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )
    request = _make_request(past_interactions=["RES-001"])

    result = recommend_cbf(
        request, sample_resources_df, encoder, matrix, weight_vector, top_n=5
    )
    assert "RES-001" not in result["resource_id"].tolist()


def test_recommend_cbf_excludes_resources_with_unmet_prerequisites(
    sample_resources_df, tmp_path
):
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, df, weight_vector = build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )
    request = _make_request(academic_level="advanced")  # cible RES-003 (prereq RES-001)

    result = recommend_cbf(
        request, sample_resources_df, encoder, matrix, weight_vector, top_n=5
    )
    # RES-003 nécessite RES-001, non complété -> absent des résultats
    assert "RES-003" not in result["resource_id"].tolist()


def test_recommend_cbf_flags_fallback_when_type_differs(sample_resources_df, tmp_path):
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, df, weight_vector = build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )
    # kinesthetic -> "exercise" attendu ; RES-001 (video) sera donc fallback
    request = _make_request(learning_style="kinesthetic")

    result = recommend_cbf(
        request, sample_resources_df, encoder, matrix, weight_vector, top_n=5
    )
    video_rows = result[result["resource_id"] == "RES-001"]
    if not video_rows.empty:
        assert bool(video_rows.iloc[0]["fallback_used"]) is True


def test_recommend_cbf_returns_empty_dataframe_when_no_candidates(tmp_path):
    df = pd.DataFrame(
        [
            {
                "resource_id": "RES-999",
                "title": "Only one",
                "subject": "math",
                "concept": "fractions",
                "difficulty": "beginner",
                "type": "video",
                "estimated_time_min": 10,
                "prerequisites": None,
                "description": "",
            }
        ]
    )
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, _, weight_vector = build_vectorizer(df, save_path=str(save_path))
    request = _make_request(past_interactions=["RES-999"])  # exclut le seul candidat

    result = recommend_cbf(request, df, encoder, matrix, weight_vector, top_n=5)
    assert result.empty


def test_recommend_cbf_recomputes_weight_vector_when_not_provided(
    sample_resources_df, tmp_path
):
    """Chemin de rétro-compatibilité: weight_vector=None recalcule via
    _build_weight_vector(encoder) au lieu de planter."""
    save_path = tmp_path / "encoder.pkl"
    encoder, matrix, df, _ = build_vectorizer(
        sample_resources_df, save_path=str(save_path)
    )
    request = _make_request()

    result = recommend_cbf(
        request, sample_resources_df, encoder, matrix, weight_vector=None, top_n=5
    )
    assert not result.empty
