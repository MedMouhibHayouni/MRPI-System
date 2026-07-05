"""
Test d'intégration : HybridEngine complet (CBF + CF réels, construits à
partir des vraies données de data/), sans aucun mock des sous-moteurs.
Vérifie que le pipeline complet (chargement CSV -> encodage CBF ->
KNN/SVD CF -> fusion) produit des recommandations valides de bout en
bout, telles qu'appelées par l'endpoint /recommendations de main.py.
"""
import pytest

from src.hybrid import HybridEngine, get_hybrid_engine


@pytest.fixture(scope="module")
def engine():
    return get_hybrid_engine()


def test_hybrid_engine_returns_recommendations_for_known_student(engine, make_request):
    request = make_request(student_id="STU-2026-0001")
    recommendations = engine.get_recommendations(request)

    assert len(recommendations) > 0
    for rec in recommendations:
        assert rec.relevance_score >= 0
        assert rec.resource_id
        assert rec.description is not None  # FIX régression: description propagée


def test_hybrid_engine_handles_unknown_student_gracefully(engine, make_request):
    """CF renverra 'student_exists: False' pour un étudiant inconnu, mais
    CBF (basé sur le profil déclaré, pas l'historique CF) doit continuer
    à produire des recommandations."""
    request = make_request(student_id="STU-9999-9999")
    recommendations = engine.get_recommendations(request)
    assert isinstance(recommendations, list)


def test_hybrid_engine_pure_cbf_weighting(engine, make_request):
    request = make_request(student_id="STU-2026-0002")
    recommendations = engine.get_recommendations(request, cf_weight=0.0, cbf_weight=1.0)
    assert len(recommendations) > 0


def test_hybrid_engine_respects_top_n(engine, make_request):
    request = make_request(student_id="STU-2026-0003")
    recommendations = engine.get_recommendations(request, top_n=3)
    assert len(recommendations) <= 3


def test_hybrid_engine_excludes_past_interactions(engine, make_request):
    request = make_request(student_id="STU-2026-0004", past_interactions=["RES-001"])
    recommendations = engine.get_recommendations(request)
    ids = {r.resource_id for r in recommendations}
    assert "RES-001" not in ids


def test_hybrid_engine_get_status(engine):
    status = engine.get_status()
    assert "resources_count" in status
    assert "cf_status" in status
    assert status["resources_count"] > 0


def test_get_hybrid_engine_is_a_singleton():
    assert get_hybrid_engine() is get_hybrid_engine()
