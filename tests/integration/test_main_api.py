"""
Tests d'intégration de l'API FastAPI (main.py).

Frontière mockée : les 3 fonctions LLM (generate_explanation,
adapt_wording_batch, suggest_study_plan) sont stubbées, car ce sont les
seules dépendances vers un service tiers externe (Groq). Tout le reste
(HybridEngine, CBF, CF, chargement CSV, exécuteurs, endpoints FastAPI)
tourne réellement, sur les vraies données de data/.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def stub_llm_calls(monkeypatch):
    """Remplace les 3 appels LLM par des valeurs déterministes, pour ne
    jamais dépendre du réseau/du provider externe dans ces tests."""
    monkeypatch.setattr(
        main, "generate_explanation", lambda profile, recs: "Explication de test."
    )
    monkeypatch.setattr(
        main,
        "adapt_wording_batch",
        lambda recs, style: [{**r, "adapted_title": r["title"]} for r in recs],
    )
    monkeypatch.setattr(
        main,
        "suggest_study_plan",
        lambda profile, recs: [
            {"resource_id": r["resource_id"], "order": i + 1, "justification": "test"}
            for i, r in enumerate(recs)
        ],
    )
    yield


VALID_PAYLOAD = {
    "student_id": "STU-2026-0001",
    "academic_level": "beginner",
    "learning_style": "visual",
    "subject": "Mathematiques",
    "weak_concept": "fractions",
    "past_interactions": [],
}


# ── POST /recommendations ──────────────────────────────────────────────


def test_recommendations_happy_path(client):
    response = client.post("/recommendations", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["student_id"] == "STU-2026-0001"
    assert body["llm_explanation"] == "Explication de test."
    assert len(body["recommendations"]) > 0
    assert body["metadata"]["latency_ms"] >= 0
    assert "request_id" in body


def test_recommendations_404_when_engine_returns_nothing(client, monkeypatch):
    monkeypatch.setattr(main.engine, "get_recommendations", lambda request: [])
    response = client.post("/recommendations", json=VALID_PAYLOAD)
    assert response.status_code == 404


def test_recommendations_422_on_invalid_payload(client):
    bad_payload = {**VALID_PAYLOAD, "academic_level": "not-a-real-level"}
    response = client.post("/recommendations", json=bad_payload)
    assert response.status_code == 422


def test_recommendations_422_on_missing_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["subject"]
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 422


# ── POST /recommendations/no-llm ───────────────────────────────────────


def test_recommendations_no_llm_does_not_call_llm_functions(client, monkeypatch):
    called = {"n": 0}

    def spy(*args, **kwargs):
        called["n"] += 1
        return "should not be called"

    monkeypatch.setattr(main, "generate_explanation", spy)
    monkeypatch.setattr(main, "adapt_wording_batch", spy)
    monkeypatch.setattr(main, "suggest_study_plan", spy)

    response = client.post("/recommendations/no-llm", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert called["n"] == 0
    body = response.json()
    assert body["llm_explanation"] == "(mode no-llm — pas d'explication générée)"
    assert body["metadata"]["model"] == "none (no-llm benchmark mode)"


def test_recommendations_no_llm_study_plan_matches_recommendation_order(client):
    response = client.post("/recommendations/no-llm", json=VALID_PAYLOAD)
    body = response.json()
    rec_ids = [r["resource_id"] for r in body["recommendations"]]
    plan_ids = [p["resource_id"] for p in body["study_plan"]]
    assert rec_ids == plan_ids


def test_recommendations_no_llm_404_when_engine_empty(client, monkeypatch):
    monkeypatch.setattr(main.engine, "get_recommendations", lambda request: [])
    response = client.post("/recommendations/no-llm", json=VALID_PAYLOAD)
    assert response.status_code == 404


# ── POST /student-profile ──────────────────────────────────────────────


def test_student_profile_validates_and_echoes_payload(client):
    response = client.post("/student-profile", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["profile"]["student_id"] == "STU-2026-0001"


def test_student_profile_422_on_invalid_enum(client):
    bad_payload = {**VALID_PAYLOAD, "learning_style": "telepathic"}
    response = client.post("/student-profile", json=bad_payload)
    assert response.status_code == 422


# ── GET /resources & /resources/all ────────────────────────────────────


def test_get_all_resources_returns_full_catalog(client):
    response = client.get("/resources/all")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) > 0
    assert "resource_id" in body[0]


def test_get_resources_filters_by_subject(client):
    response = client.get("/resources", params={"subject": "Mathematiques"})
    assert response.status_code == 200
    body = response.json()
    assert all(r["subject"] == "Mathematiques" for r in body)


def test_get_resources_filters_by_difficulty(client):
    response = client.get("/resources", params={"difficulty": "beginner"})
    assert response.status_code == 200
    body = response.json()
    assert all(r["difficulty"] == "beginner" for r in body)


def test_get_resources_no_filters_returns_everything(client):
    all_resources = client.get("/resources/all").json()
    filtered = client.get("/resources").json()
    assert len(all_resources) == len(filtered)


# ── Helpers internes (df_to_records, parsing CSV bracket-aware) ───────


def test_df_to_records_converts_nan_to_none_and_numpy_types():
    import numpy as np

    df = pd.DataFrame(
        {
            "a": [1, np.nan],
            "b": [np.int64(3), np.int64(4)],
            "c": [np.float64(1.5), np.float64(2.5)],
            "d": [np.bool_(True), np.bool_(False)],
        }
    )
    records = main.df_to_records(df)

    assert records[1]["a"] is None
    assert isinstance(records[0]["b"], int)
    assert isinstance(records[0]["c"], float)
    assert isinstance(records[0]["d"], bool)


def test_split_csv_line_respecting_brackets_ignores_commas_inside_brackets():
    line = 'STU-001,Alice,[RES-001, RES-002, RES-003]'
    fields = main._split_csv_line_respecting_brackets(line)
    assert fields == ["STU-001", "Alice", "[RES-001, RES-002, RES-003]"]


def test_split_csv_line_respecting_brackets_plain_line():
    fields = main._split_csv_line_respecting_brackets("a,b,c")
    assert fields == ["a", "b", "c"]


def test_read_csv_with_bracket_lists_skips_malformed_rows(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "student_id,name,past_interactions\n"
        "STU-001,Alice,[RES-001, RES-002]\n"
        "STU-002,Bob,TooManyFields,Extra\n"  # ligne malformée -> ignorée
        "STU-003,Chloe,[]\n"
    )
    df = main._read_csv_with_bracket_lists(str(csv_path))
    assert len(df) == 2
    assert list(df["student_id"]) == ["STU-001", "STU-003"]


def test_read_csv_with_bracket_lists_empty_file_returns_empty_df(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    df = main._read_csv_with_bracket_lists(str(csv_path))
    assert df.empty


# ── Caching (lru_cache) ────────────────────────────────────────────────


def test_load_resources_is_cached_across_calls():
    df1 = main.load_resources()
    df2 = main.load_resources()
    assert df1 is df2  # même objet -> pas relu depuis le disque


def test_load_students_is_cached_across_calls():
    df1 = main.load_students()
    df2 = main.load_students()
    assert df1 is df2
