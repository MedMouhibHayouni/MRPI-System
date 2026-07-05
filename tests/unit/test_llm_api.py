import json
from types import SimpleNamespace

import pytest

from src import LLM_API


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Chaque test repart d'un cache LLM vide et isolé sur disque."""
    monkeypatch.setattr(LLM_API, "CACHE_FILE_PATH", str(tmp_path / "llm_cache.json"))
    monkeypatch.setattr(LLM_API, "_LLM_CACHE", None)
    yield


def _fake_llm_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage={"total_tokens": 42},
    )


# ── cache helpers ──────────────────────────────────────────────────────


def test_cache_roundtrip_store_and_get(tmp_path):
    fp = "abc123"
    assert LLM_API._get_cached_result(fp) is None
    LLM_API._store_cached_result(fp, {"type": "explanation", "result": "hello"})
    assert LLM_API._get_cached_result(fp) == "hello"


def test_cache_persists_across_reloads(tmp_path, monkeypatch):
    fp = "xyz"
    LLM_API._store_cached_result(fp, {"type": "wording", "result": {"a": 1}})
    monkeypatch.setattr(LLM_API, "_LLM_CACHE", None)  # force un rechargement disque
    assert LLM_API._get_cached_result(fp) == {"a": 1}


def test_load_llm_cache_handles_corrupt_json(tmp_path, monkeypatch):
    path = tmp_path / "llm_cache.json"
    path.write_text("{not valid json")
    monkeypatch.setattr(LLM_API, "CACHE_FILE_PATH", str(path))
    monkeypatch.setattr(LLM_API, "_LLM_CACHE", None)
    assert LLM_API._load_llm_cache() == {}


# ── _strip_json_fences ─────────────────────────────────────────────────


def test_strip_json_fences_removes_markdown_fences():
    raw = '```json\n{"a": 1}\n```'
    assert LLM_API._strip_json_fences(raw) == '{"a": 1}'


def test_strip_json_fences_passthrough_when_no_fences():
    raw = '{"a": 1}'
    assert LLM_API._strip_json_fences(raw) == raw


def test_strip_json_fences_raises_on_none():
    with pytest.raises(ValueError):
        LLM_API._strip_json_fences(None)


# ── _parse_wording_response ────────────────────────────────────────────


def test_parse_wording_response_valid():
    raw = '{"adapted_title": "T", "adapted_description": "D"}'
    result = LLM_API._parse_wording_response(raw)
    assert result == {"adapted_title": "T", "adapted_description": "D"}


def test_parse_wording_response_invalid_json_raises():
    with pytest.raises(ValueError):
        LLM_API._parse_wording_response("not json")


def test_parse_wording_response_missing_keys_raises():
    with pytest.raises(ValueError, match="Clés manquantes"):
        LLM_API._parse_wording_response('{"adapted_title": "T"}')


def test_parse_wording_response_not_a_dict_raises():
    with pytest.raises(ValueError):
        LLM_API._parse_wording_response("[1, 2, 3]")


# ── _parse_study_plan_response ─────────────────────────────────────────


def test_parse_study_plan_response_valid():
    raw = '[{"resource_id": "RES-001", "order": 1, "justification": "j"}]'
    result = LLM_API._parse_study_plan_response(raw)
    assert result[0]["resource_id"] == "RES-001"


def test_parse_study_plan_response_empty_list_raises():
    with pytest.raises(ValueError, match="vide"):
        LLM_API._parse_study_plan_response("[]")


def test_parse_study_plan_response_duplicate_ids_raises():
    raw = (
        '[{"resource_id": "RES-001", "order": 1, "justification": "a"},'
        '{"resource_id": "RES-001", "order": 2, "justification": "b"}]'
    )
    with pytest.raises(ValueError, match="dupliqués"):
        LLM_API._parse_study_plan_response(raw)


def test_parse_study_plan_response_bad_order_type_raises():
    raw = '[{"resource_id": "RES-001", "order": "one", "justification": "a"}]'
    with pytest.raises(ValueError, match="order"):
        LLM_API._parse_study_plan_response(raw)


def test_parse_study_plan_response_missing_keys_raises():
    raw = '[{"resource_id": "RES-001"}]'
    with pytest.raises(ValueError, match="manquantes"):
        LLM_API._parse_study_plan_response(raw)


def test_parse_study_plan_response_not_a_list_raises():
    with pytest.raises(ValueError):
        LLM_API._parse_study_plan_response('{"a": 1}')


# ── generate_explanation ────────────────────────────────────────────────


def test_generate_explanation_success(monkeypatch):
    monkeypatch.setattr(LLM_API, "_call_llm", lambda prompt, max_tokens=300: "Explication générée.")
    result = LLM_API.generate_explanation(
        {"student_id": "STU-001", "academic_level": "beginner"},
        [{"resource_id": "RES-001", "title": "T", "type": "video", "difficulty": "beginner"}],
    )
    assert result == "Explication générée."


def test_generate_explanation_uses_cache_on_second_call(monkeypatch):
    calls = {"n": 0}

    def fake_call(prompt, max_tokens=300):
        calls["n"] += 1
        return "Résultat"

    monkeypatch.setattr(LLM_API, "_call_llm", fake_call)
    profile = {"student_id": "STU-001"}
    recs = [{"resource_id": "RES-001"}]

    LLM_API.generate_explanation(profile, recs)
    LLM_API.generate_explanation(profile, recs)

    assert calls["n"] == 1  # deuxième appel servi depuis le cache


def test_generate_explanation_falls_back_on_llm_error(monkeypatch):
    def raise_error(prompt, max_tokens=300):
        raise RuntimeError("network down")

    monkeypatch.setattr(LLM_API, "_call_llm", raise_error)
    result = LLM_API.generate_explanation({"student_id": "STU-001"}, [])
    assert result == LLM_API.FALLBACK_EXPLANATION


# ── adapt_wording / adapt_wording_batch ────────────────────────────────


def test_adapt_wording_success(monkeypatch):
    monkeypatch.setattr(
        LLM_API,
        "_call_llm",
        lambda prompt, max_tokens=300: json.dumps(
            {"adapted_title": "Titre adapté", "adapted_description": "Desc adaptée"}
        ),
    )
    resource = {"resource_id": "RES-001", "title": "T", "description": "D"}
    result = LLM_API.adapt_wording(resource, "visual")

    assert result["adapted_title"] == "Titre adapté"
    assert result["resource_id"] == "RES-001"  # champs originaux préservés


def test_adapt_wording_falls_back_when_response_invalid(monkeypatch):
    monkeypatch.setattr(LLM_API, "_call_llm", lambda prompt, max_tokens=300: "not json")
    resource = {"resource_id": "RES-001", "title": "Original title", "description": "Original desc"}
    result = LLM_API.adapt_wording(resource, "auditory")

    assert result["adapted_title"] == "Original title"
    assert result["adapted_description"] == "Original desc"


def test_adapt_wording_falls_back_when_fields_empty(monkeypatch):
    monkeypatch.setattr(
        LLM_API,
        "_call_llm",
        lambda prompt, max_tokens=300: json.dumps(
            {"adapted_title": "  ", "adapted_description": "D"}
        ),
    )
    resource = {"resource_id": "RES-002", "title": "Fallback title", "description": "Fallback desc"}
    result = LLM_API.adapt_wording(resource, "textual")
    assert result["adapted_title"] == "Fallback title"


def test_adapt_wording_batch_processes_all_resources(monkeypatch):
    resources = [{"resource_id": f"RES-{i:03d}", "title": "T", "description": "D"} for i in range(3)]

    monkeypatch.setattr(
        LLM_API,
        "_call_llm",
        lambda prompt, max_tokens=300: json.dumps(
            [
                {
                    "resource_id": resource["resource_id"],
                    "adapted_title": "AT",
                    "adapted_description": "AD",
                }
                for resource in resources
            ]
        ),
    )
    results = LLM_API.adapt_wording_batch(resources, "kinesthetic")

    assert len(results) == 3
    assert all(r["adapted_title"] == "AT" for r in results)


# ── suggest_study_plan ──────────────────────────────────────────────────


def test_suggest_study_plan_success(monkeypatch):
    monkeypatch.setattr(
        LLM_API,
        "_call_llm",
        lambda prompt, max_tokens=300: json.dumps(
            [{"resource_id": "RES-001", "order": 1, "justification": "base"}]
        ),
    )
    result = LLM_API.suggest_study_plan(
        {"student_id": "STU-001", "academic_level": "beginner"},
        [{"resource_id": "RES-001", "title": "T", "difficulty": "beginner", "estimated_time_min": 10}],
    )
    assert result[0]["resource_id"] == "RES-001"


def test_suggest_study_plan_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setattr(LLM_API, "_call_llm", lambda prompt, max_tokens=300: "garbage")
    recs = [
        {"resource_id": "RES-001", "title": "T1"},
        {"resource_id": "RES-002", "title": "T2"},
    ]
    result = LLM_API.suggest_study_plan({"student_id": "STU-001"}, recs)

    assert [r["resource_id"] for r in result] == ["RES-001", "RES-002"]
    assert result[0]["order"] == 1
    assert result[1]["order"] == 2


def test_suggest_study_plan_falls_back_on_llm_exception(monkeypatch):
    def raise_error(prompt, max_tokens=300):
        raise RuntimeError("timeout")

    monkeypatch.setattr(LLM_API, "_call_llm", raise_error)
    recs = [{"resource_id": "RES-001", "title": "T1"}]
    result = LLM_API.suggest_study_plan({"student_id": "STU-001"}, recs)
    assert result[0]["resource_id"] == "RES-001"


# ── _call_llm (test d'intégration légère avec le SDK OpenAI mocké) ─────


def test_call_llm_returns_message_content(monkeypatch):
    fake_response = _fake_llm_response("contenu du modèle")
    monkeypatch.setattr(
        LLM_API.client.chat.completions, "create", lambda **kwargs: fake_response
    )
    result = LLM_API._call_llm("un prompt", max_tokens=100)
    assert result == "contenu du modèle"
