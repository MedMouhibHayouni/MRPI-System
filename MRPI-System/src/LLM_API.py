"""
LLM_API.py — Module d'intégration de l'API LLM (CDC 3.4.5)

Aucune modification fonctionnelle ici. Le bug "description toujours vide"
qui affectait adapt_wording() n'était pas dans ce fichier — il venait de
ResourceRecommendation (schemas/response.py) et de hybrid.py qui ne
propageaient jamais ce champ. Les deux sont corrigés en amont ; ce module
reçoit maintenant des dicts avec une vraie clé "description" sans qu'aucune
ligne ici n'ait besoin de changer.
"""

import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from src.cache_utils import compute_content_fingerprint as _fingerprint_for_llm

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

LLM_MODEL = os.getenv("LLM_MODEL")

logger = logging.getLogger(__name__)

CACHE_FILE_PATH = "models/llm_cache.json"

FALLBACK_EXPLANATION = (
    "Ces ressources ont été sélectionnées en fonction de votre profil "
    "d'apprentissage et du concept à renforcer."
)


def _load_llm_cache() -> dict:
    if not os.path.exists(CACHE_FILE_PATH):
        return {}
    try:
        with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_llm_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
    with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _call_llm(prompt: str, max_tokens: int = 300) -> str:
    start = time.monotonic()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=15.0,
    )
    latency = time.monotonic() - start

    usage = getattr(response, "usage", None)
    logger.info(
        "LLM call | model=%s | latency=%.2fs | tokens=%s | prompt=%s...",
        LLM_MODEL,
        latency,
        usage,
        prompt[:80],
    )

    return response.choices[0].message.content


def _build_explanation_prompt(student_profile: dict, recommendations: list[dict]) -> str:
    academic_level = student_profile.get("academic_level", "inconnu")
    learning_style = student_profile.get("learning_style", "inconnu")
    subject = student_profile.get("subject", "inconnu")
    weak_concept = student_profile.get("weak_concept", "inconnu")

    resource_lines = []
    for r in recommendations:
        line = f"- {r.get('title')} (type: {r.get('type')}, difficulté: {r.get('difficulty')})"
        if r.get("fallback_used"):
            line += " [sélection de repli, moins ciblée que d'habitude]"
        resource_lines.append(line)
    resources_block = (
        "\n".join(resource_lines) if resource_lines else "(aucune ressource)"
    )

    return f"""Tu es un assistant pédagogique qui explique à un étudiant pourquoi
certaines ressources lui sont recommandées.

- Ne commence JAMAIS par une salutation ("Bonjour", "Salut", "Hey") ni par une
  formule à la première personne ("je te recommande", "j'ai sélectionné").
  Rédige à la voix impersonnelle, comme si le système décrivait une sélection
  déjà établie — pas un tuteur qui te parle. Formulations attendues :
  "Ces ressources ont été sélectionnées parce que...", "Cette sélection cible
  en priorité...", "Face à [concept], ces ressources permettent de...".
- N'utilise jamais "je", "tu" en position de sujet du verbe principal de la
  première phrase.

Profil de l'étudiant :
- Niveau académique : {academic_level}
- Style d'apprentissage : {learning_style}
- Matière : {subject}
- Concept à renforcer : {weak_concept}

Ressources recommandées :
{resources_block}

Consignes :
- Rédige un paragraphe de 3 à 5 phrases, en français.
- Adapte le ton et le vocabulaire au niveau académique de l'étudiant
  (reste simple pour un niveau débutant, plus technique pour un niveau avancé).
- Explique pourquoi ces ressources sont pertinentes par rapport à son profil
  et son concept à renforcer, et donne une indication sur comment les aborder
  (ex: dans quel ordre, à quoi faire attention).
- Si une ressource est marquée "sélection de repli", ne prétends pas qu'elle
  est parfaitement ciblée — reste honnête, dis par exemple qu'elle a été
  choisie sur des critères plus généraux faute de meilleure correspondance.
- Ne produis rien d'autre que le paragraphe : pas de titre, pas de liste,
  pas de formules d'introduction du type "Voici...".
"""


def _build_wording_prompt(resource: dict, learning_style: str) -> str:
    title = resource.get("title", "")
    description = resource.get("description", "")

    return f"""Tu reformules le titre et la description d'une ressource pédagogique
pour les rendre plus accessibles à un étudiant ayant un style d'apprentissage
"{learning_style}" (visual / auditory / kinesthetic / textual).

Ressource originale :
- Titre : {title}
- Description : {description}

Consignes :
- Adapte le vocabulaire et la formulation au style d'apprentissage indiqué
  (ex: pour "visual", évoque des repères visuels/schémas ; pour "kinesthetic",
  insiste sur la pratique/l'action ; pour "auditory", évoque l'écoute/l'explication
  orale ; pour "textual", privilégie une formulation claire et structurée).
- Ne change pas le sens ni le sujet de la ressource, seulement la formulation.
- Reste concis : titre court, description en une phrase.
- Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, sans
  balises markdown, au format exact suivant :
{{"adapted_title": "...", "adapted_description": "..."}}
"""


def _build_study_plan_prompt(student_profile: dict, recommendations: list[dict]) -> str:
    academic_level = student_profile.get("academic_level", "inconnu")

    resource_lines = []
    for r in recommendations:
        resource_lines.append(
            f"- resource_id: {r.get('resource_id')}, titre: {r.get('title')}, "
            f"difficulté: {r.get('difficulty')}, "
            f"durée estimée: {r.get('estimated_time_min')} min"
        )
    resources_block = (
        "\n".join(resource_lines) if resource_lines else "(aucune ressource)"
    )

    return f"""Tu proposes un ordre de consommation optimal pour un ensemble de
ressources pédagogiques déjà sélectionnées pour un étudiant.

Niveau académique de l'étudiant : {academic_level}

Ressources à ordonner :
{resources_block}

Consignes :
- Propose un ordre séquentiel logique (ex: du plus simple/prérequis au plus
  avancé, ou selon une progression pédagogique cohérente).
- Justifie chaque position en 8 mots maximum. Pas de phrase complète, un
  fragment concis.
- Utilise EXCLUSIVEMENT les resource_id fournis ci-dessus, n'en invente aucun,
  et inclus-les tous exactement une fois chacun.
- Réponds en JSON COMPACT (pas d'indentation, pas de retours à la ligne
  superflus), sans texte autour, sans balises markdown, au format exact
  suivant :
[{{"resource_id":"...","order":1,"justification":"..."}}]
"""


def _strip_json_fences(raw_text: str) -> str:
    if raw_text is None:
        raise ValueError("Réponse LLM vide (None) — impossible de parser.")

    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def _parse_wording_response(raw_text: str) -> dict:
    text = _strip_json_fences(raw_text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide dans la réponse wording: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Réponse wording attendue comme objet JSON, reçu: {type(parsed).__name__}"
        )

    required_keys = {"adapted_title", "adapted_description"}
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"Clés manquantes dans la réponse wording: {missing}")

    if not isinstance(parsed["adapted_title"], str) or not isinstance(
        parsed["adapted_description"], str
    ):
        raise ValueError("adapted_title/adapted_description doivent être des strings.")

    return {
        "adapted_title": parsed["adapted_title"],
        "adapted_description": parsed["adapted_description"],
    }


def _parse_study_plan_response(raw_text: str) -> list[dict]:
    text = _strip_json_fences(raw_text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide dans la réponse study plan: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(
            f"Réponse study plan attendue comme tableau JSON, reçu: {type(parsed).__name__}"
        )

    if not parsed:
        raise ValueError("Réponse study plan vide — déclenche le fallback.")

    required_keys = {"resource_id", "order", "justification"}
    seen_ids = set()

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Élément {i} du plan n'est pas un objet: {item!r}")

        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Élément {i} du plan — clés manquantes: {missing}")

        if not isinstance(item["order"], int):
            raise ValueError(
                f"Élément {i} du plan — 'order' doit être un int, "
                f"reçu: {type(item['order']).__name__}"
            )

        seen_ids.add(item["resource_id"])

    if len(seen_ids) != len(parsed):
        raise ValueError("resource_id dupliqués dans la réponse study plan.")

    return parsed


def generate_explanation(student_profile: dict, recommendations: list[dict]) -> str:
    fingerprint = _fingerprint_for_llm("explanation", student_profile, recommendations)
    cache = _load_llm_cache()

    if fingerprint in cache:
        return cache[fingerprint]["result"]

    prompt = _build_explanation_prompt(student_profile, recommendations)
    try:
        result = _call_llm(prompt)
    except Exception as e:
        logger.warning(f"⚠️ LLM explanation failed, fallback used: {e}")
        result = FALLBACK_EXPLANATION

    cache[fingerprint] = {
        "type": "explanation",
        "student_id": student_profile.get("student_id"),
        "resource_ids": [r.get("resource_id") for r in recommendations],
        "result": result,
    }
    _save_llm_cache(cache)
    return result


def adapt_wording(resource: dict, learning_style: str) -> dict:
    fingerprint = _fingerprint_for_llm("wording", resource, learning_style)
    cache = _load_llm_cache()

    if fingerprint in cache:
        cached = cache[fingerprint]["result"]
        return {**resource, **cached}

    prompt = _build_wording_prompt(resource, learning_style)
    try:
        raw = _call_llm(prompt)
        parsed = _parse_wording_response(raw)
        if (
            not parsed["adapted_title"].strip()
            or not parsed["adapted_description"].strip()
        ):
            raise ValueError(
                "adapted_title/adapted_description vides — réponse LLM inexploitable."
            )
    except Exception as e:
        logger.warning(f"⚠️ LLM wording failed, fallback used: {e}", exc_info=True)
        parsed = {
            "adapted_title": resource.get("title", ""),
            "adapted_description": resource.get("description", ""),
        }

    cache[fingerprint] = {
        "type": "wording",
        "resource_id": resource.get("resource_id"),
        "learning_style": learning_style,
        "result": parsed,
    }
    _save_llm_cache(cache)
    return {**resource, **parsed}


def adapt_wording_batch(recommendations: list[dict], learning_style: str) -> list[dict]:
    results = []
    for i, r in enumerate(recommendations):
        results.append(adapt_wording(r, learning_style))
        if i < len(recommendations) - 1:
            time.sleep(0.5)
    return results


def suggest_study_plan(student_profile: dict, recommendations: list[dict]) -> list[dict]:
    fingerprint = _fingerprint_for_llm("study-plan", student_profile, recommendations)
    cache = _load_llm_cache()

    if fingerprint in cache:
        return cache[fingerprint]["result"]

    prompt = _build_study_plan_prompt(student_profile, recommendations)
    try:
        raw = _call_llm(prompt, max_tokens=min(4000, 150 * len(recommendations)))
        result = _parse_study_plan_response(raw)

        cache[fingerprint] = {
            "type": "study_plan",
            "student_id": student_profile.get("student_id"),
            "resource_ids": [r.get("resource_id") for r in recommendations],
            "result": result,
        }
        _save_llm_cache(cache)
        return result

    except Exception as e:
        logger.warning(f"⚠️ LLM study plan failed, fallback used: {e}", exc_info=True)
        result = [
            {
                "resource_id": r.get("resource_id"),
                "order": i + 1,
                "justification": "Ordre basé sur le score de recommandation (LLM indisponible ou réponse invalide).",
            }
            for i, r in enumerate(recommendations)
        ]
        return result
