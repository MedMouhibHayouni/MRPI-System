"""
LLM_API.py — Module d'intégration de l'API LLM (CDC 3.4.5)

CHANGEMENT DE PERFORMANCE (voir diagnostic Locust) :
adapt_wording_batch() ne boucle plus sur N appels séquentiels à
adapt_wording(). Elle envoie UN SEUL appel LLM contenant toutes les
ressources à reformuler, avec les resource_id en clé de correspondance.
Ça réduit le nombre d'appels LLM par requête /recommendations de
~(2 + N) à 3, quel que soit top_n. C'est un changement structurel du
nombre d'appels, pas une parallélisation — la parallélisation aurait
réduit la latence perçue mais aggravé la pression sur le rate limit
Groq (6000 TPM en tier gratuit), donc n'a pas été retenue.
"""

import json
import logging
import os
import time
from threading import RLock

from dotenv import load_dotenv
from openai import OpenAI

from src.cache_utils import compute_content_fingerprint as _fingerprint_for_llm

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    max_retries=0,
)

LLM_MODEL = os.getenv("LLM_MODEL")

logger = logging.getLogger(__name__)

CACHE_FILE_PATH = "models/llm_cache.json"
_CACHE_LOCK = RLock()
_LLM_CACHE: dict | None = None

FALLBACK_EXPLANATION = (
    "Ces ressources ont été sélectionnées en fonction de votre profil "
    "d'apprentissage et du concept à renforcer."
)


def _load_llm_cache() -> dict:
    """
    Charge le cache LLM depuis disque (models/llm_cache.json) au premier
    appel du process, puis réutilise la copie en mémoire (_LLM_CACHE)
    pour tous les appels suivants — évite de relire/reparser le fichier
    JSON à chaque requête de recommandation.

    Thread-safe via _CACHE_LOCK (RLock) : plusieurs requêtes concurrentes
    (llm_executor, jusqu'à 100 threads) peuvent appeler cette fonction
    simultanément sans corruption de l'état partagé _LLM_CACHE.

    Si le fichier n'existe pas encore, ou si son contenu est illisible
    (JSON invalide, erreur disque), initialise un cache vide plutôt que
    de faire planter le process — le cache LLM est une optimisation, sa
    perte ne doit jamais bloquer une recommandation.

    Returns
    -------
    dict
        Cache complet {fingerprint: entry}.
    """
    global _LLM_CACHE
    with _CACHE_LOCK:
        if _LLM_CACHE is not None:
            return _LLM_CACHE
        if not os.path.exists(CACHE_FILE_PATH):
            _LLM_CACHE = {}
            return _LLM_CACHE
        try:
            with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                _LLM_CACHE = json.load(f)
        except (json.JSONDecodeError, OSError):
            _LLM_CACHE = {}
        return _LLM_CACHE


def _save_llm_cache(cache: dict) -> None:
    """
    Persiste l'intégralité du cache LLM sur disque (écrase le fichier
    existant), et met à jour la copie en mémoire _LLM_CACHE en même
    temps, sous le même verrou que _load_llm_cache() pour éviter les
    races entre lecture et écriture concurrentes.

    Crée le dossier parent (models/) s'il n'existe pas encore.

    Parameters
    ----------
    cache : dict
        Cache complet à écrire (remplace entièrement le contenu
        précédent, pas un merge incrémental).
    """
    global _LLM_CACHE
    with _CACHE_LOCK:
        _LLM_CACHE = cache
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


def _get_cached_result(fingerprint: str):
    """
    Recherche une entrée de cache par empreinte de contenu
    (compute_content_fingerprint, voir cache_utils.py) et retourne
    uniquement son résultat utile (entry["result"]), sans les métadonnées
    de contexte (type, student_id, resource_ids) stockées à côté.

    Parameters
    ----------
    fingerprint : str
        Empreinte calculée sur les arguments de l'appel LLM concerné.

    Returns
    -------
    Any | None
        Le résultat mis en cache, ou None si aucune entrée ne
        correspond à ce fingerprint.
    """
    cache = _load_llm_cache()
    entry = cache.get(fingerprint)
    if entry is None:
        return None
    return entry["result"]


def _store_cached_result(fingerprint: str, entry: dict):
    """
    Enregistre une nouvelle entrée dans le cache LLM sous la clé
    fingerprint, puis persiste immédiatement tout le cache sur disque
    (pas de write-behind différé — chaque nouvel appel LLM réussi ou
    tombé en fallback est immédiatement durable).

    Parameters
    ----------
    fingerprint : str
    entry : dict
        Métadonnées + résultat à stocker (structure libre selon
        l'appelant : type, student_id/resource_id, result...).
    """
    cache = _load_llm_cache()
    with _CACHE_LOCK:
        cache[fingerprint] = entry
        _save_llm_cache(cache)


def _call_llm(prompt: str, max_tokens: int = 300) -> str:
    """
    Effectue l'appel réseau brut vers l'API LLM (compatible OpenAI,
    provider configuré via LLM_BASE_URL/LLM_API_KEY, modèle
    llama-3.1-8b-instant via Groq en configuration actuelle), avec un
    timeout de 15 secondes et sans retry automatique (max_retries=0 au
    niveau du client — les retries sont gérés explicitement en amont si
    nécessaire, pas silencieusement par la lib cliente, pour éviter un
    effet cascade de latence sous rate limit, voir diagnostic Locust).

    Journalise la latence et l'usage de tokens de chaque appel pour
    permettre un diagnostic de performance a posteriori.

    Parameters
    ----------
    prompt : str
        Prompt complet à envoyer au LLM.
    max_tokens : int
        Limite de tokens en sortie (défaut 300).

    Returns
    -------
    str
        Contenu textuel brut de la réponse du LLM (non parsé).

    Raises
    ------
    Exception
        Toute exception du client OpenAI (timeout, erreur HTTP, rate
        limit) remonte telle quelle — la gestion du fallback est à la
        charge de l'appelant (generate_explanation, adapt_wording,
        suggest_study_plan), pas de cette fonction bas niveau.
    """
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


def _build_explanation_prompt(
    student_profile: dict, recommendations: list[dict]
) -> str:
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


def _build_wording_batch_prompt(
    recommendations: list[dict], learning_style: str
) -> str:
    """
    Construit le prompt destiné à adapt_wording_batch() : demande au LLM
    de reformuler le titre et la description de PLUSIEURS ressources en
    un seul appel, chaque ressource étant identifiée par son
    resource_id, avec instruction explicite de traiter chaque ressource
    listée sans en omettre ni en inventer, et de répondre en JSON
    compact (tableau d'objets, un objet par ressource) sans texte ni
    markdown autour.

    Ce regroupement en un seul prompt est ce qui permet de réduire le
    nombre d'appels LLM de ~(2 + N) à 3 par requête /recommendations,
    quel que soit le nombre de recommandations N (voir docstring
    module).

    Parameters
    ----------
    recommendations : list[dict]
        Ressources à reformuler, chacune avec resource_id, title,
        description.
    learning_style : str

    Returns
    -------
    str
        Prompt complet.
    """
    resource_lines = []

    for r in recommendations:
        resource_lines.append(
            f"- resource_id: {r.get('resource_id')}, titre: {r.get('title', '')}, "
            f"description: {r.get('description', '')}"
        )
    resources_block = (
        "\n".join(resource_lines) if resource_lines else "(aucune ressource)"
    )

    return f"""Tu reformules le titre et la description de plusieurs ressources
pédagogiques pour les rendre plus accessibles à un étudiant ayant un style
d'apprentissage "{learning_style}" (visual / auditory / kinesthetic / textual).

Ressources originales :
{resources_block}

Consignes :
- Adapte le vocabulaire et la formulation au style d'apprentissage indiqué
  (ex: pour "visual", évoque des repères visuels/schémas ; pour "kinesthetic",
  insiste sur la pratique/l'action ; pour "auditory", évoque l'écoute/l'explication
  orale ; pour "textual", privilégie une formulation claire et structurée).
- Ne change pas le sens ni le sujet de chaque ressource, seulement la formulation.
- Reste concis : titre court, description en une phrase, PAR ressource.
- Traite CHAQUE ressource listée ci-dessus, indépendamment des autres.
- Utilise EXCLUSIVEMENT les resource_id fournis ci-dessus, n'en invente aucun,
  et inclus-les tous exactement une fois chacun.
- Réponds en JSON COMPACT (pas d'indentation, pas de retours à la ligne
  superflus), sans texte autour, sans balises markdown, au format exact
  suivant :
[{{"resource_id":"...","adapted_title":"...","adapted_description":"..."}}]
"""


def _build_study_plan_prompt(student_profile: dict, recommendations: list[dict]) -> str:
    """
    Construit le prompt destiné à suggest_study_plan() : demande au LLM
    de proposer un ordre de consommation séquentiel et pédagogiquement
    cohérent pour un ensemble de ressources déjà sélectionnées, avec une
    justification courte (8 mots max, fragment) par position, en
    réponse JSON compact — tableau d'objets {resource_id, order,
    justification}, sans resource_id inventé ni omis.

    Parameters
    ----------
    student_profile : dict
        Utilisé uniquement pour academic_level ici (contrairement au
        prompt d'explication, qui utilise tout le profil).
    recommendations : list[dict]
        Ressources à ordonner (title, difficulty, estimated_time_min).

    Returns
    -------
    str
        Prompt complet.
    """
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
    """
    Retire les balises markdown de bloc de code (```...```) qu'un LLM
    ajoute parfois autour d'une réponse JSON malgré une instruction
    explicite de ne pas le faire — supprime la première et la dernière
    ligne si elles commencent par ```.

    Parameters
    ----------
    raw_text : str
        Réponse brute du LLM.

    Returns
    -------
    str
        Texte nettoyé, prêt pour un parsing JSON.

    Raises
    ------
    ValueError
        Si raw_text est None (réponse LLM vide/absente).
    """
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


def _parse_leading_json(text: str):
    """
    Parse uniquement le premier objet ou tableau JSON valide en tête de
    la chaîne, en ignorant tout ce qui suit (texte parasite, répétitions
    que le LLM ajoute parfois malgré l'instruction "réponds
    UNIQUEMENT avec du JSON").

    Utilise json.JSONDecoder().raw_decode() plutôt que json.loads(),
    qui lèverait une erreur "Extra data" sur toute traîne après le
    premier objet JSON complet — raw_decode() s'arrête dès que le
    premier objet est complet.

    Parameters
    ----------
    text : str
        Texte déjà nettoyé des balises markdown (voir _strip_json_fences).

    Returns
    -------
    Any
        Objet Python désérialisé (dict ou list selon le contenu).

    Raises
    ------
    json.JSONDecodeError
        Si aucun JSON valide n'est trouvé en tête de chaîne.
    """
    decoder = json.JSONDecoder()
    text = text.strip()
    parsed, _end_index = decoder.raw_decode(text)
    return parsed


def _parse_wording_batch_response(raw_text: str, expected_ids: set) -> dict:
    """
    Parse et valide strictement la réponse JSON batchée de
    adapt_wording_batch() : doit être un tableau d'objets, chacun
    contenant resource_id, adapted_title, adapted_description (strings).

    Vérifie en plus que TOUS les resource_id attendus (expected_ids) sont
    bien couverts par la réponse — si le LLM en a omis, lève une
    exception plutôt que de renvoyer un résultat partiel silencieusement
    incomplet (l'appelant adapt_wording_batch() gère alors le fallback
    pour l'ensemble du lot).

    Parameters
    ----------
    raw_text : str
        Réponse brute du LLM.
    expected_ids : set
        Ensemble des resource_id qui devaient figurer dans la réponse.

    Returns
    -------
    dict
        {resource_id: {"adapted_title": str, "adapted_description": str}}.

    Raises
    ------
    ValueError
        JSON invalide, type incorrect, élément malformé, clés
        manquantes, valeurs non-string, ou resource_id manquants par
        rapport à expected_ids.
    """

    text = _strip_json_fences(raw_text)

    try:
        parsed = _parse_leading_json(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide dans la réponse wording batch: {e}") from e

    if not isinstance(parsed, list):
        raise ValueError(
            f"Réponse wording batch attendue comme tableau JSON, reçu: {type(parsed).__name__}"
        )

    if not parsed:
        raise ValueError("Réponse wording batch vide — déclenche le fallback.")

    required_keys = {"resource_id", "adapted_title", "adapted_description"}
    by_id: dict = {}

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(
                f"Élément {i} du batch wording n'est pas un objet: {item!r}"
            )

        missing = required_keys - item.keys()
        if missing:
            raise ValueError(
                f"Élément {i} du batch wording — clés manquantes: {missing}"
            )

        if not isinstance(item["adapted_title"], str) or not isinstance(
            item["adapted_description"], str
        ):
            raise ValueError(
                f"Élément {i} du batch wording — adapted_title/description doivent "
                f"être des strings."
            )

        if not item["adapted_title"].strip() or not item["adapted_description"].strip():
            raise ValueError(
                f"Élément {i} du batch wording — champs vides, réponse inexploitable."
            )

        by_id[item["resource_id"]] = {
            "adapted_title": item["adapted_title"],
            "adapted_description": item["adapted_description"],
        }

    missing_ids = expected_ids - by_id.keys()
    if missing_ids:
        raise ValueError(
            f"resource_id manquants dans la réponse wording batch: {missing_ids}"
        )

    return by_id


def _parse_wording_response(raw_text: str) -> dict:
    """
    Parse et valide la réponse JSON d'une reformulation pour une seule
    ressource.

    Returns
    -------
    dict
        {"adapted_title": str, "adapted_description": str}.
    """
    text = _strip_json_fences(raw_text)

    try:
        parsed = _parse_leading_json(text)
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

    if not parsed["adapted_title"].strip() or not parsed["adapted_description"].strip():
        raise ValueError("Champs wording vides, réponse inexploitable.")

    return {
        "adapted_title": parsed["adapted_title"],
        "adapted_description": parsed["adapted_description"],
    }


def _parse_study_plan_response(raw_text: str) -> list[dict]:
    """
    Parse et valide strictement la réponse JSON de
    suggest_study_plan() : doit être un tableau non vide d'objets,
    chacun contenant resource_id, order (int), justification, sans
    resource_id dupliqué entre les éléments.

    Parameters
    ----------
    raw_text : str
        Réponse brute du LLM.

    Returns
    -------
    list[dict]
        Plan d'étude validé, un dict par ressource ordonnée.

    Raises
    ------
    ValueError
        JSON invalide, pas un tableau, tableau vide, élément malformé,
        clés manquantes, order non-entier, ou resource_id dupliqués.
    """
    text = _strip_json_fences(raw_text)

    try:
        parsed = _parse_leading_json(text)
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
    """
    Génère une explication personnalisée en langage naturel pour un jeu
    de recommandations (CDC 3.4.5, cas d'usage 1), avec cache et
    fallback.

    Vérifie d'abord le cache (fingerprint basé sur student_profile +
    recommendations) ; si absent, construit le prompt, appelle le LLM.
    En cas d'échec (timeout, erreur réseau, réponse invalide), retombe
    sur FALLBACK_EXPLANATION (texte statique générique) plutôt que de
    faire échouer toute la requête /recommendations — le CDC exige un
    mécanisme de fallback en cas d'indisponibilité de l'API LLM (CDC
    3.4.5). Le résultat (LLM ou fallback) est mis en cache dans les deux
    cas.

    Parameters
    ----------
    student_profile : dict
    recommendations : list[dict]

    Returns
    -------
    str
        Explication générée par le LLM, ou FALLBACK_EXPLANATION en cas
        d'éche"""
    fingerprint = _fingerprint_for_llm("explanation", student_profile, recommendations)
    cached = _get_cached_result(fingerprint)
    if cached is not None:
        return cached

    prompt = _build_explanation_prompt(student_profile, recommendations)
    try:
        result = _call_llm(prompt)
    except Exception as e:
        logger.warning(f"⚠️ LLM explanation failed, fallback used: {e}")
        result = FALLBACK_EXPLANATION

    _store_cached_result(
        fingerprint,
        {
            "type": "explanation",
            "student_id": student_profile.get("student_id"),
            "resource_ids": [r.get("resource_id") for r in recommendations],
            "result": result,
        },
    )
    return result


def adapt_wording_batch(recommendations: list[dict], learning_style: str) -> list[dict]:
    """
    Reformule le titre et la description de TOUTES les recommandations
    en UN SEUL appel LLM (CDC 3.4.5, cas d'usage 2 — optimisation
    perf, voir docstring module), avec cache par-ressource conservé :
    seules les ressources non déjà en cache pour ce fingerprint partent
    dans l'appel batché, les autres sont servies directement depuis le
    cache sans appel réseau.

    En cas d'échec du batch entier (timeout, JSON invalide, resource_id
    manquants), CHAQUE ressource non résolue individuellement retombe
    sur son titre/description d'origine — comportement de fallback
    identique à adapt_wording(), mais appliqué à tout le lot en échec
    plutôt qu'à une ressource isolée. Aucune exception ne remonte à
    l'appelant dans tous les cas.

    Parameters
    ----------
    recommendations : list[dict]
        Ressources à reformuler, chacune avec resource_id, title,
        description.
    learning_style : str

    Returns
    -------
    list[dict]
        Une entrée par recommandation d'entrée, dans le même ordre,
        fusionnée avec adapted_title/adapted_description (LLM ou
        fallback selon la ressource).
    """
    results: list[dict | None] = [None] * len(recommendations)
    to_call = []

    for i, r in enumerate(recommendations):
        fingerprint = _fingerprint_for_llm("wording", r, learning_style)
        cached = _get_cached_result(fingerprint)
        if cached is not None:
            results[i] = {**r, **cached}
        else:
            to_call.append((i, r, fingerprint))

    if not to_call:
        return results

    expected_ids = {r.get("resource_id") for _, r, _ in to_call}
    prompt = _build_wording_batch_prompt([r for _, r, _ in to_call], learning_style)

    try:
        raw = _call_llm(prompt, max_tokens=min(4000, 200 * len(to_call)))
        by_id = _parse_wording_batch_response(raw, expected_ids)

        for i, r, fingerprint in to_call:
            parsed = by_id[r.get("resource_id")]
            _store_cached_result(
                fingerprint,
                {
                    "type": "wording",
                    "resource_id": r.get("resource_id"),
                    "learning_style": learning_style,
                    "result": parsed,
                },
            )
            results[i] = {**r, **parsed}

    except Exception as e:
        logger.warning(f"⚠️ LLM wording batch failed, fallback used: {e}", exc_info=True)
        for i, r, fingerprint in to_call:
            fallback = {
                "adapted_title": r.get("title", ""),
                "adapted_description": r.get("description", ""),
            }
            _store_cached_result(
                fingerprint,
                {
                    "type": "wording",
                    "resource_id": r.get("resource_id"),
                    "learning_style": learning_style,
                    "result": fallback,
                },
            )
            results[i] = {**r, **fallback}

    return results


def adapt_wording(resource: dict, learning_style: str) -> dict:
    """
    Reformule le titre et la description d'une seule ressource.

    Cette fonction conserve l'API unitaire et délègue au même cache que
    le chemin batch.
    """
    fingerprint = _fingerprint_for_llm("wording", resource, learning_style)
    cached = _get_cached_result(fingerprint)
    if cached is not None:
        return {**resource, **cached}

    prompt = _build_wording_batch_prompt([resource], learning_style)
    try:
        raw = _call_llm(prompt)
        parsed = _parse_wording_response(raw)
    except Exception as e:
        logger.warning(f"⚠️ LLM wording failed, fallback used: {e}", exc_info=True)
        parsed = {
            "adapted_title": resource.get("title", ""),
            "adapted_description": resource.get("description", ""),
        }

    _store_cached_result(
        fingerprint,
        {
            "type": "wording",
            "resource_id": resource.get("resource_id"),
            "learning_style": learning_style,
            "result": parsed,
        },
    )
    return {**resource, **parsed}


def suggest_study_plan(
    student_profile: dict, recommendations: list[dict]
) -> list[dict]:
    """
    Reformule le titre et la description de TOUTES les recommandations
    en UN SEUL appel LLM (CDC 3.4.5, cas d'usage 2 — optimisation
    perf, voir docstring module), avec cache par-ressource conservé :
    seules les ressources non déjà en cache pour ce fingerprint partent
    dans l'appel batché, les autres sont servies directement depuis le
    cache sans appel réseau.

    En cas d'échec du batch entier (timeout, JSON invalide, resource_id
    manquants), CHAQUE ressource non résolue individuellement retombe
    sur son titre/description d'origine — comportement de fallback
    identique à adapt_wording(), mais appliqué à tout le lot en échec
    plutôt qu'à une ressource isolée. Aucune exception ne remonte à
    l'appelant dans tous les cas.

    Parameters
    ----------
    recommendations : list[dict]
        Ressources à reformuler, chacune avec resource_id, title,
        description.
    learning_style : str

    Returns
    -------
    list[dict]
        Une entrée par recommandation d'entrée, dans le même ordre,
        fusionnée avec adapted_title/adapted_description (LLM ou
        fallback selon la ressource).
    """
    fingerprint = _fingerprint_for_llm("study-plan", student_profile, recommendations)
    cached = _get_cached_result(fingerprint)
    if cached is not None:
        return cached

    prompt = _build_study_plan_prompt(student_profile, recommendations)
    try:
        raw = _call_llm(prompt, max_tokens=min(4000, 150 * len(recommendations)))
        result = _parse_study_plan_response(raw)

        _store_cached_result(
            fingerprint,
            {
                "type": "study_plan",
                "student_id": student_profile.get("student_id"),
                "resource_ids": [r.get("resource_id") for r in recommendations],
                "result": result,
            },
        )
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
