"""API FastAPI du MRPI-System."""

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.hybrid import get_hybrid_engine
from src.LLM_API import (
    LLM_MODEL,
    adapt_wording_batch,
    generate_explanation,
    suggest_study_plan,
)
from src.schemas.request import RecommendationRequest
from src.schemas.response import RecommendationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

# --------------------
# Executors séparés par nature de charge.
# --------------------

# CPU-bound : HybridEngine (pandas/CBF/CF).
ML_EXECUTOR_WORKERS = int(os.environ.get("ML_EXECUTOR_WORKERS", os.cpu_count() or 4))
ml_executor = ThreadPoolExecutor(
    max_workers=ML_EXECUTOR_WORKERS, thread_name_prefix="ml"
)

# I/O-bound : appels LLM, dominés par l'attente réseau.
LLM_EXECUTOR_WORKERS = int(os.environ.get("LLM_EXECUTOR_WORKERS", 100))
llm_executor = ThreadPoolExecutor(
    max_workers=LLM_EXECUTOR_WORKERS, thread_name_prefix="llm"
)

engine = get_hybrid_engine()


# --------------------
# CSV loading helpers — mise en cache (voir docstring module)
# --------------------


def _split_csv_line_respecting_brackets(line: str) -> list[str]:
    """
    Découpe une ligne CSV par virgule, en traitant tout segment délimité
    par des crochets [ ] comme un champ atomique unique — même si ce
    champ contient des virgules internes non échappées (cas de la
    colonne past_interactions au format [RES-001, RES-004] sans
    guillemets CSV englobants, ce qui casserait un parsing CSV standard).

    Maintient un compteur de profondeur (depth) incrémenté sur [ et
    décrémenté sur ] : une virgule n'est un séparateur de champ que si
    depth == 0 (c'est-à-dire hors de toute liste entre crochets).

    Parameters
    ----------
    line : str
        Une ligne brute du fichier CSV (sans le retour à la ligne).

    Returns
    -------
    list[str]
        Champs de la ligne, dans l'ordre, crochets internes préservés
        tels quels dans le champ correspondant.
    """
    fields = []
    current = []
    depth = 0
    for ch in line:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def _read_csv_with_bracket_lists(path: str) -> pd.DataFrame:
    """
    Lit un fichier CSV potentiellement mal formé à cause d'une colonne
    contenant des listes entre crochets non échappées, en utilisant
    _split_csv_line_respecting_brackets() ligne par ligne au lieu du
    parser CSV standard de pandas (qui casserait sur ces virgules
    internes).

    Toute ligne dont le nombre de champs ne correspond pas au nombre de
    colonnes de l'en-tête est ignorée (et journalisée en warning avec
    son numéro de ligne) plutôt que de faire planter tout le chargement
    — une ligne malformée isolée ne doit pas bloquer l'accès à tout le
    reste du fichier. Ne modifie jamais le fichier source sur disque.

    Parameters
    ----------
    path : str
        Chemin du fichier CSV à lire.

    Returns
    -------
    pd.DataFrame
        DataFrame construit à partir des lignes valides uniquement.
        Vide si le fichier est vide.
    """
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n").rstrip("\r") for l in f if l.strip()]

    if not lines:
        return pd.DataFrame()

    header = _split_csv_line_respecting_brackets(lines[0])
    rows = [_split_csv_line_respecting_brackets(l) for l in lines[1:]]

    clean_rows = []
    for i, row in enumerate(rows, start=2):
        if len(row) != len(header):
            logger.warning(
                f"[{os.path.basename(path)}] Ligne {i} ignorée: "
                f"{len(row)} champs au lieu de {len(header)} attendus."
            )
            continue
        clean_rows.append(row)

    return pd.DataFrame(clean_rows, columns=header)


@lru_cache(maxsize=1)
def load_students() -> pd.DataFrame:
    """
    Charge students.csv via _read_csv_with_bracket_lists() et met le
    résultat en cache mémoire pour tout le cycle de vie du process
    (lru_cache(maxsize=1) : un seul appel disque, tous les appels
    suivants réutilisent le même DataFrame en mémoire).

    Le cache n'est invalidé qu'au redémarrage du process — contrairement
    aux modèles CF/CBF (invalidation par fingerprint des fichiers
    source), ce niveau de sophistication n'a pas été jugé nécessaire ici
    car cet endpoint sert de la donnée de consultation brute, pas un
    modèle entraîné dont la fraîcheur affecte la qualité des
    recommandations.

    Returns
    -------
    pd.DataFrame
        Contenu de students.csv.
    """
    path = os.path.join(DATA_PATH, "students.csv")
    return _read_csv_with_bracket_lists(path)


@lru_cache(maxsize=1)
def load_resources() -> pd.DataFrame:
    """
    Charge resources.csv via pandas standard (pas besoin du parsing
    spécial à crochets ici — students.csv est la seule source concernée
    par past_interactions) et met le résultat en cache mémoire pour tout
    le cycle de vie du process. Voir load_students() pour la
    justification du choix lru_cache plutôt qu'une invalidation par
    fingerprint.

    Returns
    -------
    pd.DataFrame
        Contenu de resources.csv.
    """
    path = os.path.join(DATA_PATH, "resources.csv")
    return pd.read_csv(path)


def df_to_records(df: pd.DataFrame):
    """
    Convertit un DataFrame pandas en liste de dictionnaires
    JSON-sérialisables, en remplaçant les NaN par None et en convertissant
    les types numpy natifs (np.integer, np.floating, np.bool_) en types
    Python natifs (int, float, bool) — FastAPI/Pydantic ne sérialise pas
    correctement les types numpy tels quels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à convertir (typiquement issu de load_resources()).

    Returns
    -------
    list[dict]
        Une entrée par ligne, valeurs JSON-safe.
    """
    df = df.astype(object).where(pd.notnull(df), None)
    records = df.to_dict(orient="records")

    def clean_value(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    return [{k: clean_value(v) for k, v in record.items()} for record in records]


# --------------------
# Recommendation endpoints
# --------------------


@app.post("/recommendations", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Endpoint principal du moteur : génère des recommandations hybrides
    enrichies par trois appels LLM distincts (explication, adaptation de
    wording, plan d'étude — CDC 3.4.5).

    Répartition du travail sur deux ThreadPoolExecutor dédiés (voir
    docstring module pour la justification complète) :
    - Le calcul hybride (engine.get_recommendations, CPU-bound,
      pandas/numpy) passe par ml_executor, dimensionné sur le nombre de
      coeurs CPU disponibles.
    - Les 3 appels LLM (I/O-bound, attente réseau Groq) sont lancés en
      parallèle via llm_executor (asyncio.gather), dimensionné bien
      plus large que ml_executor puisqu'un thread en attente réseau ne
      consomme quasiment pas de CPU.

    Si le hybrid engine ne retourne aucune recommandation, répond 404
    plutôt que de renvoyer une réponse vide avec un statut 200 trompeur.

    Parameters
    ----------
    request : RecommendationRequest
        Profil apprenant validé par Pydantic (CDC Tableau 6).

    Returns
    -------
    RecommendationResponse
        request_id (généré), student_id, recommendations,
        adapted_recommendations, study_plan, llm_explanation, metadata
        (latency_ms, model).

    Raises
    ------
    HTTPException
        404 si aucune recommandation n'a pu être générée pour ce profil.
    """

    start = time.monotonic()
    loop = asyncio.get_event_loop()

    recommendations = await loop.run_in_executor(
        ml_executor, engine.get_recommendations, request
    )

    if not recommendations:
        raise HTTPException(
            status_code=404, detail="Aucune recommandation trouvée pour ce profil."
        )
    recs_as_dicts = [r.model_dump() for r in recommendations]
    profile_as_dict = request.model_dump()

    explanation_task = loop.run_in_executor(
        llm_executor, generate_explanation, profile_as_dict, recs_as_dicts
    )
    wording_task = loop.run_in_executor(
        llm_executor,
        adapt_wording_batch,
        recs_as_dicts,
        request.learning_style.value,
    )
    study_plan_task = loop.run_in_executor(
        llm_executor, suggest_study_plan, profile_as_dict, recs_as_dicts
    )
    llm_explanation, adapted_recommendations, study_plan = await asyncio.gather(
        explanation_task, wording_task, study_plan_task
    )

    latency_ms = int((time.monotonic() - start) * 1000)
    return RecommendationResponse(
        request_id=f"REQ-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        student_id=request.student_id,
        recommendations=recommendations,
        adapted_recommendations=adapted_recommendations,
        study_plan=study_plan,
        llm_explanation=llm_explanation,
        metadata={"latency_ms": latency_ms, "model": LLM_MODEL},
    )


# --------------------
# Profile endpoint
# --------------------


@app.post("/student-profile")
async def validate_student_profile(profile: RecommendationRequest):
    """Accepte un JSON, le valide avec RecommendationRequest, le retourne."""
    profile_dict = profile.dict()
    return {
        "status": "success",
        "message": "Profil validé avec succès",
        "profile": profile_dict,
    }


# --------------------
# Resources endpoints
# --------------------


@app.get("/resources/all")
def get_all_resources():
    """
    Retourne l'intégralité du catalogue de ressources pédagogiques, sans
    filtrage, sous forme de liste de dictionnaires JSON.

    Returns
    -------
    list[dict]
        Voir df_to_records(load_resources()).
    """
    df = load_resources()
    return df_to_records(df)


@app.get("/resources")
def get_resources(subject: str | None = None, difficulty: str | None = None):
    """
    Retourne le catalogue de ressources, filtré optionnellement par
    matière et/ou difficulté (correspondance exacte, pas de recherche
    approximative). Les deux filtres sont combinables (ET logique) et
    indépendamment optionnels — absence des deux paramètres équivaut à
    /resources/all.

    Parameters
    ----------
    subject : str | None
        Filtre exact sur la colonne subject, si fourni.
    difficulty : str | None
        Filtre exact sur la colonne difficulty, si fourni.

    Returns
    -------
    list[dict]
        Sous-ensemble du catalogue correspondant aux filtres.
    """
    df = load_resources()
    if subject:
        df = df[df["subject"] == subject]
    if difficulty:
        df = df[df["difficulty"] == difficulty]
    return df_to_records(df)


# --------------------
# Lancement
# uvicorn src.main:app --workers 4 --host 0.0.0.0 --port 8000
# --reload est dev only, jamais en charge/prod (recharge le process et
# invalide tous les caches lru_cache / singletons à chaque modif de fichier)
#
# Pour A/B tester les tailles d'executor sans modifier le code :
#   ML_EXECUTOR_WORKERS=8 LLM_EXECUTOR_WORKERS=150 uvicorn src.main:app ...
# --------------------


@app.post("/recommendations/no-llm", response_model=RecommendationResponse)
async def recommend_no_llm(request: RecommendationRequest):
    """
    Variante de /recommendations qui exécute UNIQUEMENT le hybrid engine
    (CBF+CF via ml_executor), sans aucun des trois appels LLM — endpoint
    de benchmark, pas destiné à un usage client final.

    Objectif : isoler la capacité réelle et la latence du moteur de
    recommandation hybride sous charge (Locust), indépendamment du rate
    limit du provider LLM externe (Groq, 6000 TPM en tier gratuit — voir
    diagnostic Tests et Évaluation : le LLM était le facteur limitant
    apparent sous charge, ce endpoint permet de prouver que non, le vrai
    goulot est ailleurs).

    Les champs llm_explanation/adapted_recommendations/study_plan du
    schéma de réponse sont remplis avec des valeurs de repli statiques
    et déterministes (aucun appel réseau) uniquement pour respecter le
    contrat RecommendationResponse sans fausser la mesure de latence
    avec un travail non pertinent à ce benchmark précis.

    Parameters
    ----------
    request : RecommendationRequest

    Returns
    -------
    RecommendationResponse
        Identique en structure à /recommendations, mais
        adapted_recommendations = recommandations brutes non reformulées,
        study_plan = ordre basé uniquement sur le score hybride,
        llm_explanation = message statique indiquant le mode no-llm,
        metadata.model = "none (no-llm benchmark mode)".

    Raises
    ------
    HTTPException
        404 si aucune recommandation n'a pu être générée.
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    recommendations = await loop.run_in_executor(
        ml_executor, engine.get_recommendations, request
    )

    if not recommendations:
        raise HTTPException(
            status_code=404, detail="Aucune recommandation trouvée pour ce profil."
        )

    recs_as_dicts = [r.model_dump() for r in recommendations]

    latency_ms = int((time.monotonic() - start) * 1000)
    return RecommendationResponse(
        request_id=f"REQ-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        student_id=request.student_id,
        recommendations=recommendations,
        adapted_recommendations=recs_as_dicts,  # pas d'adaptation, valeurs brutes
        study_plan=[
            {
                "resource_id": r.get("resource_id"),
                "order": i + 1,
                "justification": "Ordre basé sur le score de recommandation (mode no-llm).",
            }
            for i, r in enumerate(recs_as_dicts)
        ],
        llm_explanation="(mode no-llm — pas d'explication générée)",
        metadata={"latency_ms": latency_ms, "model": "none (no-llm benchmark mode)"},
    )
