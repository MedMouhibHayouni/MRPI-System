"""
main.py — API FastAPI du MRPI-System.

FIX vs version précédente : le fichier reçu était littéralement DEUX
versions concaténées (l'ancienne, puis le "fix ThreadPoolExecutor" présenté
en commentaire comme "à intégrer, ne remplace pas main.py"). Résultat concret :
- Plusieurs fonctions avaient leur docstring ouverte par du texte sans les
  guillemets triples d'ouverture (_read_csv_with_bracket_lists,
  df_to_records, validate_student_profile, recommend_without_llm) ->
  SyntaxError pure et simple à l'import. Ce fichier ne pouvait pas tourner
  tel quel.
- Deux `app = FastAPI()` et deux définitions de route `/recommendations`
  auraient coexisté si le fichier avait été syntaxiquement valide —
  Starlette ne garde que la PREMIÈRE route déclarée pour un path donné,
  donc le fix ThreadPoolExecutor(max_workers=20) n'aurait jamais été
  exécuté même sans le bug de syntaxe.
Ce fichier ci-dessous est la version consolidée, unique, avec l'executor
dédié appliqué aux DEUX endpoints de recommandation.

FIX perf supplémentaire : load_resources()/load_students() relisaient le
CSV depuis le disque à CHAQUE appel des endpoints /resources, /resources/all,
/resources/{id}, /student-profile — sous charge (500 users, CDC 3.3) c'est
de l'I/O disque + re-parsing pandas redondant à chaque requête. Mis en
cache via lru_cache(maxsize=1) : chargé une fois, réutilisé ensuite (même
logique que le singleton `engine` déjà en place pour HybridEngine).
"""

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
from src.LLM_API import LLM_MODEL, generate_explanation
from src.schemas.request import RecommendationRequest
from src.schemas.response import RecommendationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:4200",  # dev Angular
    "http://127.0.0.1:4200",
    # ajoute ton domaine de prod ici
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # jamais "*" si allow_credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

# Executor dédié, sizé explicitement — ne pas laisser Python choisir le
# défaut (min(32, cpu_count()+4)), qui sature sous 500 users concurrents
# (CDC 3.3, benchmark Locust). Ajuster selon profiling réel du CPU cible.
recommendation_executor = ThreadPoolExecutor(max_workers=20)

engine = get_hybrid_engine()


# --------------------
# CSV loading helpers — mise en cache (voir docstring module)
# --------------------


def _split_csv_line_respecting_brackets(line: str) -> list[str]:
    """
    Split une ligne CSV par virgule, en traitant tout ce qui se trouve
    entre [ et ] comme un seul champ atomique, même si ce champ contient
    des virgules non échappées (ex: colonne 'past_interactions' au format
    [RES-001, RES-004, ...] sans guillemets CSV englobants).

    Cela permet de lire le fichier tel quel, sans le modifier.
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
    Lecture robuste d'un CSV potentiellement mal formé à cause d'une
    colonne contenant des listes entre crochets non échappées.
    Ne modifie jamais le fichier source.
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
    """Charge students.csv, mis en cache — un seul appel disque par process.

    Redémarrer le process après une modification de students.csv (cohérent
    avec le pattern déjà en place pour le pickle CF/CBF, qui, eux,
    invalident via fingerprint plutôt qu'un simple cache mémoire — ce
    niveau de sophistication n'est pas nécessaire ici, ce endpoint sert de
    la donnée de consultation, pas un modèle entraîné)."""
    path = os.path.join(DATA_PATH, "students.csv")
    return _read_csv_with_bracket_lists(path)


@lru_cache(maxsize=1)
def load_resources() -> pd.DataFrame:
    """Charge resources.csv, mis en cache — voir load_students()."""
    path = os.path.join(DATA_PATH, "resources.csv")
    return pd.read_csv(path)


def df_to_records(df: pd.DataFrame):
    """
    Convertit un DataFrame en liste de dicts JSON-safe :
    - remplace les NaN par None
    - convertit les types numpy (int64, float64...) en types Python natifs
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
    """Recommandations hybrides + explication LLM.

    request.alpha (si fourni) est lu directement par
    HybridEngine.get_recommendations — voir hybrid.py. Aucun câblage
    supplémentaire nécessaire ici : le champ existe dans le schéma et
    HybridEngine sait déjà le consommer.
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    recommendations = await loop.run_in_executor(
        recommendation_executor, engine.get_recommendations, request
    )

    if not recommendations:
        raise HTTPException(
            status_code=404, detail="Aucune recommandation trouvée pour ce profil."
        )

    recs_as_dicts = [r.dict() for r in recommendations]
    profile_as_dict = request.dict()

    llm_explanation = await loop.run_in_executor(
        recommendation_executor, generate_explanation, profile_as_dict, recs_as_dicts
    )

    latency_ms = int((time.monotonic() - start) * 1000)
    return RecommendationResponse(
        request_id=f"REQ-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        student_id=request.student_id,
        recommendations=recommendations,
        llm_explanation=llm_explanation,
        metadata={"latency_ms": latency_ms, "model": LLM_MODEL},
    )


@app.post("/recommendations/no-llm", response_model=RecommendationResponse)
async def recommend_without_llm(request: RecommendationRequest):
    """Version sans LLM, pour benchmark A/B latence (CDC 3.3).

    FIX : llm_explanation=None plantait systématiquement avant que
    RecommendationResponse.llm_explanation soit rendu Optional[str]
    (voir schemas/response.py) — cet endpoint renvoyait une
    ValidationError / 500 à chaque appel.
    """
    start = time.monotonic()
    loop = asyncio.get_event_loop()

    recommendations = await loop.run_in_executor(
        recommendation_executor, engine.get_recommendations, request
    )

    if not recommendations:
        raise HTTPException(
            status_code=404, detail="Aucune recommandation trouvée pour ce profil."
        )

    latency_ms = int((time.monotonic() - start) * 1000)

    return RecommendationResponse(
        request_id=f"REQ-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        student_id=request.student_id,
        recommendations=recommendations,
        llm_explanation=None,
        metadata={"latency_ms": latency_ms, "llm_enabled": False},
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
    df = load_resources()
    return df_to_records(df)


@app.get("/resources")
def get_resources(subject: str | None = None, difficulty: str | None = None):
    df = load_resources()
    if subject:
        df = df[df["subject"] == subject]
    if difficulty:
        df = df[df["difficulty"] == difficulty]
    return df_to_records(df)


@app.get("/resources/{resource_id}")
def get_resource(resource_id: str):
    df = load_resources()
    resource = df[df["resource_id"] == resource_id]

    if resource.empty:
        raise HTTPException(status_code=404, detail="Resource not found")

    return df_to_records(resource)[0]


# --------------------
# Lancement
# uvicorn src.main:app --workers 4 --host 0.0.0.0 --port 8000
# --reload est dev only, jamais en charge/prod (recharge le process et
# invalide tous les caches lru_cache / singletons à chaque modif de fichier)
# --------------------
