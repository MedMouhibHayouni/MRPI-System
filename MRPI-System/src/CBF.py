"""
Content-Based Filtering Engine (CBF).

Encode les ressources pédagogiques via One-Hot Encoding pondéré sur
quatre attributs (subject, concept, difficulty, type), puis calcule la
similarité cosinus entre le profil de l'apprenant et chaque ressource
du catalogue.

Per CDC p.12 : ces quatre dimensions sont des dimensions NOTÉES du
vecteur, pas des filtres d'admission durs. Aucun filtre exact-match
n'est appliqué — le classement se fait entièrement par similarité
pondérée.

FIX vs version précédente :
- get_or_build_vectorizer() dupliquait la logique de load_vectorizer()
  en inline au lieu de l'appeler -> load_vectorizer() était du code mort.
  Corrigé : get_or_build_vectorizer() appelle désormais load_vectorizer().
- Le weight_vector est calculé une fois et persisté dans le pickle
  (model_data["weight_vector"]) mais était recalculé à chaque appel de
  recommend_cbf() via _build_weight_vector(encoder). Corrigé : le
  weight_vector est maintenant retourné par get_or_build_vectorizer() et
  réutilisé partout, au lieu d'être recalculé sur le chemin chaud
  (recommend_cbf est appelé à CHAQUE requête /recommendations).
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from src.cache_utils import compute_source_fingerprint, is_cache_valid
from src.schemas.request import RecommendationRequest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.pkl")

SOURCE_FILES = [os.path.join(DATA_DIR, "resources.csv")]

STYLE_TO_TYPE = {
    "visual": "video",
    "auditory": "video",
    "kinesthetic": "exercise",
    "textual": "micro_lesson",
}

FEATURE_ORDER = ["subject", "concept", "difficulty", "type"]
FEATURE_WEIGHTS = {
    "subject": 2.0,
    "concept": 3.0,
    "difficulty": 2.0,
    "type": 0.5,
}


def _build_weight_vector(encoder: OneHotEncoder) -> np.ndarray:
    """Construit un vecteur de poids aligné colonne-par-colonne avec la sortie
    de l'encodeur. N'est appelé QUE lors d'une (re)construction du modèle
    (build_vectorizer) — plus jamais sur le chemin de requête."""
    blocks = []
    for feature_name, categories in zip(FEATURE_ORDER, encoder.categories_):
        weight = FEATURE_WEIGHTS[feature_name]
        blocks.append(np.full(len(categories), weight))
    return np.concatenate(blocks)


# ─────────────────────────────────────────
# STEP 1 — Build / load encoder (avec cache pickle + fingerprint)
# ─────────────────────────────────────────


def build_vectorizer(resources_df, save_path=ENCODER_PATH):
    """Fit un OneHotEncoder, pondère, persiste (encoder + matrice + weight_vector)."""
    resources_df = resources_df.reset_index(drop=True).copy()
    features = resources_df[FEATURE_ORDER]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    raw_matrix = encoder.fit_transform(features)

    weight_vector = _build_weight_vector(encoder)
    resource_matrix = raw_matrix * weight_vector

    model_data = {
        "encoder": encoder,
        "resource_matrix": resource_matrix,
        "weight_vector": weight_vector,
        "resources_df": resources_df,
        "source_fingerprint": compute_source_fingerprint(SOURCE_FILES),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    print(
        f"✓ OneHot encoder fitted on {len(resources_df)} resources (weighted). "
        f"Saved to {save_path}"
    )
    return encoder, resource_matrix, resources_df, weight_vector


def load_vectorizer(save_path=ENCODER_PATH):
    """Charge un encodeur + données pondérées depuis disque.

    FIX : cette fonction existait mais n'était appelée par personne —
    get_or_build_vectorizer() réimplémentait sa propre lecture pickle en
    inline. Maintenant c'est le seul point de lecture du pickle, comme
    load_cf_model() l'est côté CF.py (les deux fichiers suivent enfin le
    même pattern).
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Encoder not found at '{save_path}'. "
            "Run build_vectorizer() first to train and save the encoder."
        )
    with open(save_path, "rb") as f:
        data = pickle.load(f)

    return (
        data["encoder"],
        data["resource_matrix"],
        data["resources_df"],
        data["weight_vector"],
    )


def get_or_build_vectorizer(resources_df, save_path=ENCODER_PATH):
    """Charge le cache pickle s'il est valide, reconstruit sinon.

    Returns
    -------
    encoder, resource_matrix, resources_df, weight_vector
    """
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            model_data = pickle.load(f)
        if is_cache_valid(model_data, SOURCE_FILES):
            return load_vectorizer(save_path)
        print("⚠ Cache CBF périmé (resources.csv modifié) — reconstruction.")
    else:
        print("🔧 Aucun cache CBF trouvé — construction initiale.")

    return build_vectorizer(resources_df, save_path)


# ─────────────────────────────────────────
# STEP 2 — Prerequisites check
# ─────────────────────────────────────────


def prerequisites_met(prereq, completed_ids):
    """True si tous les prérequis sont satisfaits ou s'il n'y en a pas."""
    if pd.isna(prereq) or prereq == "":
        return True
    required = [r.strip() for r in prereq.split(",")]
    return all(r in completed_ids for r in required)


# ─────────────────────────────────────────
# STEP 3 — Recommend CBF
# ─────────────────────────────────────────


def recommend_cbf(
    request: RecommendationRequest,
    resources_df: pd.DataFrame,
    encoder: OneHotEncoder,
    resource_matrix,
    weight_vector: np.ndarray = None,
    top_n: int = 5,
):
    """Génère les recommandations CBF pour un étudiant.

    FIX : accepte maintenant `weight_vector` en paramètre (calculé une
    seule fois par get_or_build_vectorizer, propagé par HybridEngine).
    Si non fourni (rétro-compatibilité), il est recalculé comme avant —
    mais le chemin chaud (HybridEngine) ne passera plus jamais par ce
    recalcul redondant.
    """
    weak_concept = request.weak_concept
    lesson_type = STYLE_TO_TYPE[request.learning_style.value]
    subject = request.subject
    difficulty = request.academic_level.value
    completed_ids = request.past_interactions

    df = resources_df.reset_index(drop=True).copy()

    base_filter = df[~df["resource_id"].isin(completed_ids)]

    if not completed_ids:
        base_filter = base_filter[
            base_filter["prerequisites"].isna() | (base_filter["prerequisites"] == "")
        ]
    else:
        base_filter = base_filter[
            base_filter["prerequisites"].apply(
                lambda p: prerequisites_met(p, completed_ids)
            )
        ]

    if base_filter.empty:
        print("⚠ No resources available for this profile.")
        return pd.DataFrame()

    filtered_indices = base_filter.index.tolist()
    filtered_matrix = resource_matrix[filtered_indices]

    profile = pd.DataFrame(
        [[subject, weak_concept, difficulty, lesson_type]],
        columns=FEATURE_ORDER,
    )
    raw_query = encoder.transform(profile)

    if weight_vector is None:
        weight_vector = _build_weight_vector(encoder)
    query_vector = raw_query * weight_vector

    scores = cosine_similarity(query_vector, filtered_matrix)[0]

    top_n_safe = min(top_n, len(scores))
    top_idx = np.argsort(scores)[::-1][:top_n_safe]

    result = base_filter.iloc[top_idx].copy()
    result["cbf_score"] = scores[top_idx]
    result["fallback_used"] = result["type"] != lesson_type

    return result[
        [
            "resource_id",
            "title",
            "concept",
            "type",
            "difficulty",
            "cbf_score",
            "fallback_used",
        ]
    ]
