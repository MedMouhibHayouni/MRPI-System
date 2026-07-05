"""
Content-Based Filtering Engine (CBF).

Encode les ressources pédagogiques via One-Hot Encoding pondéré sur
quatre attributs (subject, concept, difficulty, type), puis calcule la
similarité cosinus entre le profil de l'apprenant et chaque ressource
du catalogue.

Per CDC p.12 : ces quatre dimensions sont des dimensions notées du
vecteur, pas des filtres d'admission durs. Aucun filtre exact-match
n'est appliqué sur ces dimensions : le classement se fait par similarité
pondérée, après exclusion des ressources déjà consommées ou inéligibles.
"""

import ast
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
    """
    Construit un vecteur de poids aligné colonne-par-colonne avec la
    sortie du OneHotEncoder, pour pondérer différemment chacune des
    quatre dimensions (subject, concept, difficulty, type) dans le
    calcul de similarité cosinus.

    Pour chaque feature (dans l'ordre FEATURE_ORDER), répète son poids
    (FEATURE_WEIGHTS) une fois par catégorie encodée pour cette feature
    (encoder.categories_), puis concatène tous les blocs. Le résultat a
    la même dimension que la sortie one-hot de l'encoder, colonne par
    colonne.

    Appelée uniquement lors de la (re)construction du modèle
    (build_vectorizer) — jamais recalculée sur le chemin de requête
    (recommend_cbf reçoit weight_vector déjà calculé en paramètre).

    Parameters
    ----------
    encoder : OneHotEncoder
        Encodeur déjà fit sur les ressources (encoder.categories_ doit
        être peuplé).

    Returns
    -------
    np.ndarray
        Vecteur de poids, une valeur par colonne one-hot.
    """

    blocks = []
    for feature_name, categories in zip(FEATURE_ORDER, encoder.categories_):
        weight = FEATURE_WEIGHTS[feature_name]
        blocks.append(np.full(len(categories), weight))
    return np.concatenate(blocks)


# ─────────────────────────────────────────
# STEP 1 — Build / load encoder (avec cache pickle + fingerprint)
# ─────────────────────────────────────────


def build_vectorizer(resources_df, save_path=ENCODER_PATH):
    """
    Entraîne un OneHotEncoder sur les quatre dimensions catégorielles
    des ressources (subject, concept, difficulty, type), calcule la
    matrice de ressources pondérée (resource_matrix = one-hot brut ×
    weight_vector), et persiste l'ensemble (encoder, matrice pondérée,
    weight_vector, DataFrame source, fingerprint) sur disque.

    handle_unknown="ignore" : une catégorie non vue à l'entraînement
    (ex: nouvelle matière ajoutée après le fit) produit un vecteur nul
    pour cette dimension au lieu de lever une exception au moment du
    transform() sur une requête.

    Parameters
    ----------
    resources_df : pd.DataFrame
        Catalogue complet des ressources, doit contenir les colonnes
        FEATURE_ORDER.
    save_path : str
        Chemin de sauvegarde du pickle.

    Returns
    -------
    Tuple[OneHotEncoder, np.ndarray, pd.DataFrame, np.ndarray]
        encoder, resource_matrix (pondérée), resources_df (index reset),
        weight_vector.
    """
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
    """
    Charge un encodeur CBF et ses données associées depuis un pickle
    déjà construit, sans vérifier sa validité vis-à-vis de
    resources.csv (voir get_or_build_vectorizer() pour la version avec
    invalidation par fingerprint).

    Point de lecture unique du pickle CBF — get_or_build_vectorizer()
    délègue systématiquement ici plutôt que de relire le pickle en
    inline, pour garder une seule implémentation du format de
    désérialisation.

    Parameters
    ----------
    save_path : str
        Chemin du pickle à charger.

    Returns
    -------
    Tuple[OneHotEncoder, np.ndarray, pd.DataFrame, np.ndarray]
        encoder, resource_matrix, resources_df, weight_vector.

    Raises
    ------
    FileNotFoundError
        Si aucun fichier n'existe à save_path.
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
    """
    Point d'entrée principal pour obtenir un vectorizer CBF utilisable :
    charge le cache pickle existant si présent ET valide (fingerprint
    de resources.csv inchangé), sinon reconstruit l'encodeur et la
    matrice pondérée depuis resources_df.

    Parameters
    ----------
    resources_df : pd.DataFrame
        Catalogue de ressources, utilisé seulement si une
        reconstruction est nécessaire.
    save_path : str
        Chemin du pickle.

    Returns
    -------
    Tuple[OneHotEncoder, np.ndarray, pd.DataFrame, np.ndarray]
        encoder, resource_matrix, resources_df, weight_vector — toujours
        les quatre valeurs, que le chemin soit cache-hit ou
        reconstruction.
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


def _normalize_prerequisites(prerequisites):
    if prerequisites is None:
        return []
    if isinstance(prerequisites, float) and pd.isna(prerequisites):
        return []
    if isinstance(prerequisites, (list, tuple, set)):
        return [str(resource_id).strip() for resource_id in prerequisites if str(resource_id).strip()]
    if isinstance(prerequisites, str):
        value = prerequisites.strip()
        if not value or value.lower() == "nan":
            return []
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value.strip("[]")
            else:
                if isinstance(parsed, (list, tuple, set)):
                    return [
                        str(resource_id).strip()
                        for resource_id in parsed
                        if str(resource_id).strip()
                    ]
        separators = [";", ","]
        values = [value]
        for separator in separators:
            if separator in value:
                values = value.split(separator)
                break
        return [resource_id.strip().strip("'\"") for resource_id in values if resource_id.strip().strip("'\"")]
    return []


def prerequisites_met(prerequisites, completed_ids):
    """
    Détermine si une ressource est accessible à un étudiant compte tenu
    de ses ressources déjà complétées (completed_ids).

    Si prerequisites n'est pas une collection itérable exploitable
    (list, tuple, set — ex: NaN, None, ou une chaîne mal formée issue du
    CSV), la ressource est considérée accessible par défaut (pas de
    prérequis interprétable = pas de blocage). Sinon, tous les
    identifiants listés dans prerequisites doivent être présents dans
    completed_ids.

    Utilisée à la fois par CBF.py (filtre d'éligibilité avant scoring)
    et par metrics.py (définition du ground truth de pertinence,
    get_relevant_resources).

    Parameters
    ----------
    prerequisites : Any
        Valeur brute de la colonne 'prerequisites' pour une ressource
        (liste d'IDs attendue, mais peut être NaN/malformée).
    completed_ids : Iterable[str]
        IDs de ressources déjà complétées par l'étudiant.

    Returns
    -------
    bool
        True si la ressource est accessible (tous les prérequis sont
        satisfaits, ou aucun prérequis interprétable), False sinon.
    """

    required_ids = _normalize_prerequisites(prerequisites)
    if not required_ids:
        return True
    completed = {str(resource_id).strip() for resource_id in (completed_ids or [])}
    return all(resource_id in completed for resource_id in required_ids)


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
    """
    Génère les recommandations Content-Based Filtering pour un profil
    apprenant donné.

    Pipeline :
    1. Construit un profil de requête à 4 dimensions (subject,
       weak_concept, academic_level, type de contenu dérivé du
       learning_style via STYLE_TO_TYPE).
    2. Filtre le catalogue en excluant : les ressources déjà consommées
       (past_interactions), et les ressources dont les prérequis ne
       sont pas satisfaits (prerequisites_met) — ou, si l'étudiant n'a
       aucune interaction passée, uniquement les ressources sans
       prérequis du tout (cold-start : impossible de vérifier des
       prérequis sans historique, donc on ne propose que l'accessible
       d'office). Ce filtrage est un filtre d'ÉLIGIBILITÉ dur — il
       détermine le pool de candidats, il n'affecte pas leur score.
    3. Sur ce pool filtré, encode le profil de requête via le même
       OneHotEncoder que les ressources, applique le même weight_vector
       (recalculé seulement si non fourni, sinon réutilisé tel quel —
       voir docstring module), et calcule la similarité cosinus entre
       le vecteur de profil et chaque ressource candidate.
    4. Retourne les top_n ressources les mieux notées, avec un flag
       fallback_used indiquant si le type de contenu de la ressource
       diffère du type dérivé du learning_style demandé (utilisé en
       aval par hybrid.py pour appliquer une pénalité de fallback).

    Note : les 4 dimensions notées (subject, concept, difficulty, type)
    ne sont JAMAIS filtrées en dur — seuls past_interactions et
    prerequisites le sont. Si le pool filtré est vide, retourne un
    DataFrame vide plutôt que de lever une exception.

    Parameters
    ----------
    request : RecommendationRequest
        Profil apprenant validé (Pydantic).
    resources_df : pd.DataFrame
        Catalogue complet des ressources.
    encoder : OneHotEncoder
        Encodeur CBF déjà fit.
    resource_matrix
        Matrice pondérée des ressources (toutes, pas seulement le pool
        filtré — le filtrage se fait après, par indexation).
    weight_vector : np.ndarray, optional
        Vecteur de poids déjà calculé. Si None, recalculé (chemin lent,
        rétrocompatibilité uniquement).
    top_n : int
        Nombre maximum de ressources à retourner.

    Returns
    -------
    pd.DataFrame
        Colonnes : resource_id, title, concept, type, difficulty,
        cbf_score, fallback_used. Vide si aucun candidat éligible.
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
            base_filter["prerequisites"].apply(
                lambda prerequisites: not _normalize_prerequisites(prerequisites)
            )
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
