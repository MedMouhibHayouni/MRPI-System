"""
Hybrid Recommendation Engine.

Combine les scores du moteur CBF (Content-Based Filtering) et du moteur CF
(Collaborative Filtering) en un score hybride pondéré :

    hybrid_score = 0.6 × CBF_score + 0.4 × CF_score

Les deux scores sont normalisés indépendamment à [0, 1] (min-max) avant fusion
pour éviter qu'une échelle domine l'autre. Une pénalité (×0.5) est appliquée
aux ressources CBF issues d'un fallback de type de contenu.

Comportement CF validé (cf. session de debug, vérifié sur données réelles) :
    - Quand CBF retourne au moins un candidat : CF n'est PAS filtré par
      matière/difficulté dans la fusion. C'est intentionnel — le CDC décrit
      le filtrage collaboratif comme devant produire un effet de sérendipité
      (ressources inattendues mais pertinentes, issues du comportement de
      pairs similaires). Vérifié concrètement : pour STU-2026-0001, une
      ressource hors-matière (français) provenait de son voisin le plus
      proche (même matière/concept/niveau), donc d'un signal réel, pas de
      bruit.
    - Quand CBF est vide : CF seul est filtré par matière (garde-fou),
      car sans CBF il n'y a plus aucune notion de pertinence de contenu
      pour contrebalancer la sérendipité. Ce filtre est intentionnellement
      moins strict que la fusion normale — il ne filtre pas la difficulté,
      seulement la matière.

Pipeline :
    1. HybridEngine.get_recommendations(request)
       ├── recommend_cbf(...)      → cbf_df    (DataFrame)
       ├── cf_engine.get_recommendations(...) → cf_results (List[Dict])
       └── hybrid_fusion(...)      → List[ResourceRecommendation]

Cas dégradés gérés :
    - CF vide  → CBF seul
    - CBF vide → CF seul (filtre matière appliqué, pas de filtre difficulté)
    - Les deux vides → liste vide

Singleton :
    get_hybrid_engine() retourne l'instance unique pour éviter
    de recharger les modèles à chaque requête.

Dépendances :
    - src/CBF.py             : recommend_cbf, get_or_build_vectorizer
    - src/CF.py               : get_cf_engine
    - src/schemas/request.py : RecommendationRequest
    - src/schemas/response.py: ResourceRecommendation, ResourceType, AcademicLevel
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional

from src.schemas.request import RecommendationRequest
from src.schemas.response import ResourceRecommendation, ResourceType, AcademicLevel
from src.CF import get_cf_engine
from src.CBF import recommend_cbf, get_or_build_vectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# CONFIGURATION
# ============================================================

# alpha=0.6 (CB) / 0.4 (CF) — décision documentée dans le CDC/journal.
DEFAULT_CBF_WEIGHT = 0.6
DEFAULT_CF_WEIGHT = 0.4

FALLBACK_PENALTY = 0.5  # pénalité appliquée quand fallback_used=True

# Seuil sous lequel min-max normalization est considérée dégénérée
# (à 1-2 candidats, min-max n'a pas de sens statistique).
MIN_CANDIDATES_FOR_NORM = 3


TARGET_RECOMMENDATIONS = (
    5  # cible — agit actuellement comme un plancher, pas un plafond (point ouvert #2)
)
MAX_RECOMMENDATIONS = 20

# ============================================================
# UTILITAIRES
# ============================================================


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalise une liste de scores à [0, 1] (min-max par liste).

    Si len(scores) < MIN_CANDIDATES_FOR_NORM, la normalisation min-max
    est statistiquement dégénérée (ex: 2 scores identiques → tous à 1.0
    sans signal réel). On le loggue pour que ce ne soit pas silencieux.

    Parameters
    ----------
    scores : List[float]
        Scores bruts à normaliser (cosine similarity pour CBF, score
        pondéré pour CF).

    Returns
    -------
    List[float]
        Scores normalisés dans [0, 1], même ordre que l'entrée.
        Liste vide si l'entrée est vide.
    """
    if not scores:
        return []
    if len(scores) < MIN_CANDIDATES_FOR_NORM:
        logger.info(
            f"ℹ️ Normalisation sur seulement {len(scores)} candidat(s) — "
            f"résultat peu fiable statistiquement"
        )
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def apply_fallback_penalty(weight: float, fallback_used: bool) -> float:
    """Réduit le poids d'une source si elle a utilisé un fallback générique.

    Parameters
    ----------
    weight : float
        Poids initial (avant pénalité).
    fallback_used : bool
        True si la source a dû recourir à un type de contenu de repli.

    Returns
    -------
    float
        Poids inchangé si fallback_used=False, sinon weight * FALLBACK_PENALTY.
    """
    return weight * FALLBACK_PENALTY if fallback_used else weight


def get_resource_type(value: str) -> ResourceType:
    """Convertit une chaîne en ResourceType, avec fallback loggé si inconnu.

    Parameters
    ----------
    value : str
        Valeur brute du champ 'type' (ex: 'video', 'exercise', 'micro_lesson').

    Returns
    -------
    ResourceType
        Enum correspondant, ou ResourceType('video') si la valeur est
        inconnue (avec warning loggé).
    """
    try:
        return ResourceType(value.lower())
    except ValueError:
        logger.warning(f"⚠️ Type inconnu: {value}, utilisation de 'video'")
        return ResourceType("video")


def _resource_metadata(resource_id: str, resources_df: pd.DataFrame) -> Dict[str, Any]:
    """Récupère les métadonnées d'une ressource depuis le DataFrame déjà chargé.

    Remplace toute relecture de CSV : resources_df est chargé une seule fois
    dans HybridEngine.__init__ (et normalisé en casse à ce moment-là) et
    propagé partout.

    Parameters
    ----------
    resource_id : str
        Identifiant de la ressource (ex: 'RES-001').
    resources_df : pd.DataFrame
        Catalogue complet des ressources, déjà chargé et normalisé.

    Returns
    -------
    Dict[str, Any]
        Métadonnées (title, type, difficulty, subject, estimated_time_min).
        Valeurs par défaut si resource_id introuvable (ne devrait pas
        arriver en pratique si resources_df est cohérent avec CF/CBF).
    """
    row = resources_df[resources_df["resource_id"] == resource_id]
    if row.empty:
        return {
            "title": f"Resource {resource_id}",
            "type": "video",
            "difficulty": "beginner",
            "subject": None,
            "estimated_time_min": 30,
        }
    r = row.iloc[0]
    return {
        "title": r["title"],
        "type": r["type"],
        "difficulty": r["difficulty"],
        "subject": r.get("subject"),
        "estimated_time_min": r.get("estimated_time_min", 30),
    }


# ============================================================
# FUSION HYBRIDE
# ============================================================


def hybrid_fusion(
    cf_results: List[Dict[str, Any]],
    cbf_df: pd.DataFrame,
    resources_df: pd.DataFrame,
    request_subject: Optional[str] = None,
    cf_weight: float = DEFAULT_CF_WEIGHT,
    cbf_weight: float = DEFAULT_CBF_WEIGHT,
    target_recommendations: int = TARGET_RECOMMENDATIONS,
    max_recommendations: int = MAX_RECOMMENDATIONS,
) -> List[ResourceRecommendation]:
    """Fusionne les résultats CF et CBF en appliquant alpha et la pénalité fallback.

    Comportement CF :
        - Si CBF a au moins un candidat : CF entre dans la fusion SANS
          filtre matière/difficulté. C'est la sérendipité voulue par le CDC.
        - Si CBF est vide : CF passe par _convert_cf_to_recommendations,
          qui filtre par matière (mais pas par difficulté).

    Parameters
    ----------
    cf_results : List[Dict[str, Any]]
        Résultats bruts du moteur CF (resource_id, predicted_score).
    cbf_df : pd.DataFrame
        Résultats bruts du moteur CBF (resource_id, title, type, difficulty,
        cbf_score, fallback_used).
    resources_df : pd.DataFrame
        Catalogue complet des ressources (normalisé en casse).
    request_subject : Optional[str]
        Matière demandée par l'apprenant (déjà lowercase via le validator
        Pydantic). Utilisé uniquement dans le cas CBF vide.
    cf_weight, cbf_weight : float
        Poids de base avant pénalité fallback (défauts : 0.4 / 0.6).
    target_recommendations : int
        Cible de résultats — agit actuellement comme plancher (point ouvert #2).
    max_recommendations : int
        Plafond dur du nombre de résultats retournés.

    Returns
    -------
    List[ResourceRecommendation]
        Liste triée par hybrid_score décroissant. Vide si CF et CBF sont
        tous deux vides.
    """

    cf_empty = not cf_results
    cbf_empty = cbf_df.empty

    if cf_empty and cbf_empty:
        logger.warning("❌ CF et CBF sont vides")
        return []

    if cf_empty and not cbf_empty:
        logger.info("⚠️ CF vide → CBF seul")
        return _convert_cbf_to_recommendations(cbf_df, max_recommendations)

    if not cf_empty and cbf_empty:
        logger.info("⚠️ CBF vide → CF seul (garde-fou matière appliqué)")
        return _convert_cf_to_recommendations(
            cf_results, resources_df, request_subject, max_recommendations
        )

    logger.info("✅ Fusion CF + CBF")

    has_fallback_col = "fallback_used" in cbf_df.columns
    if not has_fallback_col:
        logger.warning(
            "⚠️ Colonne 'fallback_used' absente de cbf_df — pénalité fallback désactivée"
        )

    # CF n'est PAS filtré par matière/difficulté ici — voir docstring + note
    # de module en haut du fichier. Comportement validé, intentionnel.
    cf_scores_norm = normalize_scores([r["predicted_score"] for r in cf_results])
    cbf_scores_norm = normalize_scores(cbf_df["cbf_score"].tolist())

    merged: Dict[str, Dict[str, Any]] = {}

    # Résultats CF
    for i, result in enumerate(cf_results):
        if i >= len(cf_scores_norm):
            break
        merged[result["resource_id"]] = {
            "resource_id": result["resource_id"],
            "cf_score": cf_scores_norm[i],
            "cbf_score": 0.0,
            "has_cf": True,
            "has_cbf": False,
            "fallback_used": False,
            "cf_data": result,
            "cbf_data": None,
        }

    # Résultats CBF
    for idx in range(len(cbf_df)):
        if idx >= len(cbf_scores_norm):
            break
        row = cbf_df.iloc[idx]
        res_id = row["resource_id"]
        row_fallback = bool(row["fallback_used"]) if has_fallback_col else False

        if res_id in merged:
            merged[res_id]["cbf_score"] = cbf_scores_norm[idx]
            merged[res_id]["has_cbf"] = True
            merged[res_id]["fallback_used"] = row_fallback
            merged[res_id]["cbf_data"] = row
        else:
            merged[res_id] = {
                "resource_id": res_id,
                "cf_score": 0.0,
                "cbf_score": cbf_scores_norm[idx],
                "has_cf": False,
                "has_cbf": True,
                "fallback_used": row_fallback,
                "cf_data": None,
                "cbf_data": row,
            }

    # Scores hybrides — pénalité appliquée PAR ITEM (dépend de la ressource
    # CBF spécifique, pas de l'étudiant globalement).
    for res_id, data in merged.items():
        if data["has_cf"] and data["has_cbf"]:
            effective_cbf_weight = apply_fallback_penalty(
                cbf_weight, data["fallback_used"]
            )
            effective_cf_weight = cf_weight
            total_weight = effective_cf_weight + effective_cbf_weight

            if total_weight > 0:
                cf_w = effective_cf_weight / total_weight
                cbf_w = effective_cbf_weight / total_weight
            else:
                cf_w = cbf_w = 0.5

            data["hybrid_score"] = cf_w * data["cf_score"] + cbf_w * data["cbf_score"]

        elif data["has_cf"]:
            data["hybrid_score"] = data["cf_score"]

        else:
            # CBF seul présent pour cet item : fallback générique = signal plus faible
            penalty = FALLBACK_PENALTY if data["fallback_used"] else 1.0
            data["hybrid_score"] = data["cbf_score"] * penalty

    sorted_items = sorted(
        merged.values(), key=lambda x: x["hybrid_score"], reverse=True
    )

    num_recommendations = max(
        target_recommendations, min(max_recommendations, len(sorted_items))
    )
    if len(sorted_items) < target_recommendations:
        logger.info(
            f"ℹ️ Seulement {len(sorted_items)} candidats disponibles "
            f"(< cible {target_recommendations}) — pas de complément automatique."
        )

    recommendations = []
    for item in sorted_items[:num_recommendations]:
        try:
            if item["has_cbf"] and item["cbf_data"] is not None:
                row = item["cbf_data"]
                recommendations.append(
                    ResourceRecommendation(
                        resource_id=item["resource_id"],
                        title=row["title"],
                        type=get_resource_type(row["type"]),
                        relevance_score=item["hybrid_score"],
                        difficulty=AcademicLevel(row["difficulty"]),
                        estimated_time_min=row.get("estimated_time_min", 30),
                    )
                )
            elif item["has_cf"] and item["cf_data"] is not None:
                meta = _resource_metadata(item["resource_id"], resources_df)
                recommendations.append(
                    ResourceRecommendation(
                        resource_id=item["resource_id"],
                        title=meta["title"],
                        type=get_resource_type(meta["type"]),
                        relevance_score=item["hybrid_score"],
                        difficulty=AcademicLevel(meta["difficulty"]),
                        estimated_time_min=meta["estimated_time_min"],
                    )
                )
        except Exception as e:
            logger.warning(f"⚠️ Erreur pour {item['resource_id']}: {e}")
            continue

    logger.info(f"✅ {len(recommendations)} recommandations")
    return recommendations


def _convert_cbf_to_recommendations(
    cbf_df: pd.DataFrame, max_recommendations: int
) -> List[ResourceRecommendation]:
    """Convertit les résultats CBF seuls (CF vide) en ResourceRecommendation.

    Parameters
    ----------
    cbf_df : pd.DataFrame
        Résultats CBF (déjà triés par cbf_score décroissant en amont).
    max_recommendations : int
        Nombre maximum de résultats à retourner.

    Returns
    -------
    List[ResourceRecommendation]
        Liste convertie, scores renormalisés à [0, 1] sur ce sous-ensemble.
    """
    recommendations = []
    top_n = min(max_recommendations, len(cbf_df))

    scores_norm = normalize_scores(cbf_df["cbf_score"].tolist()[:top_n])

    for idx, score in zip(range(top_n), scores_norm):
        row = cbf_df.iloc[idx]
        try:
            recommendations.append(
                ResourceRecommendation(
                    resource_id=row["resource_id"],
                    title=row["title"],
                    type=get_resource_type(row["type"]),
                    relevance_score=score,
                    difficulty=AcademicLevel(row["difficulty"]),
                    estimated_time_min=row.get("estimated_time_min", 30),
                )
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur pour {row['resource_id']}: {e}")
            continue

    return recommendations


def _convert_cf_to_recommendations(
    cf_results: List[Dict[str, Any]],
    resources_df: pd.DataFrame,
    request_subject: Optional[str],
    max_recommendations: int,
) -> List[ResourceRecommendation]:
    """Convertit les résultats CF seuls (CBF vide) en ResourceRecommendation.

    Garde-fou matière : sans CBF, il n'y a plus de signal de contenu pour
    contrebalancer la sérendipité de CF. On filtre par matière avant de
    classer ; si le filtre vide tout, fallback total + warning.

    Parameters
    ----------
    cf_results : List[Dict[str, Any]]
        Résultats bruts du moteur CF.
    resources_df : pd.DataFrame
        Catalogue des ressources, pour récupérer la matière de chaque candidat.
    request_subject : Optional[str]
        Matière demandée (lowercase). Si None, le garde-fou est désactivé
        (warning loggé).
    max_recommendations : int
        Nombre maximum de résultats à retourner.

    Returns
    -------
    List[ResourceRecommendation]
        Liste filtrée (ou non filtrée si le filtre vide tout le pool).
    """
    if request_subject:
        filtered = [
            r
            for r in cf_results
            if _resource_metadata(r["resource_id"], resources_df)["subject"]
            == request_subject
        ]
    else:
        logger.warning("⚠️ request_subject non fourni — garde-fou matière désactivé")
        filtered = []

    if not filtered:
        if request_subject:
            logger.warning(
                f"⚠️ Aucun résultat CF ne correspond à la matière '{request_subject}' — "
                f"fallback sur CF non filtré (recommandations possiblement hors-sujet)"
            )
        pool = cf_results
    else:
        pool = filtered

    top_n = min(max_recommendations, len(pool))
    scores_norm = normalize_scores([r["predicted_score"] for r in pool][:top_n])

    recommendations = []
    for result, score in zip(pool[:top_n], scores_norm):
        meta = _resource_metadata(result["resource_id"], resources_df)
        recommendations.append(
            ResourceRecommendation(
                resource_id=result["resource_id"],
                title=meta["title"],
                type=get_resource_type(meta["type"]),
                relevance_score=score,
                difficulty=AcademicLevel(meta["difficulty"]),
                estimated_time_min=meta["estimated_time_min"],
            )
        )

    return recommendations


# ============================================================
# MOTEUR HYBRIDE
# ============================================================


class HybridEngine:
    """
    Moteur de recommandation hybride (CBF + CF).

    Charge en mémoire au démarrage :
        - resources.csv      : catalogue complet des ressources (normalisé
          en casse : subject et concept passés en lowercase pour matcher
          le contrat imposé par RecommendationRequest.strip_strings)
        - encoder CBF         : via get_or_build_vectorizer(self.resources_df),
          garantit que l'encodeur est entraîné sur les données normalisées
        - modèle CF           : via get_cf_engine()

    Point d'entrée : get_recommendations(request) → List[ResourceRecommendation]
    """

    def __init__(self):
        """
        Initialise le moteur hybride en chargeant les trois composants.

        Important : resources_df est normalisé en casse (subject, concept
        en lowercase) AVANT d'être passé à get_or_build_vectorizer, pour que
        l'encodeur soit entraîné sur les mêmes valeurs que celles produites
        par RecommendationRequest (dont le validator lowercase déjà subject
        et weak_concept). difficulty, type et learning_style n'ont pas
        besoin de cette normalisation : ce sont soit des Enums contraints
        à un vocabulaire fixe (AcademicLevel, LearningStyle), soit déjà
        cohérents en casse dans resources.csv (vérifié empiriquement).

        Raises
        ------
        FileNotFoundError
            Ne devrait plus se produire en pratique : get_or_build_vectorizer
            reconstruit automatiquement si le cache est absent ou périmé.
        """
        self.resources_df = pd.read_csv(
            os.path.join(DATA_DIR, "resources.csv"), quotechar='"'
        )
        # Normalisation canonique — doit matcher le contrat déjà imposé par
        # RecommendationRequest.strip_strings (subject/weak_concept en lowercase).
        self.resources_df["subject"] = (
            self.resources_df["subject"].str.strip().str.lower()
        )
        self.resources_df["concept"] = (
            self.resources_df["concept"].str.strip().str.lower()
        )

        self.cbf_encoder, self.cbf_matrix, _ = get_or_build_vectorizer(
            self.resources_df
        )
        self.cf_engine = get_cf_engine()
        logger.info(
            f"✅ HybridEngine initialisé avec {len(self.resources_df)} ressources"
        )

    def get_recommendations(
        self,
        request: RecommendationRequest,
        cf_weight: float = DEFAULT_CF_WEIGHT,
        cbf_weight: float = DEFAULT_CBF_WEIGHT,
        top_n: int = MAX_RECOMMENDATIONS,
    ) -> List[ResourceRecommendation]:
        """
        Génère les recommandations hybrides pour un apprenant.

        Appelle successivement CBF et CF, puis fusionne leurs résultats
        via hybrid_fusion() avec pondération alpha.

        Parameters
        ----------
        request : RecommendationRequest
            Profil apprenant validé par Pydantic.
        cf_weight : float, optional
            Poids du score CF dans la fusion (défaut : 0.4).
        cbf_weight : float, optional
            Poids du score CBF dans la fusion (défaut : 0.6).
        top_n : int, optional
            Nombre maximum de recommandations à retourner (défaut : 20).

        Returns
        -------
        List[ResourceRecommendation]
            Liste ordonnée par hybrid_score décroissant.
            Liste vide si aucune ressource ne passe les filtres.
        """
        logger.info(f"🔍 Génération pour {request.student_id}")

        logger.info("  → CBF Engine...")
        cbf_df = recommend_cbf(
            request=request,
            resources_df=self.resources_df,
            encoder=self.cbf_encoder,
            resource_matrix=self.cbf_matrix,
            top_n=top_n,
        )
        logger.info(f"    CBF: {len(cbf_df)} recommandations")

        logger.info("  → CF Engine...")
        cf_result = self.cf_engine.get_recommendations(
            student_id=request.student_id, n_recommendations=top_n
        )
        cf_results = cf_result.get("recommended_resources", [])
        logger.info(f"    CF: {len(cf_results)} recommandations")

        logger.info("  → Fusion...")

        recommendations = hybrid_fusion(
            cf_results=cf_results,
            cbf_df=cbf_df,
            resources_df=self.resources_df,
            request_subject=request.subject,
            cf_weight=cf_weight,
            cbf_weight=cbf_weight,
            target_recommendations=TARGET_RECOMMENDATIONS,
            max_recommendations=top_n,
        )

        logger.info(f"✅ {len(recommendations)} recommandations finales")
        return recommendations

    def get_status(self) -> Dict[str, Any]:
        """
        Retourne un snapshot de la configuration active du moteur.

        Returns
        -------
        Dict[str, Any]
            Clés : resources_count, cf_status, target_recommendations,
            max_recommendations, cf_weight, cbf_weight.
        """
        return {
            "resources_count": len(self.resources_df),
            "cf_status": self.cf_engine.get_status(),
            "target_recommendations": TARGET_RECOMMENDATIONS,
            "max_recommendations": MAX_RECOMMENDATIONS,
            "cf_weight": DEFAULT_CF_WEIGHT,
            "cbf_weight": DEFAULT_CBF_WEIGHT,
        }


# ============================================================
# SINGLETON
# ============================================================

_hybrid_engine: Optional[HybridEngine] = None


def get_hybrid_engine() -> HybridEngine:
    """
    Retourne l'instance singleton du HybridEngine.

    Crée l'instance au premier appel (chargement des modèles),
    puis retourne la même instance pour tous les appels suivants.

    Returns
    -------
    HybridEngine
        Instance unique partagée dans le process.
    """
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridEngine()
    return _hybrid_engine
