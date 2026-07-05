"""
Hybrid Recommendation Engine.

Combine les scores du moteur CBF et du moteur CF en un score hybride pondéré :

    hybrid_score = alpha × CBF_score + (1 - alpha) × CF_score

Le poids CBF par défaut est 0.6 et le poids CF par défaut est 0.4.
"""

import pandas as pd
import os
import logging
from typing import List, Dict, Any, Optional

from src.schemas.request import RecommendationRequest
from src.schemas.response import ResourceRecommendation, ResourceType, AcademicLevel
from src.CF import get_cf_engine
from src.CBF import recommend_cbf, get_or_build_vectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DEFAULT_CBF_WEIGHT = 0.6
DEFAULT_CF_WEIGHT = 0.4
CF_SCORE_MAX = 5.0
FALLBACK_PENALTY = 0.8
MIN_CANDIDATES_FOR_NORM = 3

TARGET_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 20

# ============================================================
# UTILITAIRES
# ============================================================


def normalize_scores(
    scores: List[float], max_value: Optional[float] = None
) -> List[float]:
    """Normalise une liste de scores à [0, 1].

    Si len(scores) >= MIN_CANDIDATES_FOR_NORM (3) : min-max classique sur
    la liste (comportement inchangé).

    Si len(scores) < MIN_CANDIDATES_FOR_NORM :
    - max_value fourni (borne théorique connue, ex: CF_SCORE_MAX pour les
      scores CF bruts) : division par max_value, garantit un résultat
      dans [0, 1] même sur un seul candidat, sans prétendre à une
      normalisation statistique qu'un échantillon aussi petit ne permet
      pas.
    - max_value=None (scores déjà bornés [0,1] par construction, ex: CBF
      cosine similarity sur vecteurs non-négatifs) : scores retournés
      tels quels.

    Les scores CF peuvent fournir une borne théorique pour garder le
    résultat dans l'intervalle attendu même avec très peu de candidats.
    """
    if not scores:
        return []
    if len(scores) < MIN_CANDIDATES_FOR_NORM:
        if max_value is not None and max_value > 0:
            logger.info(
                f"ℹ️ Normalisation sur seulement {len(scores)} candidat(s) — "
                f"division par borne théorique {max_value}"
            )
            return [min(s / max_value, 1.0) for s in scores]
        logger.info(
            f"ℹ️ Normalisation sur seulement {len(scores)} candidat(s) — "
            f"scores bruts conservés (déjà bornés par construction)"
        )
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def apply_fallback_penalty(weight: float, fallback_used: bool) -> float:
    """
    Réduit un poids de fusion (cbf_weight typiquement) d'un facteur
    FALLBACK_PENALTY (0.8) si la ressource concernée est un résultat de
    repli CBF (fallback_used=True, c'est-à-dire que son type de contenu
    ne correspond pas au style d'apprentissage demandé). Une ressource
    non-fallback conserve son poids intact.

    Parameters
    ----------
    weight : float
        Poids d'origine (avant pénalité).
    fallback_used : bool
        Si True, applique la pénalité.

    Returns
    -------
    float
        Poids ajusté.
    """
    return weight * FALLBACK_PENALTY if fallback_used else weight


def get_resource_type(value: str) -> ResourceType:
    """
    Convertit une valeur brute de type de contenu (string, potentiellement
    mal formée ou absente du catalogue) en enum ResourceType valide.

    Si la conversion échoue (valeur inconnue de l'enum), journalise un
    avertissement et retombe sur ResourceType("video") comme valeur par
    défaut plutôt que de laisser une exception remonter et faire échouer
    toute la génération de recommandations pour une seule ressource mal
    cataloguée.

    Parameters
    ----------
    value : str
        Type de contenu brut (ex: depuis une colonne CSV).

    Returns
    -------
    ResourceType
        Enum validé, ou ResourceType.video en repli.
    """
    try:
        return ResourceType(value.lower())
    except ValueError:
        logger.warning(f"⚠️ Type inconnu: {value}, utilisation de 'video'")
        return ResourceType("video")


def _resource_metadata(resource_id: str, resources_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Récupère les métadonnées d'affichage d'une ressource (title, type,
    difficulty, subject, estimated_time_min, description) à partir de
    son resource_id, en cherchant dans le DataFrame déjà chargé en
    mémoire (pas d'accès disque).

    Si la ressource n'existe pas dans resources_df (incohérence
    possible entre le résultat CF/CBF et le catalogue courant), retourne
    un jeu de métadonnées de repli générique plutôt que de lever une
    exception — la recommandation reste affichable, avec un titre
    générique "Resource <id>".

    Le champ description est toujours présent dans le résultat (chaîne
    vide si absente du DataFrame source), car il alimente
    adapt_wording()/adapt_wording_batch() en aval — une description
    manquante ne doit jamais faire planter l'appel LLM, seulement lui
    donner moins de matière à reformuler.

    Parameters
    ----------
    resource_id : str
    resources_df : pd.DataFrame
        Catalogue complet des ressources.

    Returns
    -------
    Dict[str, Any]
        {"title", "type", "difficulty", "subject",
        "estimated_time_min", "description"}.
    """

    row = resources_df[resources_df["resource_id"] == resource_id]
    if row.empty:
        return {
            "title": f"Resource {resource_id}",
            "type": "video",
            "difficulty": "beginner",
            "subject": None,
            "estimated_time_min": 30,
            "description": "",
        }
    r = row.iloc[0]
    return {
        "title": r["title"],
        "type": r["type"],
        "difficulty": r["difficulty"],
        "subject": r.get("subject"),
        "estimated_time_min": r.get("estimated_time_min", 30),
        "description": r.get("description", "") or "",
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
    """
    Fusionne les résultats du moteur CF et du moteur CBF en une liste
    unique de recommandations classées par score hybride, selon
    Score_hybride = cbf_weight * Score_CB + cf_weight * Score_CF (CDC
    3.4.4).

    Trois cas de figure :
    1. CF et CBF tous deux vides : retourne une liste vide.
    2. CF vide, CBF non vide : délègue entièrement à
       _convert_cbf_to_recommendations() (pas de fusion à faire).
    3. CBF vide, CF non vide : délègue à _convert_cf_to_recommendations(),
       qui applique un garde-fou matière (les résultats CF ne sont
       filtrés sur request_subject QUE dans cette branche — dans le cas
       de fusion normale ci-dessous, CF n'est jamais filtré par matière,
       volontairement, pour préserver l'effet de sérendipité voulu par
       le CDC 3.4.4).
    4. Cas général (les deux non vides) : normalise indépendamment les
       scores CF et CBF (min-max, voir normalize_scores), fusionne les
       deux ensembles de résultats par resource_id (une ressource peut
       apparaître dans les deux, une seule, ou l'autre), calcule un
       score hybride par ressource :
       - Présente dans les deux : moyenne pondérée normalisée des deux
         scores, où le poids CBF est réduit par apply_fallback_penalty
         si la ressource est un résultat de repli CBF.
       - Uniquement CF : hybrid_score = cf_score (le poids de fusion
         n'entre pas en jeu, faute de contrepartie CBF).
       - Uniquement CBF : hybrid_score = cbf_score, pénalisé par
         FALLBACK_PENALTY si fallback_used.
       Trie par hybrid_score décroissant, tronque à
       max_recommendations, puis construit les objets
       ResourceRecommendation finaux en récupérant les métadonnées via
       _resource_metadata() pour CHAQUE ressource (que sa source soit
       CBF ou CF), garantissant que description (et les autres champs
       d'affichage) sont toujours peuplés indépendamment de la source.

    Si moins de target_recommendations candidats sont disponibles après
    fusion, aucune complémentation automatique n'est appliquée.

    Parameters
    ----------
    cf_results : List[Dict[str, Any]]
        Résultats bruts du CF (voir CFEngine._compute_recommendations).
    cbf_df : pd.DataFrame
        Résultats bruts du CBF (voir recommend_cbf).
    resources_df : pd.DataFrame
        Catalogue complet, pour résoudre les métadonnées manquantes.
    request_subject : Optional[str]
        Matière de la requête, utilisée uniquement comme garde-fou
        quand CBF est vide et CF seul doit être servi.
    cf_weight, cbf_weight : float
        Poids de fusion (1 - alpha et alpha respectivement, CDC 3.4.4).
    target_recommendations : int
        Cible indicative (CDC : minimum 5), non garantie.
    max_recommendations : int
        Plafond dur du nombre de résultats retournés (CDC : maximum 20).

    Returns
    -------
    List[ResourceRecommendation]
        Liste triée par pertinence décroissante, taille entre 0 et
        max_recommendations.
    """
    cf_empty = not cf_results
    cbf_empty = cbf_df.empty

    if cf_empty and cbf_empty:
        logger.warning("❌ CF et CBF sont vides")
        return []

    if cf_empty and not cbf_empty:
        logger.info("⚠️ CF vide → CBF seul")
        return _convert_cbf_to_recommendations(
            cbf_df, resources_df, max_recommendations
        )

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

    cf_scores_norm = normalize_scores(
        [r["predicted_score"] for r in cf_results], max_value=CF_SCORE_MAX
    )
    cbf_scores_norm = normalize_scores(cbf_df["cbf_score"].tolist())
    merged: Dict[str, Dict[str, Any]] = {}

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
            penalty = FALLBACK_PENALTY if data["fallback_used"] else 1.0
            data["hybrid_score"] = data["cbf_score"] * penalty

    sorted_items = sorted(
        merged.values(), key=lambda x: x["hybrid_score"], reverse=True
    )

    num_recommendations = min(max_recommendations, len(sorted_items))
    if len(sorted_items) < target_recommendations:
        logger.info(
            f"ℹ️ Seulement {len(sorted_items)} candidats disponibles "
            f"(< cible {target_recommendations}) — pas de complément automatique."
        )

    recommendations = []
    for item in sorted_items[:num_recommendations]:
        try:
            meta = _resource_metadata(item["resource_id"], resources_df)

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
                        description=meta["description"],
                    )
                )
            elif item["has_cf"] and item["cf_data"] is not None:
                recommendations.append(
                    ResourceRecommendation(
                        resource_id=item["resource_id"],
                        title=meta["title"],
                        type=get_resource_type(meta["type"]),
                        relevance_score=item["hybrid_score"],
                        difficulty=AcademicLevel(meta["difficulty"]),
                        estimated_time_min=meta["estimated_time_min"],
                        description=meta["description"],
                    )
                )
        except Exception as e:
            logger.warning(f"⚠️ Erreur pour {item['resource_id']}: {e}")
            continue

    logger.info(f"✅ {len(recommendations)} recommandations")
    return recommendations


def _convert_cbf_to_recommendations(
    cbf_df: pd.DataFrame, resources_df: pd.DataFrame, max_recommendations: int
) -> List[ResourceRecommendation]:
    """
    Convertit des résultats CBF seuls (cas où le CF n'a rien produit,
    typiquement un étudiant en cold-start CF) en objets
    ResourceRecommendation, en normalisant les scores CBF sur le
    sous-ensemble effectivement retourné (top_n premiers, pas
    l'ensemble du DataFrame cbf_df).

    Parameters
    ----------
    cbf_df : pd.DataFrame
        Résultats CBF (voir recommend_cbf), déjà triés par cbf_score
        décroissant.
    resources_df : pd.DataFrame
        Catalogue complet, pour les métadonnées d'affichage.
    max_recommendations : int
        Plafond de résultats à convertir.

    Returns
    -------
    List[ResourceRecommendation]
        Liste triée par relevance_score décroissant.
    """
    recommendations = []
    top_n = min(max_recommendations, len(cbf_df))

    scores_norm = normalize_scores(cbf_df["cbf_score"].tolist()[:top_n])

    for idx, score in zip(range(top_n), scores_norm):
        row = cbf_df.iloc[idx]
        try:
            meta = _resource_metadata(row["resource_id"], resources_df)
            recommendations.append(
                ResourceRecommendation(
                    resource_id=row["resource_id"],
                    title=row["title"],
                    type=get_resource_type(row["type"]),
                    relevance_score=score,
                    difficulty=AcademicLevel(row["difficulty"]),
                    estimated_time_min=row.get("estimated_time_min", 30),
                    description=meta["description"],
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
    """
    Convertit des résultats CF seuls (cas où le CBF n'a rien produit —
    ex: aucune ressource éligible après filtrage prérequis/déjà-vues)
    en objets ResourceRecommendation, en appliquant un garde-fou matière
    optionnel.

    Si request_subject est fourni : ne conserve que les résultats CF
    dont la matière (résolue via _resource_metadata) correspond
    exactement à request_subject. Si ce filtrage ne laisse aucun
    résultat, retombe sur l'ensemble non filtré des résultats CF (avec
    avertissement journalisé — les recommandations peuvent alors être
    hors-sujet, mais on préfère ça à une liste vide).

    Si request_subject n'est pas fourni : le garde-fou est
    explicitement désactivé (filtered reste vide dès le départ), donc
    le pool CF non filtré est utilisé directement.

    Parameters
    ----------
    cf_results : List[Dict[str, Any]]
        Résultats bruts du CF.
    resources_df : pd.DataFrame
        Catalogue complet.
    request_subject : Optional[str]
        Matière à privilégier, ou None pour désactiver le filtrage.
    max_recommendations : int
        Plafond de résultats.

    Returns
    -------
    List[ResourceRecommendation]
        Liste triée par relevance_score décroissant.
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
                description=meta["description"],
            )
        )

    return recommendations


# ============================================================
# MOTEUR HYBRIDE
# ============================================================


class HybridEngine:
    """
    Initialise le moteur hybride : charge le catalogue complet des
    ressources depuis resources.csv, normalise la casse des colonnes
    subject et concept (strip + lowercase) pour garantir une
    correspondance fiable avec le profil apprenant (qui peut arriver
    avec une casse différente), puis construit/charge le vectorizer
    CBF (encoder, matrice pondérée, weight_vector) et le moteur CF
    via leurs fonctions get_or_build respectives.

    weight_vector est récupéré ici une seule fois par instance et
    propagé à chaque appel de recommend_cbf() dans
    get_recommendations(), pour éviter de le recalculer à chaque
    requête (voir CBF.py).
    """

    def __init__(self):
        self.resources_df = pd.read_csv(
            os.path.join(DATA_DIR, "resources.csv"), quotechar='"'
        )
        self.resources_df["subject"] = (
            self.resources_df["subject"].str.strip().str.lower()
        )
        self.resources_df["concept"] = (
            self.resources_df["concept"].str.strip().str.lower()
        )

        (
            self.cbf_encoder,
            self.cbf_matrix,
            _,
            self.cbf_weight_vector,
        ) = get_or_build_vectorizer(self.resources_df)
        self.cf_engine = get_cf_engine()
        logger.info(
            f"✅ HybridEngine initialisé avec {len(self.resources_df)} ressources"
        )

    def get_recommendations(
        self,
        request: RecommendationRequest,
        cf_weight: Optional[float] = None,
        cbf_weight: Optional[float] = None,
        top_n: int = MAX_RECOMMENDATIONS,
    ) -> List[ResourceRecommendation]:
        """
        Point d'entrée principal : génère les recommandations hybrides
        complètes pour un profil apprenant (appelé depuis main.py via
        ml_executor, hors event loop async — c'est un appel CPU-bound
        synchrone).

        cf_weight / cbf_weight, si non fournis explicitement, retombent
        sur les constantes DEFAULT_CF_WEIGHT (0.4) / DEFAULT_CBF_WEIGHT
        (0.6) — CDC 3.4.4, alpha = 0.6 par défaut.

        Pipeline : appelle recommend_cbf() (CBF), puis
        self.cf_engine.get_recommendations() (CF), puis fusionne les
        deux via hybrid_fusion().

        Parameters
        ----------
        request : RecommendationRequest
            Profil apprenant validé.
        cf_weight, cbf_weight : Optional[float]
            Poids de fusion explicites, prioritaires sur les défauts
            CDC s'ils sont fournis par l'appelant direct (pas par le
            client API — voir avertissement ci-dessus).
        top_n : int
            Nombre maximum de recommandations (CDC : max 20).

        Returns
        -------
        List[ResourceRecommendation]
            Recommandations finales triées, éventuellement vide.
        """
        if cbf_weight is None:
            cbf_weight = DEFAULT_CBF_WEIGHT
        if cf_weight is None:
            cf_weight = DEFAULT_CF_WEIGHT
        logger.info(f"🔍 Génération pour {request.student_id} (alpha={cbf_weight})")

        logger.info("  → CBF Engine...")
        cbf_df = recommend_cbf(
            request=request,
            resources_df=self.resources_df,
            encoder=self.cbf_encoder,
            resource_matrix=self.cbf_matrix,
            weight_vector=self.cbf_weight_vector,
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
        Retourne un instantané de configuration du moteur hybride
        (nombre de ressources, statut du sous-moteur CF, cibles et
        poids par défaut) — diagnostic/monitoring uniquement.

        Returns
        -------
        Dict[str, Any]
            resources_count, cf_status, target_recommendations,
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


_hybrid_engine: Optional[HybridEngine] = None


def get_hybrid_engine() -> HybridEngine:
    """
    Retourne l'instance singleton globale de HybridEngine, construite
    paresseusement au premier appel — évite de recharger le catalogue
    de ressources et de reconstruire/recharger CBF+CF à chaque requête
    /recommendations.

    Returns
    -------
    HybridEngine
        Instance partagée à l'échelle du process (instanciée une seule
        fois au démarrage de main.py via `engine = get_hybrid_engine()`).
    """
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridEngine()
    return _hybrid_engine
