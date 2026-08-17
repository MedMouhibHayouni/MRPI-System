"""
Hybrid Recommendation Engine.

Combine les scores du moteur CBF et du moteur CF en un score hybride pondéré :

    hybrid_score = alpha × CBF_score + (1 - alpha) × CF_score

alpha défaut = 0.6 (CDC 3.4.4), ajustable via l'API (voir FIX plus bas).

FIX vs version précédente :
1. `alpha` n'était nulle part exposé au client de l'API alors que le CDC
   l'exige explicitement ("ajustable via l'API"). RecommendationRequest a
   maintenant un champ `alpha` optionnel ; HybridEngine.get_recommendations
   le lit et l'utilise si fourni, sinon retombe sur les défauts CDC.
2. `description` ne circulait jusqu'à aucun ResourceRecommendation ->
   adapt_wording() (LLM_API.py) recevait systématiquement une description
   vide, peu importe la qualité du LLM. _resource_metadata() et les deux
   branches de construction de recommandations (CBF et CF) portent
   maintenant ce champ.
3. `import numpy as np` était mort (aucun np.* utilisé dans ce fichier) —
   supprimé.
4. get_or_build_vectorizer() renvoie maintenant aussi `weight_vector` (voir
   fix CBF.py) — HybridEngine le récupère et le transmet à recommend_cbf()
   pour éviter un recalcul redondant à chaque requête.
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

FALLBACK_PENALTY = 0.8
MIN_CANDIDATES_FOR_NORM = 3

TARGET_RECOMMENDATIONS = 5
# NOTE (non résolu, pas un bug de code) : agit toujours comme un plancher
# indicatif, pas un plafond garanti. Si moins de candidats existent, la
# liste renvoyée est plus courte que 5 — violation CDC (min 5) qui est un
# problème de méthodologie/dataset, pas quelque chose que je corrige en
# silence ici. À trancher avec Ikram (même dossier que Precision@K).
MAX_RECOMMENDATIONS = 20

# ============================================================
# UTILITAIRES
# ============================================================


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalise une liste de scores à [0, 1] (min-max par liste)."""
    if not scores:
        return []
    if len(scores) < MIN_CANDIDATES_FOR_NORM:
        logger.info(
            f"ℹ️ Normalisation sur seulement {len(scores)} candidat(s) — "
            f"scores bruts conservés (pas de min-max fiable)"
        )
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def apply_fallback_penalty(weight: float, fallback_used: bool) -> float:
    return weight * FALLBACK_PENALTY if fallback_used else weight


def get_resource_type(value: str) -> ResourceType:
    try:
        return ResourceType(value.lower())
    except ValueError:
        logger.warning(f"⚠️ Type inconnu: {value}, utilisation de 'video'")
        return ResourceType("video")


def _resource_metadata(resource_id: str, resources_df: pd.DataFrame) -> Dict[str, Any]:
    """Récupère les métadonnées d'une ressource depuis le DataFrame déjà chargé.

    FIX : ajout de 'description' — absente avant, ce qui cassait
    adapt_wording() en aval. resources_df n'a pas nécessairement de colonne
    'description' selon la version du dataset ; .get() renvoie "" si absente,
    donc ce fix ne plante jamais même sur un ancien resources.csv.
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
    """Fusionne les résultats CF et CBF (voir docstring module pour le détail
    du comportement CF non filtré / garde-fou matière)."""

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

    cf_scores_norm = normalize_scores([r["predicted_score"] for r in cf_results])
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

    cbf_ids = set(cbf_df["resource_id"])
    cf_ids = {r["resource_id"] for r in cf_results}
    overlap = cbf_ids & cf_ids
    logger.info(
        f"Chevauchement CBF∩CF : {len(overlap)} / CBF={len(cbf_ids)}, CF={len(cf_ids)}"
    )
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
            # FIX : description récupérée dans TOUS les cas via
            # _resource_metadata, que la ressource vienne de CBF ou de CF.
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
    """Convertit les résultats CBF seuls (CF vide) en ResourceRecommendation."""
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
    """Convertit les résultats CF seuls (CBF vide) en ResourceRecommendation."""
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
    """Moteur de recommandation hybride (CBF + CF)."""

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

        # FIX : get_or_build_vectorizer() renvoie maintenant aussi
        # weight_vector — récupéré ici et propagé à recommend_cbf() pour
        # éviter un recalcul par requête (voir CBF.py).
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
        """Génère les recommandations hybrides pour un apprenant.

        FIX (CDC 3.4.4, alpha ajustable via l'API) : si cf_weight/cbf_weight
        ne sont pas fournis explicitement par l'appelant, on regarde
        request.alpha. S'il est présent, alpha = cbf_weight et
        (1 - alpha) = cf_weight. Sinon, retombe sur les défauts CDC
        (0.6 / 0.4) — comportement strictement identique à avant pour
        tout client qui n'envoie pas ce champ.
        """
        if cbf_weight is None or cf_weight is None:
            if request.alpha is not None:
                cbf_weight = request.alpha
                cf_weight = 1.0 - request.alpha
            else:
                cbf_weight = DEFAULT_CBF_WEIGHT
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
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridEngine()
    return _hybrid_engine
