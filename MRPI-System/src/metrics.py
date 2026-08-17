"""
metrics.py — Métriques d'évaluation du moteur de recommandation.

Implémente les 5 métriques du cahier des charges (Tableau 9) :
    - Precision@K
    - Recall@K
    - NDCG@K
    - Diversité intra-liste
    - Couverture (FIX : manquait totalement, livrable CDC explicite —
      "Couverture >= 0.25", ajoutée en fin de fichier)

Ground truth (pertinence) — définition stricte, indépendante du moteur
testé, voir docstring de get_relevant_resources().
"""

import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.CBF import prerequisites_met

# ─────────────────────────────────────────
# GROUND TRUTH
# ─────────────────────────────────────────


def get_relevant_resources(profile: dict, resources_df) -> set:
    """
    Ensemble des resource_id jugés pertinents pour un profil étudiant,
    selon une définition STRICTE : correspondance exacte subject +
    concept + difficulty, plus prérequis satisfaits.

    Cette fonction ne doit JAMAIS être modifiée pour "corriger" un score
    de précision décevant — si le moteur note mal des ressources qui
    correspondent à cette définition, c'est le moteur qu'il faut
    corriger, pas cette fonction.
    """
    subject = profile["subject"]
    concept = profile["weak_concept"]
    difficulty = profile["academic_level"]
    completed_ids = profile.get("past_interactions") or []

    candidates = resources_df[
        (resources_df["subject"] == subject)
        & (resources_df["concept"] == concept)
        & (resources_df["difficulty"] == difficulty)
    ]

    if candidates.empty:
        return set()

    relevant = candidates[
        candidates["prerequisites"].apply(lambda p: prerequisites_met(p, completed_ids))
    ]

    return set(relevant["resource_id"].tolist())


# ─────────────────────────────────────────
# METRIQUES DE RANG
# ─────────────────────────────────────────


def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Precision@K = |recommandées_top_k ∩ pertinentes| / K"""
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int):
    """Recall@K. None si relevant_ids est vide (métrique indéfinie)."""
    if not relevant_ids:
        return None
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int):
    """NDCG@K avec pertinence binaire (0/1). None si aucune ressource
    pertinente n'existe pour ce profil (IDCG = 0, indéfini)."""
    top_k = recommended_ids[:k]

    def dcg(ids):
        return sum(
            (1.0 if rid in relevant_ids else 0.0) / math.log2(i + 2)
            for i, rid in enumerate(ids)
        )

    actual_dcg = dcg(top_k)

    n_relevant = min(len(relevant_ids), k)
    if n_relevant == 0:
        return None

    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ─────────────────────────────────────────
# DIVERSITE INTRA-LISTE
# ─────────────────────────────────────────


def intra_list_diversity(recommended_ids: list, resources_df, encoder, resource_matrix):
    """
    Moyenne des distances cosinus entre chaque paire de ressources
    recommandées.

    ATTENTION méthodologique (non un bug, une décision à trancher) :
    si resource_matrix est la matrice CBF PONDÉRÉE (FEATURE_WEIGHTS,
    concept=3.0 / subject=2.0 / difficulty=2.0 / type=0.5), ces poids ont
    été choisis pour optimiser le SCORING de pertinence, pas pour définir
    ce qu'est la diversité. Deux ressources qui ne diffèrent que par leur
    type (poids 0.5) auront une distance artificiellement compressée,
    ce qui peut faire baisser ce score indépendamment de la diversité
    réelle du catalogue recommandé. Si le score de diversité est sous le
    seuil CDC (0.30), vérifier d'abord si resource_matrix ici devrait
    être une matrice one-hot NON pondérée dédiée à l'évaluation, avant de
    toucher à l'algorithme de recommandation. Décision à documenter dans
    le rapport, pas un ajustement silencieux.
    """
    id_to_index = {
        rid: idx for idx, rid in enumerate(resources_df["resource_id"].tolist())
    }
    indices = [id_to_index[rid] for rid in recommended_ids if rid in id_to_index]

    if len(indices) < 2:
        return None

    vectors = resource_matrix[indices]
    sim_matrix = cosine_similarity(vectors)

    n = len(indices)
    total_distance = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += 1 - sim_matrix[i, j]
            n_pairs += 1

    return total_distance / n_pairs if n_pairs > 0 else None


# ─────────────────────────────────────────
# COUVERTURE (FIX : métrique CDC manquante — Tableau 9, seuil >= 0.25)
# ─────────────────────────────────────────


def coverage(all_recommended_ids: list, resources_df) -> float:
    """
    Couverture = |ressources distinctes recommandées sur tout le dataset
    de test| / |catalogue total|.

    CDC (Tableau 9) : "Proportion du catalogue total de ressources qui
    apparaissent dans les recommandations sur l'ensemble du dataset de
    test. Couverture >= 0.25."

    Cette métrique était totalement absente du module — aucune fonction
    ne la calculait, alors que c'est un livrable explicite du rapport
    d'évaluation final (section 3.4.6).

    Parameters
    ----------
    all_recommended_ids : list
        Concaténation de TOUS les resource_id recommandés à TOUS les
        étudiants du dataset de test (avec doublons — dédupliqués ici).
        Ne pas passer seulement les recommandations d'un seul étudiant :
        la couverture se mesure sur l'ensemble du run d'évaluation, pas
        par profil individuel.
    resources_df : pd.DataFrame
        Catalogue complet des ressources (doit contenir 'resource_id').

    Returns
    -------
    float
        Couverture dans [0, 1]. 0.0 si le catalogue est vide (évite une
        division par zéro plutôt que de lever une exception).
    """
    total_catalog = set(resources_df["resource_id"].tolist())
    if not total_catalog:
        return 0.0

    recommended_distinct = set(all_recommended_ids)
    covered = recommended_distinct & total_catalog

    return len(covered) / len(total_catalog)


# ─────────────────────────────────────────
# AGREGATION
# ─────────────────────────────────────────


def average_ignoring_none(values: list) -> float:
    """Moyenne d'une liste en ignorant les None (métriques indéfinies)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)
