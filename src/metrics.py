"""
metrics.py — Métriques d'évaluation du moteur de recommandation.

Implémente les 5 métriques du cahier des charges (Tableau 9) :
    - Precision@K
    - Recall@K
    - NDCG@K
    - Diversité intra-liste
    - Couverture

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
    Définit le ground truth de pertinence pour un profil apprenant,
    indépendamment de tout moteur de recommandation testé — cette
    définition ne doit jamais être modifiée pour améliorer
    artificiellement un score de précision décevant.

    Une ressource est jugée pertinente si et seulement si : sa matière,
    son concept ET sa difficulté correspondent EXACTEMENT à ceux du
    profil (correspondance stricte des trois champs, pas de similarité
    approximative), ET ses prérequis sont satisfaits par
    past_interactions (via prerequisites_met, importé de CBF.py — même
    logique d'éligibilité que le moteur réel).

    Parameters
    ----------
    profile : dict
        Profil apprenant (subject, weak_concept, academic_level,
        past_interactions optionnel).
    resources_df : pd.DataFrame
        Catalogue complet des ressources.

    Returns
    -------
    set
        Ensemble des resource_id jugés pertinents. Vide si aucune
        ressource ne correspond exactement aux trois critères.
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
    """
    Precision@K = proportion des K premières ressources recommandées
    qui figurent dans l'ensemble des ressources pertinentes.

    Retourne 0.0 si les k premières recommandations sont vides (pas de
    division par zéro). Ne pénalise pas l'absence de ressources
    pertinentes dans le dataset — si relevant_ids est vide, toute
    recommandation compte comme un non-hit, la précision tombe
    mécaniquement à 0.0 pour ce profil.

    Parameters
    ----------
    recommended_ids : list
        Liste ordonnée des resource_id recommandés.
    relevant_ids : set
        Ground truth (voir get_relevant_resources).
    k : int
        Profondeur de coupe.

    Returns
    -------
    float
        Precision@K dans [0, 1].
    """
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int):
    """
    Recall@K = proportion des ressources pertinentes effectivement
    retrouvées parmi les K premières recommandations.

    Retourne None (métrique indéfinie, pas 0.0) si relevant_ids est
    vide — un rappel de 0.0 laisserait croire que le moteur a échoué,
    alors qu'il n'existe simplement aucune ressource pertinente à
    retrouver pour ce profil (cas fréquent vu la structure du dataset,
    voir mémoire de contexte sur les 56% de profils à un seul ground
    truth).

    Parameters
    ----------
    recommended_ids : list
    relevant_ids : set
    k : int

    Returns
    -------
    Optional[float]
        Recall@K dans [0, 1], ou None si relevant_ids est vide.
    """
    if not relevant_ids:
        return None
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int):
    """
    NDCG@K (Normalized Discounted Cumulative Gain) avec pertinence
    binaire (0/1) — mesure non seulement SI les ressources pertinentes
    sont présentes dans le top-K, mais aussi À QUELLE POSITION (un hit
    en position 1 vaut plus qu'un hit en position 10, pondération en
    1/log2(rang+2)).

    Calcule le DCG réel de la liste recommandée, puis le divise par
    l'IDCG (DCG idéal : toutes les ressources pertinentes disponibles,
    dans la limite de k, placées en tête de liste) pour normaliser sur
    [0, 1] indépendamment du nombre de ressources pertinentes existant
    pour ce profil.

    Retourne None si aucune ressource pertinente n'existe pour ce
    profil (IDCG = 0, ratio indéfini, pas 0.0).

    Parameters
    ----------
    recommended_ids : list
    relevant_ids : set
    k : int

    Returns
    -------
    Optional[float]
        NDCG@K dans [0, 1], ou None si relevant_ids est vide.
    """
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
    Mesure la diversité d'une liste de recommandations comme la
    distance cosinus moyenne entre chaque paire de ressources
    recommandées (1 - similarité), sur la base de leur représentation
    dans resource_matrix.

    Nécessite au moins 2 ressources valides (présentes dans
    resources_df) pour être calculable — retourne None sinon (pas 0.0 :
    l'absence de diversité mesurable n'est pas la même chose qu'une
    diversité nulle).

    Parameters
    ----------
    recommended_ids : list
        Resource_id recommandés à évaluer.
    resources_df : pd.DataFrame
        Catalogue complet, pour résoudre les index.
    encoder : OneHotEncoder
        Non utilisé directement dans le calcul (résidu de signature),
        conservé pour cohérence d'appel avec le reste du module.
    resource_matrix
        Matrice de représentation des ressources (pondérée ou non selon
        l'appelant — voir avertissement ci-dessus).

    Returns
    -------
    Optional[float]
        Diversité moyenne dans [0, 1] (0 = ressources identiques, 1 =
        orthogonales), ou None si moins de 2 ressources valides.
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
# COUVERTURE
# ─────────────────────────────────────────


def coverage(all_recommended_ids: list, resources_df) -> float:
    """
    Couverture = proportion du catalogue total de ressources qui
    apparaît au moins une fois dans les recommandations, sur l'ENSEMBLE
    du dataset de test (CDC Tableau 9, seuil >= 0.25).

    Ne pas appeler avec les recommandations d'un seul étudiant — la
    couverture se mesure sur la concaténation de tous les resource_id
    recommandés à tous les étudiants du run d'évaluation
    (all_recommended_ids doit contenir des doublons entre étudiants,
    dédupliqués en interne par cette fonction).

    Parameters
    ----------
    all_recommended_ids : list
        Concaténation de tous les resource_id recommandés à tous les
        profils testés (avec doublons).
    resources_df : pd.DataFrame
        Catalogue complet de référence.

    Returns
    -------
    float
        Couverture dans [0, 1]. 0.0 si le catalogue est vide.
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
    """
    Calcule la moyenne arithmétique d'une liste en ignorant
    silencieusement les valeurs None (métriques indéfinies pour
    certains profils — ex: recall_at_k ou ndcg_at_k quand
    relevant_ids est vide).

    Retourne 0.0 si la liste ne contient aucune valeur exploitable
    (tout est None), plutôt que de lever une exception de division par
    zéro.

    Parameters
    ----------
    values : list
        Valeurs numériques, potentiellement mêlées de None.

    Returns
    -------
    float
        Moyenne des valeurs non-None, ou 0.0 si aucune.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)
