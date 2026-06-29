"""
Moteur de Filtrage Collaboratif (Collaborative Filtering Engine).

Ce module:
    1. Teste automatiquement si la SVD est utile sur les données
    2. Sélectionne la méthode la plus adaptée (SVD-KNN ou KNN brut)
    3. Fournit une interface unifiée pour les recommandations
    4. Met en cache le modèle (pickle) et invalide le cache si les données source changent

Architecture:
    - Une seule classe CFEngine
    - __init__ décide SVD vs Raw basé sur test de variance (ou charge le cache si valide)
    - Helpers partagés pour l'agrégation et le filtrage

Portée fonctionnelle (important, validé par le CDC) :
    Ce moteur n'a et ne doit avoir AUCUNE notion de matière, concept ou
    difficulté. Il opère uniquement sur la matrice utilisateur-item
    construite depuis interactions.csv : similarité comportementale entre
    étudiants, point final. Le CDC décrit explicitement cette absence de
    contexte de contenu comme la source de l'effet de sérendipité
    recherché (ressources inattendues mais pertinentes via des pairs
    similaires). Ne pas ajouter de filtre matière/difficulté dans ce
    fichier — si un filtrage est nécessaire, il doit vivre dans la couche
    de fusion (hybrid.py), qui a accès au contexte de la requête et peut
    décider au cas par cas (CBF présent vs CBF vide) où appliquer ce filtre.

Cold start :
    Si un étudiant n'a aucune ligne dans interactions.csv (aucune
    interaction passée), il n'apparaît pas dans la matrice utilisateur-item
    et get_recommendations() retourne student_exists=False avec une liste
    vide. C'est un comportement attendu, pas un bug — le filtrage
    collaboratif ne peut structurellement pas produire de signal pour un
    utilisateur sans historique. La couche de fusion (hybrid.py) gère ce
    cas en retombant sur CBF seul.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .user_itemMatrix import get_user_item_matrix
from .cf_config import (
    KNN_NEIGHBORS,
    MAX_NEIGHBORS,
    SVD_COMPONENTS,
    SVD_TEST_K,
    SVD_VARIANCE_THRESHOLD,
)
from .cache_utils import compute_source_fingerprint, is_cache_valid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CF_MODEL_PATH = os.path.join(MODELS_DIR, "cf_model.pkl")

# Fichiers source dont dépend la matrice user-item.
# IMPORTANT: adapte cette liste si get_user_item_matrix() lit d'autres fichiers
# (ex: students.csv) que ceux listés ici. Le cache ne sera invalidé que si
# un de CES fichiers change.
SOURCE_FILES = [
    os.path.join(DATA_DIR, "interactions.csv"),
    os.path.join(DATA_DIR, "resources.csv"),
]


# ============================================
# CONSTRUCTION / CHARGEMENT DU MODÈLE
# ============================================


def build_cf_model(save_path: str = CF_MODEL_PATH) -> Dict[str, Any]:
    """
    Construit le modèle CF (test SVD, choix de méthode, fit KNN) et le picklé.

    Returns
    -------
    Dict[str, Any]
        Métadonnées et objets du modèle entraîné (type, matrix, knn,
        svd_variance, et svd/U_reduced si SVD activé).
    """
    matrix = get_user_item_matrix()

    # Test de variance expliquée par SVD à k=SVD_TEST_K
    svd_test = TruncatedSVD(n_components=SVD_TEST_K, random_state=42)
    svd_test.fit(matrix.values)
    svd_variance = float(svd_test.explained_variance_ratio_.sum())

    use_svd = svd_variance >= SVD_VARIANCE_THRESHOLD

    if use_svd:
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
        U_reduced = svd.fit_transform(matrix.values)
        knn = NearestNeighbors(
            n_neighbors=min(MAX_NEIGHBORS, len(U_reduced)), metric="cosine"
        )
        knn.fit(U_reduced)
        model_data = {
            "type": "svd",
            "matrix": matrix,
            "svd": svd,
            "U_reduced": U_reduced,
            "knn": knn,
            "svd_variance": svd_variance,
        }
        logger.info(
            f"SVD-KNN construit: {matrix.shape[0]} étudiants → "
            f"{SVD_COMPONENTS} facteurs latents, variance expliquée: {svd_variance:.4f}"
        )
    else:
        knn = NearestNeighbors(
            n_neighbors=min(MAX_NEIGHBORS, len(matrix)), metric="cosine"
        )
        knn.fit(matrix.values)
        model_data = {
            "type": "raw",
            "matrix": matrix,
            "knn": knn,
            "svd_variance": svd_variance,
        }
        logger.info(
            f"Raw-KNN construit: {matrix.shape[0]} étudiants, "
            f"{matrix.shape[1]} ressources (variance SVD={svd_variance:.4f} "
            f"< seuil {SVD_VARIANCE_THRESHOLD})"
        )

    model_data["source_fingerprint"] = compute_source_fingerprint(SOURCE_FILES)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"✓ Modèle CF sauvegardé: {save_path}")

    return model_data


def load_cf_model(save_path: str = CF_MODEL_PATH) -> Dict[str, Any]:
    """Charge le modèle CF picklé depuis le disque.

    Parameters
    ----------
    save_path : str, optional
        Chemin du fichier pickle.

    Returns
    -------
    Dict[str, Any]
        Données du modèle (voir build_cf_model).

    Raises
    ------
    FileNotFoundError
        Si save_path n'existe pas.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"CF model not found at '{save_path}'. Run build_cf_model() first."
        )
    with open(save_path, "rb") as f:
        model_data = pickle.load(f)
    logger.info(f"✓ Modèle CF chargé: {save_path}")
    return model_data


def get_or_build_cf_model(save_path: str = CF_MODEL_PATH) -> Dict[str, Any]:
    """
    Charge le modèle pickle s'il existe ET s'il est à jour par rapport aux
    fichiers source. Sinon, reconstruit le modèle.

    Parameters
    ----------
    save_path : str, optional
        Chemin du fichier pickle.

    Returns
    -------
    Dict[str, Any]
        Données du modèle, chargées ou nouvellement construites.
    """
    if os.path.exists(save_path):
        model_data = load_cf_model(save_path)
        if is_cache_valid(model_data, SOURCE_FILES):
            return model_data
        logger.info("⚠️ Cache CF périmé (données source modifiées) — reconstruction.")
    else:
        logger.info("🔧 Aucun cache CF trouvé — construction initiale.")

    return build_cf_model(save_path)


# ============================================
# MOTEUR CF PRINCIPAL
# ============================================


class CFEngine:
    """
    Moteur de Filtrage Collaboratif.

    Sélectionne entre Raw KNN et SVD-KNN en fonction de la variance expliquée
    par SVD, avec mise en cache automatique (pickle invalidé si les données
    source changent).

    N'a aucune connaissance de matière/concept/difficulté — voir note de
    module en haut du fichier.
    """

    def __init__(self, model_path: str = CF_MODEL_PATH):
        model_data = get_or_build_cf_model(model_path)
        self._load_from_pickle(model_data)

    def _load_from_pickle(self, model_data: Dict[str, Any]) -> None:
        """Charge les données et reconstruit l'état runtime depuis le modèle.

        Parameters
        ----------
        model_data : Dict[str, Any]
            Données issues de build_cf_model / load_cf_model.
        """
        self.matrix = model_data["matrix"]
        self.svd_variance = model_data["svd_variance"]
        self.n_neighbors = KNN_NEIGHBORS

        if model_data["type"] == "svd":
            self.use_svd = True
            self.active_method = "svd_knn"
            self.svd = model_data["svd"]
            self.U_reduced = model_data["U_reduced"]
            self.U_reduced_df = pd.DataFrame(
                self.U_reduced,
                index=self.matrix.index,
                columns=[f"factor_{i + 1}" for i in range(SVD_COMPONENTS)],
            )
        else:
            self.use_svd = False
            self.active_method = "raw_knn"

        self.knn = model_data["knn"]
        logger.info(f"✅ CF chargé: {self.active_method}")

    def get_recommendations(
        self, student_id: str, n_recommendations: int = 3
    ) -> Dict[str, Any]:
        """
        Point d'entrée principal pour les recommandations.

        Délègue à _recommend() avec les matrices appropriées selon la méthode active:
        - SVD-KNN: feature_matrix = espace réduit, raw_matrix = matrice brute
        - Raw KNN: feature_matrix = raw_matrix = matrice brute

        Parameters
        ----------
        student_id : str
            Identifiant de l'étudiant (ex: 'STU-2026-0001').
        n_recommendations : int, optional
            Nombre de ressources à recommander (défaut : 3).

        Returns
        -------
        Dict[str, Any]
            Voir _recommend() / _empty_result() pour la structure complète.
            student_exists=False si l'étudiant n'a aucune ligne dans la
            matrice (cold start total — comportement attendu, pas un bug).
        """
        if self.use_svd:
            return self._recommend(
                student_id, n_recommendations, self.U_reduced_df, self.matrix
            )
        return self._recommend(student_id, n_recommendations, self.matrix, self.matrix)

    def _recommend(
        self,
        student_id: str,
        n_recommendations: int,
        feature_matrix: pd.DataFrame,
        raw_matrix: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Méthode unifiée de recommandation (Raw KNN et SVD-KNN).

        Parameters
        ----------
        student_id : str
            Identifiant de l'étudiant.
        n_recommendations : int
            Nombre de ressources à recommander.
        feature_matrix : pd.DataFrame
            Matrice utilisée pour le calcul des voisins (espace réduit
            si SVD, matrice brute sinon).
        raw_matrix : pd.DataFrame
            Matrice brute utilisateur-item, utilisée pour le calcul des
            scores pondérés une fois les voisins identifiés.

        Returns
        -------
        Dict[str, Any]
            Résultat complet (voir get_recommendations) ou résultat vide
            via _empty_result si l'étudiant est absent ou sans voisin valide.
        """
        if student_id not in feature_matrix.index:
            return self._empty_result(student_id, exists=False)

        student_idx = feature_matrix.index.get_loc(student_id)

        k = min(self.n_neighbors + 1, len(feature_matrix))
        distances, indices = self.knn.kneighbors(
            feature_matrix.values[student_idx].reshape(1, -1), n_neighbors=k
        )

        neighbor_indices = indices[0][1:]
        neighbor_distances = distances[0][1:]

        similar_students = self._filter_similar_students(
            neighbor_indices, neighbor_distances, feature_matrix
        )

        if not similar_students:
            return self._empty_result(
                student_id, exists=True, message="Aucun voisin similaire trouvé"
            )

        recommendations = self._compute_recommendations(
            student_idx, similar_students, n_recommendations, raw_matrix
        )

        return {
            "student_id": student_id,
            "student_exists": True,
            "method": self.active_method,
            "similar_students": similar_students,
            "recommended_resources": recommendations,
            "num_similar_students": len(similar_students),
            "num_recommendations": len(recommendations),
        }

    def _filter_similar_students(
        self, indices: np.ndarray, distances: np.ndarray, feature_matrix: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Filtre les voisins avec similarité > 0.

        Parameters
        ----------
        indices : np.ndarray
            Indices des voisins retournés par KNN.
        distances : np.ndarray
            Distances cosinus correspondantes.
        feature_matrix : pd.DataFrame
            Matrice utilisée pour résoudre les indices en student_id.

        Returns
        -------
        List[Dict[str, Any]]
            Liste de voisins valides (similarity_score > 0), avec
            student_id, distance, similarity_score.
        """
        similar_students = []
        for idx, dist in zip(indices, distances):
            similarity = 1 - float(dist)
            if similarity > 0:
                similar_students.append(
                    {
                        "student_id": feature_matrix.index[idx],
                        "distance": float(dist),
                        "similarity_score": similarity,
                    }
                )
        return similar_students

    def _compute_recommendations(
        self,
        student_idx: int,
        similar_students: List[Dict[str, Any]],
        n_recommendations: int,
        raw_matrix: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Calcule les scores pondérés des ressources à recommander.

        Les scores des voisins sont pondérés par leur similarity_score
        (normalisé pour sommer à 1), puis sommés par ressource. Les
        ressources déjà vues par l'étudiant sont exclues (mises à -inf
        puis filtrées par le test > 0 final).

        Parameters
        ----------
        student_idx : int
            Index de l'étudiant dans raw_matrix.
        similar_students : List[Dict[str, Any]]
            Voisins valides avec leurs similarity_score.
        n_recommendations : int
            Nombre de ressources à retourner.
        raw_matrix : pd.DataFrame
            Matrice brute utilisateur-item.

        Returns
        -------
        List[Dict[str, Any]]
            Ressources recommandées (resource_id, predicted_score), triées
            par score décroissant, score > 0 uniquement.
        """
        X_raw = raw_matrix.values
        student_vector_raw = X_raw[student_idx]
        student_rated = student_vector_raw > 0

        similar_indices_raw = [
            raw_matrix.index.get_loc(s["student_id"]) for s in similar_students
        ]
        similar_vectors_raw = X_raw[similar_indices_raw]

        weights = np.array([s["similarity_score"] for s in similar_students])
        weights = weights / weights.sum()

        weighted_scores = np.zeros(X_raw.shape[1])
        for i, vec in enumerate(similar_vectors_raw):
            weighted_scores += weights[i] * vec

        weighted_scores[student_rated] = -np.inf

        top_indices = np.argsort(weighted_scores)[::-1][:n_recommendations]

        return [
            {
                "resource_id": raw_matrix.columns[idx],
                "predicted_score": float(weighted_scores[idx]),
            }
            for idx in top_indices
            if weighted_scores[idx] > 0
        ]

    def _empty_result(
        self, student_id: str, exists: bool = False, message: str = ""
    ) -> Dict[str, Any]:
        """Retourne un résultat vide formaté (étudiant non trouvé ou pas de voisins).

        Parameters
        ----------
        student_id : str
            Identifiant de l'étudiant.
        exists : bool, optional
            False si l'étudiant n'a aucune ligne dans la matrice (cold
            start total). True si présent mais sans voisin valide.
        message : str, optional
            Message explicatif (ex: "Aucun voisin similaire trouvé").

        Returns
        -------
        Dict[str, Any]
            Résultat vide, structure identique à get_recommendations.
        """
        return {
            "student_id": student_id,
            "student_exists": exists,
            "method": self.active_method,
            "message": message,
            "similar_students": [],
            "recommended_resources": [],
            "num_similar_students": 0,
            "num_recommendations": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Retourne l'état du moteur CF.

        Returns
        -------
        Dict[str, Any]
            matrix_shape, svd_enabled, svd_variance_at_k5,
            svd_variance_threshold, active_method, svd_components,
            knn_neighbors.
        """
        return {
            "matrix_shape": self.matrix.shape,
            "svd_enabled": self.use_svd,
            "svd_variance_at_k5": self.svd_variance,
            "svd_variance_threshold": SVD_VARIANCE_THRESHOLD,
            "active_method": self.active_method,
            "svd_components": SVD_COMPONENTS if self.use_svd else None,
            "knn_neighbors": KNN_NEIGHBORS,
        }


# ============================================
# SINGLETON
# ============================================

_cf_engine: Optional[CFEngine] = None


def get_cf_engine() -> CFEngine:
    """Retourne l'instance unique du moteur CF.

    Returns
    -------
    CFEngine
        Instance singleton, créée au premier appel.
    """
    global _cf_engine
    if _cf_engine is None:
        _cf_engine = CFEngine()
    return _cf_engine


def reset_cf_engine() -> None:
    """
    Force la réinitialisation du singleton en mémoire.

    Utile en tests ou si tu modifies les données dans le même process
    sans redémarrer — sinon le singleton garde l'ancien état même si
    le fichier pickle a été régénéré.
    """
    global _cf_engine
    _cf_engine = None


def get_recommendations(student_id: str, n_recommendations: int = 3) -> Dict[str, Any]:
    """Fonction de convenance pour obtenir des recommandations.

    Parameters
    ----------
    student_id : str
        Identifiant de l'étudiant.
    n_recommendations : int, optional
        Nombre de ressources à recommander (défaut : 3).

    Returns
    -------
    Dict[str, Any]
        Voir CFEngine.get_recommendations.
    """
    engine = get_cf_engine()
    return engine.get_recommendations(student_id, n_recommendations)
