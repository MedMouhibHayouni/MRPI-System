"""
Moteur de Filtrage Collaboratif (Collaborative Filtering Engine).

Construit ou charge un modèle CF (raw KNN ou SVD+KNN selon la variance
expliquée) à partir de la matrice utilisateur-ressource. Le cache sur
disque (cf_model.pkl) est invalidé automatiquement si les fichiers
sources (interactions.csv, resources.csv) changent, via fingerprint.

CFEngine.get_recommendations(student_id) retourne les k plus proches
voisins d'un étudiant et une liste de ressources recommandées, pondérée
par similarité. Si l'étudiant n'existe pas dans la matrice, retourne
student_exists=False sans erreur.

model_path est résolu au moment de l'instanciation (pas figé comme
valeur par défaut à l'import), ce qui permet de pointer l'engine vers
un modèle différent — utile pour l'isolation des tests et pour
reset_cf_engine(), qui force une reconstruction propre.
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

SOURCE_FILES = [
    os.path.join(DATA_DIR, "interactions.csv"),
    os.path.join(DATA_DIR, "resources.csv"),
]


def build_cf_model(save_path: str = CF_MODEL_PATH) -> Dict[str, Any]:
    """
    Construit un modèle de Filtrage Collaboratif à partir de la matrice
    utilisateur-ressource actuelle, en choisissant automatiquement entre
    deux stratégies selon la variance expliquée par une SVD test.

    Étapes :
    1. Charge la matrice via get_user_item_matrix().
    2. Fit une TruncatedSVD à SVD_TEST_K composantes uniquement pour
       mesurer la variance expliquée cumulée (svd_variance) — ce n'est
       pas la SVD finale utilisée pour les recommandations, seulement
       un diagnostic pour décider si une réduction dimensionnelle est
       pertinente sur ce dataset.
    3. Si svd_variance >= SVD_VARIANCE_THRESHOLD (0.50) : fit une
       TruncatedSVD à SVD_COMPONENTS (10) composantes, puis un KNN
       cosine sur l'espace réduit (U_reduced). Stratégie "svd".
    4. Sinon : fit un KNN cosine directement sur la matrice brute
       (pas de réduction dimensionnelle — avec peu d'étudiants, la SVD
       n'apporte pas de signal fiable, voir cf_config.py). Stratégie
       "raw".
    5. Attache une empreinte des fichiers source (fingerprint) au
       modèle pour permettre l'invalidation automatique du cache
       (voir cache_utils.py), puis sérialise tout le dict model_data
       vers save_path via pickle.

    Parameters
    ----------
    save_path : str
        Chemin de sauvegarde du modèle pickled. Défaut : CF_MODEL_PATH.

    Returns
    -------
    Dict[str, Any]
        model_data contenant : type ("svd"|"raw"), matrix, knn,
        svd_variance, et selon le type : svd, U_reduced. Toujours
        source_fingerprint.
    """

    matrix = get_user_item_matrix()

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
    """
    Charge un modèle CF précédemment sérialisé depuis le disque, sans
    vérifier sa validité vis-à-vis des fichiers source (voir
    get_or_build_cf_model() pour la version avec invalidation).

    Parameters
    ----------
    save_path : str
        Chemin du fichier pickle à charger.

    Returns
    -------
    Dict[str, Any]
        model_data tel que sérialisé par build_cf_model().

    Raises
    ------
    FileNotFoundError
        Si aucun fichier n'existe à save_path — le modèle doit d'abord
        être construit via build_cf_model().
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
    Point d'entrée principal pour obtenir un modèle CF utilisable :
    charge le cache pickle existant si présent ET toujours valide
    (fingerprint des fichiers source inchangé depuis sa construction),
    sinon reconstruit le modèle depuis zéro.

    Ce pattern évite de refaire un fit KNN/SVD à chaque redémarrage du
    process tant que interactions.csv et resources.csv n'ont pas changé,
    tout en garantissant qu'un modèle jamais périmé silencieusement
    n'est jamais servi après une modification des données source.

    Parameters
    ----------
    save_path : str
        Chemin du modèle pickled à charger ou construire.

    Returns
    -------
    Dict[str, Any]
        model_data valide et à jour.
    """
    if os.path.exists(save_path):
        model_data = load_cf_model(save_path)
        if is_cache_valid(model_data, SOURCE_FILES):
            return model_data
        logger.info("⚠️ Cache CF périmé (données source modifiées) — reconstruction.")
    else:
        logger.info("🔧 Aucun cache CF trouvé — construction initiale.")

    return build_cf_model(save_path)


class CFEngine:
    # after
    def __init__(self, model_path: Optional[str] = None):
        """
        Instancie un moteur CF prêt à servir des recommandations.

        Résout le chemin du modèle au moment de l'appel (pas figé comme
        valeur par défaut au niveau de la signature de la classe), ce
        qui permet de pointer une instance vers un modèle différent du
        chemin global CF_MODEL_PATH — utile pour l'isolation des tests
        unitaires (chaque test peut utiliser son propre fichier pickle
        sans polluer le modèle de production) et pour reset_cf_engine(),
        qui force une reconstruction propre du singleton global.

        Charge (ou construit si absent/périmé) le modèle via
        get_or_build_cf_model(), puis hydrate l'état interne de
        l'instance via _load_from_pickle().

        Parameters
        ----------
        model_path : Optional[str]
            Chemin du modèle CF. Si None, utilise CF_MODEL_PATH.
        """

        resolved_path = model_path if model_path is not None else CF_MODEL_PATH
        model_data = get_or_build_cf_model(resolved_path)
        self._load_from_pickle(model_data)

    def _load_from_pickle(self, model_data: Dict[str, Any]) -> None:
        """
        Hydrate les attributs d'instance à partir d'un model_data chargé
        (matrix, svd_variance, n_neighbors), et bascule entre les deux
        modes de fonctionnement selon model_data["type"] :

        - "svd" : reconstruit un DataFrame U_reduced_df (facteurs
          latents indexés par student_id) à partir de U_reduced pour
          permettre des lookups par student_id cohérents avec la
          matrice brute, active_method = "svd_knn".
        - autre ("raw") : pas de réduction, la matrice brute sert
          directement de feature_matrix pour le KNN, active_method =
          "raw_knn".

        Parameters
        ----------
        model_data : Dict[str, Any]
            Modèle CF chargé depuis pickle (build_cf_model /load_cf_model).
        """
        self.matrix = model_data["matrix"]
        self.svd_variance = model_data["svd_variance"]
        self.n_neighbors = KNN_NEIGHBORS

        if model_data["type"] == "svd":
            self.use_svd = True
            self.active_method = "svd_knn"
            self.svd = model_data["svd"]
            self.U_reduced = model_data["U_reduced"]
            actual_components = self.U_reduced.shape[1]
            self.U_reduced_df = pd.DataFrame(
                self.U_reduced,
                index=self.matrix.index,
                columns=[f"factor_{i + 1}" for i in range(actual_components)],
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
        Point d'entrée public pour obtenir des recommandations CF pour
        un étudiant donné. Sélectionne automatiquement la bonne paire
        (feature_matrix, raw_matrix) selon que le modèle actif utilise
        la SVD ou non, puis délègue à _recommend().

        - Mode SVD : la recherche de voisins (KNN) se fait dans
          l'espace réduit (U_reduced_df), mais les scores de
          recommandation finaux sont calculés sur la matrice brute
          (self.matrix) — la réduction dimensionnelle sert uniquement
          à trouver des voisins, pas à calculer les scores prédits.
        - Mode raw : les deux rôles (recherche de voisins et calcul de
          score) utilisent la même matrice brute.

        Parameters
        ----------
        student_id : str
            Identifiant de l'étudiant (doit correspondre à un index de
            la matrice utilisateur-ressource).
        n_recommendations : int
            Nombre de ressources à recommander (défaut 3).

        Returns
        -------
        Dict[str, Any]
            Voir _recommend() / _empty_result() pour la structure
            exacte selon les cas.
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
        """
        Implémentation générique de la recommandation CF, paramétrée
        par la matrice utilisée pour la recherche de voisins
        (feature_matrix) et celle utilisée pour le calcul de score
        (raw_matrix) — permet de partager exactement la même logique
        entre le mode SVD et le mode raw (voir get_recommendations()).

        Étapes :
        1. Si student_id absent de feature_matrix.index : retourne un
           résultat vide avec student_exists=False (pas d'exception,
           le cold-start utilisateur est un cas géré, pas une erreur).
        2. Recherche les k plus proches voisins de l'étudiant via KNN
           (k = n_neighbors + 1 pour inclure puis exclure l'étudiant
           lui-même, qui sera toujours son propre plus proche voisin à
           distance 0).
        3. Filtre les voisins à similarité positive uniquement via
           _filter_similar_students().
        4. Si aucun voisin similaire : résultat vide avec
           student_exists=True (l'étudiant existe mais est isolé dans
           l'espace des interactions).
        5. Calcule les recommandations pondérées par similarité via
           _compute_recommendations().

        Parameters
        ----------
        student_id : str
            Identifiant de l'étudiant.
        n_recommendations : int
            Nombre de ressources à recommander.
        feature_matrix : pd.DataFrame
            Matrice utilisée pour la recherche KNN (U_reduced_df en
            mode SVD, matrix brute en mode raw).
        raw_matrix : pd.DataFrame
            Matrice utilisée pour le calcul des scores prédits
            (toujours la matrice brute d'interactions).

        Returns
        -------
        Dict[str, Any]
            Structure complète : student_id, student_exists, method,
            similar_students, recommended_resources,
            num_similar_students, num_recommendations.
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
        """
        Convertit les distances cosinus brutes retournées par le KNN
        (distances) en scores de similarité (similarity = 1 - distance),
        et exclut tout voisin dont la similarité est nulle ou négative
        (aucune corrélation exploitable pour la recommandation).

        Parameters
        ----------
        indices : np.ndarray
            Indices des voisins retournés par NearestNeighbors.kneighbors().
        distances : np.ndarray
            Distances cosinus correspondantes.
        feature_matrix : pd.DataFrame
            Matrice utilisée pour résoudre les indices en student_id
            via feature_matrix.index.

        Returns
        -------
        List[Dict[str, Any]]
            Liste de {"student_id": str, "distance": float,
            "similarity_score": float}, triée dans l'ordre retourné par
            le KNN (le plus proche voisin en premier).
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
        """
        Calcule un score prédit par ressource, agrégé sur les voisins
        similaires pondérés par leur score de similarité normalisé
        (les poids somment à 1), puis exclut les ressources déjà
        consommées par l'étudiant cible (student_rated, score forcé à
        -inf pour les exclure du top-N sans les retirer du tableau).

        Le score prédit d'une ressource = moyenne pondérée des scores
        d'interaction de cette ressource chez les voisins similaires,
        pondération = similarity_score normalisé de chaque voisin.

        Ne renvoie que les ressources à score strictement positif — une
        ressource jamais interagie par aucun voisin similaire (score
        agrégé = 0) n'est pas recommandée, même si elle apparaît dans
        le top-N trié.

        Parameters
        ----------
        student_idx : int
            Position (positionnelle, pas label) de l'étudiant cible
            dans raw_matrix.
        similar_students : List[Dict[str, Any]]
            Voisins filtrés (voir _filter_similar_students).
        n_recommendations : int
            Nombre maximum de ressources à retourner.
        raw_matrix : pd.DataFrame
            Matrice d'interactions brute (toujours la matrice
            d'origine, pas l'espace réduit SVD).

        Returns
        -------
        List[Dict[str, Any]]
            Liste de {"resource_id": str, "predicted_score": float},
            triée par score décroissant, plafonnée à n_recommendations,
            filtrée à score > 0.
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
        """
        Retourne une réponse vide structurée pour les cas sans
        recommandation CF disponible.

        Returns
        -------
        Dict[str, Any]
            Structure de réponse CF avec listes vides et compteurs à 0.
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
        """
        Retourne un instantané de la configuration et de l'état courant
        du moteur CF.

        Returns
        -------
        Dict[str, Any]
            Paramètres principaux du moteur CF actif.
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


_cf_engine: Optional[CFEngine] = None


def get_cf_engine() -> CFEngine:
    """
    Retourne l'instance singleton globale de CFEngine, construite au
    premier appel puis réutilisée.
    """
    global _cf_engine
    if _cf_engine is None:
        _cf_engine = CFEngine()
    return _cf_engine


def reset_cf_engine() -> None:
    global _cf_engine
    _cf_engine = None


def get_recommendations(student_id: str, n_recommendations: int = 3) -> Dict[str, Any]:
    """
    Fonction de convenance au niveau module : récupère le singleton
    CFEngine via get_cf_engine() et délègue directement l'appel — évite
    à l'appelant de gérer explicitement le cycle de vie de l'engine
    pour un usage simple, hors classe HybridEngine.

    Parameters
    ----------
    student_id : str
    n_recommendations : int

    Returns
    -------
    Dict[str, Any]
        Voir CFEngine.get_recommendations().
    """
    engine = get_cf_engine()
    return engine.get_recommendations(student_id, n_recommendations)
