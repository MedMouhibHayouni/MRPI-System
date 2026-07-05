"""
Construit la matrice utilisateur-ressource (user-item matrix) utilisée
par le moteur de Filtrage Collaboratif (CF.py).

Exécuté une seule fois à l'import du module : charge interactions.csv et
resources.csv, fusionne les deux sur resource_id pour récupérer
estimated_time_min, puis calcule un score d'interaction composite par
paire (student_id, resource_id) à partir de trois signaux normalisés :

    interaction_score = 0.5 * rating_norm
                       + 0.3 * completed_norm
                       + 0.2 * time_norm

- rating_norm : note brute de l'étudiant, ou la moyenne globale des
  notes si absente (imputation simple, pas de biais par étudiant/ressource).
- completed_norm : 5 si complété, 0 sinon.
- time_norm : ratio (temps passé / temps estimé), plafonné à 2.0
  (TIME_RATIO_CAP) pour qu'un étudiant qui traîne 10x plus longtemps que
  prévu sur une ressource ne domine pas le score, puis remis à l'échelle
  sur [0, 5].

Le résultat est pivoté (pivot_table) en une matrice student_id × resource_id,
valeurs = interaction_score, cellules manquantes = 0 (absence
d'interaction, pas un score neutre).

Cette matrice est construite au niveau module (pas dans une fonction),
donc son coût de calcul est payé une seule fois par process, pas à
chaque appel de get_user_item_matrix().
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
resources = pd.read_csv(os.path.join(DATA_DIR, "resources.csv"))

df = interactions.merge(
    resources[["resource_id", "estimated_time_min"]],
    on="resource_id",
    how="left",
)

df["rating_norm"] = df["rating"].fillna(df["rating"].mean())

TIME_RATIO_CAP = 2.0
df["time_ratio"] = df["time_spent_minutes"] / df["estimated_time_min"]
df["time_norm"] = df["time_ratio"].clip(upper=TIME_RATIO_CAP) / TIME_RATIO_CAP * 5

df["completed_norm"] = df["completed"].astype(int) * 5

W_RATING = 0.5
W_COMPLETED = 0.3
W_TIME = 0.2

df["interaction_score"] = (
    df["rating_norm"] * W_RATING
    + df["completed_norm"] * W_COMPLETED
    + df["time_norm"] * W_TIME
)

user_item_matrix = df.pivot_table(
    index="student_id",
    columns="resource_id",
    values="interaction_score",
    fill_value=0,
)


def get_user_item_matrix():
    """
    Retourne la matrice utilisateur-ressource déjà construite à l'import
    du module (pas de recalcul).

    Returns
    -------
    pd.DataFrame
        Matrice indexée par student_id (lignes) et resource_id
        (colonnes), valeurs = interaction_score composite, 0 si aucune
        interaction enregistrée pour cette paire.
    """
    return user_item_matrix


def get_matrix_info():
    """
    Calcule des statistiques descriptives sur la matrice
    utilisateur-ressource : dimensions, nombre d'étudiants, nombre de
    ressources, densité (proportion de cellules non nulles) et sparsité
    (proportion de cellules nulles, complément de la densité).

    Utile pour diagnostiquer si le dataset a assez d'interactions
    réelles pour que le CF (KNN, SVD) produise des résultats fiables —
    une matrice très creuse (sparsity élevée) dégrade la qualité du CF
    indépendamment de la qualité de l'implémentation.

    Returns
    -------
    Dict[str, Any]
        {"shape": tuple, "n_students": int, "n_resources": int,
         "density": float, "sparsity": float}
    """
    matrix = user_item_matrix
    total_cells = matrix.shape[0] * matrix.shape[1]
    non_zero = (matrix.values != 0).sum()

    return {
        "shape": matrix.shape,
        "n_students": matrix.shape[0],
        "n_resources": matrix.shape[1],
        "density": non_zero / total_cells,
        "sparsity": 1 - (non_zero / total_cells),
    }
