"""
Module de construction de la matrice utilisateur-ressource (user-item matrix).

Ce module charge les données d'interaction et construit la matrice
utilisée par le filtrage collaboratif (CF).
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================

interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
resources = pd.read_csv(os.path.join(DATA_DIR, "resources.csv"))

# Fusion pour récupérer estimated_time_min
df = interactions.merge(
    resources[["resource_id", "estimated_time_min"]],
    on="resource_id",
    how="left"
)

# ============================================
# CONSTRUCTION DES SIGNAUX D'INTERACTION
# ============================================

# Signal 1 : Rating normalisé (moyenne pour les valeurs manquantes)
df["rating_norm"] = df["rating"].fillna(df["rating"].mean())

# Signal 2 : Temps normalisé par rapport à la durée estimée
TIME_RATIO_CAP = 2.0
df["time_ratio"] = df["time_spent_minutes"] / df["estimated_time_min"]
df["time_norm"] = df["time_ratio"].clip(upper=TIME_RATIO_CAP) / TIME_RATIO_CAP * 5

# Signal 3 : Complétion binaire
df["completed_norm"] = df["completed"].astype(int) * 5

# ============================================
# SCORE D'INTERACTION PONDÉRÉ
# ============================================

W_RATING = 0.5      # Poids du rating
W_COMPLETED = 0.3   # Poids de la complétion
W_TIME = 0.2        # Poids du temps

df["interaction_score"] = (
    df["rating_norm"] * W_RATING
    + df["completed_norm"] * W_COMPLETED
    + df["time_norm"] * W_TIME
)

# ============================================
# MATRICE UTILISATEUR-RESSOURCE
# ============================================

user_item_matrix = df.pivot_table(
    index="student_id",
    columns="resource_id",
    values="interaction_score",
    fill_value=0
)


def get_user_item_matrix():
    """
    Retourne la matrice utilisateur-ressource.

    Returns:
        pandas.DataFrame: Matrice avec les étudiants en index,
                         les ressources en colonnes,
                         et les scores d'interaction comme valeurs.
    
    Exemple:
        >>> matrix = get_user_item_matrix()
        >>> print(matrix.shape)
        (30, 100)
    """
    return user_item_matrix


def get_matrix_info():
    """
    Retourne des informations sur la matrice.

    Returns:
        dict: Dictionnaire contenant:
            - shape: (n_students, n_resources)
            - n_students: Nombre d'étudiants
            - n_resources: Nombre de ressources
            - density: Densité (proportion de valeurs non nulles)
            - sparsity: Sparsité (proportion de zéros)
    
    Exemple:
        >>> info = get_matrix_info()
        >>> print(f"{info['n_students']} étudiants, {info['n_resources']} ressources")
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