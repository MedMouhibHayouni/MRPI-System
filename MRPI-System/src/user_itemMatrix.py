"""
Module de construction de la matrice utilisateur-ressource (user-item matrix).
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
    """Retourne la matrice utilisateur-ressource (calculée une fois à l'import)."""
    return user_item_matrix


def get_matrix_info():
    """Retourne des informations sur la matrice."""
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
