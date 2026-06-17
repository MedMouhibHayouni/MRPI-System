import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from src.schemas.request import RecommendationRequest

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────
# MAPPINGS
# ─────────────────────────────────────────
STYLE_TO_TYPE = {
    "visual": "video",
    "auditory": "video",
    "kinesthetic": "exercise",
    "textual": "micro_lesson",
}


# ─────────────────────────────────────────
# STEP 1 — Build encoder
# ─────────────────────────────────────────


def build_vectorizer(resources_df, save_path=os.path.join(MODELS_DIR, "encoder.pkl")):
    """
    Fit a OneHotEncoder on resource features and persist it to disk.

    Each resource is encoded as a binary one-hot vector over four categorical
    features: subject, concept, difficulty, and type. The fitted encoder,
    the resulting resource matrix, and the cleaned DataFrame are serialized
    together so they can be reloaded without retraining.

    Parameters
    ----------
    resources_df : pd.DataFrame
        DataFrame loaded from resources.csv. Contain the columns:
        'subject', 'concept', 'difficulty', 'type'.
    save_path : str, optional
        Destination path for the serialized encoder pickle file.
        Defaults to <MODELS_DIR>/encoder.pkl.

    Returns
    -------
    encoder : OneHotEncoder
        Fitted scikit-learn OneHotEncoder instance.
    resource_matrix : np.ndarray of shape (n_resources, n_features_encoded)
        One-hot encoded matrix for all resources.
    resources_df : pd.DataFrame
        Reset-indexed copy of the input DataFrame.
    """
    resources_df = resources_df.reset_index(drop=True).copy()
    features = resources_df[["subject", "concept", "difficulty", "type"]]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    resource_matrix = encoder.fit_transform(features)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump((encoder, resource_matrix, resources_df), f)

    print(f"✓ OneHot encoder fitted on {len(resources_df)} resources.")
    return encoder, resource_matrix, resources_df


def load_vectorizer(save_path=os.path.join(MODELS_DIR, "encoder.pkl")):
    """
    Load a previously fitted encoder and its associated data from disk.

    Parameters
    ----------
    save_path : str, optional
        Path to the pickle file produced by build_vectorizer().
        Defaults to <MODELS_DIR>/encoder.pkl.

    Returns
    -------
    encoder : OneHotEncoder
        Fitted scikit-learn OneHotEncoder instance.
    resource_matrix : np.ndarray of shape (n_resources, n_features_encoded)
        One-hot encoded matrix for all resources.
    resources_df : pd.DataFrame
        DataFrame of resources aligned with resource_matrix row indices.

    Raises
    ------
    FileNotFoundError
        If save_path does not exist (i.e., build_vectorizer has not been run yet).
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Encoder not found at '{save_path}'. "
            "Run build_vectorizer() first to train and save the encoder."
        )
    with open(save_path, "rb") as f:
        encoder, resource_matrix, resources_df = pickle.load(f)
    return encoder, resource_matrix, resources_df


# ─────────────────────────────────────────
# STEP 2 — Prerequisites check
# ─────────────────────────────────────────


def prerequisites_met(prereq, completed_ids):
    """
    Check whether all prerequisites for a resource have been completed.

    Parameters
    ----------
    prereq : str
        Comma-separated string of prerequisite resource IDs, or NaN/empty
        string if the resource has no prerequisites.
    completed_ids : list[str]
        List of resource IDs the student has already interacted with.

    Returns
    -------
    bool
        True if all prerequisites are satisfied or if there are none.
        False if at least one required resource ID is missing from completed_ids.
    """
    if pd.isna(prereq) or prereq == "":
        return True
    required = [r.strip() for r in prereq.split(",")]
    return all(r in completed_ids for r in required)


# ─────────────────────────────────────────
# STEP 3 — Recommend CBF
# ─────────────────────────────────────────


def recommend_cbf(
    request: RecommendationRequest,
    resources_df: pd.DataFrame,
    encoder: OneHotEncoder,
    resource_matrix,
    top_n: int = 5,
):
    """
    Generate content-based filtering (CBF) recommendations for a student.

    The function applies three successive filters — subject/difficulty/concept,
    exclusion of already-seen resources, and prerequisite validation — then
    ranks the remaining candidates by cosine similarity between a learner
    profile vector and the one-hot resource matrix.

    The learner profile vector is built from the same four features used to
    encode resources (subject, weak_concept, academic_level, preferred type),
    which ensures the similarity score reflects how closely each resource
    matches the student's current learning context.

    If the student's preferred content type yields no results, a fallback
    order is applied before considering all remaining types. The 'fallback_used'
    column in the result signals whether degraded results were returned.
    Fallback strategy
    ------------------
    The hard filters (subject, difficulty, concept, excluded resources,
    prerequisites) are never relaxed — they define the pool of pedagogically
    valid candidates regardless of content type.

    Once that pool (`base_filter`) is established, the function tries to
    match the student's preferred content type (`lesson_type`, derived from
    their learning style). If no resource of that type survives the hard
    filters, a secondary preference order is consulted:

        video        -> micro_lesson -> exercise
        exercise     -> micro_lesson -> video
        micro_lesson -> exercise     -> video

    The first non-empty type in this order is used. If none of the fallback
    types yield results either, the function returns the entire `base_filter`
    pool unfiltered by type, as a last resort.

    Rationale: relevance to the student's actual learning context (subject,
    concept, difficulty, prerequisites) is prioritized over matching their
    preferred content format. Returning a relevant resource in a suboptimal
    format is preferable to returning no recommendation at all.

    The `fallback_used` column in the result signals whether a fallback was
    triggered (True) or the preferred type was matched directly (False),
    allowing downstream consumers (API layer, frontend, evaluation metrics)
    to distinguish optimal from degraded recommendations.

    Parameters
    ----------
    request : RecommendationRequest
        Pydantic request object containing:
        - weak_concept (str): The concept the student is struggling with.
        - learning_style (LearningStyle): Enum mapped to a content type via STYLE_TO_TYPE.
        - subject (str): The subject area to filter on.
        - academic_level (AcademicLevel): Enum whose value maps to a difficulty string.
        - past_interactions (list[str]): Resource IDs the student has already seen.
    resources_df : pd.DataFrame
        Full resource catalog. Must contain: 'resource_id', 'subject', 'concept',
        'difficulty', 'type', 'prerequisites', 'title'.
    encoder : OneHotEncoder
        Fitted encoder returned by build_vectorizer() or load_vectorizer().
    resource_matrix : np.ndarray of shape (n_resources, n_features_encoded)
        One-hot matrix for all resources, row-aligned with resources_df.
    top_n : int, optional
        Maximum number of recommendations to return. Defaults to 5.
        Capped automatically if fewer candidates are available.

    Returns
    -------
    pd.DataFrame
        Top-N recommended resources with columns:
        ['resource_id', 'title', 'concept', 'type', 'difficulty', 'cbf_score', 'fallback_used'].
        Returns an empty DataFrame if no resources pass the filters.
    """
    weak_concept = request.weak_concept
    lesson_type = STYLE_TO_TYPE[request.learning_style.value]
    subject = request.subject
    difficulty = request.academic_level.value
    completed_ids = request.past_interactions

    df = resources_df.reset_index(drop=True).copy()

    # Base filter — subject + difficulty + concept
    base_filter = df[
        (df["subject"] == subject)
        & (df["difficulty"] == difficulty)
        & (df["concept"] == weak_concept)
    ]

    # Exclude already seen
    base_filter = base_filter[~base_filter["resource_id"].isin(completed_ids)]

    # Prerequisites check
    if not completed_ids:
        base_filter = base_filter[
            base_filter["prerequisites"].isna() | (base_filter["prerequisites"] == "")
        ]
    else:
        base_filter = base_filter[
            base_filter["prerequisites"].apply(
                lambda p: prerequisites_met(p, completed_ids)
            )
        ]

    if base_filter.empty:
        print("⚠ No resources available for this profile.")
        return pd.DataFrame()

    # Preferred type
    fallback_used = False
    filtered = base_filter[base_filter["type"] == lesson_type].copy()

    # Fallback — if preferred type unavailable, try alternatives in order
    if filtered.empty:
        fallback_order = {
            "video": ["micro_lesson", "exercise"],
            "exercise": ["micro_lesson", "video"],
            "micro_lesson": ["exercise", "video"],
        }
        for fallback_type in fallback_order.get(lesson_type, ["micro_lesson"]):
            filtered = base_filter[base_filter["type"] == fallback_type].copy()
            if not filtered.empty:
                fallback_used = True
                print(f"⚠ Fallback → {fallback_type}")
                break

    if filtered.empty:
        filtered = base_filter.copy()
        fallback_used = True

    filtered_indices = filtered.index.tolist()
    filtered_matrix = resource_matrix[filtered_indices]

    # Build learner profile vector and compute cosine similarity
    profile = pd.DataFrame(
        [[subject, weak_concept, difficulty, lesson_type]],
        columns=["subject", "concept", "difficulty", "type"],
    )
    query_vector = encoder.transform(profile)
    scores = cosine_similarity(query_vector, filtered_matrix)[0]
    top_n_safe = min(top_n, len(scores))
    top_idx = np.argsort(scores)[::-1][:top_n_safe]

    result = filtered.iloc[top_idx].copy()
    result["cbf_score"] = scores[top_idx]
    result["fallback_used"] = fallback_used

    return result[
        [
            "resource_id",
            "title",
            "concept",
            "type",
            "difficulty",
            "cbf_score",
            "fallback_used",
        ]
    ]
