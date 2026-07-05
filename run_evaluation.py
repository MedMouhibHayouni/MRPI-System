"""
run_evaluation.py — Calcule les 4 métriques CDC sur le dataset réel complet.
Produit les chiffres à mettre dans le rapport d'évaluation.
"""

import pandas as pd
from src.hybrid import get_hybrid_engine
from src.schemas.request import RecommendationRequest
from src.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    intra_list_diversity,
    coverage,
    average_ignoring_none,
    get_relevant_resources,
)

DATA_DIR = "data"

students = pd.read_csv(f"{DATA_DIR}/students.csv")
resources = pd.read_csv(f"{DATA_DIR}/resources.csv")
resources["subject"] = resources["subject"].str.strip().str.lower()
resources["concept"] = resources["concept"].str.strip().str.lower()

engine = get_hybrid_engine()

all_recommended_lists = []
precisions_5, precisions_10 = [], []
recalls_5 = []
ndcgs_10 = []
diversities_10 = []

for _, student in students.iterrows():
    request = RecommendationRequest(
        student_id=student["student_id"],
        subject=student["subject"],
        weak_concept=student["weak_concept"],
        academic_level=student["academic_level"],
        learning_style=student["learning_style"],
    )

    recs = engine.get_recommendations(request)
    rec_ids = [r.resource_id for r in recs]
    all_recommended_lists.append(rec_ids)

    profile_dict = {
        "student_id": student["student_id"],
        "subject": str(student["subject"]).strip().lower(),
        "weak_concept": str(student["weak_concept"]).strip().lower(),
        "academic_level": student["academic_level"],
        "past_interactions": student.get("past_interactions", []),
    }
    relevant = get_relevant_resources(profile_dict, resources)

    precisions_5.append(precision_at_k(rec_ids, relevant, k=5))
    precisions_10.append(precision_at_k(rec_ids, relevant, k=10))
    recalls_5.append(recall_at_k(rec_ids, relevant, k=5))
    ndcgs_10.append(ndcg_at_k(rec_ids, relevant, k=10))
    diversities_10.append(
        intra_list_diversity(
            rec_ids[:10], engine.resources_df, engine.cbf_encoder, engine.cbf_matrix
        )
    )

p5 = average_ignoring_none(precisions_5)
p10 = average_ignoring_none(precisions_10)
r5 = average_ignoring_none(recalls_5)
ndcg10 = average_ignoring_none(ndcgs_10)
div10 = average_ignoring_none(diversities_10)

# coverage() veut UNE liste plate d'IDs, pas une liste de listes
flat_recommended_ids = [
    rid for student_list in all_recommended_lists for rid in student_list
]
cov = coverage(flat_recommended_ids, resources)

print("=" * 50)
print("RÉSULTATS ÉVALUATION MRPI — CDC Tableau 9")
print("=" * 50)
print(f"Precision@5   : {p5:.3f}   (seuil CDC: 0.60)")
print(f"Precision@10  : {p10:.3f}   (seuil CDC: 0.50)")
print(f"Recall@5      : {r5:.3f}")
print(f"NDCG@10       : {ndcg10:.3f}   (seuil CDC: 0.55)")
print(f"Diversité@10  : {div10:.3f}")
print(f"Couverture    : {cov:.3f}   (seuil CDC: 0.25)")
print("=" * 50)
