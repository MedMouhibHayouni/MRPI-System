"""
test_llm_integration.py — Test end-to-end : HybridEngine → LLM_API

Usage :
    python tests/integration/test_integration_llm.py

Teste dans l'ordre :
    1. HybridEngine produit de vraies recommandations pour un étudiant réel
    2. generate_explanation() sur ces recommandations
    3. adapt_wording_batch() sur les premières ressources recommandées
    4. suggest_study_plan() sur l'ensemble des recommandations

Ce script est gardé comme outil manuel d'intégration LLM.
"""

import json
import logging
import os
import ast
import csv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.hybrid import get_hybrid_engine
from src.schemas.request import RecommendationRequest
from src.LLM_API import generate_explanation, suggest_study_plan

import pandas as pd


def load_students_data():
    """Load students.csv and parse past_interactions properly."""
    csv_path = Path("data/students.csv")

    # Read CSV manually to handle the array format
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []

        for row in reader:
            # If row has more than 6 columns, the array got split by commas
            if len(row) > 6:
                # Reconstruct the array
                array_parts = row[5:]
                array_str = ",".join(array_parts)
                row = row[:5] + [array_str]
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=header)

    # Parse past_interactions using ast.literal_eval
    df["past_interactions"] = df["past_interactions"].apply(
        lambda x: (
            ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
        )
    )

    return df


def main():
    # ------------------------------------------------------------
    # STEP 1 — construire une requête réelle
    # ------------------------------------------------------------

    students_df = load_students_data()
    row = students_df.loc[students_df["student_id"] == "STU-2026-0001"].iloc[0]

    request = RecommendationRequest(
        student_id=row["student_id"],
        subject=row["subject"],
        weak_concept=row["weak_concept"],
        academic_level=row["academic_level"],
        learning_style=row["learning_style"],
        past_interactions=row["past_interactions"],  # Now properly parsed
    )

    print("=" * 60)
    print("STEP 1 — HybridEngine.get_recommendations()")
    print("=" * 60)

    engine = get_hybrid_engine()
    recommendations = engine.get_recommendations(request)

    if not recommendations:
        print(
            "❌ Aucune recommandation retournée — vérifie subject/weak_concept/academic_level"
        )
        print(
            "   contre les valeurs réelles dans data/resources.csv avant de continuer."
        )
        return

    print(f"✓ {len(recommendations)} recommandations obtenues")
    for r in recommendations[:3]:
        print(f"   - {r.title} (score: {r.relevance_score:.3f})")

    # Convertit les objets Pydantic en dicts pour LLM_API (qui attend des dicts)
    recs_as_dicts = [r.dict() for r in recommendations]
    profile_as_dict = request.dict()

    # ------------------------------------------------------------
    # STEP 2 — generate_explanation
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — generate_explanation()")
    print("=" * 60)

    explanation = generate_explanation(profile_as_dict, recs_as_dicts)
    print(explanation)

    # ------------------------------------------------------------
    # STEP 3 — adapt_wording (sur la première ressource uniquement)
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — adapt_wording()")
    print("=" * 60)

    from src.LLM_API import adapt_wording_batch

    LLM_ENRICHMENT_TOP_N = 5  # ne pas envoyer les 20 recommandations au LLM

    first_resource = recs_as_dicts[0]
    if "description" not in first_resource:
        print(
            "⚠️ 'description' absent du dict ressource — vérifie resources.csv "
            "et ResourceRecommendation."
        )

    # Seul le top N est enrichi par le LLM — le reste garde ses données brutes
    top_recs_for_llm = recs_as_dicts[:LLM_ENRICHMENT_TOP_N]
    adapted_resources = adapt_wording_batch(top_recs_for_llm, request.learning_style.value)

    print(f"({len(adapted_resources)}/{len(recs_as_dicts)} ressources enrichies)")
    for r in adapted_resources[:3]:
        print(f"   - {r['title']} → {r.get('adapted_title', '(pas adapté)')}")

    # Détail complet de la première ressource, pour inspecter le JSON exact
    print("\nDétail de la première ressource adaptée :")
    print(json.dumps(adapted_resources[0], indent=2, ensure_ascii=False))
    
    # ------------------------------------------------------------
    # STEP 4 — suggest_study_plan
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — suggest_study_plan()")
    print("=" * 60)

    plan = suggest_study_plan(profile_as_dict, top_recs_for_llm)  # top_recs_for_llm, pas recs_as_dicts
    print(json.dumps(plan, indent=2, ensure_ascii=False))
        # ------------------------------------------------------------
    # STEP 5 — confirmer que le cache a bien été écrit sur disque
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — vérification du cache")
    print("=" * 60)
    from src.LLM_API import CACHE_FILE_PATH

    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"✓ Cache trouvé: {CACHE_FILE_PATH} ({len(cache)} entrée(s))")
    else:
        print(
            f"❌ Cache introuvable à {CACHE_FILE_PATH} — bug potentiel dans _save_llm_cache"
        )


if __name__ == "__main__":
    main()
