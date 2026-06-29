"""
Script de test bout-en-bout : students.csv -> validation Pydantic -> moteur hybride -> JSON.

Pipeline :
    1. Lecture de students.csv (toutes les lignes, ou un sous-ensemble via STUDENT_IDS_FILTER)
    2. Parsing de past_interactions (string -> list[str])
    3. Construction + validation de chaque ligne via RecommendationRequest (Pydantic)
       - Lignes invalides : capturées, loggées, exclues du traitement (pas de crash global)
    4. Pour chaque requête valide : appel à HybridEngine.get_recommendations()
    5. Export : un objet JSON par étudiant, incluant la requête validée (dump Pydantic)
       et les recommandations produites

Usage :
    python test_students_to_hybrid.py

Sortie :
    students_hybrid_results.json — un objet par étudiant, succès ou échec de validation
"""

import sys
import os
import json
import importlib
import ast
import csv
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
from pydantic import ValidationError

import src.CBF
import src.CF
import src.hybrid

importlib.reload(src.CBF)
importlib.reload(src.CF)
importlib.reload(src.hybrid)

from src.hybrid import HybridEngine, MAX_RECOMMENDATIONS
from src.schemas.request import RecommendationRequest, AcademicLevel, LearningStyle

DATA_DIR = os.path.join(os.path.abspath(".."), "data")

# Mettre une liste d'IDs ici pour limiter le test, ou None pour traiter tout students.csv
STUDENT_IDS_FILTER = [
    "STU-2026-0001",
    "STU-2026-0032",
    "STU-2026-0011",
]  # ex: ["STU-2026-0001", "STU-2026-0033"]


def load_students_csv(filepath):
    """
    Charge students.csv en gérant la colonne past_interactions qui peut être :
    - Un JSON array avec guillemets : "[\"RES-101\", \"RES-205\"]"
    - Un JSON array sans guillemets : [RES-101, RES-205]
    - Une liste Python : ['RES-101', 'RES-205']

    Conforme au CDC qui spécifie : past_interaction | array | ["RES-101", "RES-205"]
    """
    rows = []

    with open(filepath, "r", encoding="utf-8") as f:
        # Lire l'en-tête
        header_line = f.readline().strip()
        headers = header_line.split(",")

        # Pour chaque ligne de données
        for line_num, line in enumerate(f, start=2):
            if not line.strip():
                continue

            # Si la ligne contient un crochet, on sait que c'est la colonne past_interactions
            if "[" in line and "]" in line:
                # Trouver la position du premier crochet
                bracket_start = line.find("[")
                bracket_end = line.rfind("]") + 1

                # Extraire les colonnes avant la liste
                prefix = line[:bracket_start].strip()
                if prefix.endswith(","):
                    prefix = prefix[:-1]

                # Extraire la liste
                list_part = line[bracket_start:bracket_end].strip()

                # Extraire les colonnes après la liste (s'il y en a)
                suffix = line[bracket_end:].strip()
                if suffix.startswith(","):
                    suffix = suffix[1:]

                # Split le prefix par les virgules pour obtenir les 5 premières colonnes
                if prefix:
                    prefix_parts = prefix.split(",")
                else:
                    prefix_parts = []

                # S'assurer qu'on a exactement 5 colonnes avant past_interactions
                while len(prefix_parts) < 5:
                    prefix_parts.append("")

                # Si plus de 5, c'est qu'il y a des virgules dans les valeurs
                if len(prefix_parts) > 5:
                    # On reconstitue les colonnes correctement
                    # Les 4 premières colonnes : student_id, subject, weak_concept, academic_level
                    # La 5ème colonne peut contenir des virgules (learning_style)
                    # On garde les 4 premières et on joint le reste pour learning_style
                    first_4 = prefix_parts[:4]
                    rest = prefix_parts[4:]
                    learning_style = ",".join(rest)
                    prefix_parts = first_4 + [learning_style]

                # Ajouter la liste comme 6ème colonne
                row = prefix_parts + [list_part]
            else:
                # Pas de liste, split normal
                parts = line.strip().split(",")
                if len(parts) < 6:
                    parts.extend([""] * (6 - len(parts)))
                row = parts[:6]

            # S'assurer qu'on a 6 colonnes
            if len(row) < 6:
                row.extend([""] * (6 - len(row)))
            elif len(row) > 6:
                row = row[:6]

            rows.append(row)

    # Créer le DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Convertir past_interactions de string à liste
    df["past_interactions"] = df["past_interactions"].apply(parse_past_interactions)

    return df


def parse_past_interactions(raw) -> list:
    """
    Parse le champ past_interactions en liste de resource_id.

    Formats supportés (conforme CDC) :
    - JSON array avec guillemets : "[\"RES-101\", \"RES-205\"]"
    - JSON array sans guillemets : [RES-101, RES-205]
    - Liste Python : ['RES-101', 'RES-205']
    - Chaîne vide ou NaN : []

    Returns:
        list[str]: Liste de resource_id
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return []

    raw_str = str(raw).strip()

    # Si c'est un JSON array ou une liste Python (avec ou sans guillemets)
    if raw_str.startswith("[") and raw_str.endswith("]"):
        # Remplacer les guillemets simples par des doubles pour le parsing JSON
        cleaned = raw_str.replace("'", '"')

        try:
            # Essayer avec json.loads pour le format JSON standard
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except json.JSONDecodeError:
            pass

        try:
            # Essayer avec ast.literal_eval pour les listes Python
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except:
            pass

        # Si tout échoue, parsing manuel
        # Enlever les crochets
        inner = raw_str[1:-1].strip()
        if not inner:
            return []
        # Split par les virgules
        items = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
        return items

    # Si c'est une chaîne avec des séparateurs
    if ";" in raw_str:
        return [r.strip() for r in raw_str.split(";") if r.strip()]
    if "," in raw_str:
        return [r.strip() for r in raw_str.split(",") if r.strip()]

    # Cas par défaut
    return [raw_str] if raw_str else []


def validate_row(row: pd.Series) -> tuple:
    """Construit et valide une RecommendationRequest depuis une ligne students.csv."""
    try:
        request = RecommendationRequest(
            student_id=row["student_id"],
            subject=row["subject"],
            weak_concept=row["weak_concept"],
            academic_level=AcademicLevel(row["academic_level"]),
            learning_style=LearningStyle(row["learning_style"]),
            past_interactions=row["past_interactions"]
            if isinstance(row["past_interactions"], list)
            else parse_past_interactions(row["past_interactions"]),
        )
        return request, None
    except ValidationError as e:
        return None, f"ValidationError: {e}"
    except ValueError as e:
        return None, f"ValueError: {e}"
    except Exception as e:
        return None, f"UnexpectedError: {type(e).__name__}: {e}"


def recommendation_to_dict(rec) -> dict:
    """Convertit un ResourceRecommendation en dict JSON-sérialisable."""
    return {
        "resource_id": rec.resource_id,
        "title": rec.title,
        "type": rec.type.value,
        "difficulty": rec.difficulty.value,
        "relevance_score": round(rec.relevance_score, 4),
        "estimated_time_min": rec.estimated_time_min,
    }


def main():
    print("=" * 70)
    print("TEST : students.csv -> Pydantic -> Moteur Hybride -> JSON")
    print(f"Exécuté le : {datetime.now().isoformat()}")
    print("=" * 70)

    # Utiliser la fonction de chargement personnalisée
    students_df = load_students_csv(os.path.join(DATA_DIR, "students.csv"))

    if STUDENT_IDS_FILTER:
        students_df = students_df[students_df["student_id"].isin(STUDENT_IDS_FILTER)]

    print(f"\n[1/3] {len(students_df)} ligne(s) à traiter depuis students.csv")

    print("\n[2/3] Initialisation du HybridEngine...")
    engine = HybridEngine()

    results = []
    n_valid = 0
    n_invalid = 0
    n_with_recommendations = 0

    print(f"\n[3/3] Validation + génération pour chaque étudiant...\n")

    for idx, row in students_df.iterrows():
        student_id = row.get("student_id", f"UNKNOWN_{idx}")
        request, error = validate_row(row)

        if error is not None:
            print(f"  ❌ {student_id} — validation échouée : {error}")
            results.append(
                {
                    "student_id": student_id,
                    "valid": False,
                    "validation_error": error,
                    "request": None,
                    "recommendations": None,
                    "num_recommendations": 0,
                }
            )
            n_invalid += 1
            continue

        n_valid += 1

        try:
            recommendations = engine.get_recommendations(
                request, top_n=MAX_RECOMMENDATIONS
            )
        except Exception as e:
            print(
                f"  ⚠️  {student_id} — validé mais erreur moteur : {type(e).__name__}: {e}"
            )
            results.append(
                {
                    "student_id": student_id,
                    "valid": True,
                    "validation_error": None,
                    "request": request.model_dump(mode="json"),
                    "recommendations": None,
                    "engine_error": f"{type(e).__name__}: {e}",
                    "num_recommendations": 0,
                }
            )
            continue

        if recommendations:
            n_with_recommendations += 1

        print(f"  ✓ {student_id} — {len(recommendations)} recommandation(s)")

        results.append(
            {
                "student_id": student_id,
                "valid": True,
                "validation_error": None,
                "request": request.model_dump(mode="json"),
                "recommendations": [recommendation_to_dict(r) for r in recommendations],
                "num_recommendations": len(recommendations),
            }
        )

    output_path = "students_hybrid_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"  Lignes traitées              : {len(students_df)}")
    print(f"  Validations réussies         : {n_valid}")
    print(f"  Validations échouées         : {n_invalid}")
    print(f"  Avec au moins 1 recommandation : {n_with_recommendations}")
    print(f"  Sans recommandation (valide) : {n_valid - n_with_recommendations}")
    print(f"\n✓ Résultats sauvegardés : {output_path}")


if __name__ == "__main__":
    main()
