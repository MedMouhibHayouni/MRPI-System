# AiKup Tech — Stage Été 2026
## Exercice S1 : Fondamentaux Python pour la Data

**Stagiaire :** Emna Harhouri  
**Branche :** `learning/week1-python`  
**Due date :** Fin Semaine 1  

---

## Contexte

Tu travailles chez AiKup Tech sur un projet d'analyse 
de données scolaires. On t'a fourni un fichier CSV 
contenant les notes de 10 élèves répartis en 2 classes 
pour 6 matières au premier trimestre.

Ton rôle : analyser ces données comme le ferait 
un système intelligent qui aide les directeurs 
d'école à comprendre les performances de leurs élèves.

---

## Structure du Repo

```
/
├── data/
│   └── notes_eleves.csv
├── exercice/
│   └── analyse_notes.py   ← ton fichier de travail
├── output/
│   └── (tes fichiers générés ici)
└── README.md
```

---

## Consignes Générales

- Tout le code dans `exercice/analyse_notes.py`
- Chaque partie dans une fonction séparée
- Commentaires obligatoires sur chaque bloc
- Les résultats exportés dans `/output/`
- Commit après chaque partie complétée

FICHIER 3 — exercice/analyse_notes.py
C'est le fichier que tu fournis à Emna — il contient l'énoncé complet structuré en parties avec le code de départ :
python"""
========================================================
AiKup Tech SUARL — Stage Été 2026
Exercice S1 : Fondamentaux Python pour la Data
Stagiaire : Emna Harhouri
Branche   : learning/week1-python
========================================================

CONTEXTE :
Tu travailles sur un système d'analyse de données scolaires.
Un fichier CSV contient les notes de 10 élèves de 3ème
pour 6 matières au T1. Ton objectif est d'explorer,
nettoyer, analyser et exporter ces données.

INSTRUCTIONS :
- Complète chaque fonction marquée TODO
- Ne supprime pas les commentaires existants
- Commit après chaque partie terminée avec le message :
  "feat: S1-partie-X terminée"
- Dépose tes exports dans le dossier /output/
========================================================
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# SETUP — Ne pas modifier
# ─────────────────────────────────────────────
DATA_PATH   = "../data/notes_eleves.csv"
OUTPUT_PATH = "../output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ═══════════════════════════════════════════════
# PARTIE 1 — CHARGEMENT ET EXPLORATION DES DONNÉES
# ═══════════════════════════════════════════════
"""
OBJECTIF : Charger le CSV et comprendre sa structure.

NOTIONS COUVERTES :
- pd.read_csv()
- .head(), .tail(), .shape, .dtypes
- .info(), .describe()
- .isnull(), .nunique()
"""

def partie1_exploration(path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV et affiche les informations
    de base du DataFrame.

    Args:
        path: chemin vers le fichier CSV

    Returns:
        DataFrame chargé
    """

    print("=" * 50)
    print("PARTIE 1 — EXPLORATION")
    print("=" * 50)

    # TODO 1.1 — Charger le CSV dans un DataFrame
    df = None  # remplace None par le bon code

    # TODO 1.2 — Afficher les 5 premières lignes
    # print(...)

    # TODO 1.3 — Afficher les dimensions du DataFrame
    # (nombre de lignes et colonnes)
    # print(f"Dimensions : ...")

    # TODO 1.4 — Afficher les types de chaque colonne
    # print(...)

    # TODO 1.5 — Afficher les statistiques descriptives
    # (min, max, mean, std pour les colonnes numériques)
    # print(...)

    # TODO 1.6 — Vérifier s'il y a des valeurs manquantes
    # Affiche le nombre de valeurs nulles par colonne
    # print(...)

    # TODO 1.7 — Afficher le nombre de valeurs uniques
    # pour les colonnes : classe, matiere, eleve_id
    # print(...)

    return df


# ═══════════════════════════════════════════════
# PARTIE 2 — FILTRAGE ET SÉLECTION
# ═══════════════════════════════════════════════
"""
OBJECTIF : Extraire des sous-ensembles précis de données.

NOTIONS COUVERTES :
- Filtrage avec conditions booléennes
- .loc[] et .iloc[]
- Filtres multiples avec & et |
- .isin()
"""

def partie2_filtrage(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("PARTIE 2 — FILTRAGE")
    print("=" * 50)

    # TODO 2.1 — Afficher toutes les notes de la classe "3ème A"
    # classe_a = ...
    # print(classe_a)

    # TODO 2.2 — Afficher toutes les notes de Mathématiques
    # notes_maths = ...
    # print(notes_maths)

    # TODO 2.3 — Afficher les élèves ayant eu une note
    # strictement inférieure à 10 (toutes matières confondues)
    # en_difficulte = ...
    # print(en_difficulte)

    # TODO 2.4 — Afficher les élèves ayant eu une note
    # supérieure ou égale à 17 (les excellents)
    # excellents = ...
    # print(excellents)

    # TODO 2.5 — Afficher uniquement les colonnes
    # nom, prenom, matiere, note pour les élèves
    # de 3ème B avec une note >= 15
    # bons_eleves_b = ...
    # print(bons_eleves_b)

    # TODO 2.6 — Afficher les notes des matières
    # ["Mathématiques", "Sciences", "Informatique"]
    # uniquement — utilise .isin()
    # matieres_scientifiques = ...
    # print(matieres_scientifiques)

    pass


# ═══════════════════════════════════════════════
# PARTIE 3 — AGRÉGATION ET GROUPBY
# ═══════════════════════════════════════════════
"""
OBJECTIF : Calculer des statistiques par groupe.

NOTIONS COUVERTES :
- .groupby()
- .mean(), .min(), .max(), .count()
- .agg() avec dictionnaire
- .sort_values()
- .reset_index()
"""

def partie3_agregation(df: pd.DataFrame) -> dict:

    print("\n" + "=" * 50)
    print("PARTIE 3 — AGRÉGATION")
    print("=" * 50)

    resultats = {}

    # TODO 3.1 — Calculer la moyenne des notes par matière
    # Afficher le résultat trié du plus haut au plus bas
    # moy_par_matiere = ...
    # print(moy_par_matiere)

    # TODO 3.2 — Calculer pour chaque élève :
    # sa moyenne simple (non pondérée) sur toutes les matières
    # Afficher trié de la meilleure à la moins bonne moyenne
    # moy_par_eleve = ...
    # print(moy_par_eleve)

    # TODO 3.3 — Calculer la moyenne par classe
    # (3ème A vs 3ème B) — quelle classe performe mieux ?
    # moy_par_classe = ...
    # print(moy_par_classe)

    # TODO 3.4 — Pour chaque matière, afficher
    # la note min, la note max, et la moyenne
    # Utilise .agg({'note': ['min', 'max', 'mean']})
    # stats_matieres = ...
    # print(stats_matieres)

    # TODO 3.5 — Trouver l'élève avec la meilleure moyenne
    # et l'élève avec la moins bonne moyenne
    # Affiche : "Meilleur élève : [nom] avec [moyenne]"
    # Affiche : "Élève en difficulté : [nom] avec [moyenne]"
    # meilleur = ...
    # moins_bon = ...

    return resultats


# ═══════════════════════════════════════════════
# PARTIE 4 — MOYENNE PONDÉRÉE AVEC NUMPY
# ═══════════════════════════════════════════════
"""
OBJECTIF : Calculer la vraie moyenne pondérée par coefficients.

NOTIONS COUVERTES :
- np.array()
- np.dot() — produit scalaire
- np.sum()
- Opérations vectorisées numpy
- Comparaison avec la moyenne simple pandas
"""

def partie4_moyenne_ponderee(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 50)
    print("PARTIE 4 — MOYENNE PONDÉRÉE (NUMPY)")
    print("=" * 50)

    resultats = []

    # Récupère la liste des élèves uniques
    eleves = df['eleve_id'].unique()

    for eleve_id in eleves:

        # Filtre les données de cet élève
        data_eleve = df[df['eleve_id'] == eleve_id]

        # TODO 4.1 — Extraire les notes et coefficients
        # sous forme de np.array
        # notes = np.array(...)
        # coefficients = np.array(...)

        # TODO 4.2 — Calculer la moyenne pondérée avec np.dot()
        # Formule : sum(note * coeff) / sum(coeff)
        # Indice : np.dot(notes, coefficients) fait la somme 
        # des produits note*coeff automatiquement
        # moyenne_ponderee = ...

        # TODO 4.3 — Calculer aussi la moyenne simple
        # pour comparer
        # moyenne_simple = np.mean(notes)

        # TODO 4.4 — Récupérer nom et prénom de l'élève
        # nom = ...
        # prenom = ...

        # TODO 4.5 — Ajouter à la liste resultats :
        # un dictionnaire avec eleve_id, nom, prenom,
        # moyenne_simple, moyenne_ponderee
        # resultats.append({...})

        pass

    # TODO 4.6 — Convertir la liste en DataFrame
    # df_resultats = pd.DataFrame(resultats)

    # TODO 4.7 — Trier par moyenne_ponderee décroissante
    # df_resultats = df_resultats.sort_values(...)

    # TODO 4.8 — Afficher le classement final
    # print(df_resultats)

    # TODO 4.9 — Afficher la différence max entre
    # moyenne simple et moyenne pondérée
    # (quel élève est le plus impacté par les coefficients ?)
    # print(...)

    # return df_resultats
    return pd.DataFrame()


# ═══════════════════════════════════════════════
# PARTIE 5 — MERGE ET ENRICHISSEMENT
# ═══════════════════════════════════════════════
"""
OBJECTIF : Combiner plusieurs DataFrames et enrichir les données.

NOTIONS COUVERTES :
- pd.DataFrame() depuis un dictionnaire
- pd.merge()
- Création de nouvelles colonnes calculées
- np.where() et np.select() pour les conditions
"""

def partie5_merge_enrichissement(df: pd.DataFrame,
                                  df_ponderee: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 50)
    print("PARTIE 5 — MERGE ET ENRICHISSEMENT")
    print("=" * 50)

    # On te donne ce DataFrame de profils élèves
    profils = pd.DataFrame({
        'eleve_id': ['E001','E002','E003','E004','E005',
                     'E006','E007','E008','E009','E010'],
        'age':      [14, 15, 14, 15, 14, 15, 14, 15, 14, 15],
        'ville':    ['Kasserine','Tunis','Sfax','Sousse',
                     'Kasserine','Tunis','Sfax','Sousse',
                     'Kasserine','Tunis'],
        'statut_bourse': [True, False, False, True, False,
                          False, True, True, False, False]
    })

    # TODO 5.1 — Fusionner df_ponderee avec profils
    # sur la clé commune eleve_id
    # df_complet = pd.merge(...)
    # print(df_complet)

    # TODO 5.2 — Créer une colonne "mention" selon ces règles :
    # moyenne_ponderee >= 18 → "Excellent"
    # moyenne_ponderee >= 15 → "Bien"
    # moyenne_ponderee >= 12 → "Assez Bien"
    # moyenne_ponderee >= 10 → "Passable"
    # moyenne_ponderee <  10 → "Insuffisant"
    # Utilise np.select() avec conditions et choix
    # conditions = [...]
    # choix = [...]
    # df_complet['mention'] = np.select(conditions, choix)

    # TODO 5.3 — Créer une colonne "statut_academique" :
    # Si moyenne_ponderee < 10 → "En difficulté"
    # Sinon → "Dans les normes"
    # Utilise np.where()
    # df_complet['statut_academique'] = np.where(...)

    # TODO 5.4 — Afficher le nombre d'élèves par mention
    # print(df_complet['mention'].value_counts())

    # TODO 5.5 — Afficher les élèves boursiers
    # et leur mention — sont-ils bien soutenus ?
    # boursiers = ...
    # print(boursiers[['nom','prenom','moyenne_ponderee','mention']])

    # return df_complet
    return pd.DataFrame()


# ═══════════════════════════════════════════════
# PARTIE 6 — EXPORT DES RÉSULTATS
# ═══════════════════════════════════════════════
"""
OBJECTIF : Exporter les analyses produites en fichiers réutilisables.

NOTIONS COUVERTES :
- .to_csv()
- .to_json()
- Formatage des floats
- Organisation des outputs
"""

def partie6_export(df_complet: pd.DataFrame,
                   df_ponderee: pd.DataFrame,
                   df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("PARTIE 6 — EXPORT")
    print("=" * 50)

    # TODO 6.1 — Exporter le classement complet en CSV
    # Fichier : output/classement_final.csv
    # Sans l'index pandas (index=False)
    # df_complet.to_csv(...)
    # print("✅ classement_final.csv exporté")

    # TODO 6.2 — Exporter les moyennes par matière en JSON
    # Fichier : output/moyennes_par_matiere.json
    # df.groupby('matiere')['note'].mean()
    #   .sort_values(ascending=False)
    #   .to_json(...)
    # print("✅ moyennes_par_matiere.json exporté")

    # TODO 6.3 — Exporter uniquement les élèves
    # en difficulté (moyenne < 10) en CSV
    # Fichier : output/eleves_en_difficulte.csv
    # en_difficulte = df_ponderee[...]
    # en_difficulte.to_csv(...)
    # print("✅ eleves_en_difficulte.csv exporté")

    # TODO 6.4 — Afficher un résumé final :
    # "Analyse terminée. X élèves analysés.
    #  Meilleure classe : [classe].
    #  Matière la plus faible : [matiere]."
    pass


# ═══════════════════════════════════════════════
# MAIN — Point d'entrée
# ═══════════════════════════════════════════════

if __name__ == "__main__":

    # Lance les parties dans l'ordre
    # Décommente chaque ligne au fur et à mesure
    # que tu complètes les parties

    df = partie1_exploration(DATA_PATH)

    # partie2_filtrage(df)

    # partie3_agregation(df)

    # df_ponderee = partie4_moyenne_ponderee(df)

    # df_complet = partie5_merge_enrichissement(df, df_ponderee)

    # partie6_export(df_complet, df_ponderee, df)

    print("\n✅ Exercice S1 terminé — pense à commit !")

Structure Finale du Repo à Créer
aikup-intern-emna/
├── data/
│   └── notes_eleves.csv        ← tu fournis ce fichier
├── exercice/
│   └── analyse_notes.py        ← tu fournis ce fichier
├── output/
│   └── .gitkeep                ← dossier vide versionné
└── README.md                   ← tu fournis ce fichier