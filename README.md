# 🧠 AiKup Tech SUARL — Stage Été 2026
### Exercice S1 : Fondamentaux Python pour la Data

---

<div align="center">

![AiKup Tech](https://img.shields.io/badge/AiKup%20Tech-SUARL-blue?style=for-the-badge)
![Stage](https://img.shields.io/badge/Stage-Été%202026-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Statut-En%20cours-yellow?style=for-the-badge)

</div>

---

## 👤 Informations Stagiaire

| Champ | Détail |
|---|---|
| **Nom complet** | Emna Harhouri |
| **Email** | emnahr109@gmail.com |
| **Établissement** | ESSTHS — Hammam Sousse |
| **Niveau** | Licence Génie Logiciel — 2ème année |
| **Superviseur** | Ikram Ajlani CTO AiKup Tech |
| **Branche de travail** | `learning/week1-python` |
| **Due date** | Fin Semaine 1 |

---

## 🏢 À Propos d'AiKup Tech

AiKup Tech SUARL est une startup tunisienne spécialisée en Intelligence Artificielle
et en transformation digitale, incubée à la Pépinière d'Entreprises APII de Kasserine
depuis février 2026. Nos projets couvrent les domaines de l'EdTech, l'AgriTech,
et la digitalisation des PME tunisiennes.

> **"Built in Kasserine. Designed for the World."**

---

## 📋 Contexte de l'Exercice

Tu travailles chez AiKup Tech sur un module d'analyse de données scolaires.
On t'a fourni un fichier CSV contenant les notes de **10 élèves** répartis
en **2 classes** (3ème A et 3ème B) pour **6 matières** au premier trimestre.

Ton rôle est d'analyser ces données comme le ferait un système intelligent
qui aide les directeurs d'école à comprendre les performances de leurs élèves —
détecter les élèves en difficulté, identifier les matières faibles, et produire
des rapports exploitables.

---

## 📁 Structure du Repo

```
aikup-intern-emna/
│
├── 📂 data/
│   └── notes_eleves.csv          ← Données fournies par AiKup (ne pas modifier)
│
├── 📂 exercice/
│   └── analyse_notes.py          ← TON fichier de travail principal
│
├── 📂 output/                    ← Tes fichiers générés vont ici
│   ├── classement_final.csv      ← Généré en Partie 6
│   ├── moyennes_par_matiere.json ← Généré en Partie 6
│   ├── eleves_en_difficulte.csv  ← Généré en Partie 6
│   └── .gitkeep                  ← Garde ce fichier (permet de versionner le dossier vide)
│
└── README.md                     ← Ce fichier
```

---

## 🗂️ Description du Dataset

**Fichier :** `data/notes_eleves.csv`

| Colonne | Type | Description |
|---|---|---|
| `eleve_id` | string | Identifiant unique de l'élève (ex: E001) |
| `nom` | string | Nom de famille |
| `prenom` | string | Prénom |
| `classe` | string | Classe de l'élève (3ème A ou 3ème B) |
| `matiere` | string | Matière concernée |
| `note` | float | Note sur 20 |
| `coefficient` | int | Coefficient de la matière (2, 3 ou 4) |
| `trimestre` | string | Trimestre (T1) |

**Matières disponibles :**
- Mathématiques (coeff. 4)
- Français (coeff. 3)
- Sciences (coeff. 3)
- Histoire-Géo (coeff. 2)
- Anglais (coeff. 2)
- Informatique (coeff. 2)

---

## 🎯 Objectifs de l'Exercice

À la fin de cet exercice, tu seras capable de :

- ✅ Charger et explorer un dataset CSV avec **Pandas**
- ✅ Filtrer des données avec des conditions simples et multiples
- ✅ Agréger et grouper des données avec **groupby**
- ✅ Calculer une **moyenne pondérée** avec **NumPy** (dot product)
- ✅ Fusionner deux DataFrames avec **merge**
- ✅ Créer des colonnes calculées avec **np.where** et **np.select**
- ✅ Exporter des résultats en **CSV** et **JSON**

---

## 📐 Structure de l'Exercice

L'exercice est divisé en **6 parties progressives** dans le fichier
`exercice/analyse_notes.py`.

| Partie | Titre | Notions clés | Difficulté |
|---|---|---|---|
| 1 | Chargement & Exploration | read_csv, info, describe, isnull | ⭐ |
| 2 | Filtrage & Sélection | Conditions booléennes, loc, isin | ⭐⭐ |
| 3 | Agrégation & Groupby | groupby, agg, sort_values | ⭐⭐ |
| 4 | Moyenne Pondérée NumPy | np.array, np.dot, np.sum | ⭐⭐⭐ |
| 5 | Merge & Enrichissement | pd.merge, np.where, np.select | ⭐⭐⭐ |
| 6 | Export des Résultats | to_csv, to_json | ⭐ |

---

## 🚀 Comment Démarrer

### 1. Cloner le repo et se positionner sur la bonne branche

```bash
git clone https://github.com/aikup-tech/[nom-du-repo].git
cd [nom-du-repo]
git checkout learning/week1-python
```

### 2. Créer un environnement virtuel

```bash
# Créer l'environnement
python -m venv venv

# L'activer (Windows)
venv\Scripts\activate

# L'activer (macOS / Linux)
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install pandas numpy
```

### 4. Lancer l'exercice

```bash
cd exercice
python analyse_notes.py
```

---

## 📦 Dépendances

```txt
pandas>=1.5.0
numpy>=1.23.0
```

> Pas de requirements.txt fourni volontairement — c'est à toi de le créer
> et de l'ajouter au repo. C'est une des tâches implicites de l'exercice.

---

## 📤 Règles de Livraison

### Convention de commits obligatoire

Chaque partie terminée = **un commit** avec ce format exact :

```
feat: S1-partie-1 terminée
feat: S1-partie-2 terminée
feat: S1-partie-3 terminée
feat: S1-partie-4 terminée
feat: S1-partie-5 terminée
feat: S1-partie-6 terminée
```

> ❌ Ne pas faire un seul commit pour tout l'exercice.
> ✅ Un commit par partie = progression visible = meilleure évaluation.

### Checklist avant de déclarer l'exercice terminé

```
☐ Les 6 parties sont complétées et fonctionnent sans erreur
☐ Le dossier /output/ contient les 3 fichiers exportés
☐ Un fichier requirements.txt a été créé et commité
☐ Le code est commenté (chaque bloc a un commentaire)
☐ 6 commits minimum avec les messages corrects
☐ La carte Trello a été déplacée en "EN RÉVISION"
☐ Un commentaire de fin a été ajouté sur la carte Trello
```

---

## 📊 Critères d'Évaluation

| Critère | Poids |
|---|---|
| Exactitude des résultats (calculs corrects) | 40% |
| Qualité du code (lisibilité, commentaires) | 25% |
| Respect des conventions Git (commits) | 20% |
| Organisation et propreté du repo | 15% |

---

## 💡 Ressources d'Apprentissage

Si tu bloques sur une notion, voici les ressources officielles recommandées :

| Notion | Ressource |
|---|---|
| Pandas complet | https://pandas.pydata.org/docs/user_guide/index.html |
| NumPy débutant | https://numpy.org/doc/stable/user/absolute_beginners.html |
| pd.merge() | https://pandas.pydata.org/docs/reference/api/pandas.merge.html |
| np.dot() | https://numpy.org/doc/stable/reference/generated/numpy.dot.html |
| np.where() | https://numpy.org/doc/stable/reference/generated/numpy.where.html |
| np.select() | https://numpy.org/doc/stable/reference/generated/numpy.select.html |

---

## ⚠️ Règles Importantes

> **Confidentialité :** Le contenu de ce repo est strictement confidentiel.
> Ne pas partager le code, les données, ou les résultats en dehors
> du cadre du stage AiKup Tech SUARL.

> **Propriété intellectuelle :** Tout code produit dans le cadre de ce stage
> appartient à AiKup Tech SUARL conformément au Cahier des Charges signé.

> **Intégrité :** Le travail doit être personnel. L'utilisation d'IA générative
> pour compléter les TODO est tolérée uniquement comme aide à la compréhension,
> pas comme remplacement du raisonnement. Tu devras expliquer chaque ligne
> lors du call de review.

---

## 📞 Contact & Support

Si tu es bloquée depuis **plus de 2 heures** sur un point :

1. Ajoute le label 🔴 **Bloqué** sur la carte Trello concernée
2. Laisse un commentaire détaillant le problème et ce que tu as déjà essayé
3. Envoie un message WhatsApp au superviseur

> Ne reste jamais bloquée en silence. Signaler un blocage = professionnalisme.

---

<div align="center">

**AiKup Tech SUARL** — Kasserine, Tunisie
`aikup.tn` | `linkedin.com/company/aikup`

*Document interne — Confidentiel — © 2026 AiKup Tech SUARL*

</div>
