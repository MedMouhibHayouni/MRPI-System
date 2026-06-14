# AiKup Recommendation System

Moteur de recommandation pédagogique hybride (Content-Based + Collaborative Filtering).  
Développé dans le cadre du stage été 2026 — AiKup Tech SUARL.

---

## À Propos d'AiKup Tech

AiKup Tech SUARL est une startup tunisienne spécialisée en Intelligence Artificielle
et en transformation digitale, incubée à la Pépinière d'Entreprises APII de Kasserine
depuis février 2026.

> **"Built in Kasserine. Designed for the World."**

---

## Prérequis

- Python 3.10+
- Git

---

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/MedMouhibHayouni/MRPI-System.git
cd MRPI-System
```

### 2. Créer et activer l'environnement virtuel

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancer Jupyter Notebook
```bash
jupyter notebook
```
> Sélectionner le kernel `venv` dans Jupyter pour utiliser les bonnes dépendances.

---

## Structure du projet
MRPI-System/
├── data/
│   ├── resources.csv
│   └── students.csv
├── src/
│   ├── __init__.py
│   └── schemas/
│       ├── init.py
│       ├── request.py
│       └── response.py
├── notebook/
├── test/
├── venv/
├── requirements.txt
└── README.md

---

## Data Model

### INPUT — RecommendationRequest

| Champ | Type | Obligatoire | Valeurs acceptées |
|-------|------|-------------|-------------------|
| student_id | string | Oui | ex: "STU-2026-0001" |
| subject | string | Oui | ex: "Mathematiques" |
| weak_concept | string | Oui | ex: "fractions" |
| academic_level | enum | Oui | beginner / intermediate / advanced |
| learning_style | enum | Oui | visual / auditory / kinesthetic |
| past_interactions | list[string] | Non | ex: ["RES-001", "RES-002"] |

### OUTPUT — RecommendationResponse

| Champ | Type | Description |
|-------|------|-------------|
| request_id | string | ID unique de la requête |
| student_id | string | ID de l'apprenant |
| recommendations | list | Liste ordonnée de ressources |
| llm_explanation | string | Explication générée par le LLM |
| metadata | dict | latency_ms, model_version |

### Structure d'une recommandation

| Champ | Type | Description |
|-------|------|-------------|
| resource_id | string | ex: "RES-001" |
| title | string | Titre de la ressource |
| type | enum | exercise / micro_lesson / video |
| relevance_score | float | Score hybride 0.0 → 1.0 |
| difficulty | enum | beginner / intermediate / advanced |
| estimated_time_min | int | Temps estimé en minutes |

### Pourquoi Pydantic pour les schémas ?
Validation automatique des schémas d'entrée/sortie — type-safe, intégration native FastAPI

### Pourquoi resources.csv et students.csv ?
resources.csv = catalogue statique des ressources pédagogiques.
students.csv = profils apprenants à alimenter au moteur de recommandation.
Les deux sources alimentent des parties différentes du moteur hybride.

### Pourquoi __init__.py
Dans un projet Python structuré en packages, `__init__.py` sert à indiquer que le dossier est un package importable. Sans lui, `from src.schemas.request import RecommendationRequest` échoue.