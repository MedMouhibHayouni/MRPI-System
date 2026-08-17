# MRPI-System — Moteur de Recommandation Pédagogique Intelligent

Moteur de recommandation hybride (Content-Based Filtering + Collaborative Filtering),
enrichi par une intégration LLM pour la génération d'explications, l'adaptation du
wording et la suggestion de plans d'étude.

Développé dans le cadre du stage été 2026 — AiKup Tech SUARL.

---

## À Propos d'AiKup Tech

AiKup Tech SUARL est une startup tunisienne spécialisée en Intelligence Artificielle
et en transformation digitale, incubée à la Pépinière d'Entreprises APII de Kasserine
depuis février 2026.

> **"Built in Kasserine. Designed for the World."**

---

## Architecture du moteur

Le système combine deux approches complémentaires (CDC 3.4.4) :

- **Content-Based Filtering (CBF)** — `src/CBF.py` : compare le profil de
  l'apprenant (matière, concept faible, niveau, style d'apprentissage) aux
  métadonnées des ressources via un encodage one-hot pondéré sur 4 dimensions
  (subject, concept, difficulty, type) et une similarité cosinus. Les ressources
  déjà consommées et celles dont les prérequis ne sont pas satisfaits sont
  exclues du pool de candidats avant scoring (filtre d'éligibilité, pas une
  dimension notée).

- **Collaborative Filtering (CF)** — `src/CF.py` : exploite la matrice
  utilisateur-ressource (`src/user_itemMatrix.py`) pour identifier des étudiants
  similaires (KNN, cosine) et recommander les ressources qui leur ont été utiles.
  Une SVD est appliquée automatiquement en amont si sa variance expliquée dépasse
  un seuil (`cf_config.py` — `SVD_VARIANCE_THRESHOLD`), sinon un KNN brut est
  utilisé directement.

- **Fusion hybride** — `src/hybrid.py` : combine les deux scores selon
  `Score_hybride = alpha × Score_CB + (1 - alpha) × Score_CF`.

- **Intégration LLM** — `src/LLM_API.py` : génère une explication personnalisée,
  reformule le wording des ressources selon le style d'apprentissage, et suggère
  un plan d'étude séquentiel (CDC 3.4.5). Chaque appel dispose d'un cache
  (fingerprint de contenu) et d'un fallback statique en cas d'échec réseau.



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

### 4. Variables d'environnement

Créer un fichier `.env` à la racine avec :
```
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=llama-3.1-8b-instant
```

### 5. Lancer l'API
```bash
uvicorn src.main:app --workers 4 --host 0.0.0.0 --port 8000
```

Tailles des executors ajustables sans modifier le code (voir `src/main.py`) :
```bash
ML_EXECUTOR_WORKERS=8 LLM_EXECUTOR_WORKERS=150 uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 6. Lancer Jupyter Notebook (exploration)
```bash
jupyter notebook
```
> Sélectionner le kernel `venv` dans Jupyter pour utiliser les bonnes dépendances.

---

## Structure du projet

```
MRPI-System/
├── data/
│   ├── resources.csv
│   ├── students.csv
│   └── interactions.csv
├── models/
│   ├── cf_model.pkl        (généré, invalidé par fingerprint)
│   ├── encoder.pkl         (généré, invalidé par fingerprint)
│   └── llm_cache.json      (généré, cache des appels LLM)
├── src/
│   ├── __init__.py
│   ├── main.py             # API FastAPI, endpoints, executors
│   ├── hybrid.py           # Fusion CBF + CF
│   ├── CBF.py              # Content-Based Filtering
│   ├── CF.py               # Collaborative Filtering
│   ├── LLM_API.py          # Intégration LLM (explication, wording, study plan)
│   ├── metrics.py          # Précision@K, Recall@K, NDCG@K, diversité, couverture
│   ├── cache_utils.py      # Fingerprinting partagé CF/CBF/LLM
│   ├── cf_config.py        # Configuration KNN/SVD
│   ├── user_itemMatrix.py  # Construction matrice utilisateur-ressource
│   └── schemas/
│       ├── __init__.py
│       ├── request.py
│       └── response.py
├── notebook/
├── tests/
├── venv/
├── requirements.txt
└── README.md
```

---

## Endpoints API

| Méthode | Route | Description |
|---|---|---|
| POST | `/recommendations` | Recommandations hybrides + explication/wording/plan LLM |
| POST | `/recommendations/no-llm` | Identique, sans les appels LLM (benchmark de charge) |
| POST | `/student-profile` | Valide un profil apprenant sans générer de recommandation |
| GET | `/resources` | Catalogue filtré par `subject`/`difficulty` |
| GET | `/resources/all` | Catalogue complet |

---

## Data Model

### INPUT — RecommendationRequest (CDC Tableau 6)

| Champ | Type | Obligatoire | Valeurs acceptées |
|-------|------|-------------|-------------------|
| student_id | string | Oui | ex: "STU-2026-0042" |
| subject | string | Oui | ex: "Mathematiques" |
| weak_concept | string | Oui | ex: "Derivees partielles" |
| academic_level | enum | Oui | beginner / intermediate / advanced |
| learning_style | enum | Oui | visual / auditory / kinesthetic / textual |
| past_interactions | list[string] | Non | ex: ["RES-101", "RES-205"] |

> Pas de champ `alpha` actuellement — voir « Écarts connus par rapport au CDC ».

### OUTPUT — RecommendationResponse (CDC Tableau 7/8)

| Champ | Type | Description |
|-------|------|-------------|
| request_id | string | ID unique de la requête |
| student_id | string | ID de l'apprenant |
| recommendations | list | Liste ordonnée de ressources (voir ci-dessous) |
| adapted_recommendations | list | Recommandations reformulées par le LLM |
| study_plan | list | Ordre de consommation suggéré |
| llm_explanation | string | Explication générée par le LLM |
| metadata | object | `{"latency_ms": int, "model": str}` |

### Structure d'une recommandation

| Champ | Type | Description |
|-------|------|-------------|
| resource_id | string | Identifiant unique de la ressource |
| title | string | Titre de la ressource |
| type | enum | exercise / micro_lesson / video |
| relevance_score | float | Score hybride pondéré — voir écarts connus |
| difficulty | enum | beginner / intermediate / advanced |
| estimated_time_min | integer | Temps estimé (minutes) |

---

## Métriques d'évaluation (CDC Tableau 9)

Implémentées dans `src/metrics.py` :

- **Precision@K** — proportion de recommandations pertinentes dans le top-K.
- **Recall@K** — proportion des ressources pertinentes retrouvées (None si
  aucune ressource pertinente n'existe pour le profil).
- **NDCG@K** — qualité du classement, pondérée par position.
- **Diversité intra-liste** — distance cosinus moyenne entre paires de
  recommandations.
- **Couverture** — proportion du catalogue total apparaissant dans les
  recommandations sur l'ensemble du dataset de test (seuil CDC ≥ 0.25).

Le ground truth de pertinence (`get_relevant_resources`) est défini
indépendamment du moteur testé : correspondance exacte matière + concept +
difficulté, prérequis satisfaits. Cette définition ne doit jamais être ajustée
pour améliorer artificiellement un score.

---

## Tests et vérification

Les tests sont regroupés dans `tests/`, séparés entre tests unitaires et tests
d'intégration. Ils servent à vérifier les objectifs non fonctionnels du projet :
couverture minimale, reproductibilité, qualité de l'API, robustesse des filtres
et absence d'appels réseau non contrôlés pendant les tests automatiques.

### Lancer les tests

```bash
python -m pytest
```

Avec couverture de code, selon `pytest.ini` :

```bash
python -m pytest --cov=src --cov-report=term-missing
```

Sur Windows, si le dossier temporaire global de pytest est bloqué, utiliser un
dossier temporaire local au projet :

```bash
python -m pytest -o addopts= --basetemp .tmp/pytest -p no:cacheprovider
```

### Rôle des fichiers de test

| Fichier | Ce qui est vérifié |
|---|---|
| `tests/conftest.py` | Fixtures partagées : mini catalogue de ressources, factory de `RecommendationRequest`, imports stables depuis le dossier `tests/`. |
| `tests/unit/test_cbf.py` | Moteur CBF : construction/chargement du vectorizer, cache encoder, exclusion des ressources déjà vues, filtre des prérequis, parsing des prérequis CSV (`RES-001`, listes séparées), fallback de type. |
| `tests/unit/test_cf.py` | Moteur CF : construction du modèle KNN/SVD, cache modèle, singleton/reset, étudiants inconnus, exclusion des ressources déjà notées, cas SVD avec moins de composantes que demandé. |
| `tests/unit/test_hybrid_fusion.py` | Fusion hybride : normalisation des scores, pénalité fallback, fusion CF+CBF, limites `max_recommendations`, garde-fou matière quand CF est seul. |
| `tests/unit/test_llm_api.py` | Module LLM : cache JSON, parsing JSON, fallbacks si réponse invalide ou erreur réseau, wording batch, génération d'explication, plan d'étude, appel SDK mocké. |
| `tests/unit/test_metrics.py` | Métriques d'évaluation : Precision@K, Recall@K, NDCG@K, diversité intra-liste, couverture, ground truth avec prérequis. |
| `tests/unit/test_cache_utils.py` | Utilitaires de cache : fingerprint des fichiers source, validation/invalidation de cache, fingerprint de contenu pour le cache LLM. |
| `tests/integration/test_main_api.py` | API FastAPI complète : `/recommendations`, `/recommendations/no-llm`, `/student-profile`, `/resources`, parsing CSV, cache `load_resources/load_students`; les appels LLM sont mockés. |
| `tests/integration/test_hybrid_engine.py` | Pipeline réel `HybridEngine` : chargement des vraies données, CBF + CF + fusion, top_n, étudiant inconnu, exclusion des interactions passées. |
| `tests/integration/test_user_item_matrix.py` | Construction réelle de la matrice utilisateur-ressource : dimensions, densité/sparsité, colonnes ressources, index étudiants. |
| `tests/integration/test_llm_connection.py` | Smoke test live du provider LLM, désactivé par défaut. À lancer seulement avec `RUN_LIVE_LLM_TESTS=1` et les variables `.env`. |
| `tests/integration/test_integration_llm.py` | Script manuel end-to-end HybridEngine -> LLM_API pour vérifier les vraies explications, wording et plans d'étude. |

### Correspondance avec les objectifs non fonctionnels

| Critère | Méthode de vérification dans le projet |
|---|---|
| Temps de réponse | `locustfile.py`, endpoint `/recommendations/no-llm`, fichiers `results_*_stats.csv`. |
| Scalabilité | Tests d'intégration API et moteur hybride, séparation `ml_executor` / `llm_executor`. |
| Qualité de documentation | README, docstrings utiles, schémas Pydantic, description des tests dans cette section. |
| Couverture de tests | `pytest --cov`, seuil configuré à 70% dans `pytest.ini`. |
| Reproductibilité | `requirements.txt`, `requirements-test.txt`, tests sans réseau par défaut, données CSV versionnées. |
| Sécurité | Validation Pydantic des entrées, tests 422 sur payload invalide, absence d'appel LLM live sauf opt-in. |

---

## Choix techniques approfondis

**Deux stratégies de cache différentes, pour deux raisons différentes.**
CF (`cf_model.pkl`) et CBF (`encoder.pkl`) utilisent une invalidation par
fingerprint SHA-256 du contenu des fichiers source (`cache_utils.py`) : ce
sont des modèles entraînés, dont la fraîcheur affecte directement la qualité
des recommandations — un modèle périmé silencieusement fausserait le scoring
sans qu'on s'en aperçoive. `load_students()`/`load_resources()` (`main.py`)
utilisent un simple `lru_cache`, sans invalidation : ce sont des endpoints de
consultation brute, pas un modèle, donc l'incohérence en cas de modification
du CSV en cours de run est jugée acceptable (redémarrage du process requis
pour rafraîchir). Le cache LLM (`llm_cache.json`) est indexé par empreinte de
**contenu** (profil + recommandations, pas juste un TTL) : deux requêtes avec
un input strictement identique retombent sur la même clé, donc pas de
staleness possible par construction — au prix d'un fichier qui grossit sans
purge (limitation connue, voir plus bas).

**Dette technique assumée : `resources.csv` est chargé indépendamment à
trois endroits** (`main.py::load_resources`, `hybrid.py::HybridEngine.__init__`,
`user_itemMatrix.py`), avec des invalidations différentes (lru_cache jamais
invalidé / aucune invalidation / aucune invalidation) et un traitement
différent (seul `hybrid.py` normalise la casse de subject/concept). Aucun
gain de performance à unifier ces trois lectures (chacune est déjà un
singleton, donc un seul `read_csv` par process dans les trois cas), mais un
vrai risque de divergence de données si `resources.csv` est modifié en cours
d'exécution sans redémarrage du process. Non corrigé dans cette version — à
trancher : soit un chargeur central partagé avec sa propre invalidation par
fingerprint, soit documenté comme contrainte opérationnelle ("redémarrer le
service après toute modification manuelle des CSV source").

**Normalisation par-requête plutôt que globale.** `normalize_scores()`
(`hybrid.py`) normalise les scores CF/CBF en min-max sur la liste de
candidats de la requête courante, pas sur une distribution globale
pré-calculée sur tout le dataset. Choix pragmatique (pas de passe de
calibration séparée à maintenir), mais qui a une conséquence directe : avec
moins de 3 candidats, un min-max par liste n'a aucune validité statistique.
Le code retombe alors sur une division par une borne théorique connue
(`CF_SCORE_MAX = 5.0` pour les scores CF bruts) plutôt que sur les valeurs
brutes non bornées — condition nécessaire pour respecter le contrat
`relevance_score ∈ [0.0, 1.0]` du CDC (Tableau 8) même sur les profils à
faible nombre de voisins similaires (le cas le plus fréquent avec un
dataset de test de cette taille).

**Garde-fou matière asymétrique entre le CF pur et la fusion normale.**
Quand CF et CBF sont fusionnés ensemble, les résultats CF ne sont **jamais**
filtrés par matière — volontairement, pour préserver l'effet de sérendipité
que le CDC 3.4.4 attribue explicitement au filtrage collaboratif. Mais quand
CBF est vide et que CF doit être servi seul (`_convert_cf_to_recommendations`),
un filtre de matière est appliqué en repli : sans CBF pour ancrer la
pertinence thématique, du CF non filtré seul risquerait de recommander des
ressources totalement hors-sujet. Ce n'est pas une incohérence, c'est une
asymétrie voulue : la sérendipité est un bonus tant qu'un signal de
pertinence thématique (CBF) existe déjà en parallèle, pas un substitut à ce
signal quand il est absent.

**SVD conditionnelle, pas systématique.** Le CDC prescrit KNN + SVD pour le
CF, mais `CF.py::build_cf_model` ne fit la SVD "réelle" que si une SVD test
à 5 composantes explique au moins 50% de variance (`SVD_VARIANCE_THRESHOLD`).
En dessous, KNN tourne directement sur la matrice brute. Justification :
avec ~30 étudiants (taille actuelle du dataset), une factorisation en facteurs
latents n'a pas assez de signal pour être fiable — la courbe de variance
expliquée est quasi-linéaire, sans coude (elbow) exploitable. Réévaluer ce
seuil quand le dataset atteint ~80-100 étudiants.

**Pydantic pour les schémas** — validation automatique des entrées/sorties,
type-safe, intégration native FastAPI, erreurs 422 générées automatiquement
sans code de validation manuel.

**resources.csv / students.csv / interactions.csv séparés** — catalogue
statique (resources), profils apprenants (students), historique
d'interactions (interactions) : trois sources de vérité distinctes alimentant
des parties différentes du moteur hybride.

**Deux ThreadPoolExecutor séparés (`ml_executor` / `llm_executor`)** — le
calcul hybride est CPU-bound (pandas/numpy), les appels LLM sont I/O-bound
(attente réseau). Les mélanger dans un seul pool fait attendre le travail CPU
derrière des threads bloqués sur le réseau, et inversement — voir docstring de
`src/main.py` pour le diagnostic complet (Locust).

**`__init__.py`** — indique que le dossier est un package Python importable.
Sans lui, `from src.schemas.request import RecommendationRequest` échoue.
