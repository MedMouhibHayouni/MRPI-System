"""
Fixtures partagées pour les tests unitaires et d'intégration du
MRPI-System.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Permet `import src...` peu importe le répertoire d'où pytest est lancé.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schemas.request import AcademicLevel, LearningStyle, RecommendationRequest


@pytest.fixture
def sample_resources_df() -> pd.DataFrame:
    """Petit catalogue de ressources en mémoire, utilisé par les tests
    unitaires CBF/metrics pour éviter toute dépendance au disque."""
    return pd.DataFrame(
        [
            {
                "resource_id": "RES-001",
                "title": "Intro Fractions Video",
                "subject": "math",
                "concept": "fractions",
                "difficulty": "beginner",
                "type": "video",
                "estimated_time_min": 15,
                "prerequisites": np.nan,
                "description": "Introduction video aux fractions",
            },
            {
                "resource_id": "RES-002",
                "title": "Fractions Exercise",
                "subject": "math",
                "concept": "fractions",
                "difficulty": "beginner",
                "type": "exercise",
                "estimated_time_min": 20,
                "prerequisites": np.nan,
                "description": "Exercice sur les fractions",
            },
            {
                "resource_id": "RES-003",
                "title": "Advanced Fractions Video",
                "subject": "math",
                "concept": "fractions",
                "difficulty": "advanced",
                "type": "video",
                "estimated_time_min": 25,
                "prerequisites": ["RES-001"],
                "description": "Fractions avancees",
            },
            {
                "resource_id": "RES-004",
                "title": "Algebra Video",
                "subject": "math",
                "concept": "algebra",
                "difficulty": "beginner",
                "type": "video",
                "estimated_time_min": 15,
                "prerequisites": np.nan,
                "description": "Introduction a l'algebre",
            },
            {
                "resource_id": "RES-005",
                "title": "Grammar Exercise",
                "subject": "french",
                "concept": "grammar",
                "difficulty": "beginner",
                "type": "exercise",
                "estimated_time_min": 20,
                "prerequisites": np.nan,
                "description": "Exercice de grammaire",
            },
        ]
    )


@pytest.fixture
def make_request():
    """Factory pour construire un RecommendationRequest rapidement, avec
    des valeurs par défaut raisonnables, surchargeables via **overrides."""

    def _make(**overrides) -> RecommendationRequest:
        defaults = dict(
            student_id="STU-2026-0001",
            academic_level=AcademicLevel.beginner,
            learning_style=LearningStyle.visual,
            subject="math",
            weak_concept="fractions",
            past_interactions=[],
        )
        defaults.update(overrides)
        return RecommendationRequest(**defaults)

    return _make
