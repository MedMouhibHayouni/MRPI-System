"""
MRPI-System - Module principal
"""

# Import des modules avec des chemins relatifs
from .CF import get_recommendations, get_cf_engine, CFEngine
from .CBF import build_vectorizer, load_vectorizer, recommend_cbf, prerequisites_met
from .hybrid import HybridEngine, get_hybrid_engine, hybrid_fusion
from .schemas.request import RecommendationRequest, AcademicLevel, LearningStyle
from .schemas.response import ResourceRecommendation, ResourceType

__all__ = [
    # CF
    "get_recommendations",
    "get_cf_engine",
    "CFEngine",
    # CBF
    "build_vectorizer",
    "load_vectorizer",
    "recommend_cbf",
    "prerequisites_met",
    # Hybrid
    "HybridEngine",
    "get_hybrid_engine",
    "hybrid_fusion",
    # Schemas
    "RecommendationRequest",
    "AcademicLevel",
    "LearningStyle",
    "ResourceRecommendation",
    "ResourceType",
]
