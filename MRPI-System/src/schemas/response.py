from pydantic import BaseModel
from typing import List, Optional
from src.schemas.request import AcademicLevel
from enum import Enum


class ResourceType(str, Enum):
    exercise = "exercise"
    micro_lesson = "micro_lesson"
    video = "video"


class ResourceRecommendation(BaseModel):
    resource_id: str
    title: str
    type: ResourceType
    relevance_score: float
    difficulty: AcademicLevel
    estimated_time_min: int
    # FIX : ce champ manquait totalement. adapt_wording() (LLM_API.py) lit
    # resource.get("description", "") — sans ce champ dans le schéma, la
    # description était TOUJOURS vide dès que les objets passaient par
    # ResourceRecommendation.dict(), quelle que soit la qualité du LLM.
    # Défaut "" pour ne rien casser côté clients existants qui ne l'envoient pas.
    description: Optional[str] = ""


class RecommendationResponse(BaseModel):
    request_id: str
    student_id: str
    recommendations: List[ResourceRecommendation]
    # BUG CRITIQUE : était `str` (non-Optional). L'endpoint /recommendations/no-llm
    # renvoie llm_explanation=None -> Pydantic v2 lève une erreur de validation
    # à CHAQUE appel de cet endpoint (ValidationError, pas un warning silencieux).
    # C'est probablement pour ça que le benchmark A/B LLM-on/LLM-off sur lequel
    # tu t'es appuyée pour diagnostiquer le bottleneck hybrid tournait — ou pas —
    # sur cet endpoint. À vérifier une fois ce fix déployé.
    llm_explanation: Optional[str] = None
    metadata: dict
