from pydantic import BaseModel
from typing import Any, List, Optional
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
    description: Optional[str] = ""


class RecommendationResponse(BaseModel):
    request_id: str
    student_id: str
    recommendations: List[ResourceRecommendation]
    adapted_recommendations: Optional[List[dict[str, Any]]] = None
    study_plan: Optional[List[dict[str, Any]]] = None
    llm_explanation: Optional[str] = None
    metadata: dict
