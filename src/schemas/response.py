from pydantic import BaseModel
from typing import List
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


class RecommendationResponse(BaseModel):
    request_id: str
    student_id: str
    recommendations: List[ResourceRecommendation]
    llm_explanation: str
    metadata: dict
