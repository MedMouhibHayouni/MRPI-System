from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class AcademicLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class LearningStyle(str, Enum):
    visual = "visual"
    auditory = "auditory"
    kinesthetic = "kinesthetic"


class RecommendationRequest(BaseModel):
    student_id: str
    subject: str
    weak_concept: str
    academic_level: AcademicLevel
    learning_style: LearningStyle
    past_interactions: Optional[List[str]] = []
