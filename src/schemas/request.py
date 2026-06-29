from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum
import re


class AcademicLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class LearningStyle(str, Enum):
    textual = "textual"
    visual = "visual"
    auditory = "auditory"
    kinesthetic = "kinesthetic"


class RecommendationRequest(BaseModel):
    student_id: str
    subject: str
    weak_concept: str
    academic_level: AcademicLevel
    learning_style: LearningStyle
    past_interactions: Optional[List[str]] = Field(default_factory=list)

    @field_validator("student_id")
    @classmethod
    def validate_student_id(cls, v):
        if not re.match(r"^STU-\d{4}-\d{4}$", v):
            raise ValueError("Format attendu : STU-YYYY-XXXX")
        return v

    @field_validator("subject", "weak_concept")
    @classmethod
    def strip_strings(cls, v):
        return v.strip().lower()
 