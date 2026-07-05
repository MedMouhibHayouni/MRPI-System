from pydantic import BaseModel, Field, field_validator, StringConstraints
from typing import List, Optional
from enum import Enum
import re
from typing import Annotated


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
    subject: Annotated[str, StringConstraints(max_length=100)]
    weak_concept: Annotated[str, StringConstraints(max_length=100)]
    academic_level: AcademicLevel
    learning_style: LearningStyle
    past_interactions: Optional[
        List[Annotated[str, StringConstraints(max_length=20)]]
    ] = Field(default_factory=list, max_length=500)

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

    @field_validator("past_interactions", mode="before")
    @classmethod
    def parse_interactions(cls, v):
        if isinstance(v, list):
            return v
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return []
        return [r.strip() for r in str(v).split(";") if r.strip()]
