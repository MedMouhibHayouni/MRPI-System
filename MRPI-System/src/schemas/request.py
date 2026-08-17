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

    # FIX (CDC 3.4.4): "alpha ... ajustable via l'API" était implémenté dans
    # hybrid.py (cf_weight/cbf_weight) mais jamais exposé au client — aucun
    # champ de requête ne permettait de le faire varier. Ajouté ici, avec
    # borne [0, 1] et défaut None (None = HybridEngine utilise ses défauts
    # 0.6/0.4, comportement identique à avant pour tout client existant qui
    # n'envoie pas ce champ).
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Poids du score CBF dans la fusion hybride (CF = 1 - alpha). "
        "Défaut CDC : 0.6. Si omis, HybridEngine applique ses valeurs par défaut.",
    )

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
