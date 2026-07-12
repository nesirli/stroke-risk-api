from pydantic import BaseModel, Field
from typing import Literal

class Patient(BaseModel):
    """Input schema for a single stroke-risk prediction request."""

    id: int = Field(gt=0)
    age: float = Field(gt=0.0, lt=120.0)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    avg_glucose_level: float = Field(gt=50.0)
    bmi: float | None = None
    gender: Literal['male', 'female', 'other']
    ever_married: Literal['no', 'yes']
    work_type: Literal['children', 'govt_job', 'never_worked', 'private', 'self_employed']
    residence_type: Literal['rural', 'urban']
    smoking_status: Literal['formerly_smoked', 'never_smoked', 'smokes', 'unknown']