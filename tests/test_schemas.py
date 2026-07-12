import pytest
from pydantic import ValidationError

from stroke_risk.app.schemas import Patient

BASE_PATIENT = dict(
    id=1,
    age=45.0,
    hypertension=0,
    heart_disease=0,
    avg_glucose_level=100.0,
    bmi=25.0,
    gender="male",
    ever_married="yes",
    work_type="private",
    residence_type="urban",
    smoking_status="never_smoked",
)


def test_valid_patient_is_accepted():
    patient = Patient(**BASE_PATIENT)
    assert patient.id == 1
    assert patient.age == 45.0


def test_bmi_is_optional():
    patient = Patient(**{**BASE_PATIENT, "bmi": None})
    assert patient.bmi is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 0),
        ("age", 0.0),
        ("age", 120.0),
        ("avg_glucose_level", 50.0),
        ("hypertension", 2),
        ("heart_disease", -1),
        ("gender", "unknown"),
        ("ever_married", "maybe"),
        ("work_type", "freelance"),
        ("residence_type", "suburban"),
        ("smoking_status", "vaping"),
    ],
)
def test_invalid_field_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Patient(**{**BASE_PATIENT, field: value})
