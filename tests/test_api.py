"""Unit tests for the FastAPI clinical endpoints."""

import pytest
from xautonet.api.server import (
    health_check,
    get_features_info,
    get_sample_patients,
    predict_patient_risk,
    PatientInput,
)

def test_health_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert data["decision_threshold"] == 0.30
    assert data["features_accepted"] == 19

def test_features_info():
    data = get_features_info()
    assert len(data["features"]) == 19
    assert len(data["bottleneck_features"]) == 11
    assert data["decision_threshold"] == 0.30

def test_predict_full_vitals():
    payload = PatientInput(
        vitals={
            "BaseExcess": 6.0,
            "Temp": 40.22,
            "Chloride": 105.0,
            "Hct": 26.50,
            "Hgb": 9.0,
            "Resp": 28.0,
            "HCO3": 24.0,
            "SIRS": 0.0,
            "Potassium": 3.8,
            "Creatinine": 0.7,
            "Phosphate": 4.2,
            "FiO2": 0.60,
            "O2Sat": 91.0,
            "Magnesium": 1.7,
            "SaO2": 90.0,
            "Lactate": 3.8,
            "pH": 7.25,
            "Calcium": 7.9,
            "BUN": 42.0,
        }
    )
    data = predict_patient_risk(payload)
    assert "prediction_probability" in data
    assert 0.0 <= data["prediction_probability"] <= 1.0
    assert data["decision_threshold"] == 0.30
    assert "risk of sepsis onset within the next 6 hours" in data["interpretation_6h"]
    assert "shap_waterfall" in data
    assert len(data["shap_waterfall"]) == 11
    assert "disclaimer" in data

def test_predict_missing_vitals_warnings():
    payload = PatientInput(
        vitals={
            "Temp": 37.0,
            "Resp": 16.0,
        }
    )
    data = predict_patient_risk(payload)
    assert len(data["warnings"]) > 0
    assert "defaulted to physiological normal" in data["warnings"][0]
    assert "risk_classification" in data
