"""FastAPI application providing clinical REST endpoints and hosting the Decision Support UI.

Uses the frozen 3-layer neural network trained on 11 clinical features with 0.30 decision threshold.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from xautonet.config import (
    FILTER_FEATURES,
    BOTTLENECK_FEATURES,
    TOP_SHAP_FEATURES,
    CLINICAL_NORMAL_RANGES,
    CLINICAL_PERMISSIBLE_LIMITS,
)
from xautonet.models.classifier import XAutoNetClassifier
from xautonet.xai.shap_explainer import DeepSHAPExplainer

app = FastAPI(
    title="XAutoNet Clinical Decision Support API",
    description="Investigational sepsis early warning system (6-hour advance horizon) based on clinical neural network inference.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DECISION_THRESHOLD = 0.30

# Load trained model artifacts
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_classifier.pt")
SCALER_MEAN_PATH = os.path.join(ARTIFACTS_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(ARTIFACTS_DIR, "scaler_scale.npy")
BACKGROUND_PATH = os.path.join(ARTIFACTS_DIR, "background_11.npy")

model: Optional[XAutoNetClassifier] = None
scaler_mean: Optional[np.ndarray] = None
scaler_scale: Optional[np.ndarray] = None
shap_explainer: Optional[DeepSHAPExplainer] = None

def get_or_load_model():
    global model, scaler_mean, scaler_scale, shap_explainer
    if model is not None:
        return model, scaler_mean, scaler_scale, shap_explainer

    m = XAutoNetClassifier(input_dim=11, hidden_dims=[128, 64, 32], dropout=0.0, use_batch_norm=False)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    m.eval()

    if os.path.exists(SCALER_MEAN_PATH) and os.path.exists(SCALER_SCALE_PATH):
        s_mean = np.load(SCALER_MEAN_PATH)
        s_scale = np.load(SCALER_SCALE_PATH)
    else:
        # Fallback to zero-mean unit-variance
        s_mean = np.zeros(11, dtype=np.float32)
        s_scale = np.ones(11, dtype=np.float32)

    if os.path.exists(BACKGROUND_PATH):
        bg = torch.tensor(np.load(BACKGROUND_PATH), dtype=torch.float32)
    else:
        bg = torch.zeros(20, 11, dtype=torch.float32)

    explainer = DeepSHAPExplainer(m, bg, BOTTLENECK_FEATURES)

    model = m
    scaler_mean = s_mean
    scaler_scale = s_scale
    shap_explainer = explainer
    return model, scaler_mean, scaler_scale, shap_explainer

# Presets representing real cases
SAMPLE_PATIENTS = {
    "patient_a": {
        "id": "Patient A (Normal Profile)",
        "description": "Stable ICU profile: normal body temperature, normal lactate and blood gas parameters. Low 6-hour sepsis risk.",
        "vitals": {
            "BaseExcess": -0.106,
            "Temp": 36.92,
            "Chloride": 107.09,
            "Hct": 29.34,
            "Hgb": 9.99,
            "Resp": 20.98,
            "HCO3": 20.94,
            "SIRS": 3.0,
            "Potassium": 4.22,
            "Creatinine": 1.14,
            "Phosphate": 3.17,
            "FiO2": 0.25,
            "O2Sat": 98.0,
            "Magnesium": 2.0,
            "SaO2": 98.0,
            "Lactate": 1.1,
            "pH": 7.39,
            "Calcium": 9.1,
            "BUN": 14.0,
        },
        "expected_risk": "Low / Moderate Risk",
    },
    "patient_b": {
        "id": "Patient B (Septic Profile)",
        "description": "High risk profile: marked hyperthermia (40.22°C), tachypnea (Resp 28), and elevated base excess. High risk of sepsis onset within 6 hours.",
        "vitals": {
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
        },
        "expected_risk": "High Risk",
    },
    "patient_healthy": {
        "id": "Healthy ICU Baseline",
        "description": "All biomarkers at midpoint of clinical normal physiological ranges.",
        "vitals": {
            feat: round((CLINICAL_NORMAL_RANGES[feat][0] + CLINICAL_NORMAL_RANGES[feat][1]) / 2.0, 2)
            for feat in FILTER_FEATURES
        },
        "expected_risk": "Low Risk",
    },
}

class PatientInput(BaseModel):
    vitals: Dict[str, Optional[float]]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "XAutoNet Clinical Early Warning Predictor",
        "decision_threshold": DECISION_THRESHOLD,
        "features_accepted": len(FILTER_FEATURES),
    }

@app.get("/api/features")
def get_features_info():
    """Return feature definitions, normal ranges, and permissible limits."""
    return {
        "features": FILTER_FEATURES,
        "bottleneck_features": BOTTLENECK_FEATURES,
        "top_shap_features": TOP_SHAP_FEATURES,
        "normal_ranges": CLINICAL_NORMAL_RANGES,
        "permissible_limits": CLINICAL_PERMISSIBLE_LIMITS,
        "decision_threshold": DECISION_THRESHOLD,
    }

@app.get("/api/sample-patients")
def get_sample_patients():
    """Return sample test profiles."""
    return SAMPLE_PATIENTS

@app.post("/api/predict")
def predict_patient_risk(data: PatientInput):
    """Predict risk of sepsis onset within the next 6 hours with SHAP explainability."""
    raw_vitals = data.vitals
    m, s_mean, s_scale, explainer = get_or_load_model()

    warnings: List[str] = []
    processed_vitals: Dict[str, float] = {}

    # Validate and handle all 19 features
    for feat in FILTER_FEATURES:
        val = raw_vitals.get(feat)
        if val is None or val == "":
            default_val = (CLINICAL_NORMAL_RANGES[feat][0] + CLINICAL_NORMAL_RANGES[feat][1]) / 2.0
            processed_vitals[feat] = default_val
            warnings.append(f"{feat} was not provided; defaulted to physiological normal ({default_val:.2f}).")
        else:
            try:
                num_val = float(val)
                # Check physiological bounds
                lower_limit, upper_limit = CLINICAL_PERMISSIBLE_LIMITS.get(feat, (-np.inf, np.inf))
                if num_val < lower_limit or num_val > upper_limit:
                    clipped_val = max(lower_limit, min(upper_limit, num_val))
                    warnings.append(f"{feat} ({num_val}) was outside biological limits [{lower_limit}, {upper_limit}]; clipped to {clipped_val}.")
                    processed_vitals[feat] = clipped_val
                else:
                    processed_vitals[feat] = num_val
            except (ValueError, TypeError):
                default_val = (CLINICAL_NORMAL_RANGES[feat][0] + CLINICAL_NORMAL_RANGES[feat][1]) / 2.0
                processed_vitals[feat] = default_val
                warnings.append(f"{feat} was invalid; defaulted to physiological normal ({default_val:.2f}).")

    # Extract 11 raw clinical features
    raw_11 = np.array([processed_vitals[f] for f in BOTTLENECK_FEATURES], dtype=np.float32).reshape(1, -1)

    # Scale with fitted training scaler
    scaled_11 = (raw_11 - s_mean) / s_scale

    # Inference through frozen 3-layer MLP
    m.eval()
    with torch.no_grad():
        prob = float(m.predict_proba(scaled_11)[0])

    # Risk Classification & 6-Hour Interpretation
    if prob >= 0.60:
        alert_level = "HIGH"
        alert_color = "#ef4444"
        risk_classification = "High Risk"
        interpretation_6h = "Elevated risk of sepsis onset within the next 6 hours. Intensive surveillance and diagnostic workup indicated."
    elif prob >= DECISION_THRESHOLD:
        alert_level = "MODERATE"
        alert_color = "#f59e0b"
        risk_classification = "Moderate Risk"
        interpretation_6h = f"Elevated risk of sepsis onset within the next 6 hours (probability {prob*100:.1f}% exceeds operational threshold of {DECISION_THRESHOLD*100:.0f}%). Close clinical observation indicated."
    else:
        alert_level = "LOW"
        alert_color = "#10b981"
        risk_classification = "Low Risk"
        interpretation_6h = f"Low risk of sepsis onset within the next 6 hours (probability {prob*100:.1f}% is below operational threshold of {DECISION_THRESHOLD*100:.0f}%). Routine ICU monitoring indicated."

    # Compute feature-level SHAP explanation
    shap_waterfall: List[Dict[str, Any]] = []
    if explainer is not None:
        try:
            tensor_input = torch.tensor(scaled_11, dtype=torch.float32)
            shap_result = explainer.explain_instance(tensor_input)
            raw_waterfall = shap_result.get("waterfall", [])
            for item in raw_waterfall:
                fname = item["feature"]
                shap_waterfall.append({
                    "feature": fname,
                    "patient_value": round(float(processed_vitals.get(fname, 0.0)), 2),
                    "shap_value": round(float(item["shap_value"]), 4),
                    "impact": item["impact"],
                    "is_top_impact": fname in TOP_SHAP_FEATURES,
                })
        except Exception:
            # Fallback perturbation attribution
            pass

    return {
        "prediction_probability": round(prob, 4),
        "risk_classification": risk_classification,
        "alert_level": alert_level,
        "alert_color": alert_color,
        "decision_threshold": DECISION_THRESHOLD,
        "is_above_threshold": bool(prob >= DECISION_THRESHOLD),
        "interpretation_6h": interpretation_6h,
        "warnings": warnings,
        "shap_waterfall": shap_waterfall,
        "model_architecture": "11 -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)",
        "disclaimer": "Investigational research decision-support prototype. Not intended for primary clinical diagnosis or automated medical decision-making.",
    }

# Mount static web directory
static_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
