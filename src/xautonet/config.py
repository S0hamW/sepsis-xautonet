"""Configuration, clinical constants, feature definitions, and default hyperparameters.

Based on IEEE IRI 2023 paper:
"An Explainable AI based Clinical Assistance Model for Identifying Patients with the Onset of Sepsis"
(Chakraborty et al., Jio Institute & Reliance Jio AICoE)
"""

from typing import Dict, List, Tuple

# The 19 features selected via Filter Methods (18 continuous via MI + 1 categorical via Chi2)
FILTER_FEATURES: List[str] = [
    "BaseExcess",
    "Temp",
    "Chloride",
    "Hct",
    "Hgb",
    "Resp",
    "HCO3",
    "SIRS",
    "Potassium",
    "Creatinine",
    "Phosphate",
    "FiO2",
    "O2Sat",
    "Magnesium",
    "SaO2",
    "Lactate",
    "pH",
    "Calcium",
    "BUN",
]

# The top 11 bottleneck features identified by GradCAM Discounting HeatMap (DHM)
BOTTLENECK_FEATURES: List[str] = [
    "Creatinine",
    "Hct",
    "Phosphate",
    "HCO3",
    "Resp",
    "Temp",
    "Hgb",
    "SIRS",
    "Potassium",
    "BaseExcess",
    "Chloride",
]

# Top 5 most impactful features identified via DeepSHAP (paper Section III.C)
TOP_SHAP_FEATURES: List[str] = [
    "Creatinine",
    "Hct",
    "Phosphate",
    "HCO3",
    "Resp",
]

# GradCAM Discounting HeatMap beta factor from equation (9) in research paper
DHM_BETA: float = 0.9

# Standard clinical normal reference ranges (engineering addition based on clinical literature)
CLINICAL_NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "BaseExcess": (-2.0, 2.0),
    "Temp": (36.1, 37.2),
    "Chloride": (96.0, 106.0),
    "Hct": (36.0, 50.0),
    "Hgb": (12.0, 17.5),
    "Resp": (12.0, 20.0),
    "HCO3": (22.0, 28.0),
    "SIRS": (0.0, 1.0),
    "Potassium": (3.5, 5.0),
    "Creatinine": (0.6, 1.2),
    "Phosphate": (2.5, 4.5),
    "FiO2": (0.21, 0.50),
    "O2Sat": (95.0, 100.0),
    "Magnesium": (1.7, 2.2),
    "SaO2": (95.0, 100.0),
    "Lactate": (0.5, 2.0),
    "pH": (7.35, 7.45),
    "Calcium": (8.5, 10.5),
    "BUN": (7.0, 20.0),
}

# Permissible physiological clipping limits for outlier handling
# (domain expert boundaries referenced in Section II.B.2 of paper)
CLINICAL_PERMISSIBLE_LIMITS: Dict[str, Tuple[float, float]] = {
    "BaseExcess": (-30.0, 30.0),
    "Temp": (30.0, 43.0),
    "Chloride": (60.0, 140.0),
    "Hct": (10.0, 65.0),
    "Hgb": (3.0, 25.0),
    "Resp": (4.0, 60.0),
    "HCO3": (5.0, 50.0),
    "SIRS": (0.0, 4.0),
    "Potassium": (1.5, 9.0),
    "Creatinine": (0.1, 15.0),
    "Phosphate": (0.5, 12.0),
    "FiO2": (0.21, 1.0),
    "O2Sat": (50.0, 100.0),
    "Magnesium": (0.5, 6.0),
    "SaO2": (50.0, 100.0),
    "Lactate": (0.2, 25.0),
    "pH": (6.8, 7.8),
    "Calcium": (4.0, 18.0),
    "BUN": (1.0, 180.0),
}

# Autoencoder architecture configurations (4 Conv1D layers decreasing in encoder, symmetric in decoder)
AUTOENCODER_CONFIG = {
    "input_dim": 19,
    "latent_dim": 11,
    "encoder_filters": [64, 32, 16, 8],
    "decoder_filters": [8, 16, 32, 64],
    "kernel_size": 3,
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 50,
}

# Classifier architecture configurations (deep neural network operating on 11 latent features)
CLASSIFIER_CONFIG = {
    "input_dim": 11,
    "hidden_layers": [64, 32, 16],
    "output_dim": 1,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 40,
}

# Dataset balancing targets explicitly stated in Section II.B.4 of paper
COHORT_TARGETS = {
    "normal_records": 32000,
    "sepsis_records": 25000,
    "total_records": 57000,
}
