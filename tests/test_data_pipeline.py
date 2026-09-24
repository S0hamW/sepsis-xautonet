"""Unit tests for the data preprocessing, outlier clipping, SIRS engineering, and balancing."""

import pytest
import numpy as np
import pandas as pd
from xautonet.config import FILTER_FEATURES, CLINICAL_PERMISSIBLE_LIMITS
from xautonet.data.feature_engineering import calculate_sirs_criteria_row, compute_sirs_score
from xautonet.data.outliers import handle_clinical_outliers
from xautonet.data.loader import generate_synthetic_physionet_cohort
from xautonet.data.balancing import balance_clinical_dataset


def test_sirs_calculation_normal_and_abnormal():
    # Normal vitals: Temp 36.8, Resp 16 -> SIRS = 0
    normal_patient = {"Temp": 36.8, "Resp": 16.0}
    assert calculate_sirs_criteria_row(normal_patient) == 0

    # Fever (39.5) and Tachypnea (28) -> SIRS = 2 (meets clinical SIRS threshold >= 2)
    septic_patient = {"Temp": 39.5, "Resp": 28.0}
    assert calculate_sirs_criteria_row(septic_patient) == 2

    # Hypothermia (35.2) and Tachypnea (24) -> SIRS = 2
    hypo_patient = {"Temp": 35.2, "Resp": 24.0}
    assert calculate_sirs_criteria_row(hypo_patient) == 2


def test_outlier_clipping():
    df = pd.DataFrame({
        "Temp": [36.5, 37.0, 36.8, 37.2, 55.0, 15.0],  # 55.0 and 15.0 are physiologically impossible
        "pH": [7.4, 7.38, 7.41, 7.39, 9.5, 5.0],        # 9.5 and 5.0 violate biological bounds
    })

    cleaned = handle_clinical_outliers(df, CLINICAL_PERMISSIBLE_LIMITS)
    
    # Temperature permissible limits are 30.0 to 43.0
    assert cleaned["Temp"].max() <= 43.0
    assert cleaned["Temp"].min() >= 30.0

    # pH limits are 6.8 to 7.8
    assert cleaned["pH"].max() <= 7.8
    assert cleaned["pH"].min() >= 6.8


def test_synthetic_cohort_generation():
    df, y = generate_synthetic_physionet_cohort(n_normal=100, n_sepsis=80, missing_rate=0.01)
    
    assert len(df) == 180
    assert len(y) == 180
    assert list(df.columns) == FILTER_FEATURES
    assert "SIRS" in df.columns
    assert set(y.unique()) == {0, 1}


def test_class_balancing():
    X = np.random.randn(200, 19)
    # Severe imbalance: 190 normal, 10 sepsis (~5%)
    y = np.array([0] * 190 + [1] * 10)

    X_bal, y_bal = balance_clinical_dataset(X, y, target_normal=50, target_sepsis=40)
    
    assert np.sum(y_bal == 1) == 40
    assert np.sum(y_bal == 0) == 50
    assert len(X_bal) == 90
