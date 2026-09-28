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


def test_cv_pipeline_patient_group_separation_and_leakage_prevention():
    from xautonet.pipeline import XAutoNetPipeline
    from sklearn.model_selection import StratifiedGroupKFold

    # 10 patients with multiple hourly records each (total 60 rows)
    patient_ids = np.repeat([f"pat_{i:02d}" for i in range(10)], 6)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 19))
    # Some missing values to verify MICE fitting within fold
    X[0, 0] = np.nan
    X[10, 2] = np.nan
    y = np.array([0] * 36 + [1] * 24)

    pipeline = XAutoNetPipeline()
    # Run with 2 splits and 1 epoch for rapid execution
    df_results = pipeline.run_5fold_cross_validation(
        X=X,
        y=y,
        groups=patient_ids,
        n_splits=2,
        epochs_ae=1,
        epochs_clf=1,
        balance_training=True,
    )

    assert "F1 Score" in df_results.columns
    assert "Accuracy" in df_results.columns
    assert len(df_results) == 3  # Fold 1, Fold 2, Mean ± SD

    # Verify patient separation logic: no patient can be in both train and val
    sgkf = StratifiedGroupKFold(n_splits=2)
    for tr_idx, val_idx in sgkf.split(X, y, groups=patient_ids):
        tr_patients = set(patient_ids[tr_idx])
        val_patients = set(patient_ids[val_idx])
        assert tr_patients.isdisjoint(val_patients), "Patient leaked between train and validation folds!"


def test_load_physionet_cohort_6hour_boundary_prevents_leakage(tmp_path):
    from xautonet.data.loader import load_physionet_cohort

    # 1. Sepsis patient: First SepsisLabel == 1 is at hour 2
    # Hours 3 and 4 are post-boundary (after 6-hour warning point)
    sepsis_psv = tmp_path / "p_sepsis.psv"
    sepsis_psv.write_text(
        "ICULOS|HR|Temp|Resp|WBC|Lactate|BaseExcess|SepsisLabel\n"
        "1|75|36.5|16|7.0|NaN|0.0|0\n"
        "2|95|38.2|22|13.0|NaN|0.0|1\n"  # ← First positive row (prediction boundary)
        "3|140|41.5|35|25.0|9.9|5.0|1\n" # ← Post-boundary (must NOT leak)
        "4|150|42.0|40|30.0|12.0|8.0|1\n" # ← Post-boundary (must NOT leak)
    )

    # 2. Control patient: SepsisLabel == 0 throughout
    control_psv = tmp_path / "p_control.psv"
    control_psv.write_text(
        "ICULOS|HR|Temp|Resp|WBC|Lactate|BaseExcess|SepsisLabel\n"
        "1|70|36.8|15|6.0|1.0|0.0|0\n"
        "2|72|37.0|16|6.5|1.1|0.0|0\n"
    )

    X, y, pids = load_physionet_cohort(str(tmp_path), return_patient_ids=True)

    # Each patient must yield exactly one patient-level vector
    assert len(X) == 2
    assert len(y) == 2
    assert len(pids) == 2

    # Map results by PatientID
    patient_data = {pid: (X.iloc[i], y.iloc[i]) for i, pid in enumerate(pids)}

    # Verify Sepsis patient
    x_sep, y_sep = patient_data["p_sepsis"]
    assert y_sep == 1, "Sepsis patient must be labeled 1"

    # Measurements must come strictly from hour 2 boundary, NEVER from post-boundary hours 3 or 4
    assert x_sep["Temp"] == 38.2, f"Expected Temp=38.2 from boundary, got {x_sep['Temp']} (post-boundary leaked!)"
    assert x_sep["Resp"] == 22.0, f"Expected Resp=22.0 from boundary, got {x_sep['Resp']} (post-boundary leaked!)"
    assert x_sep["BaseExcess"] == 0.0, f"Expected BaseExcess=0.0, got {x_sep['BaseExcess']}"

    # Lactate was NaN at hours 1 and 2, but 9.9 at hour 3: it must remain NaN (no backward fill from future)
    assert np.isnan(x_sep["Lactate"]), (
        f"Lactate was filled with {x_sep['Lactate']} from post-boundary hour 3! Look-ahead leakage detected."
    )

    # SIRS must be computed strictly from boundary state (Temp 38.2 > 38, Resp 22 > 20, HR 95 > 90, WBC 13 > 12 -> SIRS = 4)
    assert x_sep["SIRS"] == 4.0, f"Expected SIRS=4 at boundary, got {x_sep['SIRS']}"

    # Verify Control patient
    x_ctrl, y_ctrl = patient_data["p_control"]
    assert y_ctrl == 0, "Control patient must be labeled 0"
    assert x_ctrl["Temp"] == 37.0
    assert x_ctrl["Lactate"] == 1.1


def test_load_physionet_cohort_parent_directory_with_subdirectories(tmp_path):
    from xautonet.data.loader import load_physionet_cohort

    set_a = tmp_path / "training_setA"
    set_b = tmp_path / "training_setB"
    set_a.mkdir()
    set_b.mkdir()

    (set_a / "p00001.psv").write_text(
        "ICULOS|HR|Temp|Resp|WBC|Lactate|BaseExcess|SepsisLabel\n"
        "1|80|37.0|18|8.0|1.2|0.0|0\n"
    )
    (set_b / "p10001.psv").write_text(
        "ICULOS|HR|Temp|Resp|WBC|Lactate|BaseExcess|SepsisLabel\n"
        "1|95|38.5|24|14.0|2.5|1.0|1\n"
    )

    # 1. Loading from parent directory (tmp_path) directly
    X, y, pids = load_physionet_cohort(str(tmp_path), return_patient_ids=True)
    assert len(X) == 2
    assert len(y) == 2
    assert set(pids) == {"p00001", "p10001"}
    assert X.shape[1] == 19
    assert set(y.values) == {0, 1}

    # 2. Loading with a list of subdirectories
    X_list, y_list, pids_list = load_physionet_cohort([str(set_a), str(set_b)], return_patient_ids=True)
    assert len(X_list) == 2
    assert set(pids_list) == {"p00001", "p10001"}


def test_load_paper_hourly_cohort_and_paper_5fold_evaluation(tmp_path):
    from xautonet.data.loader import load_paper_hourly_cohort
    from xautonet.pipeline import XAutoNetPipeline

    test_file = tmp_path / "p00001.psv"
    test_file.write_text(
        "ICULOS|HR|Temp|Resp|WBC|Lactate|BaseExcess|SepsisLabel\n"
        "1|75|36.8|16|7.0|1.2|0.0|0\n"
        "2|78|37.0|17|7.5|1.3|0.0|0\n"
        "3|95|38.5|24|13.0|2.5|1.0|1\n"
        "4|105|39.0|28|15.0|3.0|2.0|1\n"
    )

    X, y = load_paper_hourly_cohort(
        data_dir=str(tmp_path),
        target_normal=10,
        target_sepsis=8,
        max_patients=1,
        random_state=42,
    )

    assert X.shape[1] == 19
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}

    pipeline = XAutoNetPipeline()
    df_eval = pipeline.run_paper_5fold_evaluation(
        X, y,
        n_splits=2,
        epochs_ae=1,
        epochs_clf=1,
    )

    assert "F1 Score" in df_eval.columns
    assert "Precision" in df_eval.columns
    assert "Recall" in df_eval.columns
    assert "Accuracy" in df_eval.columns
    assert len(df_eval) == 3  # Fold 1, Fold 2, Mean ± SD




