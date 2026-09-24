"""PhysioNet 2019 dataset loader and synthetic clinical cohort generator.

Provides loaders for the raw PhysioNet Computing in Cardiology Challenge 2019 data
(20,336 ICU patient files, hourly measurements) and a statistically aligned synthetic
cohort generator reproducing the distributions described in the IEEE IRI 2023 paper.
"""

import os
import glob
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from xautonet.config import (
    FILTER_FEATURES,
    CLINICAL_PERMISSIBLE_LIMITS,
)
from xautonet.data.feature_engineering import compute_sirs_score
from xautonet.data.outliers import handle_clinical_outliers


def load_physionet_directory(data_dir: str) -> pd.DataFrame:
    """Load raw PhysioNet challenge .psv or .csv files from directory.

    Each file represents a single patient's hourly ICU time-series.
    Label is SepsisLabel (1 = onset of sepsis 6 hours in advance, 0 = non-sepsis).
    """
    files = glob.glob(os.path.join(data_dir, "*.psv")) + glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No .psv or .csv patient files found in directory: {data_dir}")

    records = []
    for fpath in files:
        pid = os.path.splitext(os.path.basename(fpath))[0]
        sep = "|" if fpath.endswith(".psv") else ","
        patient_df = pd.read_csv(fpath, sep=sep)
        patient_df["PatientID"] = pid
        records.append(patient_df)

    full_df = pd.concat(records, ignore_index=True)
    return full_df


def generate_synthetic_physionet_cohort(
    n_normal: int = 32000,
    n_sepsis: int = 25000,
    missing_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a clinically realistic cohort mirroring the PhysioNet challenge statistics.

    Normal records reflect physiological values mostly within standard reference ranges.
    Sepsis onset records reflect signature multi-system inflammatory and metabolic derangements
    (hyperthermia/hypothermia, tachypnea, renal impairment, acid-base shifts, hyperlactatemia).
    """
    rng = np.random.default_rng(random_state)
    labels = np.array([0] * n_normal + [1] * n_sepsis)

    data = {}

    # 1. BaseExcess: normal (-2 to 2), sepsis (-10 to -3 or metabolic stress)
    data["BaseExcess"] = np.concatenate([
        rng.normal(loc=0.0, scale=1.4, size=n_normal),
        rng.normal(loc=-4.5, scale=3.5, size=n_sepsis),
    ])

    # 2. Temp (Celsius): normal (36.1 - 37.2), sepsis (fever >38.5 or hypothermia <36)
    sepsis_temp = np.where(
        rng.random(n_sepsis) > 0.3,
        rng.normal(loc=39.2, scale=1.1, size=n_sepsis),  # Fever
        rng.normal(loc=35.5, scale=0.8, size=n_sepsis),  # Hypothermia
    )
    data["Temp"] = np.concatenate([
        rng.normal(loc=36.8, scale=0.35, size=n_normal),
        sepsis_temp,
    ])

    # 3. Chloride: normal (96 - 106), sepsis shifted
    data["Chloride"] = np.concatenate([
        rng.normal(loc=101.0, scale=3.0, size=n_normal),
        rng.normal(loc=106.0, scale=6.0, size=n_sepsis),
    ])

    # 4. Hct (Hematocrit %): normal (36 - 50), sepsis anemia/hemodilution (25 - 33)
    data["Hct"] = np.concatenate([
        rng.normal(loc=40.0, scale=3.8, size=n_normal),
        rng.normal(loc=28.5, scale=4.5, size=n_sepsis),
    ])

    # 5. Hgb (Hemoglobin g/dL): normal (12 - 16), sepsis lower (8 - 11)
    data["Hgb"] = np.concatenate([
        rng.normal(loc=13.5, scale=1.4, size=n_normal),
        rng.normal(loc=9.5, scale=1.8, size=n_sepsis),
    ])

    # 6. Resp (Respiration Rate bpm): normal (12 - 20), sepsis tachypnea (22 - 36)
    data["Resp"] = np.concatenate([
        rng.normal(loc=16.5, scale=2.4, size=n_normal),
        rng.normal(loc=26.5, scale=5.0, size=n_sepsis),
    ])

    # 7. HCO3 (Bicarbonate mEq/L): normal (22 - 28), sepsis metabolic acidosis (14 - 22)
    data["HCO3"] = np.concatenate([
        rng.normal(loc=25.0, scale=2.0, size=n_normal),
        rng.normal(loc=18.5, scale=4.0, size=n_sepsis),
    ])

    # 8. Potassium (mEq/L): normal (3.5 - 5.0)
    data["Potassium"] = np.concatenate([
        rng.normal(loc=4.1, scale=0.38, size=n_normal),
        rng.normal(loc=4.6, scale=0.8, size=n_sepsis),
    ])

    # 9. Creatinine (mg/dL): normal (0.6 - 1.1), sepsis acute kidney injury (>1.4 - 4.0)
    data["Creatinine"] = np.concatenate([
        rng.normal(loc=0.85, scale=0.18, size=n_normal),
        rng.exponential(scale=1.4, size=n_sepsis) + 1.2,
    ])

    # 10. Phosphate (mg/dL): normal (2.5 - 4.5), sepsis shifted
    data["Phosphate"] = np.concatenate([
        rng.normal(loc=3.4, scale=0.48, size=n_normal),
        rng.normal(loc=4.5, scale=1.2, size=n_sepsis),
    ])

    # 11. FiO2 (fraction): normal (0.21 - 0.35), sepsis supplemental oxygen (0.4 - 0.8)
    data["FiO2"] = np.concatenate([
        rng.normal(loc=0.28, scale=0.08, size=n_normal),
        rng.normal(loc=0.55, scale=0.18, size=n_sepsis),
    ])

    # 12. O2Sat (%): normal (96 - 100), sepsis hypoxia
    data["O2Sat"] = np.concatenate([
        rng.normal(loc=98.0, scale=1.5, size=n_normal),
        rng.normal(loc=92.5, scale=4.0, size=n_sepsis),
    ])

    # 13. Magnesium (mg/dL): normal (1.7 - 2.2)
    data["Magnesium"] = np.concatenate([
        rng.normal(loc=2.0, scale=0.22, size=n_normal),
        rng.normal(loc=1.8, scale=0.4, size=n_sepsis),
    ])

    # 14. SaO2 (%): normal (95 - 100)
    data["SaO2"] = np.concatenate([
        rng.normal(loc=97.5, scale=1.8, size=n_normal),
        rng.normal(loc=91.0, scale=5.0, size=n_sepsis),
    ])

    # 15. Lactate (mmol/L): normal (0.5 - 1.8), sepsis hyperlactatemia (>2.2 - 6.0)
    data["Lactate"] = np.concatenate([
        rng.normal(loc=1.2, scale=0.38, size=n_normal),
        rng.exponential(scale=1.8, size=n_sepsis) + 2.2,
    ])

    # 16. pH: normal (7.35 - 7.45), sepsis acidosis (<7.32)
    data["pH"] = np.concatenate([
        rng.normal(loc=7.40, scale=0.035, size=n_normal),
        rng.normal(loc=7.28, scale=0.08, size=n_sepsis),
    ])

    # 17. Calcium (mg/dL): normal (8.5 - 10.5)
    data["Calcium"] = np.concatenate([
        rng.normal(loc=9.2, scale=0.55, size=n_normal),
        rng.normal(loc=8.0, scale=1.1, size=n_sepsis),
    ])

    # 18. BUN (mg/dL): normal (10 - 20), sepsis uremia (>30 - 65)
    data["BUN"] = np.concatenate([
        rng.normal(loc=15.0, scale=3.8, size=n_normal),
        rng.normal(loc=38.0, scale=14.0, size=n_sepsis),
    ])

    # Build DataFrame
    df = pd.DataFrame(data)

    # Compute engineered SIRS feature
    df = compute_sirs_score(df)

    # Reorder columns to exactly match FILTER_FEATURES
    df = df[FILTER_FEATURES]

    # Clip values within biological permissible limits
    df = handle_clinical_outliers(df, CLINICAL_PERMISSIBLE_LIMITS)

    # Introduce sparse missingness if requested
    if missing_rate > 0:
        mask = rng.random(df.shape) < missing_rate
        df = df.mask(mask)

    # Shuffle instances
    shuffle_idx = rng.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    target_series = pd.Series(labels[shuffle_idx], name="SepsisLabel")

    return df, target_series
