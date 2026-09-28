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


def load_physionet_cohort(
    data_dir: str,
    max_patients: Optional[int] = None,
    random_state: int = 42,
    return_patient_ids: bool = False,
) -> Tuple:
    """Load PhysioNet .psv files into patient-level 6-hour-ahead feature vectors.

    For sepsis patients:
      - Identifies the FIRST hourly row where SepsisLabel == 1 as the 6-hour-ahead warning point.
      - Uses ONLY observations up to and including that boundary (LOCF forward fill).
      - Strictly excludes all post-boundary observations (no post-onset look-ahead).
      - Yields a single patient-level vector labeled 1.

    For non-sepsis patients:
      - Uses all available observations over their ICU stay (LOCF forward fill).
      - Yields a single patient-level vector labeled 0.

    SIRS is computed from measurements up to the boundary (HR, Temp, Resp, WBC).

    Args:
        data_dir: Path to directory containing .psv (or .csv) patient files.
        max_patients: If set, sample this many patient files randomly.
        random_state: Seed for reproducible patient sampling.
        return_patient_ids: If True, also return array of patient IDs.

    Returns:
        X: DataFrame of shape (n_patients, 19) with FILTER_FEATURES columns.
           Residual NaN values remain for downstream MICE imputation.
        y: Series of shape (n_patients,) with integer SepsisLabel (0 or 1).
        (Optional) patient_ids: ndarray of shape (n_patients,) with patient identifier.
    """
    if isinstance(data_dir, (list, tuple)):
        candidate_files = []
        for d in data_dir:
            candidate_files.extend(glob.glob(os.path.join(d, "*.psv")))
            candidate_files.extend(glob.glob(os.path.join(d, "*.csv")))
            candidate_files.extend(glob.glob(os.path.join(d, "**", "*.psv"), recursive=True))
            candidate_files.extend(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    else:
        candidate_files = (
            glob.glob(os.path.join(data_dir, "*.psv"))
            + glob.glob(os.path.join(data_dir, "*.csv"))
            + glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
            + glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        )
    files = sorted(list(dict.fromkeys(candidate_files)))
    if not files:
        raise FileNotFoundError(f"No .psv or .csv patient files found in: {data_dir}")

    if max_patients is not None and max_patients < len(files):
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(files), size=max_patients, replace=False)
        files = [files[i] for i in sorted(chosen)]

    # 18 raw continuous features (SIRS is engineered, not a raw column)
    RAW_FEATURES = [f for f in FILTER_FEATURES if f != "SIRS"]

    # Auxiliary columns needed for SIRS but not in FILTER_FEATURES
    SIRS_AUX = ["HR", "WBC"]

    all_rows: list = []

    for fpath in files:
        pid = os.path.splitext(os.path.basename(fpath))[0]
        sep = "|" if fpath.endswith(".psv") else ","
        try:
            patient_df = pd.read_csv(fpath, sep=sep)
        except Exception:
            continue

        if "SepsisLabel" not in patient_df.columns:
            continue

        # Guarantee temporal order within patient
        if "ICULOS" in patient_df.columns:
            patient_df = patient_df.sort_values("ICULOS").reset_index(drop=True)
        else:
            patient_df = patient_df.reset_index(drop=True)

        # ── 6-Hour-Ahead Temporal Boundary ──────────────────────────────────
        # Identify the FIRST hourly row where SepsisLabel == 1.
        # This row represents the 6-hour warning point before clinical onset.
        sepsis_mask = (patient_df["SepsisLabel"] == 1)
        is_sepsis = bool(sepsis_mask.any())

        if is_sepsis:
            # Cut off history strictly at the first SepsisLabel == 1 row (prediction boundary).
            # Do NOT allow any observations after this boundary into features.
            first_pos_idx = int(patient_df.index[sepsis_mask][0])
            patient_history = patient_df.iloc[:first_pos_idx + 1].copy()
            label = 1
        else:
            # Non-sepsis patient: use all available observations.
            patient_history = patient_df.copy()
            label = 0

        # ── LOCF fill strictly within allowable history ──────────────────────
        fill_cols = [c for c in RAW_FEATURES + SIRS_AUX if c in patient_history.columns]
        filled = patient_history[fill_cols].ffill().bfill()

        # ── Compute SIRS on LOCF-filled data ─────────────────────────────────
        sirs_input = filled.copy()
        sirs_input = compute_sirs_score(sirs_input)      # adds / overwrites "SIRS"

        # ── Select the 19 FILTER_FEATURES; drop aux cols not in FILTER_FEATURES
        for col in RAW_FEATURES:
            if col not in sirs_input.columns:
                sirs_input[col] = np.nan               # placeholder for MICE later

        # ── Patient-level feature vector: state at prediction boundary ───────
        patient_vector = sirs_input[FILTER_FEATURES].iloc[-1:].copy()
        patient_vector["SepsisLabel"] = label
        patient_vector["PatientID"] = pid

        # Only include if at least one feature was measured/computed
        if patient_vector[FILTER_FEATURES].notna().any(axis=1).iloc[0]:
            all_rows.append(patient_vector)

    if not all_rows:
        raise ValueError(
            f"No valid patient records could be loaded from {data_dir}. "
            "Ensure the directory contains correctly formatted .psv files."
        )

    combined = pd.concat(all_rows, ignore_index=True)
    X = combined[FILTER_FEATURES]
    y = combined["SepsisLabel"].astype(int)

    if return_patient_ids:
        return X, y, combined["PatientID"].values
    return X, y


def load_paper_hourly_cohort(
    data_dir: str = "data/physionet-2019/training_setA",
    target_normal: int = 32000,
    target_sepsis: int = 25000,
    max_patients: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load PhysioNet 2019 Set A hourly time-series instances for paper reproduction.

    Methodology reproduced from IEEE IRI 2023 paper (Section II.A & II.B):
    1. Set A only (20,336 patients).
    2. Hourly records as individual sample rows (not 1 row per patient).
    3. LOCF per patient stay + SIRS feature engineering.
    4. 19 features (18 continuous from MI + 1 categorical SIRS).
    5. MICE imputation + domain outlier clipping.
    6. Global class balancing to 32,000 normal + 25,000 sepsis records (SMOTE + undersampling).

    Args:
        data_dir: Path to directory containing Set A .psv files.
        target_normal: Desired normal records after undersampling (default 32000).
        target_sepsis: Desired sepsis records after SMOTE oversampling (default 25000).
        max_patients: If set, sample this many patient files for quick smoke testing.
        random_state: Seed for reproducible sampling.

    Returns:
        X: DataFrame of shape (target_normal + target_sepsis, 19).
        y: Series of shape (target_normal + target_sepsis,).
    """
    from xautonet.data.imputation import MICEImputerWrapper
    from xautonet.data.balancing import balance_clinical_dataset

    # Find Set A files
    candidate_files = (
        glob.glob(os.path.join(data_dir, "*.psv"))
        + glob.glob(os.path.join(data_dir, "*.csv"))
        + glob.glob(os.path.join(data_dir, "**", "*.psv"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    )
    files = sorted(list(dict.fromkeys(candidate_files)))
    if not files:
        raise FileNotFoundError(f"No .psv or .csv patient files found in: {data_dir}")

    if max_patients is not None and max_patients < len(files):
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(files), size=max_patients, replace=False)
        files = [files[i] for i in sorted(chosen)]

    RAW_FEATURES = [f for f in FILTER_FEATURES if f != "SIRS"]
    SIRS_AUX = ["HR", "WBC"]

    normal_records = []
    sepsis_records = []

    for fpath in files:
        sep = "|" if fpath.endswith(".psv") else ","
        try:
            patient_df = pd.read_csv(fpath, sep=sep)
        except Exception:
            continue

        if "SepsisLabel" not in patient_df.columns:
            continue

        if "ICULOS" in patient_df.columns:
            patient_df = patient_df.sort_values("ICULOS").reset_index(drop=True)
        else:
            patient_df = patient_df.reset_index(drop=True)

        fill_cols = [c for c in RAW_FEATURES + SIRS_AUX if c in patient_df.columns]
        filled = patient_df[fill_cols].ffill().bfill()
        sirs_input = compute_sirs_score(filled)

        for col in RAW_FEATURES:
            if col not in sirs_input.columns:
                sirs_input[col] = np.nan

        sirs_input["SepsisLabel"] = patient_df["SepsisLabel"].astype(int)
        
        # Partition hourly records by SepsisLabel
        pos_mask = (sirs_input["SepsisLabel"] == 1)
        neg_mask = (sirs_input["SepsisLabel"] == 0)

        pos_subset = sirs_input.loc[pos_mask, FILTER_FEATURES + ["SepsisLabel"]]
        neg_subset = sirs_input.loc[neg_mask, FILTER_FEATURES + ["SepsisLabel"]]

        if len(pos_subset) > 0:
            sepsis_records.append(pos_subset)
        if len(neg_subset) > 0:
            normal_records.append(neg_subset)

    if not normal_records and not sepsis_records:
        raise ValueError(f"No hourly records extracted from {data_dir}.")

    df_norm_all = pd.concat(normal_records, ignore_index=True) if normal_records else pd.DataFrame(columns=FILTER_FEATURES + ["SepsisLabel"])
    df_sep_all = pd.concat(sepsis_records, ignore_index=True) if sepsis_records else pd.DataFrame(columns=FILTER_FEATURES + ["SepsisLabel"])

    # If max_patients is small (e.g. quick testing), scale targets to available counts
    if max_patients is not None and max_patients < 5000:
        eff_target_normal = min(target_normal, max(10, int(len(df_norm_all) * 0.8)))
        eff_target_sepsis = min(target_sepsis, max(5, int(eff_target_normal * 0.78)))
    else:
        eff_target_normal = target_normal
        eff_target_sepsis = target_sepsis

    # Subsample majority normal records to eff_target_normal
    rng = np.random.default_rng(random_state)
    if len(df_norm_all) > eff_target_normal:
        sample_idx = rng.choice(len(df_norm_all), size=eff_target_normal, replace=False)
        df_norm_sampled = df_norm_all.iloc[sample_idx].reset_index(drop=True)
    else:
        df_norm_sampled = df_norm_all.copy()

    # Pre-imputation combined cohort
    combined = pd.concat([df_norm_sampled, df_sep_all], ignore_index=True)

    # Impute missing values with MICE wrapper (Section II.B.1)
    mice = MICEImputerWrapper(
        max_iter=5 if max_patients is not None and max_patients < 5000 else 10,
        random_state=random_state,
        features=FILTER_FEATURES,
    )
    df_imputed = mice.fit_transform(combined[FILTER_FEATURES])

    # Outlier clipping within clinical bounds (Section II.B.2)
    df_clean = handle_clinical_outliers(df_imputed)

    # Balance with SMOTE oversampling to eff_target_sepsis and cluster undersampling (Section II.B.4)
    X_mat = df_clean.values.astype(np.float64)
    y_vec = combined["SepsisLabel"].values.astype(int)

    X_bal, y_bal = balance_clinical_dataset(
        X_mat,
        y_vec,
        target_normal=eff_target_normal,
        target_sepsis=eff_target_sepsis,
        random_state=random_state,
    )

    X_final = pd.DataFrame(X_bal, columns=FILTER_FEATURES)
    y_final = pd.Series(y_bal, name="SepsisLabel", dtype=int)

    return X_final, y_final




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
