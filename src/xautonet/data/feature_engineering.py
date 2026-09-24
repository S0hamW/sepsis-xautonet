"""Feature engineering module for calculating the Systemic Inflammatory Response Syndrome (SIRS) score.

As described in Section II.B.3 of the IEEE IRI 2023 paper:
"Generation of additional feature named SIRS was done by combining existing features
as it is considered as one of the helping tool in the identification of sepsis and
hence can provides useful information while model building."
"""

from typing import Union, Dict, Any
import numpy as np
import pandas as pd


def calculate_sirs_criteria_row(row: Union[pd.Series, Dict[str, Any]]) -> int:
    """Calculate SIRS criteria score (0-4) for a single record.

    Clinical SIRS consensus criteria:
    1. Body Temperature: Temp > 38.0 C or Temp < 36.0 C
    2. Heart Rate: HR > 90 bpm (if available in record)
    3. Respiratory Rate: Resp > 20 breaths/min or PaCO2 < 32 mmHg
    4. White Blood Cell count: WBC > 12,000 /uL or < 4,000 /uL (or > 12.0 / < 4.0 in 10^3/uL)
    """
    sirs_count = 0

    # 1. Temperature criterion
    temp = row.get("Temp", None)
    if temp is not None and not pd.isna(temp):
        if float(temp) > 38.0 or float(temp) < 36.0:
            sirs_count += 1

    # 2. Heart Rate criterion (if present in raw record)
    hr = row.get("HR", None)
    if hr is not None and not pd.isna(hr):
        if float(hr) > 90.0:
            sirs_count += 1

    # 3. Respiration Rate criterion
    resp = row.get("Resp", None)
    if resp is not None and not pd.isna(resp):
        if float(resp) > 20.0:
            sirs_count += 1

    # 4. White Blood Cell count (if present)
    wbc = row.get("WBC", None)
    if wbc is not None and not pd.isna(wbc):
        val = float(wbc)
        # Check standard units (10^3 / uL) and raw units (/uL)
        if (val > 12.0 and val <= 100.0) or (val < 4.0 and val > 0.0):
            sirs_count += 1
        elif val > 12000.0 or (val < 4000.0 and val > 100.0):
            sirs_count += 1

    return sirs_count


def compute_sirs_score(data: pd.DataFrame) -> pd.DataFrame:
    """Compute and append or update the 'SIRS' feature column in the DataFrame."""
    df = data.copy()
    if "SIRS" not in df.columns or df["SIRS"].isna().any():
        sirs_values = [calculate_sirs_criteria_row(row) for _, row in df.iterrows()]
        df["SIRS"] = sirs_values
    return df
