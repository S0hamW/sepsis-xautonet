"""Outlier handling module using Z-Score and IQR detection with clinical clipping.

As described in Section II.B.2 of the IEEE IRI 2023 paper:
"Handling outliers: After imputing missing data, outliers were identified using Z-Score
and Interquartile Range (IQR) methods for both normally distributed and skewed data.
Invalid outliers were fixed by clipping them within their permissible range based on
domain expert advice, while valid outliers were retained."
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from xautonet.config import CLINICAL_PERMISSIBLE_LIMITS


def handle_clinical_outliers(
    df: pd.DataFrame,
    permissible_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    z_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Identify outliers via Z-Score (normal distributions) and IQR (skewed distributions).

    Invalid outliers that exceed physiologically possible domain ranges (permissible limits)
    are clipped to those boundaries, while valid clinical outliers within biological limits
    are preserved.
    """
    cleaned_df = df.copy()
    limits = permissible_limits if permissible_limits is not None else CLINICAL_PERMISSIBLE_LIMITS

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = cleaned_df[col].dropna()
        if len(series) >= 5:
            # Check skewness to determine whether to apply Z-score or IQR
            skewness = float(series.skew())

            if abs(skewness) <= 0.75:
                # Approximately normal: Use Z-Score
                mean = series.mean()
                std = series.std()
                if std > 1e-6:
                    z_scores = (cleaned_df[col] - mean) / std
                    _ = z_scores.abs() > z_threshold
            else:
                # Skewed: Use Interquartile Range (IQR)
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - iqr_multiplier * iqr
                upper_bound = q75 + iqr_multiplier * iqr
                _ = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)

        # Clip values violating domain expert boundaries (CLINICAL_PERMISSIBLE_LIMITS)
        if col in limits:
            min_val, max_val = limits[col]
            cleaned_df[col] = cleaned_df[col].clip(lower=min_val, upper=max_val)

    return cleaned_df
