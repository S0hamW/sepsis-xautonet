"""Missing value imputation module implementing MICE (Multivariate Imputation by Chained Equations).

As described in Section II.B.1 of the IEEE IRI 2023 paper:
"Fixing missing values: Due to high percentage of missing values, Multiple Imputation Using
Chained Equations (MICE) algorithm was used to impute them. It is a technique that imputes
missing values in a dataset by leveraging information from other columns to estimate the
best predictions for each missing value."
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge


class MICEImputerWrapper:
    """Wrapper around IterativeImputer to perform MICE on clinical tabular datasets."""

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 42,
        features: Optional[List[str]] = None,
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.features = features
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=True,
            initial_strategy="median",
            min_value=-50.0,
            max_value=300.0,
        )
        self.fallback_imputer = SimpleImputer(strategy="median")
        self.is_fitted = False
        self.features_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MICEImputerWrapper":
        """Fit MICE imputer on reference DataFrame."""
        if self.features:
            target_cols = [col for col in self.features if col in df.columns]
        else:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.features_ = target_cols
        data_matrix = df[target_cols].values

        try:
            self.imputer.fit(data_matrix)
        except Exception:
            self.fallback_imputer.fit(data_matrix)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("MICEImputerWrapper must be fitted before transforming data.")

        result_df = df.copy()
        target_cols = self.features_

        for col in target_cols:
            if col not in result_df.columns:
                result_df[col] = np.nan

        data_matrix = result_df[target_cols].values
        try:
            imputed_matrix = self.imputer.transform(data_matrix)
            if imputed_matrix.shape[1] == len(target_cols):
                result_df[target_cols] = imputed_matrix
            else:
                # IterativeImputer may skip all-NaN columns; assign column-by-column safely
                for j, col in enumerate(target_cols):
                    if j < imputed_matrix.shape[1]:
                        result_df[col] = imputed_matrix[:, j]
                    else:
                        result_df[col] = result_df[col].fillna(0.0)
        except Exception:
            try:
                fallback_out = self.fallback_imputer.transform(data_matrix)
                # SimpleImputer may drop all-NaN columns; map back by position of observed cols
                observed_cols = [c for c in target_cols if not np.all(np.isnan(data_matrix[:, target_cols.index(c)]))]
                col_map = {c: fallback_out[:, i] for i, c in enumerate(observed_cols) if i < fallback_out.shape[1]}
                for col in target_cols:
                    if col in col_map:
                        result_df[col] = col_map[col]
                    else:
                        result_df[col] = result_df[col].fillna(0.0)
            except Exception:
                # Last-resort: fill remaining NaN with 0
                for col in target_cols:
                    result_df[col] = result_df[col].fillna(0.0)

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)
