"""Feature selection module implementing Mutual Information (MI) and Chi-square (chi2) filter methods.

As described in Section II.B.5 of the IEEE IRI 2023 paper:
"Filter Method: To select the best 19 features from the total feature space, methods
like Mutual Information (MI) score and Chi-square (χ2) were used. MI helped us to get
the best 18 features from the continuous feature space with the highest MI scores...
On the other hand, χ2 test was used to select the best feature from the categorical
feature space (SIRS and Gender). Notable features for label determination have high
χ2 statistics and p-values below 0.05."
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from xautonet.config import FILTER_FEATURES


class TwoTierFeatureSelector:
    """Filter method selector:
    1. Mutual Information (MI) for continuous features (selects top 18).
    2. Chi-Square (chi2) test for categorical features (assesses SIRS & Gender, selects p < 0.05).
    """

    def __init__(
        self,
        n_continuous: int = 18,
        categorical_candidates: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.n_continuous = n_continuous
        self.categorical_candidates = categorical_candidates or ["SIRS", "Gender"]
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.mi_scores_: dict = {}
        self.chi2_scores_: dict = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TwoTierFeatureSelector":
        """Compute MI and Chi-Square scores to select top features."""
        cat_cols = [col for col in self.categorical_candidates if col in X.columns]
        cont_cols = [col for col in X.columns if col not in cat_cols]

        # 1. Mutual Information on continuous features
        if cont_cols:
            X_cont = X[cont_cols].fillna(X[cont_cols].median())
            mi = mutual_info_classif(X_cont.values, y, random_state=self.random_state)
            self.mi_scores_ = dict(zip(cont_cols, mi))
            sorted_cont = sorted(self.mi_scores_.items(), key=lambda item: item[1], reverse=True)
            selected_cont = [k for k, _ in sorted_cont[: self.n_continuous]]
        else:
            selected_cont = []

        # 2. Chi-square test on categorical candidates
        selected_cat = []
        if cat_cols:
            X_cat = X[cat_cols].fillna(0)
            # Shift positive if necessary for chi2 (non-negative required)
            X_cat_pos = X_cat - X_cat.min()
            chi2_stats, p_values = chi2(X_cat_pos.values, y)
            for col, stat, p in zip(cat_cols, chi2_stats, p_values):
                self.chi2_scores_[col] = {"statistic": float(stat), "p_value": float(p)}
                if p < 0.05:
                    selected_cat.append(col)

            if len(selected_cat) > 1:
                selected_cat = [max(selected_cat, key=lambda c: self.chi2_scores_[c]["statistic"])]

        self.selected_features_ = selected_cont + selected_cat

        # Preserve canonical order if the set matches FILTER_FEATURES
        if set(self.selected_features_) == set(FILTER_FEATURES):
            self.selected_features_ = [f for f in FILTER_FEATURES if f in self.selected_features_]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame down to the selected features."""
        features = self.selected_features_ if self.selected_features_ else FILTER_FEATURES
        available = [f for f in features if f in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
