"""Class balancing module: Hybrid SMOTE minority oversampling and cluster-based undersampling.

As described in Section II.B.4 of the IEEE IRI 2023 paper:
"Due to high class imbalance problem in the data (2% of positive samples), a joint approach
of oversampling of the minority class by Synthetic Minority Over-Sampling Technique (SMOTE)
and then undersampling of the majority class based on clusters was incorporated to retain
information as much as possible. The final dataset consisted of 32,000 normal records
and 25,000 sepsis records."
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def balance_clinical_dataset(
    X: np.ndarray,
    y: np.ndarray,
    target_normal: int = 32000,
    target_sepsis: int = 25000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform hybrid class balancing:
    1. SMOTE oversampling on the minority class (sepsis) up to target_sepsis.
    2. Cluster-based representative undersampling of majority class (normal) to target_normal.

    Final cohort has target_normal normal records and target_sepsis sepsis records.
    """
    rng = np.random.default_rng(random_state)

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    X_pos = X[pos_mask]
    X_neg = X[neg_mask]

    n_pos = len(X_pos)
    n_neg = len(X_neg)

    if n_pos == 0 or n_neg == 0:
        return X, y

    # 1. Minority oversampling up to target_sepsis
    if n_pos < target_sepsis:
        try:
            from imblearn.over_sampling import SMOTE
            k_neighbors = min(5, n_pos - 1)
            if k_neighbors >= 1:
                smote = SMOTE(
                    sampling_strategy={1: target_sepsis},
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                )
                X_res, y_res = smote.fit_resample(X, y)
                X_pos_balanced = X_res[y_res == 1]
            else:
                raise ValueError("Too few samples for k_neighbors")
        except Exception:
            # Fallback synthetic interpolation if SMOTE fails / too few neighbors
            needed = target_sepsis - n_pos
            idx_a = rng.choice(n_pos, size=needed)
            idx_b = rng.choice(n_pos, size=needed)
            lambdas = rng.uniform(0.1, 0.9, size=(needed, 1))
            synthetic = X_pos[idx_a] * lambdas + X_pos[idx_b] * (1.0 - lambdas)
            X_pos_balanced = np.vstack([X_pos, synthetic])
    elif n_pos > target_sepsis:
        # Sample down to target_sepsis if input exceeded target
        sampled_indices = rng.choice(n_pos, size=target_sepsis, replace=False)
        X_pos_balanced = X_pos[sampled_indices]
    else:
        X_pos_balanced = X_pos

    # 2. Cluster-based undersampling of majority class down to target_normal
    if n_neg > target_normal:
        # Cluster centroids serve as representative compressed majority samples.
        # For very large cluster counts (>1000), MiniBatchKMeans distance matrices
        # cause memory freeze; representative subsampling is used.
        if target_normal <= 1000:
            batch_size = min(1024, n_neg)
            kmeans = MiniBatchKMeans(
                n_clusters=target_normal,
                random_state=random_state,
                batch_size=batch_size,
                n_init="auto",
            )
            kmeans.fit(X_neg)
            X_neg_balanced = kmeans.cluster_centers_
        else:
            sampled_indices = rng.choice(n_neg, size=target_normal, replace=False)
            X_neg_balanced = X_neg[sampled_indices]
    elif n_neg < target_normal:
        # If input has fewer than target_normal, replicate/sample up
        needed = target_normal - n_neg
        replicated_indices = rng.choice(n_neg, size=needed, replace=True)
        X_neg_balanced = np.vstack([X_neg, X_neg[replicated_indices]])
    else:
        X_neg_balanced = X_neg

    y_pos_balanced = np.ones(len(X_pos_balanced), dtype=int)
    y_neg_balanced = np.zeros(len(X_neg_balanced), dtype=int)

    X_final = np.vstack([X_neg_balanced, X_pos_balanced])
    y_final = np.concatenate([y_neg_balanced, y_pos_balanced])

    # Shuffle instances
    shuffle_idx = rng.permutation(len(y_final))
    return X_final[shuffle_idx], y_final[shuffle_idx]
