"""XAutoNet deep neural network classifier operating on 11-dimensional latent features.

As described in Section II.C of the IEEE IRI 2023 paper:
"A deep neural network model named XAutoNet was trained on the latent dimension
outputted from the encoder block of the autoencoder (once it was trained) to classify
them between normal or sepsis 6 hour before, shown in Fig. 2. The objective is to
minimize the binary cross entropy between its output (P'(theta')) and ground truth (Q)
with respect to all parameters (theta')."
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class XAutoNetClassifier(nn.Module):
    """Deep neural network classifier head operating on the 11 latent features from Conv1D autoencoder.

    Predicts onset of sepsis 6 hours in advance.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32, 16]
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Final classification neuron with Sigmoid activation
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting sepsis onset probability [0, 1].

        Args:
            latent: Tensor of shape (Batch, 11)
        Returns:
            prob: Tensor of shape (Batch, 1) representing P'(theta')
        """
        return self.network(latent)

    def predict_proba(self, latent: torch.Tensor) -> np.ndarray:
        """Return numpy array of probabilities."""
        self.eval()
        with torch.no_grad():
            if isinstance(latent, np.ndarray):
                latent = torch.tensor(latent, dtype=torch.float32)
            probs = self.forward(latent).cpu().numpy().squeeze()
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            return probs

    def predict(self, latent: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions (0 = normal, 1 = sepsis onset 6h ahead)."""
        probs = self.predict_proba(latent)
        return (probs >= threshold).astype(int)


def evaluate_classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate clinical benchmark evaluation metrics matching Table I & II in paper."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    }
