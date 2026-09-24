"""1D GradCAM implementation for CNN Autoencoder and Discounting HeatMap (DHM) aggregation.

As formulated in Section II.D.1 and Section III.B of the IEEE IRI 2023 paper:
Equation (3): nabla = d(y_cls) / d(F)
Equation (4): H = (1 / C) * sum_{i=1}^C (hm_i * grad_i)  [Hadamard product]
Equation (9): DisHM = sum_{i=1}^4 (beta^i * HM_i(x_1, ..., x_19)), with beta = 0.9
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from xautonet.config import FILTER_FEATURES, DHM_BETA
from xautonet.models.autoencoder import Conv1DAutoencoder


class GradCAM1D:
    """Computes 1D GradCAM heatmaps on Autoencoder encoder layers (E1, E2, E3, E4).

    Gradients are computed from bottleneck activations back through convolutional feature maps.
    """

    def __init__(self, model: Conv1DAutoencoder, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or FILTER_FEATURES
        self.model.eval()

    def generate_layer_heatmaps(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate GradCAM heatmaps for all 4 encoder layers for input x.

        Args:
            x: Input tensor of shape (1, 19) or (1, 1, 19)
        Returns:
            Dict mapping layer name ('E1', 'E2', 'E3', 'E4') to 19-dimensional numpy heatmap array.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        # Forward pass through encoder
        latent = self.model.encoder(x)

        # Target: sum of bottleneck activations across all 11 latent dimensions
        target_score = latent.sum()
        target_score.backward(retain_graph=True)

        heatmaps = {}
        for layer_name in ["E1", "E2", "E3", "E4"]:
            fmap = self.model.encoder.feature_maps.get(layer_name)
            grad = self.model.encoder.gradients.get(layer_name)

            if fmap is not None and grad is not None:
                # fmap shape: (1, C, 19), grad shape: (1, C, 19)
                # Equation (4): H = (1 / C) * sum(hm_i * grad_i)
                hadamard = fmap * grad
                layer_cam = hadamard.mean(dim=1).squeeze(0)  # average across C channels -> (19,)

                # Apply ReLU: only features positively contributing to bottleneck
                layer_cam = F.relu(layer_cam)

                cam_np = layer_cam.detach().cpu().numpy()
                norm = np.max(np.abs(cam_np))
                if norm > 1e-8:
                    cam_np = cam_np / norm
                heatmaps[layer_name] = cam_np
            else:
                heatmaps[layer_name] = np.ones(len(self.feature_names), dtype=np.float32) / len(self.feature_names)

        return heatmaps


def compute_discounting_heatmap(
    layer_heatmaps: Dict[str, np.ndarray],
    beta: float = DHM_BETA,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """Combine heatmaps of each encoder layer with discounting factor beta=0.9 by Equation (9):

    DisHM = sum_{i=1}^4 (beta^i * HM_i(x_1, ..., x_19))

    Returns:
        dishm_scores: numpy array of aggregated scores for the 19 features
        ranked_features: list of (feature_name, score) sorted from most active to least
    """
    names = feature_names or FILTER_FEATURES
    n_features = len(names)
    dishm = np.zeros(n_features, dtype=np.float32)

    layer_order = ["E1", "E2", "E3", "E4"]
    for i, layer_name in enumerate(layer_order, start=1):
        if layer_name in layer_heatmaps:
            dishm += (beta ** i) * layer_heatmaps[layer_name]

    ranked = sorted(zip(names, dishm.tolist()), key=lambda item: item[1], reverse=True)
    return dishm, ranked
