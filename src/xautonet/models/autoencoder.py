"""1D CNN Autoencoder architecture for compressing 19 clinical features to 11 latent representations.

As described in Section II.B.5, II.C and Fig. 2 of the IEEE IRI 2023 paper:
"Encoder block: This block encodes or compresses the input dimension into a smaller
latent dimension. It comprises of four 1D convolutional layers with decreasing number
of filters in each layer...
Decoder block: The role of this block is to decompress the latent dimension of the
encoder back to the original input dimension thereby reconstructing them. The
convolution layers used in this block are also connected sequentially with increasing
number of filters, making it symmetric with the encoder. The output of the last
convolution layer was flattened and passed to a final dense layer with nineteen
neurons representing the corresponding input dimensions...
Loss: Expected Mean Squared Error (MSE)."
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DEncoder(nn.Module):
    """4-layer 1D CNN Encoder compressing 19 input features into 11 latent dimensions."""

    def __init__(self, input_dim: int = 19, latent_dim: int = 11):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # E1: Conv1D (B, 1, 19) -> (B, 64, 19)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)

        # E2: Conv1D (B, 64, 19) -> (B, 32, 19)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        # E3: Conv1D (B, 32, 19) -> (B, 16, 19)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        # E4: Conv1D (B, 16, 19) -> (B, 8, 19)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)

        # Bottleneck projection to exact latent dimension of 11
        self.bottleneck_fc = nn.Linear(8 * input_dim, latent_dim)

        # Dictionaries to store feature maps and gradients for 1D GradCAM
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Tensor of shape (Batch, 19) or (Batch, 1, 19)
        Returns:
            latent: Tensor of shape (Batch, 11)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # E1
        out1 = F.relu(self.conv1(x))
        self.feature_maps["E1"] = out1
        if out1.requires_grad:
            out1.register_hook(lambda grad: self._save_gradient("E1", grad))

        # E2
        out2 = F.relu(self.conv2(out1))
        self.feature_maps["E2"] = out2
        if out2.requires_grad:
            out2.register_hook(lambda grad: self._save_gradient("E2", grad))

        # E3
        out3 = F.relu(self.conv3(out2))
        self.feature_maps["E3"] = out3
        if out3.requires_grad:
            out3.register_hook(lambda grad: self._save_gradient("E3", grad))

        # E4
        out4 = F.relu(self.conv4(out3))
        self.feature_maps["E4"] = out4
        if out4.requires_grad:
            out4.register_hook(lambda grad: self._save_gradient("E4", grad))

        flattened = out4.view(out4.size(0), -1)
        latent = self.bottleneck_fc(flattened)
        self.feature_maps["bottleneck"] = latent
        return latent

    def _save_gradient(self, layer_name: str, grad: torch.Tensor):
        self.gradients[layer_name] = grad


class Conv1DDecoder(nn.Module):
    """4-layer 1D CNN Decoder decompressing 11 latent dimensions back to 19 features."""

    def __init__(self, latent_dim: int = 11, output_dim: int = 19):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Project 11 latent dimensions up to (8 * output_dim)
        self.fc_expand = nn.Linear(latent_dim, 8 * output_dim)

        # D1: Conv1D (B, 8, 19) -> (B, 16, 19)
        self.deconv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

        # D2: Conv1D (B, 16, 19) -> (B, 32, 19)
        self.deconv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # D3: Conv1D (B, 32, 19) -> (B, 64, 19)
        self.deconv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # D4: Conv1D (B, 64, 19) -> (B, 64, 19)
        self.deconv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Final Dense reconstruction layer: Flatten -> Dense(19)
        self.reconstruction_dense = nn.Linear(64 * output_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            latent: Tensor of shape (Batch, 11)
        Returns:
            reconstruction: Tensor of shape (Batch, 19)
        """
        expanded = F.relu(self.fc_expand(latent))
        x = expanded.view(-1, 8, self.output_dim)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        flattened = x.view(x.size(0), -1)
        reconstruction = self.reconstruction_dense(flattened)
        return reconstruction


class Conv1DAutoencoder(nn.Module):
    """Complete Autoencoder combining 1D CNN Encoder and Decoder.

    Trained with Mean Squared Error (MSE) loss:
    L(theta) = E[(I - Output(theta))^2]
    """

    def __init__(self, input_dim: int = 19, latent_dim: int = 11):
        super().__init__()
        self.encoder = Conv1DEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Conv1DDecoder(latent_dim=latent_dim, output_dim=input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (latent_representation, reconstructed_features)."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the 11 latent bottleneck features."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct original 19 features from latent representation."""
        return self.decoder(latent)
