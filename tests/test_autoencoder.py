"""Unit tests for the Conv1D Autoencoder architecture and latent dimensionality reduction."""

import pytest
import torch
import numpy as np
from xautonet.models.autoencoder import Conv1DAutoencoder


def test_autoencoder_dimensions():
    batch_size = 16
    input_dim = 19
    latent_dim = 11

    model = Conv1DAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    dummy_input = torch.randn(batch_size, input_dim)

    # 1. Forward pass
    latent, recon = model(dummy_input)

    assert latent.shape == (batch_size, latent_dim), f"Expected latent shape {(batch_size, latent_dim)}, got {latent.shape}"
    assert recon.shape == (batch_size, input_dim), f"Expected recon shape {(batch_size, input_dim)}, got {recon.shape}"


def test_autoencoder_reconstruction_loss_descent():
    model = Conv1DAutoencoder(input_dim=19, latent_dim=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    x = torch.randn(32, 19)

    # Compute initial loss
    _, recon_init = model(x)
    loss_init = criterion(recon_init, x).item()

    # Train for 5 gradient steps
    for _ in range(5):
        optimizer.zero_grad()
        _, recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

    _, recon_final = model(x)
    loss_final = criterion(recon_final, x).item()

    assert loss_final < loss_init, f"Expected loss descent from {loss_init} to be lower, got {loss_final}"
