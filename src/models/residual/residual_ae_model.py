"""Dense Autoencoder for 24-hour, 15-minute residual-load sequences.

Input/output shape is flattened [batch, 96 * 15] to remain compatible with
training, inference, and timestamp-scoring utilities.
"""
from __future__ import annotations

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """Fully connected autoencoder operating on a flattened 96 x 15 window."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, bottleneck_dim: int = 32):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or bottleneck_dim < 1:
            raise ValueError("input_dim, hidden_dim and bottleneck_dim must all be positive.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected flattened input [batch, {self.input_dim}], got {tuple(x.shape)}."
            )
        return self.decoder(self.encoder(x))
