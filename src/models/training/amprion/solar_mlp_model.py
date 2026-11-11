from __future__ import annotations

import torch
from torch import nn


class SolarMultiOutputMLP(nn.Module):
    """Feed-forward model mapping one engineered forecast-origin row to 96 PV values."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 96,
        hidden_1: int = 256,
        hidden_2: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if min(input_size, output_size, hidden_1, hidden_2) <= 0:
            raise ValueError("All layer dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
