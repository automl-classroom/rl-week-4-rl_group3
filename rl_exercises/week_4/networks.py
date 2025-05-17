from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        """
        super().__init__()

        # dynamic number of hidden layers (copilot)
        layers = []

        # Input layer
        layers.append(("fc1", nn.Linear(obs_dim, hidden_dim)))
        layers.append(("relu1", nn.ReLU()))
        # Hidden layers
        for i in range(1, num_hidden_layers):
            layers.append((f"fc{i + 1}", nn.Linear(hidden_dim, hidden_dim)))
            layers.append((f"relu{i + 1}", nn.ReLU()))
        # Output layer
        layers.append(("out", nn.Linear(hidden_dim, n_actions)))

        # Create the sequential model
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
