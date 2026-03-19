"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        B = state.shape[0]
        flat = self.net(state)
        return flat.view(B, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.

    Same MLP architecture as ObstaclePolicy; goal information is simply
    concatenated into the state vector before training, so no structural
    change is needed here.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, d_model), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d_model
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        B = state.shape[0]
        flat = self.net(state)
        return flat.view(B, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 256,
    depth: int = 4,
    dropout: float = 0.0,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
