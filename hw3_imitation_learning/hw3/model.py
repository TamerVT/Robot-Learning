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


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: scale + shift hidden features by a goal embedding.

    Initialised as identity (gamma=1, beta=0) so training starts stable.
    """

    def __init__(self, d_model: int, emb_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(emb_dim, d_model)
        self.beta  = nn.Linear(emb_dim, d_model)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.gamma(emb) * h + self.beta(emb)


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned FiLM policy for the multicube scene.

    Architecture
    ------------
    1. Extract the goal one-hot from the flat state and project it through a
       small learnable embedding (Linear(goal_dim → goal_emb_dim)).
    2. Optionally concatenate relative position features:
       ``cube_i − ee_xyz`` and ``bin_xyz − cube_i`` for each cube.
       These make the policy approximately translation-invariant.
    3. Pass the (possibly augmented) state through a deep MLP.
       After every ReLU, apply a FiLM layer conditioned on the goal embedding.
    4. Project to a flat action chunk and reshape to (B, chunk_size, action_dim).

    Layout parameters (indices into the flat, already-normalised state vector)
    --------------------------------------------------------------------------
    goal_start   : first index of the goal one-hot           (default 9)
    goal_dim     : length of the one-hot                     (default 3)
    goal_emb_dim : learnable embedding width                 (default 16)
    ee_start     : first index of EE xyz                     (default 15)
    bin_start    : first index of bin/goal-pos xyz           (default 12)
    cube_starts  : list of first indices for each cube's xyz (default [0, 3, 6])

    The defaults match the recommended training command using ``[:3]`` slicing::

        python scripts/train.py --policy multitask \\
            --state-keys "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" \\
                         "original_pos_cube_blue[:3]" state_goal goal_pos \\
                         state_ee_xyz state_gripper ...

    which yields state_dim = 19.  Relative features are disabled when any of
    *ee_start*, *bin_start*, or *cube_starts* is ``None``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        d_model: int = 512,
        depth: int = 5,
        dropout: float = 0.1,
        # ── goal conditioning ───────────────────────────────────────
        goal_start: int = 9,
        goal_dim: int = 3,
        goal_emb_dim: int = 16,
        # ── relative position features ──────────────────────────────
        ee_start: int | None = 15,
        bin_start: int | None = 12,
        cube_starts: list[int] | None = None,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.goal_start   = goal_start
        self.goal_dim     = goal_dim
        self.goal_emb_dim = goal_emb_dim
        self.ee_start     = ee_start
        self.bin_start    = bin_start
        self.cube_starts  = list(cube_starts) if cube_starts is not None else [0, 3, 6]

        self._use_rel = (
            ee_start is not None
            and bin_start is not None
            and len(self.cube_starts) > 0
        )

        # Augmented input: base state + (cube−ee, bin−cube) × n_cubes
        n_rel  = len(self.cube_starts) * 6 if self._use_rel else 0
        in_dim = state_dim + n_rel

        # Goal embedding
        self.goal_embed = nn.Linear(goal_dim, goal_emb_dim)

        # FiLM-conditioned hidden layers
        self.hidden_layers = nn.ModuleList()
        self.film_layers   = nn.ModuleList()
        self.drop_layers   = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(nn.Linear(in_dim, d_model))
            self.film_layers.append(FiLMBlock(d_model, goal_emb_dim))
            self.drop_layers.append(nn.Dropout(dropout))
            in_dim = d_model

        self.head = nn.Linear(in_dim, chunk_size * action_dim)

    # ── internal helpers ────────────────────────────────────────────

    def _rel_features(self, state: torch.Tensor) -> torch.Tensor:
        """Compute relative position features from the (normalised) state."""
        ee      = state[:, self.ee_start  : self.ee_start  + 3]  # (B, 3)
        bin_pos = state[:, self.bin_start : self.bin_start + 3]  # (B, 3)
        parts: list[torch.Tensor] = []
        for cs in self.cube_starts:
            cube = state[:, cs : cs + 3]   # (B, 3) — xyz, first 3 dims of cube feat
            parts.append(cube - ee)         # cube relative to EE
            parts.append(bin_pos - cube)    # bin  relative to cube
        return torch.cat(parts, dim=-1)

    # ── forward ─────────────────────────────────────────────────────

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        B = state.shape[0]

        goal     = state[:, self.goal_start : self.goal_start + self.goal_dim]
        goal_emb = self.goal_embed(goal)  # (B, goal_emb_dim)

        h = torch.cat([state, self._rel_features(state)], dim=-1) if self._use_rel else state

        for linear, film, drop in zip(self.hidden_layers, self.film_layers, self.drop_layers):
            h = torch.relu(linear(h))
            h = film(h, goal_emb)
            h = drop(h)

        return self.head(h).view(B, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(state), action_chunk)

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
    # MultiTaskPolicy-specific (ignored for ObstaclePolicy)
    goal_start: int = 9,
    goal_dim: int = 3,
    goal_emb_dim: int = 16,
    ee_start: int | None = 15,
    bin_start: int | None = 12,
    cube_starts: list[int] | None = None,
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
            goal_start=goal_start,
            goal_dim=goal_dim,
            goal_emb_dim=goal_emb_dim,
            ee_start=ee_start,
            bin_start=bin_start,
            cube_starts=cube_starts,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")