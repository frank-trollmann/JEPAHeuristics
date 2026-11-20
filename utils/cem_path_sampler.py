"""
Cross-Entropy Method (CEM) for path / action-sequence sampling with JEPA.

Uses the JEPA model's ``get_cost`` to evaluate how well a candidate action
sequence drives the predicted state toward a goal.  CEM iteratively refines a
Gaussian distribution over flattened action sequences, keeping the top-``elite_fraction``
candidates each generation.

Usage as a library::

    from utils.cem_path_sampler import CEMPathSampler
    from utils.load_fix_pretrained import load_fix_pretrained
    from huggingface_hub import snapshot_download
    from utils.image_preprocessing import normalise_images
    from stable_worldmodel.envs.two_room import TwoRoomEnv
    import torch

    model = load_fix_pretrained(snapshot_download(...))
    model.eval()
    env = TwoRoomEnv()
    env.reset(options={"state": [50, 50]})

    sampler = CEMPathSampler(model, history_size=3, action_dim=10,
                             action_low=-1.0, action_high=1.0)

    plan = sampler.sample_path(
        agent_image=agent_img,
        goal_image=goal_img,
        horizon=5,
        num_generations=8,
        batch_size=20,
        elite_fraction=0.2,
    )
    # plan is a torch.Tensor of shape (horizon, action_dim)

Usage from CLI::

    python -m scripts.cem_path_sampling \\
        --agent-pos 50 50 --goal-pos 180 180 \\
        --horizon 5 --generations 8 --batch-size 20
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class GenerationResult:
    """Cost and flattened action sequence for one sampled candidate."""
    params: np.ndarray   # flattened action sequence, shape (horizon * action_dim,)
    cost: float


class CEMPathSampler:
    """
    CEM optimizer over action sequences for JEPA path planning.

    Parameters
    ----------
    model : torch.nn.Module
        JEPA / LeWM model with ``encode()`` and ``get_cost()``.
    history_size : int
        Number of past frames fed to the predictor (typically 3).
    action_dim : int
        Dimension of each action vector (typically 10 for TwoRoom).
    action_low : float
        Lower bound on action values (typically -1.0).
    action_high : float
        Upper bound on action values (typically 1.0).
    elite_fraction : float
        Fraction of candidates kept as elite each generation (default 0.2).
    min_std : float
        Floor on Gaussian std to prevent premature collapse (default 1e-3).
    rng : np.random.Generator, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        history_size: int = 3,
        action_dim: int = 10,
        action_low: float = -1.0,
        action_high: float = 1.0,
        elite_fraction: float = 0.2,
        min_std: float = 1e-3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.history_size = history_size
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.elite_fraction = elite_fraction
        self.min_std = min_std
        self.rng = rng if rng is not None else np.random.default_rng()
        self._history: list[GenerationResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_path(
        self,
        agent_image: torch.Tensor,
        goal_image: torch.Tensor,
        horizon: int,
        num_generations: int = 8,
        batch_size: int = 20,
        device: str = "cpu",
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Run CEM and return the best action sequence.

        Parameters
        ----------
        agent_image : torch.Tensor
            Preprocessed agent image, shape ``(C, H, W)`` or ``(1, C, H, W)``.
        goal_image : torch.Tensor
            Preprocessed goal image, same format.
        horizon : int
            Total action horizon T (must be >= history_size).
        num_generations : int
            Number of CEM generations.
        batch_size : int
            Candidate samples per generation.
        device : str
            Torch device for model inference.
        seed : int, optional
            Random seed.

        Returns
        -------
        torch.Tensor
            Best action sequence, shape ``(horizon, action_dim)``.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        B = 1
        S = batch_size
        T = horizon
        A = self.action_dim
        H = self.history_size
        flat_dim = T * A

        # Reset distribution to uniform over action space for this horizon
        self.mean = np.zeros(flat_dim)
        self.std = np.full(flat_dim, (self.action_high - self.action_low) / 2)
        self._history = []

        # Prepare image tensors: (B, S, H, C, H_img, W_img)
        if agent_image.dim() == 3:
            agent_img = agent_image.unsqueeze(0).unsqueeze(0)
        else:
            agent_img = agent_image.unsqueeze(0)
        agent_img = agent_img.expand(B, 1, H, -1, -1, -1).to(device)

        if goal_image.dim() == 3:
            goal_img = goal_image.unsqueeze(0).unsqueeze(0)
        else:
            goal_img = goal_image.unsqueeze(0)
        goal_img = goal_img.expand(B, 1, H, -1, -1, -1).to(device)

        for gen in range(num_generations):
            # Sample batch of flattened action sequences
            raw = self.rng.normal(self.mean, self.std, size=(S, flat_dim))
            raw = np.clip(raw, self.action_low, self.action_high)

            costs: list[GenerationResult] = []

            for i in range(S):
                flat_action = raw[i]
                # Reshape to (B, 1, T, A)
                action_seq = torch.from_numpy(flat_action).float().to(device)
                action_seq = action_seq.reshape(B, 1, T, A)

                # Build info dict
                info = {
                    "pixels": agent_img,
                    "goal": goal_img,
                    "action": action_seq[:, :, :H, :],  # initial history actions
                }

                try:
                    with torch.no_grad():
                        cost = self.model.get_cost(info, action_seq)
                    cost_val = cost.item()
                except Exception as exc:
                    print(f"      ERROR: {exc}")
                    cost_val = float("inf")

                costs.append(GenerationResult(params=flat_action.copy(), cost=cost_val))

            # Select elites
            sorted_costs = sorted(costs, key=lambda c: c.cost)
            k = max(1, int(S * self.elite_fraction))
            elites = sorted_costs[:k]

            # Update distribution from elite means
            elite_arr = np.array([e.params for e in elites])
            new_mean = elite_arr.mean(axis=0)
            new_std = elite_arr.std(axis=0)
            self.mean = new_mean
            self.std = np.maximum(new_std, self.min_std)

            self._history.extend(costs)

            best_so_far = min(c.cost for c in costs)
            print(f"    Gen {gen + 1}/{num_generations}: "
                  f"best={best_so_far:.4f}  "
                  f"mean_cost={np.mean([c.cost for c in costs]):.4f}")

        # Return best sequence
        best = min(self._history, key=lambda c: c.cost)
        return torch.from_numpy(best.params).float().reshape(T, A)

    def best_params(self) -> np.ndarray:
        """Return the best flattened action sequence seen so far."""
        if not self._history:
            return np.zeros(self.mean.shape if hasattr(self, 'mean') else (0,))
        return min(self._history, key=lambda c: c.cost).params

    def best_cost(self) -> float:
        """Return the best (lowest) cost seen so far."""
        if not self._history:
            return float("inf")
        return min(c.cost for c in self._history)

    def summary(self) -> str:
        lines = [f"CEMPathSampler (horizon*action_dim={self.mean.shape if hasattr(self, 'mean') else 0})"]
        lines.append(f"Candidates evaluated: {len(self._history)}")
        if self._history:
            lines.append(f"Best cost so far: {self.best_cost():.4f}")
        return "\n".join(lines)


# =============================================================================
# Convenience: build info dict from env positions
# =============================================================================

def build_info_from_positions(
    model: torch.nn.Module,
    agent_pos: list[float],
    goal_pos: list[float],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Render agent and goal images from positions and return preprocessed tensors.

    Returns
    -------
    agent_img : torch.Tensor
        Normalised agent image, shape ``(C, H, W)``.
    goal_img : torch.Tensor
        Normalised goal image, shape ``(C, H, W)``.
    """
    from stable_worldmodel.envs.two_room import TwoRoomEnv
    from utils.image_preprocessing import normalise_images

    env = TwoRoomEnv()

    env.reset(options={"state": agent_pos})
    img_a = env.render()
    t_img_a = torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0)
    t_img_a = normalise_images(t_img_a)

    env.reset(options={"state": goal_pos})
    img_g = env.render()
    t_img_g = torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0)
    t_img_g = normalise_images(t_img_g)

    return t_img_a.squeeze(0).to(device), t_img_g.squeeze(0).to(device)
