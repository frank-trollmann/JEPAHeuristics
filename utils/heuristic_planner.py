"""
Heuristic-guided planners for JEPA planning in TwoRoomEnv.

Provides three planning strategies:
1. Greedy MLP-heuristic — uses a pretrained MLP to estimate distance from
   predicted next latent state to the goal, picks the action with lowest estimate.
2. Greedy JEPA-cost — same structure but uses the JEPA model's raw prediction
   cost (MSE between predicted and goal embedding) as the guidance signal.
3. CEM baseline — re-uses the existing Cross-Entropy Method path sampler.

The planner works entirely in the JEPA latent space (using model.predict /
model.get_cost) and optionally validates plans by executing them in the real
TwoRoomEnv.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from utils.euclid_distance_ml_model_training_utils import _Standardize
from utils.load_fix_pretrained import load_fix_pretrained
from utils.cem_path_sampler import CEMPathSampler, build_info_from_positions, GenerationResult


# =============================================================================
# MLP Heuristic Loader
# =============================================================================

def load_pretrained_mlp(
    checkpoint_path: str,
    device: str = "cpu",
) -> torch.nn.Sequential:
    """
    Load the pretrained MLP heuristic from the 1M-dataset run.

    The checkpoint stores a Sequential with the Standardizer baked in as the
    first module (keys 0.mean, 0.std, 1.weight, …).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Read standardizer stats from the standalone .npz side-car if available
    std_path = os.path.join(
        os.path.dirname(checkpoint_path),
        "2026-11-18_16-47-37_standardizer.npz",
    )
    if os.path.exists(std_path):
        std = np.load(std_path)
        mean, std_arr = std["mean"], std["std"]
    else:
        # Fallback: extract from checkpoint keys
        mean = ckpt["0.mean"].numpy()
        std_arr = ckpt["0.std"].numpy()

    hidden_layers = [128, 128, 64]
    layers: list[torch.nn.Module] = [
        _Standardize(torch.tensor(mean), torch.tensor(std_arr)),
    ]
    prev = 384
    for h in hidden_layers:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, 1))

    model = torch.nn.Sequential(*layers)
    model.load_state_dict(ckpt)
    model.eval()
    return model.to(device)


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class PlanningResult:
    """Results from one planning run."""

    method: str
    agent_pos: list[float]
    goal_pos: list[float]
    horizon: int
    success: bool
    final_distance: float  # Euclidean distance in state space
    initial_distance: float
    steps_taken: int
    planned_actions: np.ndarray | None = None  # (horizon, action_dim)
    planning_time_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Greedy Heuristic Planner
# =============================================================================

class GreedyHeuristicPlanner:
    """
    Greedy planner that selects each action by evaluating the MLP heuristic
    on the JEPA-predicted next state.

    Image input format: (B, T, C, H, W) — typically (1, 1, 3, 224, 224).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mlp_heuristic: torch.nn.Sequential,
        history_size: int = 3,
        action_dim: int = 10,
        action_low: float = -1.0,
        action_high: float = 1.0,
        num_candidates: int = 20,
        device: str = "cpu",
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.mlp = mlp_heuristic
        self.history_size = history_size
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.num_candidates = num_candidates
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()
        self.history_actions: list[np.ndarray] = []

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single image tensor to latent embedding.

        Accepts (C, H, W) or (1, 1, C, H, W) and returns (1, 1, D).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 4:
            image = image.unsqueeze(0)
        return self.model.encode({"pixels": image})["emb"]  # (1, 1, D)

    def plan(
        self,
        agent_image: torch.Tensor,
        goal_image: torch.Tensor,
        horizon: int,
    ) -> np.ndarray:
        """
        Run greedy heuristic planning and return the action sequence
        (horizon, action_dim).
        """
        H = self.history_size
        A = self.action_dim
        K = self.num_candidates
        device = self.device

        # Encode initial states
        emb_s = self._encode(agent_image)  # (1, 1, D)
        emb_g = self._encode(goal_image)  # (1, 1, D)

        planned_actions: list[np.ndarray] = []
        self.history_actions = []

        for _step in range(horizon):
            # Sample candidate actions: (K, A)
            raw = self.rng.uniform(self.action_low, self.action_high, size=(K, A))

            # Build history buffer for prediction: (K, H, A)
            # Last H-1 actions from history + current candidate
            n_hist = min(len(self.history_actions), H)
            hist_arr = np.array(self.history_actions[-n_hist:] if n_hist > 0 else [], dtype=np.float32).reshape(-1, A)
            if len(self.history_actions) < H:
                hist_arr = np.concatenate([
                    np.zeros((H - len(self.history_actions), A), dtype=np.float32),
                    hist_arr,
                ], axis=0)

            hist_tensor = torch.from_numpy(hist_arr).float().to(device)  # (H, A)
            cand_tensor = torch.from_numpy(raw).float().to(device)  # (K, A)

            # Build (K, H, A) where last entry is candidate
            history_for_pred = torch.cat([
                hist_tensor.unsqueeze(0).expand(K, -1, -1)[:, :-1, :],
                cand_tensor.unsqueeze(1),
            ], dim=1)  # (K, H, A)

            with torch.no_grad():
                # Action encoder: (K, H, A) -> (K, H, D)
                act_emb = self.model.action_encoder(history_for_pred)
                # Predict next embedding: (K, 1, D)
                emb_s_exp = emb_s.expand(K, -1, -1)  # (K, 1, D)
                emb_pred = self.model.predict(emb_s_exp, act_emb)[:, -1, :]  # (K, D)

                # Heuristic: h(emb_pred, emb_g)
                emb_g_exp = emb_g.squeeze(0).expand(K, -1)  # (K, D)
                mlp_input = torch.cat([emb_pred, emb_g_exp], dim=-1)  # (K, 2D)
                heuristic_vals = self.mlp(mlp_input).squeeze(-1)  # (K,)

            best_idx = int(heuristic_vals.argmin())
            best_action = raw[best_idx].copy()
            planned_actions.append(best_action)
            self.history_actions.append(best_action)

            # Update current embedding
            emb_s = emb_pred[best_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, D)

        return np.array(planned_actions)


def greedy_jepa_cost_planner(
    model: torch.nn.Module,
    agent_image: torch.Tensor,
    goal_image: torch.Tensor,
    horizon: int,
    num_candidates: int = 20,
    history_size: int = 3,
    action_dim: int = 10,
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Greedy planner that uses the JEPA model's raw prediction cost (MSE to
    goal embedding) as the action-selection signal, without an MLP heuristic.
    """
    H = history_size
    A = action_dim
    K = num_candidates
    rng = rng if rng is not None else np.random.default_rng()
    history_actions: list[np.ndarray] = []

    # Encode — images may be (C,H,W), (1,1,C,H,W) or (1,1,T,C,H,W)
    def _enc(img):
        if img.dim() == 3: img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 6: img = img[:, :, :1]  # take first timestep
        return model.encode({"pixels": img})["emb"]  # (1, 1, D)
    emb_s = _enc(agent_image)
    emb_g = _enc(goal_image)

    planned_actions: list[np.ndarray] = []

    for _step in range(horizon):
        raw = rng.uniform(-1.0, 1.0, size=(K, A))

        n_hist = min(len(history_actions), H)
        hist_arr = np.array(history_actions[-n_hist:] if n_hist > 0 else [], dtype=np.float32).reshape(-1, A)
        if len(history_actions) < H:
            hist_arr = np.concatenate([
                np.zeros((H - len(history_actions), A), dtype=np.float32),
                hist_arr,
            ], axis=0)

        hist_tensor = torch.from_numpy(hist_arr).float().to(device)
        cand_tensor = torch.from_numpy(raw).float().to(device)
        history_for_pred = torch.cat([
            hist_tensor.unsqueeze(0).expand(K, -1, -1)[:, :-1, :],
            cand_tensor.unsqueeze(1),
        ], dim=1)  # (K, H, A)

        with torch.no_grad():
            act_emb = model.action_encoder(history_for_pred)
            emb_pred = model.predict(emb_s.expand(K, -1, -1), act_emb)[:, -1, :]
            cost = torch.mean((emb_pred - emb_g.squeeze(0)) ** 2, dim=-1)

        best_idx = int(cost.argmin())
        planned_actions.append(raw[best_idx].copy())
        history_actions.append(raw[best_idx])
        emb_s = emb_pred[best_idx].unsqueeze(0).unsqueeze(0)

    return np.array(planned_actions)


# =============================================================================
# Env Execution & Evaluation
# =============================================================================

def execute_plan_in_env(
    plan: np.ndarray,
    agent_pos: list[float],
    goal_pos: list[float],
) -> dict[str, float]:
    """
    Execute a planned action sequence in the real TwoRoomEnv.
    Actions are 10-dim vectors; we use the first 2 components as the env action.

    Returns dict with final_distance, success, steps_taken.
    """
    from stable_worldmodel.envs.two_room import TwoRoomEnv

    env = TwoRoomEnv()
    env.reset(options={"state": agent_pos})
    env.unwrapped._set_goal_state(goal_pos)

    initial_dist = float(torch.norm(env.unwrapped.agent_position - env.unwrapped.target_position).item())
    final_dist = initial_dist
    steps = 0

    for action_10d in plan:
        env_action = action_10d[:2]  # First 2 dims -> env action space
        _, _, terminated, truncated, info = env.step(env_action)
        final_dist = info["distance_to_target"]
        steps += 1
        if terminated or truncated:
            break

    success = final_dist < 10.0
    return {
        "final_distance": float(final_dist),
        "initial_distance": float(initial_dist),
        "success": success,
        "steps_taken": steps,
    }


def evaluate_plan(
    plan: np.ndarray,
    agent_pos: list[float],
    goal_pos: list[float],
    horizon: int,
    method: str,
    planning_time_s: float = 0.0,
) -> PlanningResult:
    """Wrap env execution into a PlanningResult."""
    env_stats = execute_plan_in_env(plan, agent_pos, goal_pos)
    return PlanningResult(
        method=method,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        horizon=horizon,
        success=env_stats["success"],
        final_distance=env_stats["final_distance"],
        initial_distance=env_stats["initial_distance"],
        steps_taken=env_stats["steps_taken"],
        planned_actions=plan,
        planning_time_s=planning_time_s,
    )


# =============================================================================
# CEM with MLP Heuristic
# =============================================================================

class CEMPathSamplerMLP:
    """
    CEM path sampler that uses the MLP heuristic as the cost function
    instead of the raw JEPA prediction cost.

    For each candidate action sequence:
      1. Roll out through JEPA to get predicted final embedding
      2. Evaluate h(emb_pred, emb_g) using the pretrained MLP
      3. Use this MLP cost for elite selection in CEM
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mlp_heuristic: torch.nn.Sequential,
        history_size: int = 3,
        action_dim: int = 10,
        action_low: float = -1.0,
        action_high: float = 1.0,
        elite_fraction: float = 0.2,
        min_std: float = 1e-3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.mlp = mlp_heuristic
        self.history_size = history_size
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.elite_fraction = elite_fraction
        self.min_std = min_std
        self.rng = rng if rng is not None else np.random.default_rng()
        self._history: list[GenerationResult] = []

    def _rollout_cost(
        self,
        agent_img: torch.Tensor,
        goal_img: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Roll out action sequence through JEPA, return MLP heuristic cost.

        agent_img / goal_img: (B, T_img, C, H_img, W_img) — typically (1,1,C,H,W)
        action_seq: (B, S, T, A)
        Returns: (B, S) — MLP predicted distance for each candidate
        """
        B, S, T, A = action_seq.shape
        H = self.history_size
        device = action_seq.device

        # Normalize image tensor dimensions:
        # Accept (C,H,W), (1,C,H,W), (1,1,C,H,W) and expand to (B, S, H, C, H_img, W_img)
        def _to_6d(img):
            # Normalize to (1,1,C,H,W) then expand to (B,S,H,C,H,W)
            if img.dim() == 3:
                img = img.unsqueeze(0).unsqueeze(0)  # (1,1,C,H,W)
            elif img.dim() == 4:
                img = img.unsqueeze(0)  # (1,1,C,H,W)
            elif img.dim() == 5:
                pass  # already (1,1,C,H,W)
            elif img.dim() == 6:
                img = img[:, :1, :1]  # (1,1,C,H,W) — take first batch/sample
            # Now (1,1,C,H,W) -> (B,S,H,C,H,W)
            return img.unsqueeze(1).expand(B, S, H, -1, -1, -1)

        img_6d_a = _to_6d(agent_img)
        img_6d_g = _to_6d(goal_img)

        # Encode start and goal: (B, S, 1, D)
        emb_s = self.model.encode({"pixels": img_6d_a[:, :, 0]})["emb"]  # (B, S, 1, D)
        emb_g = self.model.encode({"pixels": img_6d_g[:, :, 0]})["emb"]  # (B, S, 1, D)

        # Flatten batch and sample dims for rollout
        emb_s_flat = emb_s.reshape(B * S, 1, -1)  # (BS, 1, D)
        act_flat = action_seq.reshape(B * S, T, A)  # (BS, T, A)

        # Autoregressive rollout
        history_actions: list[torch.Tensor] = []
        for t in range(T):
            # Build history tensor (last H actions)
            if history_actions:
                act_tensor = torch.cat(history_actions, dim=1)  # (BS, total_len, A)
                if act_tensor.shape[1] >= H:
                    hist = act_tensor[:, -H:, :]  # (BS, H, A)
                else:
                    n_pad = H - act_tensor.shape[1]
                    pad_tensor = torch.zeros(B * S, n_pad, A, device=device)
                    hist = torch.cat([pad_tensor, act_tensor], dim=1)  # (BS, H, A)
            else:
                hist = torch.zeros(B * S, H, A, device=device)

            # Predict next embedding
            act_emb = self.model.action_encoder(hist)
            pred = self.model.predict(emb_s_flat, act_emb)[:, -1:, :]  # (BS, 1, D)
            emb_s_flat = pred

            # Store action for next history
            history_actions.append(act_flat[:, t:t+1, :])

        # Final predicted embedding: (BS, D)
        pred_final = emb_s_flat[:, 0, :]

        # Compute MLP heuristic cost: h(pred_final, goal_emb)
        emb_g_flat = emb_g.reshape(B * S, -1)  # (BS, D)
        mlp_input = torch.cat([pred_final, emb_g_flat], dim=-1)  # (BS, 2D)
        costs = self.mlp(mlp_input).squeeze(-1)  # (BS,)

        return costs.reshape(B, S)  # (B, S)

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
        """Run CEM with MLP heuristic and return the best action sequence."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        B = 1
        S = batch_size
        T = horizon
        A = self.action_dim
        H = self.history_size
        flat_dim = T * A

        self.mean = np.zeros(flat_dim)
        self.std = np.full(flat_dim, (self.action_high - self.action_low) / 2)
        self._history = []

        # Prepare image tensors — normalize to (1,1,C,H,W)
        def _prep(img):
            img = img.to(device)
            if img.dim() == 3: img = img.unsqueeze(0).unsqueeze(0)
            elif img.dim() == 4: img = img.unsqueeze(0)
            elif img.dim() == 5: pass
            elif img.dim() == 6: img = img[:, 0:1, 0:1]
            elif img.dim() == 7: img = img[:, 0:1, 0:1, 0:1]
            return img
        agent_img = _prep(agent_image)
        goal_img = _prep(goal_image)

        for gen in range(num_generations):
            raw = self.rng.normal(self.mean, self.std, size=(S, flat_dim))
            raw = np.clip(raw, self.action_low, self.action_high)

            # Build action sequences: (B=1, S, T, A)
            action_seqs = torch.from_numpy(raw).float().to(device).reshape(B, S, T, A)

            with torch.no_grad():
                costs = self._rollout_cost(agent_img, goal_img, action_seqs)  # (B, S) = (1, S)

            cost_vals = costs.squeeze(0).cpu().numpy()  # (S,)

            costs_list = [GenerationResult(params=raw[i].copy(), cost=float(cost_vals[i]))
                          for i in range(S)]

            # Select elites
            sorted_costs = sorted(costs_list, key=lambda c: c.cost)
            k = max(1, int(S * self.elite_fraction))
            elites = sorted_costs[:k]

            elite_arr = np.array([e.params for e in elites])
            self.mean = elite_arr.mean(axis=0)
            self.std = np.maximum(elite_arr.std(axis=0), self.min_std)

            self._history.extend(costs_list)

            best_so_far = min(c.cost for c in costs_list)
            print(f"    Gen {gen + 1}/{num_generations}: "
                  f"best={best_so_far:.4f}  "
                  f"mean_cost={np.mean([c.cost for c in costs_list]):.4f}")

        best = min(self._history, key=lambda c: c.cost)
        return torch.from_numpy(best.params).float().reshape(T, A)




# =============================================================================
# Receding-Horizon Controller
# =============================================================================

def receding_horizon_plan(
    model: torch.nn.Module,
    mlp_heuristic: torch.nn.Sequential | None,
    agent_pos: list[float],
    goal_pos: list[float],
    plan_horizon: int,
    max_steps: int,
    method: str = "greedy_jepa",
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
    num_candidates: int = 20,
) -> tuple[np.ndarray, dict]:
    """
    Receding-horizon planning: re-plan at each step, execute only the first action.

    Parameters
    ----------
    model : JEPA model
    mlp_heuristic : MLP heuristic (None for greedy_jepa / cem)
    agent_pos : Initial agent position [x, y]
    goal_pos : Goal position [x, y]
    plan_horizon : Planning horizon H for each re-plan
    max_steps : Maximum number of execution steps
    method : "greedy_jepa", "greedy_mlp", "cem", or "cem_mlp"
    device : Torch device
    rng : Random generator
    num_candidates : Candidates per greedy step

    Returns
    -------
    plan : np.ndarray of shape (steps_taken, 10) — full action sequence
    info : dict with steps_taken, final_distance, success, re_plans
    """
    from stable_worldmodel.envs.two_room import TwoRoomEnv
    from utils.image_preprocessing import normalise_images

    if rng is None:
        rng = np.random.default_rng()

    env = TwoRoomEnv()
    env.reset(options={"state": agent_pos})
    env.unwrapped._set_goal_state(goal_pos)

    all_actions: list[np.ndarray] = []
    re_plans = 0

    for step in range(max_steps):
        # Render current state and goal
        img_a = env.render()
        img_g = env.render()  # goal image (same regardless of agent pos)
        t_a = normalise_images(
            torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(device)
        )
        t_g = normalise_images(
            torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # Re-plan with current horizon
        if method == "greedy_jepa":
            plan = greedy_jepa_cost_planner(
                model=model, agent_image=t_a, goal_image=t_g,
                horizon=plan_horizon, rng=rng,
            )
        elif method == "greedy_mlp":
            planner = GreedyHeuristicPlanner(
                model=model, mlp_heuristic=mlp_heuristic,
                device=device, rng=rng, num_candidates=num_candidates,
            )
            plan = planner.plan(t_a, t_g, plan_horizon)
        elif method == "cem":
            sampler = CEMPathSampler(
                model=model, history_size=3, action_dim=10,
                action_low=-1.0, action_high=1.0, rng=rng,
            )
            plan = sampler.sample_path(
                agent_image=t_a, goal_image=t_g,
                horizon=plan_horizon, num_generations=5, batch_size=16, device=device,
            ).cpu().numpy()
        elif method == "cem_mlp":
            sampler = CEMPathSamplerMLP(
                model=model, mlp_heuristic=mlp_heuristic,
                history_size=3, action_dim=10,
                action_low=-1.0, action_high=1.0, rng=rng,
            )
            plan = sampler.sample_path(
                agent_image=t_a, goal_image=t_g,
                horizon=plan_horizon, num_generations=5, batch_size=16, device=device,
            ).cpu().numpy()
        else:
            raise ValueError(f"Unknown method: {method}")

        re_plans += 1

        # Execute only the first action
        action_10d = plan[0]
        all_actions.append(action_10d)

        env_action = action_10d[:2]
        _, _, terminated, truncated, info = env.step(env_action)
        final_dist = info["distance_to_target"]

        # Check success
        if final_dist < 10.0 or terminated or truncated:
            return np.array(all_actions), {
                "steps_taken": step + 1,
                "final_distance": float(final_dist),
                "success": final_dist < 10.0,
                "re_plans": re_plans,
            }

    return np.array(all_actions), {
        "steps_taken": max_steps,
        "final_distance": float(final_dist),
        "success": False,
        "re_plans": re_plans,
    }


# =============================================================================
# Hybrid: Greedy-Initialized CEM
# =============================================================================

class GreedyInitCEM:
    """CEM initialized with a greedy heuristic solution instead of uniform random."""

    def __init__(
        self,
        model: torch.nn.Module,
        history_size: int = 3,
        action_dim: int = 10,
        action_low: float = -1.0,
        action_high: float = 1.0,
        elite_fraction: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = model
        self.history_size = history_size
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.elite_fraction = elite_fraction
        self.rng = rng if rng is not None else np.random.default_rng()

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 4:
            image = image.unsqueeze(0)
        return self.model.encode({"pixels": image})["emb"]

    def _greedy_plan(self, emb_s, emb_g, horizon, num_candidates=20):
        H = self.history_size
        A = self.action_dim
        K = num_candidates
        rng = self.rng
        planned_actions = []
        history = []
        current_emb = emb_s

        for _step in range(horizon):
            raw = rng.uniform(self.action_low, self.action_high, size=(K, A))
            n_hist = min(len(history), H)
            hist_arr = np.zeros((0, A), dtype=np.float32) if not history else np.array(history[-n_hist:], dtype=np.float32)
            if len(history) < H:
                hist_arr = np.concatenate([np.zeros((H - len(history), A), dtype=np.float32), hist_arr], axis=0)
            hist_tensor = torch.from_numpy(hist_arr).float().to(current_emb.device)
            cand_tensor = torch.from_numpy(raw).float().to(current_emb.device)
            history_for_pred = torch.cat([
                hist_tensor.unsqueeze(0).expand(K, -1, -1)[:, :-1, :],
                cand_tensor.unsqueeze(1),
            ], dim=1)
            with torch.no_grad():
                act_emb = self.model.action_encoder(history_for_pred)
                emb_s_exp = current_emb.expand(K, -1, -1)
                emb_pred = self.model.predict(emb_s_exp, act_emb)[:, -1, :]
                emb_g_exp = emb_g.squeeze(0).expand(K, -1)
                mse = (emb_pred - emb_g_exp).pow(2).mean(dim=-1)
            best_idx = int(mse.argmin())
            best_action = raw[best_idx].copy()
            planned_actions.append(best_action)
            history.append(best_action)
            current_emb = emb_pred[best_idx].unsqueeze(0).unsqueeze(0)
        return np.array(planned_actions)

    def sample_path(self, agent_image, goal_image, horizon, num_generations=5,
                    batch_size=32, device="cpu"):
        A = self.action_dim
        flat_dim = horizon * A
        rng = self.rng

        emb_s = self._encode(agent_image)
        emb_g = self._encode(goal_image)

        # Greedy initialization
        greedy_plan = self._greedy_plan(emb_s, emb_g, horizon, num_candidates=20)
        mu = torch.from_numpy(greedy_plan.flatten()).float().to(device)
        sigma = torch.ones(flat_dim, device=device) * 0.5
        elite_idx = max(1, int(batch_size * self.elite_fraction))

        best_cost = float("inf")
        best_candidates = None
        for gen in range(num_generations):
            candidates = torch.randn(batch_size, flat_dim, device=device) * sigma.unsqueeze(0) + mu.unsqueeze(0)
            candidates = torch.clamp(candidates, self.action_low, self.action_high)
            actions = candidates.view(batch_size, horizon, A)

            H = self.history_size
            batch_emb = emb_s.expand(batch_size, -1, -1)
            for step in range(horizon):
                step_actions = actions[:, step, :]
                n_hist = min(step, H)
                hist_arr = torch.zeros(batch_size, H, A, device=device)
                for b in range(batch_size):
                    start = max(0, step - n_hist)
                    hist_arr[b, H - n_hist:, :] = actions[b, start:step, :]
                history_for_pred = torch.cat([
                    hist_arr[:, :-1, :],
                    step_actions.unsqueeze(1),
                ], dim=1)
                with torch.no_grad():
                    act_emb = self.model.action_encoder(history_for_pred)
                    batch_emb = self.model.predict(batch_emb, act_emb)[:, -1:, :]

            costs = (batch_emb.squeeze(-2) - emb_g.expand(batch_size, -1, -1).squeeze(-2)).pow(2).mean(dim=-1)
            sorted_idx = torch.argsort(costs)
            elite_candidates = candidates[sorted_idx[:elite_idx]]
            mu = elite_candidates.mean(dim=0)
            sigma = elite_candidates.std(dim=0).clamp(min=1e-6)
            gen_best = costs[sorted_idx[0]].item()
            print(f"    Gen {gen+1}/{num_generations}: best={gen_best:.2f}  mean_cost={costs.mean().item():.2f}")
            if gen_best < best_cost:
                best_cost = gen_best
                best_candidates = candidates[sorted_idx[:1]]

        if best_candidates is None:
            return candidates.squeeze(0).view(horizon, A)
        return best_candidates.squeeze(0).view(horizon, A)
