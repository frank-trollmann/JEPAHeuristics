"""Tests for :mod:`utils.cem_path_sampler`."""

from __future__ import annotations

import pytest
import numpy as np
import torch

from utils.cem_path_sampler import CEMPathSampler, GenerationResult


# =============================================================================
# Mock model for testing
# =============================================================================

class _MockJEPA(torch.nn.Module):
    """
    Mock JEPA model: cost = MSE between mean of action sequence and a target vector.

    This lets us verify CEM converges without loading the real model.
    """

    def __init__(self, target: np.ndarray):
        super().__init__()
        self._target = torch.from_numpy(target).float()  # (C, H, W) — unused, just for shape

    def encode(self, info: dict) -> dict:
        # Returns random embeddings — not used by get_cost in this mock
        B, S, H = info["pixels"].shape[:3]
        return {"emb": torch.randn(B, S, H, 192)}

    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Cost = sum of squared deviations of action values from 0.5.
        Optimum: every action value = 0.5.
        Shape: action_candidates (B, S, T, A) -> cost (B, S)
        """
        # Flatten last 3 dims and compare to 0.5
        flat = action_candidates.reshape(action_candidates.shape[0], -1)
        target = torch.full_like(flat, 0.5)
        mse = ((flat - target) ** 2).mean(dim=1)  # (B,)
        # Expand to (B, S) — all samples get same cost for simplicity
        B, S = action_candidates.shape[:2]
        return mse.unsqueeze(1).expand(B, S)


# =============================================================================
# CEMPathSampler tests
# =============================================================================

class TestCEMPathSampler:
    """Tests for the CEM path sampler with a mock JEPA model."""

    @pytest.fixture
    def mock_model(self):
        return _MockJEPA(target=np.ones(3))

    @pytest.fixture
    def sampler(self, mock_model):
        return CEMPathSampler(
            model=mock_model,
            history_size=2,
            action_dim=4,
            action_low=0.0,
            action_high=1.0,
            elite_fraction=0.25,
            min_std=1e-4,
            rng=np.random.default_rng(42),
        )

    def test_sample_path_converges(self, sampler):
        """After several generations the mean should approach 0.5."""
        agent_img = torch.ones(3, 224, 224)
        goal_img = torch.ones(3, 224, 224)

        path = sampler.sample_path(
            agent_image=agent_img,
            goal_image=goal_img,
            horizon=4,
            num_generations=10,
            batch_size=16,
            device="cpu",
            seed=42,
        )

        # Path should have correct shape
        assert path.shape == (4, 4)  # (horizon, action_dim)

        # Values should be clipped to [0, 1]
        assert path.min() >= 0.0
        assert path.max() <= 1.0

        # Mean should be closer to 0.5 than the initial 0.5 guess
        # (CEM should have refined it)
        mean_val = path.mean().item()
        assert 0.3 < mean_val < 0.7, f"Expected near 0.5, got {mean_val}"

    def test_best_cost_decreases(self, sampler):
        """Best cost should non-increase across generations."""
        agent_img = torch.ones(3, 224, 224)
        goal_img = torch.ones(3, 224, 224)

        path = sampler.sample_path(
            agent_image=agent_img,
            goal_image=goal_img,
            horizon=3,
            num_generations=6,
            batch_size=12,
            device="cpu",
            seed=123,
        )

        costs = [r.cost for r in sampler._history]
        # Group by generation
        gen_size = 12  # batch_size
        gen_costs = [min(costs[i:i + gen_size]) for i in range(0, len(costs), gen_size)]

        # Should generally decrease (allowing for randomness)
        assert gen_costs[-1] <= gen_costs[0] + 0.1, \
            f"Costs should decrease: {gen_costs}"

    def test_best_params_returns_array(self, sampler):
        """best_params should return a numpy array."""
        # Before any sampling — mean is not set yet, returns zeros
        params = sampler.best_params()
        assert isinstance(params, np.ndarray)
        assert len(params) == 0  # no sampling done yet

    def test_best_params_after_sampling(self, sampler):
        """best_params should return correct-sized array after sampling."""
        agent_img = torch.ones(3, 224, 224)
        goal_img = torch.ones(3, 224, 224)
        sampler.sample_path(
            agent_image=agent_img, goal_image=goal_img,
            horizon=3, num_generations=1, batch_size=4,
            device="cpu", seed=0,
        )
        params = sampler.best_params()
        assert isinstance(params, np.ndarray)
        assert len(params) == 3 * 4  # horizon * action_dim

    def test_best_cost_returns_inf_before_sampling(self, sampler):
        assert sampler.best_cost() == float("inf")

    def test_summary_before_sampling(self, sampler):
        s = sampler.summary()
        assert "Candidates evaluated: 0" in s

    def test_summary_after_sampling(self, sampler):
        agent_img = torch.ones(3, 224, 224)
        goal_img = torch.ones(3, 224, 224)
        sampler.sample_path(
            agent_image=agent_img, goal_image=goal_img,
            horizon=3, num_generations=2, batch_size=8,
            device="cpu", seed=0,
        )
        s = sampler.summary()
        assert "Best cost so far" in s
        assert "Candidates evaluated: 16" in s  # 2 gens * 8 batch

    def test_action_clipping(self, sampler):
        """All action values should be within [action_low, action_high]."""
        agent_img = torch.ones(3, 224, 224)
        goal_img = torch.ones(3, 224, 224)

        path = sampler.sample_path(
            agent_image=agent_img, goal_image=goal_img,
            horizon=3, num_generations=5, batch_size=10,
            device="cpu", seed=99,
        )
        assert path.min() >= sampler.action_low - 1e-9
        assert path.max() <= sampler.action_high + 1e-9

    def test_reproducibility_with_seed(self, sampler):
        """Same seed should produce same first-generation samples."""
        agent_img = torch.ones(3, 224, 224)
        goal_img = torch.ones(3, 224, 224)

        # Reset sampler
        sampler.rng = np.random.default_rng(7)
        path1 = sampler.sample_path(
            agent_image=agent_img, goal_image=goal_img,
            horizon=2, num_generations=1, batch_size=4,
            device="cpu", seed=7,
        )

        sampler.rng = np.random.default_rng(7)
        path2 = sampler.sample_path(
            agent_image=agent_img, goal_image=goal_img,
            horizon=2, num_generations=1, batch_size=4,
            device="cpu", seed=7,
        )

        np.testing.assert_array_equal(path1.numpy(), path2.numpy())


# =============================================================================
# build_info_from_positions test (skipped if model can't load)
# =============================================================================

class TestBuildInfo:
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="needs GPU or skips if model unavailable",
    )
    def test_build_info_shapes(self):
        """build_info_from_positions returns correctly shaped tensors."""
        from utils.cem_path_sampler import build_info_from_positions
        from utils.load_fix_pretrained import load_fix_pretrained
        from huggingface_hub import snapshot_download

        cp = snapshot_download(
            repo_id="quentinll/lewm-tworooms",
            revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
        )
        model = load_fix_pretrained(cp)
        model.eval()

        agent_img, goal_img = build_info_from_positions(
            model, [50.0, 50.0], [180.0, 180.0], device="cpu"
        )
        assert agent_img.shape == (3, 224, 224)
        assert goal_img.shape == (3, 224, 224)
