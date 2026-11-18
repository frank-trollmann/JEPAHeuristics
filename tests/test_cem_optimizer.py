"""Tests for :mod:`utils.cem_optimizer`."""

from __future__ import annotations

import pytest
import numpy as np

from utils.cem_optimizer import (
    CEMOptimizer,
    ParameterConfig,
    ParamType,
    GenerationResult,
    make_xgb_param_configs,
    make_mlp_param_configs,
    make_siamese_param_configs,
    xgb_sample_to_kwargs,
    mlp_sample_to_kwargs,
    siamese_sample_to_kwargs,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_optimizer(continuous: bool = True, seed: int = 42):
    """Small fixture: 1 continuous + 1 discrete param."""
    configs = [
        ParameterConfig("lr", ParamType.CONTINUOUS, lower=0.001, upper=0.5),
        ParameterConfig("depth", ParamType.DISCRETE, values=[2, 4, 6, 8]),
    ]
    return CEMOptimizer(configs, elite_fraction=0.25,
                        rng=np.random.default_rng(seed))


def _fake_eval(params: dict) -> float:
    """Artificial objective: prefers lr≈0.1 and depth=4."""
    lr_pen = (params["lr"] - 0.1) ** 2
    depth_pen = abs(params["depth"] - 4)
    return float(lr_pen + 0.01 * depth_pen)


# =============================================================================
# ParameterConfig / ParamType
# =============================================================================

class TestParamConfig:
    def test_continuous_defaults(self):
        cfg = ParameterConfig("x", ParamType.CONTINUOUS)
        assert cfg.lower == 0.0
        assert cfg.upper == 1.0
        assert cfg.init_mean is None
        assert cfg.init_std is None

    def test_discrete_requires_values(self):
        cfg = ParameterConfig("k", ParamType.DISCRETE, values=[1, 2, 3])
        assert cfg.values == [1, 2, 3]

    def test_init_probs_length_match(self):
        cfg = ParameterConfig("k", ParamType.DISCRETE, values=[1, 2],
                              init_probs=[0.3, 0.7])
        assert cfg.init_probs == [0.3, 0.7]


# =============================================================================
# CEMOptimizer — sampling
# =============================================================================

class TestSampling:
    def test_sample_returns_correct_keys(self):
        opt = _make_optimizer(seed=0)
        s = opt.sample(batch_size=1)[0]
        assert set(s.keys()) == {"lr", "depth"}

    def test_continuous_sample_within_bounds(self):
        opt = _make_optimizer(seed=0)
        for _ in range(50):
            s = opt.sample(batch_size=1)[0]
            assert 0.001 <= s["lr"] <= 0.5
            assert s["depth"] in {2, 4, 6, 8}

    def test_sample_batch_size(self):
        opt = _make_optimizer(seed=0)
        batch = opt.sample(batch_size=7)
        assert len(batch) == 7

    def test_discrete_values_from_config(self):
        opt = _make_optimizer(seed=0)
        samples = opt.sample(batch_size=100)
        depths = {s["depth"] for s in samples}
        assert depths <= {2, 4, 6, 8}


# =============================================================================
# CEMOptimizer — update and convergence
# =============================================================================

class TestUpdate:
    def test_empty_update_noop(self):
        opt = _make_optimizer(seed=0)
        opt.update([])
        assert opt.best_params() == {}
        assert opt.best_score() == float("inf")

    def test_single_result_becomes_best(self):
        opt = _make_optimizer(seed=0)
        params = opt.sample(batch_size=1)[0]
        opt.update([GenerationResult(params=params, score=1.0)])
        assert opt.best_params() == params
        assert opt.best_score() == 1.0

    def test_convergence_to_known_optimum(self):
        """After many generations on a convex-like landscape the distribution
        should concentrate around the optimum."""
        opt = CEMOptimizer(
            [ParameterConfig("x", ParamType.CONTINUOUS, lower=0.0, upper=1.0)],
            elite_fraction=0.2,
            rng=np.random.default_rng(123),
        )

        def objective(params):
            # minimum at x=0.7
            return (params["x"] - 0.7) ** 2

        for gen in range(15):
            samples = opt.sample(batch_size=20)
            results = [GenerationResult(params=s, score=objective(s))
                       for s in samples]
            opt.update(results)

        best = opt.best_params()["x"]
        # Should be reasonably close to 0.7 after 15 generations
        assert 0.5 <= best <= 0.9, f"Expected near 0.7, got {best}"

    def test_discrete_convergence(self):
        """On a discrete space the distribution should concentrate on the best value."""
        opt = CEMOptimizer(
            [ParameterConfig("k", ParamType.DISCRETE, values=[1, 2, 3, 4, 5])],
            elite_fraction=0.2,
            rng=np.random.default_rng(456),
        )

        def objective(params):
            return abs(params["k"] - 3)  # optimum at 3

        for gen in range(10):
            samples = opt.sample(batch_size=20)
            results = [GenerationResult(params=s, score=objective(s))
                       for s in samples]
            opt.update(results)

        best_k = opt.best_params()["k"]
        assert best_k == 3

    def test_multiple_params_converge(self):
        """Both continuous and discrete params should narrow."""
        opt = _make_optimizer(seed=789)

        def objective(params):
            return (params["lr"] - 0.3) ** 2 + abs(params["depth"] - 6)

        for gen in range(12):
            samples = opt.sample(batch_size=16)
            results = [GenerationResult(params=s, score=objective(s))
                       for s in samples]
            opt.update(results)

        best = opt.best_params()
        assert 0.2 <= best["lr"] <= 0.4, f"lr={best['lr']}"
        assert best["depth"] in {4, 6, 8}, f"depth={best['depth']}"


# =============================================================================
# CEMOptimizer — edge cases
# =============================================================================

class TestEdgeCases:
    def test_all_elite_same_value_continuous(self):
        """If all elite candidates are identical, std should not collapse to 0."""
        opt = CEMOptimizer(
            [ParameterConfig("x", ParamType.CONTINUOUS, lower=0.0, upper=1.0)],
            elite_fraction=1.0,  # everything is elite
            min_std=1e-2,
            rng=np.random.default_rng(1),
        )
        params = {"x": 0.5}
        opt.update([GenerationResult(params=params, score=0.0)])
        # std should be at least min_std
        assert opt._cont_dists["x"].std >= 1e-2

    def test_resume_from_saved_state(self):
        """Sampling after update should use the new distribution."""
        opt = _make_optimizer(seed=10)
        # Force a narrow distribution
        narrow = {"lr": 0.1, "depth": 4}
        opt.update([GenerationResult(params=narrow, score=0.0)])
        # Subsequent samples should be more likely near the optimum
        samples = opt.sample(batch_size=20)
        lrs = [s["lr"] for s in samples]
        # Mean should be closer to 0.1 than to 0.25 (initial mean)
        assert abs(np.mean(lrs) - 0.1) < 0.1


# =============================================================================
# Predefined config factories
# =============================================================================

class TestParamFactories:
    def test_xgb_configs(self):
        configs = make_xgb_param_configs()
        names = [c.name for c in configs]
        assert names == ["n_estimators", "max_depth", "learning_rate",
                         "subsample", "colsample_bytree"]
        assert all(c.ptype in (ParamType.DISCRETE, ParamType.CONTINUOUS)
                   for c in configs)

    def test_mlp_configs(self):
        configs = make_mlp_param_configs()
        names = [c.name for c in configs]
        assert "layer1" in names
        assert "layer2" in names
        assert "layer3" in names
        assert "epochs" in names
        assert "lr" in names

    def test_siamese_configs(self):
        configs = make_siamese_param_configs()
        names = [c.name for c in configs]
        assert "dropout" in names
        assert "lr" in names


# =============================================================================
# Sample-to-kwargs helpers
# =============================================================================

class TestSampleToKwargs:
    def test_xgb_kwargs_types(self):
        sample = {
            "n_estimators": 500, "max_depth": 6,
            "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.9,
        }
        kw = xgb_sample_to_kwargs(sample)
        assert kw["n_estimators"] == 500
        assert isinstance(kw["n_estimators"], int)
        assert isinstance(kw["learning_rate"], float)

    def test_mlp_kwargs_empty_layers(self):
        sample = {"layer1": 0, "layer2": 0, "layer3": 0, "epochs": 100, "lr": 0.001}
        kw = mlp_sample_to_kwargs(sample)
        # Fallback to (64, 32) when all layers are 0
        assert kw["hidden_layers"] == (64, 32)
        assert kw["epochs"] == 100

    def test_mlp_kwargs_with_layers(self):
        sample = {"layer1": 64, "layer2": 32, "layer3": 0, "epochs": 50, "lr": 0.01}
        kw = mlp_sample_to_kwargs(sample)
        assert kw["hidden_layers"] == (64, 32)

    def test_siamese_kwargs(self):
        sample = {
            "layer1": 128, "layer2": 64, "layer3": 0,
            "epochs": 200, "lr": 0.001, "dropout": 0.1,
        }
        kw = siamese_sample_to_kwargs(sample)
        assert kw["hidden_layers"] == (128, 64)
        assert kw["dropout"] == 0.1


# =============================================================================
# Summary
# =============================================================================

class TestSummary:
    def test_summary_before_training(self):
        opt = _make_optimizer(seed=0)
        s = opt.summary()
        assert "Samples collected : 0" in s

    def test_summary_after_training(self):
        opt = _make_optimizer(seed=0)
        opt.update([GenerationResult(params={"lr": 0.1, "depth": 4}, score=0.5)])
        s = opt.summary()
        assert "Best score so far: 0.5000" in s
        assert "Best params" in s
