"""
Cross-Entropy Method (CEM) for hyperparameter optimization.

Mirrors the hyperparameter search in ``scripts/euclid_distance_ml_model_training.py``
but replaces exhaustive grid search with an adaptive distribution-based search.
Eachgeneration samples candidate hyperparameter sets, evaluates them, keeps the
elite (best-performing) fraction, and re-estimates the sampling distribution
from that elite set.

Usage from the CLI::

    python -m scripts.cem_hyperparameter_search \\
        --input data/2026-11-10\\ 17-44-33_10000_heuristic_dataset.parquet \\
        --model xgb --generations 10 --batch_size 20

Usage as a library::

    from utils.cem_optimizer import CEMOptimizer, ParameterConfig, ParamType
    from utils.euclid_distance_ml_model_training_utils import train_xgboost, evaluate_model

    optimizer = CEMOptimizer([
        ParameterConfig("n_estimators", ParamType.DISCRETE,
                        values=[100, 300, 500, 1000]),
        ParameterConfig("max_depth", ParamType.DISCRETE,
                        values=[3, 4, 6, 8]),
        ParameterConfig("learning_rate", ParamType.CONTINUOUS,
                        lower=0.001, upper=0.5),
    ], elite_fraction=0.2)

    for gen in range(10):
        candidates = optimizer.sample(batch_size=20)
        scores = [eval_fn(c) for c in candidates]
        optimizer.update(scores)
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np


class ParamType(Enum):
    """Whether a hyperparameter is sampled from a continuous or discrete set."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class ParameterConfig:
    """Definition of one optimizable hyperparameter."""
    name: str
    ptype: ParamType
    # For CONTINUOUS:
    lower: float = 0.0
    upper: float = 1.0
    # For DISCRETE:
    values: list[Any] = field(default_factory=list)
    # Initial distribution (overwritten on first update if not provided):
    init_mean: Optional[float] = None
    init_std: Optional[float] = None
    init_probs: Optional[list[float]] = None


@dataclass
class _ContDist:
    """Gaussian distribution with clipping to [lower, upper]."""
    mean: float
    std: float
    lower: float
    upper: float


@dataclass
class _DiscDist:
    """Categorical distribution over discrete values."""
    values: list[Any]
    probs: list[float]  # length == len(values), sums to 1.0


@dataclass
class GenerationResult:
    """Score and hyperparameters for one sampled candidate."""
    params: dict[str, Any]
    score: float  # lower is better (we minimise MAE)


class CEMOptimizer:
    """
    Cross-Entropy Method optimizer for mixed continuous/discrete hyperparameters.

    Parameters
    ----------
    param_configs : list[ParameterConfig]
        Definitions of each hyperparameter to optimise.
    elite_fraction : float
        Fraction of candidates kept as the elite set each generation
        (default 0.2 → top 20 %).
    min_std : float
        Floor on the Gaussian standard deviation to prevent premature
        collapse (default 1e-4).
    rng : np.random.Generator
        Random state for reproducibility.
    """

    def __init__(
        self,
        param_configs: list[ParameterConfig],
        elite_fraction: float = 0.2,
        min_std: float = 1e-4,
        rng: Optional[np.random.Generator] = None,
    ):
        self.param_configs = param_configs
        self.elite_fraction = elite_fraction
        self.min_std = min_std
        self.rng = rng if rng is not None else np.random.default_rng()

        self._cont_dists: dict[str, _ContDist] = {}
        self._disc_dists: dict[str, _DiscDist] = {}
        self._history: list[GenerationResult] = []

        self._init_distributions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        """Draw ``batch_size`` hyperparameter sets from the current distribution."""
        return [self._sample_one() for _ in range(batch_size)]

    def update(self, results: list[GenerationResult]) -> None:
        """
        Update the sampling distribution from the elite set.

        ``results`` should contain one entry per candidate produced by
        :meth:`sample`.  The top ``elite_fraction * len(results)`` by score
        (lower is better) determine the new distribution parameters.
        """
        if not results:
            return

        sorted_results = sorted(results, key=lambda r: r.score)
        k = max(1, int(len(sorted_results) * self.elite_fraction))
        elite = sorted_results[:k]

        for cfg in self.param_configs:
            if cfg.ptype is ParamType.CONTINUOUS:
                self._update_continuous(cfg, elite)
            else:
                self._update_discrete(cfg, elite)

        self._history.extend(results)

    def best_params(self) -> dict[str, Any]:
        """Return the hyperparameter set with the lowest score seen so far."""
        if not self._history:
            return {}
        best = min(self._history, key=lambda r: r.score)
        return best.params

    def best_score(self) -> float:
        """Return the best (lowest) score seen so far."""
        if not self._history:
            return float("inf")
        return min(r.score for r in self._history)

    def summary(self) -> str:
        """Human-readable summary of optimisation progress."""
        lines = [f"CEM optimiser ({len(self.param_configs)} params, "
                 f"elite={self.elite_fraction:.0%})"]
        lines.append(f"Samples collected : {len(self._history)}")
        if self._history:
            lines.append(f"Best score so far: {self.best_score():.4f}")
            lines.append(f"Best params      : {self.best_params()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_distributions(self) -> None:
        for cfg in self.param_configs:
            if cfg.ptype is ParamType.CONTINUOUS:
                mean = cfg.init_mean if cfg.init_mean is not None \
                    else 0.5 * (cfg.lower + cfg.upper)
                std = cfg.init_std if cfg.init_std is not None \
                    else 0.5 * (cfg.upper - cfg.lower)
                self._cont_dists[cfg.name] = _ContDist(
                    mean=mean, std=std,
                    lower=cfg.lower, upper=cfg.upper,
                )
            else:
                probs = cfg.init_probs if cfg.init_probs is not None \
                    else [1.0 / len(cfg.values)] * len(cfg.values)
                self._disc_dists[cfg.name] = _DiscDist(
                    values=list(cfg.values), probs=list(probs),
                )

    def _sample_one(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for cfg in self.param_configs:
            if cfg.ptype is ParamType.CONTINUOUS:
                dist = self._cont_dists[cfg.name]
                val = self.rng.normal(dist.mean, dist.std)
                val = float(np.clip(val, dist.lower, dist.upper))
            else:
                dist = self._disc_dists[cfg.name]
                idx = int(self.rng.choice(len(dist.values), p=dist.probs))
                val = dist.values[idx]
            params[cfg.name] = val
        return params

    def _update_continuous(self, cfg: ParameterConfig, elite: list[GenerationResult]) -> None:
        vals = np.array([r.params[cfg.name] for r in elite], dtype=float)
        mean = float(vals.mean())
        std = float(max(vals.std(), self.min_std))
        self._cont_dists[cfg.name] = _ContDist(
            mean=mean, std=std,
            lower=cfg.lower, upper=cfg.upper,
        )

    def _update_discrete(self, cfg: ParameterConfig, elite: list[GenerationResult]) -> None:
        vals = [r.params[cfg.name] for r in elite]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        total = len(vals)
        probs = [counts.get(v, 0) / total for v in cfg.values]
        self._disc_dists[cfg.name] = _DiscDist(
            values=list(cfg.values), probs=probs,
        )


# =============================================================================
# Predefined hyperparameter spaces (mirrors the grid search in the training script)
# =============================================================================

def make_xgb_param_configs() -> list[ParameterConfig]:
    """Return the XGBoost hyperparameter space used by the project."""
    return [
        ParameterConfig("n_estimators", ParamType.DISCRETE,
                        values=[100, 300, 500, 750, 1000]),
        ParameterConfig("max_depth", ParamType.DISCRETE,
                        values=[3, 4, 5, 6, 7, 8]),
        ParameterConfig("learning_rate", ParamType.CONTINUOUS,
                        lower=0.001, upper=0.5),
        ParameterConfig("subsample", ParamType.CONTINUOUS,
                        lower=0.5, upper=1.0),
        ParameterConfig("colsample_bytree", ParamType.CONTINUOUS,
                        lower=0.5, upper=1.0),
    ]


def make_mlp_param_configs() -> list[ParameterConfig]:
    """Return the MLP hyperparameter space used by the project."""
    # Hidden layers encoded as three discrete size choices; 0 means "no layer".
    layer_sizes = [0, 16, 32, 64, 128]
    return [
        ParameterConfig("layer1", ParamType.DISCRETE, values=layer_sizes),
        ParameterConfig("layer2", ParamType.DISCRETE, values=layer_sizes),
        ParameterConfig("layer3", ParamType.DISCRETE, values=layer_sizes),
        ParameterConfig("epochs", ParamType.DISCRETE,
                        values=[25, 50, 100, 150, 200]),
        ParameterConfig("lr", ParamType.CONTINUOUS,
                        lower=1e-5, upper=1e-2),
    ]


def make_siamese_param_configs() -> list[ParameterConfig]:
    """Return the Siamese-network hyperparameter space used by the project."""
    layer_sizes = [0, 32, 64, 128]
    return [
        ParameterConfig("layer1", ParamType.DISCRETE, values=layer_sizes),
        ParameterConfig("layer2", ParamType.DISCRETE, values=layer_sizes),
        ParameterConfig("layer3", ParamType.DISCRETE, values=layer_sizes),
        ParameterConfig("epochs", ParamType.DISCRETE,
                        values=[50, 100, 150, 200]),
        ParameterConfig("lr", ParamType.CONTINUOUS,
                        lower=1e-5, upper=1e-2),
        ParameterConfig("dropout", ParamType.CONTINUOUS,
                        lower=0.0, upper=0.3),
    ]


# =============================================================================
# Helpers: translate CEM samples into the kwargs expected by the model trainers
# =============================================================================

def xgb_sample_to_kwargs(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a CEM sample to :func:`train_xgboost` kwargs."""
    return {
        "n_estimators": int(sample["n_estimators"]),
        "max_depth": int(sample["max_depth"]),
        "learning_rate": float(sample["learning_rate"]),
        "subsample": float(sample["subsample"]),
        "colsample_bytree": float(sample["colsample_bytree"]),
    }


def mlp_sample_to_kwargs(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a CEM sample to :func:`train_mlp` kwargs."""
    sizes = [s for s in [sample["layer1"], sample["layer2"], sample["layer3"]] if s > 0]
    return {
        "hidden_layers": tuple(sizes) if sizes else (64, 32),
        "epochs": int(sample["epochs"]),
        "lr": float(sample["lr"]),
    }


def siamese_sample_to_kwargs(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a CEM sample to :func:`train_siamese` kwargs."""
    sizes = [s for s in [sample["layer1"], sample["layer2"], sample["layer3"]] if s > 0]
    return {
        "hidden_layers": tuple(sizes) if sizes else (128, 64),
        "epochs": int(sample["epochs"]),
        "lr": float(sample["lr"]),
        "dropout": float(sample["dropout"]),
    }


# =============================================================================
# Standalone convenience function
# =============================================================================

def cem_search(
    train_fn: Callable,
    param_configs: list[ParameterConfig],
    evaluate_fn: Callable,
    x_train, y_train, x_test, y_test,
    generations: int = 10,
    batch_size: int = 10,
    elite_fraction: float = 0.2,
    seed: Optional[int] = None,
    **train_kwargs,
) -> dict:
    """
    Run CEM hyperparameter search and return results.

    Parameters
    ----------
    train_fn : callable
        ``(x_train, y_train, **kwargs) -> model``.
    param_configs : list[ParameterConfig]
        Hyperparameter definitions.
    evaluate_fn : callable
        ``(model, x_test, y_test, **kwargs) -> dict``  (must contain ``"MAE"``).
    x_train, y_train, x_test, y_test : torch.Tensor
        Training and test features / targets.
    generations : int
        Number of CEM generations.
    batch_size : int
        Candidates per generation.
    elite_fraction : float
        Fraction kept as elite each generation.
    seed : int, optional
        Random seed.
    **train_kwargs
        Extra kwargs forwarded to ``train_fn`` (e.g. ``device``, ``batch_size``).

    Returns
    -------
    dict
        ``{"best_params": ..., "best_mae": ..., "all_results": [...]}``
    """
    optimizer = CEMOptimizer(param_configs, elite_fraction=elite_fraction,
                             rng=np.random.default_rng(seed))
    all_results: list[dict] = []

    for gen in range(generations):
        samples = optimizer.sample(batch_size=batch_size)
        gen_scores: list[GenerationResult] = []

        for i, sample in enumerate(samples):
            kwargs = {**train_kwargs, **sample}
            try:
                model = train_fn(x_train, y_train, **kwargs)
                metrics = evaluate_fn(model, x_test, y_test)
                mae = metrics["MAE"]
                print(f"    [{gen+1}/{generations}] "
                      f"cand {i+1}/{batch_size}: MAE={mae:.4f}  "
                      f"{{k:v for k,v in sample.items()}}", flush=True)
                gen_scores.append(GenerationResult(params=sample, score=mae))
                all_results.append({
                    "params": sample,
                    "MAE": mae,
                    "MSE": metrics.get("MSE", float("nan")),
                    "R2": metrics.get("R2", float("nan")),
                    "Pearson": metrics.get("Pearson", float("nan")),
                    "Spearman": metrics.get("Spearman", float("nan")),
                })
            except Exception as exc:
                print(f"    [{gen+1}/{generations}] "
                      f"cand {i+1}/{batch_size}: ERROR {exc}", flush=True)
                gen_scores.append(GenerationResult(params=sample, score=float("inf")))

        optimizer.update(gen_scores)
        print(f"    Generation {gen+1} complete. "
              f"Best MAE so far: {optimizer.best_score():.4f}", flush=True)

    return {
        "best_params": optimizer.best_params(),
        "best_mae": optimizer.best_score(),
        "all_results": all_results,
    }
