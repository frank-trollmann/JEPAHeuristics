"""
Cross-Entropy Method (CEM) hyperparameter search for JEPA heuristic models.

Mirrors ``scripts/euclid_distance_ml_model_training.py`` but replaces grid
search with the adaptive CEM optimiser from ``utils.cem_optimizer``.

Hyperparameter search is evaluated on the **validation** set; the **test** set
is reserved for the final evaluation only.

Usage::

    # XGBoost, 8 generations, 15 candidates per gen
    python -m scripts.cem_hyperparameter_search \\
        --input data/2026-11-10\\ 17-44-33_10000_heuristic_dataset.parquet \\
        --model xgb --generations 8 --batch_size 15

    # MLP with custom settings
    python -m scripts.cem_hyperparameter_search \\
        --input data/...parquet --model mlp \\
        --generations 6 --batch_size 12 --epochs 100

    # All models, default CEM settings
    python -m scripts.cem_hyperparameter_search \\
        --input data/...parquet --model all
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

from utils.cem_optimizer import (
    CEMOptimizer,
    GenerationResult,
    make_xgb_param_configs,
    make_mlp_param_configs,
    xgb_sample_to_kwargs,
    mlp_sample_to_kwargs,
)
from utils.euclid_distance_ml_model_training_utils import (
    LinearRegressor,
    convert_pickle_to_parquet,
    evaluate_model,
    load_features,
    load_split,
    resolve_input_path,
    split_three_way,
    train_mlp,
    train_xgboost,
)


# =============================================================================
# CEM search wrappers (one per model type)
# =============================================================================

def cem_search_xgboost(x_train, y_train, x_val, y_val,
                       generations: int = 10, batch_size: int = 10,
                       elite_fraction: float = 0.2, device: str = "cuda",
                       n_jobs: int = 1, max_rows: int = 0,
                       seed: int | None = None) -> dict:
    """Run CEM hyperparameter search for XGBoost. Evaluates on val set."""
    param_configs = make_xgb_param_configs()
    optimizer = CEMOptimizer(param_configs, elite_fraction=elite_fraction,
                             rng=np.random.default_rng(seed))
    all_results: list[dict] = []

    for gen in range(generations):
        samples = optimizer.sample(batch_size=batch_size)
        gen_scores: list[GenerationResult] = []

        for i, sample in enumerate(samples):
            kwargs = xgb_sample_to_kwargs(sample)
            kwargs["device"] = device
            kwargs["n_jobs"] = n_jobs
            kwargs["max_rows"] = max_rows
            try:
                model = train_xgboost(x_train, y_train, **kwargs)
                metrics = evaluate_model(model, x_val, y_val)
                mae = metrics["MAE"]
                print(f"    [{gen+1}/{generations}] "
                      f"cand {i+1}/{batch_size}: MAE={mae:.4f}  "
                      f"{{k:v for k,v in sample.items()}}", flush=True)
                gen_scores.append(GenerationResult(params=sample, score=mae))
                all_results.append({
                    "params": sample,
                    "MAE": mae,
                    "MSE": metrics["MSE"],
                    "R2": metrics["R2"],
                    "Pearson": metrics["Pearson"],
                    "Spearman": metrics["Spearman"],
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
        "all_results": all_results,
    }


def cem_search_mlp(x_train, y_train, x_val, y_val,
                   generations: int = 10, batch_size: int = 10,
                   elite_fraction: float = 0.2,
                   base_epochs: int = 100, base_batch_size: int = 4096,
                   seed: int | None = None) -> dict:
    """Run CEM hyperparameter search for MLP. Evaluates on val set."""
    param_configs = make_mlp_param_configs()
    optimizer = CEMOptimizer(param_configs, elite_fraction=elite_fraction,
                             rng=np.random.default_rng(seed))
    all_results: list[dict] = []

    for gen in range(generations):
        samples = optimizer.sample(batch_size=batch_size)
        gen_scores: list[GenerationResult] = []

        for i, sample in enumerate(samples):
            kwargs = mlp_sample_to_kwargs(sample)
            kwargs["epochs"] = sample.get("epochs", base_epochs)
            kwargs["batch_size"] = base_batch_size
            try:
                model = train_mlp(x_train, y_train, **kwargs)
                metrics = evaluate_model(model, x_val, y_val)
                mae = metrics["MAE"]
                print(f"    [{gen+1}/{generations}] "
                      f"cand {i+1}/{batch_size}: MAE={mae:.4f}  "
                      f"{{k:v for k,v in sample.items()}}", flush=True)
                gen_scores.append(GenerationResult(params=sample, score=mae))
                all_results.append({
                    "params": sample,
                    "MAE": mae,
                    "MSE": metrics["MSE"],
                    "R2": metrics["R2"],
                    "Pearson": metrics["Pearson"],
                    "Spearman": metrics["Spearman"],
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
        "all_results": all_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CEM Hyperparameter Search for JEPA Heuristic Models"
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the source .pkl file"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["linear", "xgb", "mlp", "all"],
        help="Which model to tune"
    )
    parser.add_argument(
        "--generations", type=int, default=10,
        help="Number of CEM generations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Candidates per generation"
    )
    parser.add_argument(
        "--elite_fraction", type=float, default=0.2,
        help="Fraction of candidates kept as elite (default 0.2)"
    )
    parser.add_argument(
        "--train-size", type=float, default=0.7,
        help="Training set proportion (default 0.7)"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15,
        help="Validation set proportion (default 0.15)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Base epochs for MLP (overridden by CEM per candidate)"
    )
    parser.add_argument(
        "--batch_size_train", type=int, default=4096,
        help="Batch size for MLP training"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Training device"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional seed for reproducible splits and CEM"
    )
    parser.add_argument(
        "--xgb_max_rows", type=int, default=0,
        help="Train XGBoost on at most N randomly sampled rows (0 = all)"
    )
    parser.add_argument(
        "--enhanced", action="store_true",
        help="Use enhanced features"
    )

    args = parser.parse_args()

    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be < 1.0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    n_jobs = 1 if sys.platform == "darwin" else -1

    print("=" * 60)
    print("CEM Hyperparameter Search")
    print("=" * 60)
    print(f"Device:        {device}")
    print(f"Generations:   {args.generations}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Elite frac:    {args.elite_fraction}")
    print(f"Train size:    {args.train_size}")
    print(f"Val size:      {args.val_size}")
    print(f"Test size:     {1.0 - args.train_size - args.val_size}")

    # 1. Resolve input (file or split directory)
    is_split_dir = os.path.isdir(args.input) and os.path.exists(
        os.path.join(args.input, "split_config.json")
    )

    if is_split_dir:
        print(f"Loading pre-split data from: {args.input}")
        print("Loading data...")
        x_train, x_val, x_test, y_train, y_val, y_test = load_split(
            args.input, device=device
        )
        print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    else:
        parquet_path = resolve_input_path(args.input)
        if os.path.exists(parquet_path):
            print(f"Using existing parquet: {parquet_path}")
        else:
            parquet_path = convert_pickle_to_parquet(args.input)
            print(f"Data converted to: {parquet_path}")

        print("Loading data...")
        X, y = load_features(parquet_path, target_col="heuristic",
                             device=device)
        print(f"Features: {tuple(X.shape)} "
              f"({X.numel() * X.element_size() / 1e6:.1f} MB on {X.device})")

        print("Splitting data into train/val/test sets...")
        x_train, x_val, x_test, y_train, y_val, y_test = split_three_way(
            X, y, train_size=args.train_size, val_size=args.val_size, seed=args.seed
        )
        del X, y
        print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # 3. Run CEM search per model (on val set), final eval on test set
    results_summary = {}
    tune_results = {}

    if args.model in ["xgb", "all"]:
        print("\n" + "=" * 60)
        print("XGBoost CEM Search (val set)")
        print("=" * 60)
        xgb_res = cem_search_xgboost(
            x_train, y_train, x_val, y_val,
            generations=args.generations,
            batch_size=args.batch_size,
            elite_fraction=args.elite_fraction,
            device=device, n_jobs=n_jobs,
            max_rows=args.xgb_max_rows,
            seed=args.seed,
        )
        # Train final model on train+val, evaluate on test
        best_kwargs = xgb_sample_to_kwargs(xgb_res["best_params"])
        best_kwargs.update({"device": device, "n_jobs": n_jobs, "max_rows": args.xgb_max_rows})
        model_xgb = train_xgboost(
            torch.cat([x_train, x_val], dim=0),
            torch.cat([y_train, y_val], dim=0),
            **best_kwargs,
        )
        results_summary["XGBoost"] = evaluate_model(model_xgb, x_test, y_test)
        print(f"XGBoost Final (test): MAE={results_summary['XGBoost']['MAE']:.4f}, "
              f"R2={results_summary['XGBoost']['R2']:.4f}")
        tune_results["XGBoost"] = {
            "best_params": xgb_res["best_params"],
            "all_results": xgb_res["all_results"],
        }

    if args.model in ["mlp", "all"]:
        print("\n" + "=" * 60)
        print("MLP CEM Search (val set)")
        print("=" * 60)
        mlp_res = cem_search_mlp(
            x_train, y_train, x_val, y_val,
            generations=args.generations,
            batch_size=args.batch_size,
            elite_fraction=args.elite_fraction,
            base_epochs=args.epochs,
            base_batch_size=args.batch_size_train,
            seed=args.seed,
        )
        best_kwargs = mlp_sample_to_kwargs(mlp_res["best_params"])
        best_kwargs.update({"epochs": args.epochs, "batch_size": args.batch_size_train})
        model_mlp = train_mlp(
            torch.cat([x_train, x_val], dim=0),
            torch.cat([y_train, y_val], dim=0),
            **best_kwargs,
        )
        results_summary["MLP"] = evaluate_model(model_mlp, x_test, y_test,
                                                batch_size=args.batch_size_train)
        print(f"MLP Final (test): MAE={results_summary['MLP']['MAE']:.4f}, "
              f"R2={results_summary['MLP']['R2']:.4f}")
        tune_results["MLP"] = {
            "best_params": mlp_res["best_params"],
            "all_results": mlp_res["all_results"],
        }

    if args.model in ["linear", "all"]:
        print("\nTraining Linear Regression (baseline, no CEM)...")
        model_lr = LinearRegressor().fit(x_train, y_train)
        results_summary["Linear"] = evaluate_model(model_lr, x_test, y_test)
        print(f"Linear: MAE={results_summary['Linear']['MAE']:.4f}, "
              f"R2={results_summary['Linear']['R2']:.4f}")

    # 4. Summary
    print("\n" + "=" * 30)
    print("FINAL SUMMARY (test set)")
    print("=" * 30)
    for model, metrics in results_summary.items():
        print(
            f"{model}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}, "
            f"Pearson={metrics['Pearson']:.4f}, Spearman={metrics['Spearman']:.4f}"
        )

    # 5. Save
    results_dir = "data/MLresults"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    main_save_path = os.path.join(results_dir, f"{timestamp}_cem_results.json")
    with open(main_save_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nResults saved to: {main_save_path}")

    if tune_results:
        tune_save_path = os.path.join(results_dir, f"{timestamp}_cem_tuning.json")
        with open(tune_save_path, "w") as f:
            json.dump(tune_results, f, indent=4)
        print(f"Tuning saved to: {tune_save_path}")


if __name__ == "__main__":
    main()
