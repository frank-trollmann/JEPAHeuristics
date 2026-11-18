import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import torch
import xgboost as xgb

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
    Standardizer,
    convert_pickle_to_parquet,
    evaluate_model,
    load_features,
    load_split,
    resolve_input_path,
    save_results,
    split_data,
    split_three_way,
    train_mlp,
    train_xgboost,
)


# Fixed training seed: same for all candidates to enable fair hyperparameter comparison.
# Only hyperparameters (hidden_layers, epochs, lr) vary between candidates.
TRAINING_SEED = 0

# =============================================================================
# Hyperparameter Grids
# =============================================================================

XGB_PARAM_GRIDS = [
    {"n_estimators": [300, 500, 1000], "max_depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1]},
    {"n_estimators": [500], "max_depth": [6], "learning_rate": [0.05], "subsample": [0.7, 0.8, 0.9], "colsample_bytree": [0.7, 0.8, 0.9]},
]

MLP_PARAM_GRIDS = [
    {"hidden_layers": [(32, 16), (64, 32), (128, 64)], "epochs": [50, 100, 200], "lr": [1e-4, 1e-3, 5e-3]},
    {"hidden_layers": [(64, 64, 32), (128, 128, 64)], "epochs": [100, 200], "lr": [1e-3, 5e-3]},
]


# =============================================================================
# Helper: Generate parameter combinations
# =============================================================================

def generate_combinations(grids):
    """Generate all parameter combinations from a list of grids."""
    all_combinations = []
    for grid in grids:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        for v in product(*values):
            all_combinations.append(dict(zip(keys, v)))
    return all_combinations


# =============================================================================
# Hyperparameter Search Functions — Grid Search
# =============================================================================

def grid_search_xgboost(x_train, y_train, x_val, y_val, param_grids, device, n_jobs=1, max_rows=0):
    """Grid search for XGBoost hyperparameters. Evaluates on val set."""
    best_mae = float("inf")
    best_params = None
    best_model = None
    all_results = []

    print("\n" + "=" * 60)
    print("XGBoost Hyperparameter Tuning (grid search, val set)")
    print("=" * 60)

    combinations = generate_combinations(param_grids)
    total = len(combinations)

    for i, params in enumerate(combinations):
        print(f"  [{i+1}/{total}] Training with params: {params}", end="")

        try:
            model = train_xgboost(
                x_train, y_train,
                device=device,
                max_rows=max_rows,
                **params
            )
            metrics = evaluate_model(model, x_val, y_val)
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            print(f" → MAE={mae:.4f}, R2={r2:.4f}")

            all_results.append({
                "params": params,
                "MAE": mae,
                "MSE": metrics["MSE"],
                "R2": r2,
                "Pearson": metrics["Pearson"],
                "Spearman": metrics["Spearman"],
            })

            if mae < best_mae:
                best_mae = mae
                best_params = params
                best_model = model
                print(f"    ★ New best! (saved)")

        except Exception as e:
            print(f" → ERROR: {e}")

    print(f"\nBest XGBoost params: {best_params}")
    print(f"Best XGBoost MAE: {best_mae:.4f}")

    return best_model, best_params, all_results


def grid_search_mlp(x_train, y_train, x_val, y_val, param_grids):
    """Grid search for MLP hyperparameters. Evaluates on val set.

    Each candidate is trained with per-epoch validation tracking; the returned
    model is the checkpoint with the lowest validation MAE, not the final
    epoch state.
    """
    import time
    best_mae = float("inf")
    best_params = None
    best_model = None
    best_history = None
    all_results = []

    print("\n" + "=" * 60)
    print("MLP Hyperparameter Tuning (grid search, val set)")
    print("=" * 60)

    combinations = generate_combinations(param_grids)
    total = len(combinations)
    t_start = time.time()

    for i, params in enumerate(combinations):
        t_cand = time.time()
        print(f"  [{i+1}/{total}] Training with params: {params}", end="")

        try:
            torch.manual_seed(TRAINING_SEED)
            result = train_mlp(
                x_train, y_train,
                x_val=x_val, y_val=y_val,
                **params
            )
            # train_mlp with val returns (model, history, best_mae)
            if isinstance(result, tuple):
                model, history, cand_best_mae = result
            else:
                model = result
                history = None
                cand_best_mae = float("inf")

            metrics = evaluate_model(model, x_val, y_val)
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            print(f" → MAE={mae:.4f}, R2={r2:.4f}  ({time.time()-t_cand:.1f}s)")

            all_results.append({
                "params": params,
                "MAE": mae,
                "MSE": metrics["MSE"],
                "R2": r2,
                "Pearson": metrics["Pearson"],
                "Spearman": metrics["Spearman"],
                "elapsed_s": time.time() - t_cand,
            })

            if mae < best_mae:
                best_mae = mae
                best_params = params
                best_model = model
                best_history = history
                print(f"    ★ New best! (saved)")

        except Exception as e:
            print(f" → ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\nBest MLP params: {best_params}")
    print(f"Best MLP MAE: {best_mae:.4f}")
    print(f"Search elapsed: {elapsed:.1f}s")

    return best_model, best_params, all_results, best_history


# =============================================================================
# Hyperparameter Search Functions — CEM
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
    """Run CEM hyperparameter search for MLP. Evaluates on val set.

    Candidates are trained on the training split only with per-epoch
    validation tracking; the returned model is the checkpoint with the lowest
    validation MAE.  No refit on train+val is performed.
    """
    import time
    param_configs = make_mlp_param_configs()
    optimizer = CEMOptimizer(param_configs, elite_fraction=elite_fraction,
                             rng=np.random.default_rng(seed))
    all_results: list[dict] = []
    best_model = None
    best_mae = float("inf")
    best_params = None
    best_history = None
    t_start = time.time()

    for gen in range(generations):
        samples = optimizer.sample(batch_size=batch_size)
        gen_scores: list[GenerationResult] = []

        for i, sample in enumerate(samples):
            kwargs = mlp_sample_to_kwargs(sample)
            kwargs["epochs"] = sample.get("epochs", base_epochs)
            kwargs["batch_size"] = base_batch_size
            t_cand = time.time()
            try:
                torch.manual_seed(TRAINING_SEED)
                result = train_mlp(x_train, y_train, x_val=x_val, y_val=y_val,
                                   **kwargs)
                if isinstance(result, tuple):
                    model, history, cand_best_mae = result
                else:
                    model = result
                    history = None
                    cand_best_mae = float("inf")
                metrics = evaluate_model(model, x_val, y_val)
                mae = metrics["MAE"]
                print(f"    [{gen+1}/{generations}] "
                      f"cand {i+1}/{batch_size}: MAE={mae:.4f}  "
                      f"({time.time()-t_cand:.1f}s)", flush=True)
                gen_scores.append(GenerationResult(params=sample, score=mae))
                all_results.append({
                    "params": sample,
                    "MAE": mae,
                    "MSE": metrics["MSE"],
                    "R2": metrics["R2"],
                    "Pearson": metrics["Pearson"],
                    "Spearman": metrics["Spearman"],
                    "elapsed_s": time.time() - t_cand,
                })
                if mae < best_mae:
                    best_mae = mae
                    best_params = mlp_sample_to_kwargs(sample)
                    best_model = model
                    best_history = history
            except Exception as exc:
                print(f"    [{gen+1}/{generations}] "
                      f"cand {i+1}/{batch_size}: ERROR {exc}", flush=True)
                gen_scores.append(GenerationResult(params=sample, score=float("inf")))

        optimizer.update(gen_scores)
        print(f"    Generation {gen+1} complete. "
              f"Best MAE so far: {optimizer.best_score():.4f}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nCEM search elapsed: {elapsed:.1f}s")
    print(f"Best CEM params: {best_params}")
    print(f"Best CEM MAE: {best_mae:.4f}")

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_mae": best_mae,
        "best_history": best_history,
        "all_results": all_results,
    }


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Accelerated ML Training for JEPA Heuristics (torch + XGBoost)"
    )

    parser.add_argument(
        "--input",
        type=str, required=True,
        help="Path to the source .pkl file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["linear", "xgb", "mlp", "all"],
        help="Which model to train"
    )
    parser.add_argument(
        "--train-size",
        type=float, default=0.7,
        help="Training set proportion (default 0.7)"
    )
    parser.add_argument(
        "--val-size",
        type=float, default=0.15,
        help="Validation set proportion (default 0.15)"
    )
    parser.add_argument(
        "--epochs",
        type=int, default=100,
        help="Epochs for MLP"
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=4096,
        help="Batch size for MLP training and inference"
    )
    parser.add_argument(
        "--device",
        type=str, default=None,
        choices=["cuda", "cpu"],
        help="Training device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="Optional seed for a reproducible split"
    )
    parser.add_argument(
        "--xgb_max_rows",
        type=int, default=0,
        help="Train XGBoost on at most N randomly sampled rows (0 = all rows)"
    )
    parser.add_argument(
        "--tune",
        type=str,
        default=None,
        choices=["grid", "cem"],
        help="Enable hyperparameter tuning (grid search or CEM)"
    )

    args = parser.parse_args()

    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be < 1.0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    n_jobs = 1 if sys.platform == "darwin" else -1

    print("Starting GPU Training Pipeline...")
    print(f"Device:     {device}")
    if device == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"Train size: {args.train_size}")
    print(f"Val size:   {args.val_size}")
    print(f"Test size:  {1.0 - args.train_size - args.val_size}")
    if args.tune:
        print(f"Tuning:     {args.tune}")

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
            print(f"Using existing parquet file: {parquet_path}")
        else:
            parquet_path = convert_pickle_to_parquet(args.input)
            print(f"Data converted and saved to: {parquet_path}")

        print("Loading data...")
        X, y = load_features(parquet_path, target_col="heuristic", device=device)
        print(f"Features: {tuple(X.shape)} "
              f"({X.numel() * X.element_size() / 1e6:.1f} MB on {X.device})")

        print("Splitting data into train/val/test sets...")
        x_train, x_val, x_test, y_train, y_val, y_test = split_three_way(
            X, y, train_size=args.train_size, val_size=args.val_size, seed=args.seed
        )
        del X, y
        print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # 4. Model Training & Evaluation
    results_summary = {}
    tune_results = {}

    if args.model in ["linear", "all"]:
        print("\nTraining Linear Regression...")
        model_lr = LinearRegressor().fit(x_train, y_train)
        results_summary["Linear"] = evaluate_model(model_lr, x_test, y_test)
        print(f"Linear Results: {results_summary['Linear']}")

    if args.model in ["xgb", "all"]:
        if args.tune == "grid":
            print("\n" + "=" * 60)
            print("XGBoost Hyperparameter Tuning (grid search)")
            print("=" * 60)
            model_xgb, best_params, all_results = grid_search_xgboost(
                x_train, y_train, x_val, y_val,
                XGB_PARAM_GRIDS, device, n_jobs=n_jobs,
                max_rows=args.xgb_max_rows
            )
            tune_results["XGBoost"] = {
                "best_params": best_params,
                "all_results": all_results,
            }
        elif args.tune == "cem":
            print("\n" + "=" * 60)
            print("XGBoost Hyperparameter Tuning (CEM)")
            print("=" * 60)
            xgb_res = cem_search_xgboost(
                x_train, y_train, x_val, y_val,
                device=device, n_jobs=n_jobs,
                max_rows=args.xgb_max_rows,
                seed=args.seed,
            )
            tune_results["XGBoost"] = {
                "best_params": xgb_res["best_params"],
                "all_results": xgb_res["all_results"],
            }
            model_xgb = train_xgboost(
                torch.cat([x_train, x_val], dim=0),
                torch.cat([y_train, y_val], dim=0),
                device=device, n_jobs=n_jobs, max_rows=args.xgb_max_rows,
                **xgb_sample_to_kwargs(xgb_res["best_params"]),
            )
        else:
            print("\nTraining XGBoost...")
            model_xgb = train_xgboost(
                x_train, y_train, device=device, max_rows=args.xgb_max_rows
            )
        results_summary["XGBoost"] = evaluate_model(
            model_xgb, x_test, y_test, batch_size=args.batch_size
        )
        print(f"XGBoost Results: {results_summary['XGBoost']}")

    if args.model in ["mlp", "all"]:
        if args.tune == "grid":
            print("\n" + "=" * 60)
            print("MLP Hyperparameter Tuning (grid search)")
            print("=" * 60)
            model_mlp, best_params, all_results = grid_search_mlp(
                x_train, y_train, x_val, y_val,
                MLP_PARAM_GRIDS
            )
            tune_results["MLP"] = {
                "best_params": best_params,
                "all_results": all_results,
            }
        elif args.tune == "cem":
            print("\n" + "=" * 60)
            print("MLP Hyperparameter Tuning (CEM)")
            print("=" * 60)
            mlp_res = cem_search_mlp(
                x_train, y_train, x_val, y_val,
                base_epochs=args.epochs,
                base_batch_size=args.batch_size,
                seed=args.seed,
            )
            tune_results["MLP"] = {
                "best_params": mlp_res["best_params"],
                "all_results": mlp_res["all_results"],
            }
            best_kwargs = mlp_sample_to_kwargs(mlp_res["best_params"])
            best_kwargs.update({"epochs": args.epochs, "batch_size": args.batch_size})
            model_mlp = train_mlp(
                torch.cat([x_train, x_val], dim=0),
                torch.cat([y_train, y_val], dim=0),
                **best_kwargs,
            )
        else:
            print("\nTraining MLP Neural Net...")
            model_mlp = train_mlp(
                x_train,
                y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        results_summary["MLP"] = evaluate_model(
            model_mlp, x_test, y_test, batch_size=args.batch_size
        )
        print(f"MLP Results: {results_summary['MLP']}")

    print("\n" + "=" * 30)
    print("FINAL SUMMARY (test set)")
    print("=" * 30)
    for model, metrics in results_summary.items():
        print(
            f"{model}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}, "
            f"Pearson={metrics['Pearson']:.4f}, Spearman={metrics['Spearman']:.4f}"
        )

    # Save results
    results_dir = "data/MLresults"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save main results
    main_save_path = os.path.join(results_dir, f"{timestamp}_results.json")
    with open(main_save_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nMain results saved to: {main_save_path}")

    # Save tuning results
    if tune_results:
        tune_save_path = os.path.join(results_dir, f"{timestamp}_tuning.json")
        with open(tune_save_path, "w") as f:
            json.dump(tune_results, f, indent=4)
        print(f"Tuning results saved to: {tune_save_path}")


if __name__ == "__main__":
    main()
