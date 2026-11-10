import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import torch
import xgboost as xgb

from utils.euclid_distance_ml_model_training_utils import (
    LinearRegressor,
    Standardizer,
    convert_pickle_to_parquet,
    evaluate_model,
    load_features,
    save_results,
    split_data,
    train_mlp,
    train_xgboost,
)


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
# Hyperparameter Search Functions
# =============================================================================

def grid_search_xgboost(x_train, y_train, x_test, y_test, param_grids, device, n_jobs=1, max_rows=0):
    """Grid search for XGBoost hyperparameters."""
    best_mae = float("inf")
    best_params = None
    best_model = None
    all_results = []

    print("\n" + "=" * 60)
    print("XGBoost Hyperparameter Tuning")
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
            metrics = evaluate_model(model, x_test, y_test)
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            print(f" → MAE={mae:.4f}, R2={r2:.4f}")

            all_results.append({
                "params": params,
                "MAE": mae,
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


def grid_search_mlp(x_train, y_train, x_test, y_test, param_grids):
    """Grid search for MLP hyperparameters."""
    best_mae = float("inf")
    best_params = None
    best_model = None
    all_results = []

    print("\n" + "=" * 60)
    print("MLP Hyperparameter Tuning")
    print("=" * 60)

    combinations = generate_combinations(param_grids)
    total = len(combinations)

    for i, params in enumerate(combinations):
        print(f"  [{i+1}/{total}] Training with params: {params}", end="")

        try:
            model = train_mlp(
                x_train, y_train,
                **params
            )
            metrics = evaluate_model(model, x_test, y_test)
            mae = metrics["MAE"]
            r2 = metrics["R2"]

            print(f" → MAE={mae:.4f}, R2={r2:.4f}")

            all_results.append({
                "params": params,
                "MAE": mae,
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

    print(f"\nBest MLP params: {best_params}")
    print(f"Best MLP MAE: {best_mae:.4f}")

    return best_model, best_params, all_results


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
        "--train_size",
        type=float,
        default=0.8,
        help="Training set proportion"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        help="Optional seed for a reproducible train/test split"
    )
    parser.add_argument(
        "--xgb_max_rows",
        type=int, default=0,
        help="Train XGBoost on at most N randomly sampled rows (0 = all rows)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning for XGBoost and MLP"
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    n_jobs = 1 if sys.platform == "darwin" else -1

    print("Starting GPU Training Pipeline...")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    if args.tune:
        print("Hyperparameter tuning: ENABLED")

    # 1. Convert Pickle to Parquet (if not already done)
    parquet_path = args.input.replace(".pkl", ".parquet")
    if os.path.exists(parquet_path):
        print(f"Using existing parquet file: {parquet_path}")
    else:
        parquet_path = convert_pickle_to_parquet(args.input)
        print(f"Data converted and saved to: {parquet_path}")

    # 2. Load features as tensors on the target device
    print("Loading data...")
    X, y = load_features(parquet_path, target_col="heuristic", device=device)
    print(f"Features: {tuple(X.shape)} "
          f"({X.numel() * X.element_size() / 1e6:.1f} MB on {X.device})")

    # 3. Split Data (on the device)
    print("Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = split_data(
        X, y, train_size=args.train_size, seed=args.seed
    )
    del X, y
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

    # 4. Model Training & Evaluation
    results_summary = {}
    tune_results = {}

    if args.model in ["linear", "all"]:
        print("\nTraining Linear Regression...")
        model_lr = LinearRegressor().fit(x_train, y_train)
        results_summary["Linear"] = evaluate_model(model_lr, x_test, y_test)
        print(f"Linear Results: {results_summary['Linear']}")

    if args.model in ["xgb", "all"]:
        if args.tune:
            print("\n" + "=" * 60)
            print("XGBoost Hyperparameter Tuning")
            print("=" * 60)
            model_xgb, best_params, all_results = grid_search_xgboost(
                x_train, y_train, x_test, y_test,
                XGB_PARAM_GRIDS, device, n_jobs=n_jobs,
                max_rows=args.xgb_max_rows
            )
            tune_results["XGBoost"] = {
                "best_params": best_params,
                "all_results": all_results,
            }
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
        if args.tune:
            print("\n" + "=" * 60)
            print("MLP Hyperparameter Tuning")
            print("=" * 60)
            model_mlp, best_params, all_results = grid_search_mlp(
                x_train, y_train, x_test, y_test,
                MLP_PARAM_GRIDS
            )
            tune_results["MLP"] = {
                "best_params": best_params,
                "all_results": all_results,
            }
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
    print("FINAL SUMMARY")
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
