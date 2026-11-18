"""
Focused MLP training for Euclidean distance prediction from JEPA embeddings.

Uses a fixed 70/15/15 train/val/test split, trains only MLP, supports grid
search and CEM hyperparameter tuning with per-epoch validation tracking.
The best checkpoint (lowest val MAE) is saved and evaluated on the untouched
test set only after the search completes.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.euclid_distance_ml_model_training_utils import (
    Standardizer,
    convert_pickle_to_parquet,
    evaluate_model,
    load_features,
    resolve_input_path,
    split_three_way,
)
from utils.cem_optimizer import (
    CEMOptimizer,
    GenerationResult,
    make_mlp_param_configs,
    mlp_sample_to_kwargs,
)
from scripts.euclid_distance_ml_model_training import (
    TRAINING_SEED,
    generate_combinations,
    grid_search_mlp,
    cem_search_mlp,
)


# =============================================================================
# Training with per-epoch tracking (re-exports enhanced train_mlp behaviour)
# =============================================================================

def train_mlp_with_tracking(
    x_train, y_train, x_val, y_val,
    hidden_layers=(128, 64), epochs=200, batch_size=4096, lr=1e-3,
    seed=None,
):
    """
    Train MLP with per-epoch train/val MSE & MAE tracking and checkpointing.

    Returns:
        history: list of dicts with epoch, train_mse, train_mae, val_mse, val_mae
        best_state: model state dict with lowest val MAE
        best_epoch: epoch index (0-based) of best model
        last_state: model state dict of final epoch
        standardizer: fitted Standardizer (numpy mean/std)
        model_config: dict of architecture + training hyperparameters
    """
    from utils.euclid_distance_ml_model_training_utils import train_mlp, _Standardize

    device = x_train.device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Fit standardizer exclusively on training data
    standardizer = Standardizer()
    standardizer.fit(x_train)

    y_train = y_train.reshape(-1, 1)
    input_dim = x_train.shape[1]

    layers = []
    prev_dim = input_dim
    for h_dim in hidden_layers:
        layers.append(torch.nn.Linear(prev_dim, h_dim))
        layers.append(torch.nn.ReLU())
        prev_dim = h_dim
    layers.append(torch.nn.Linear(prev_dim, 1))

    model = torch.nn.Sequential(*layers).to(device)
    model[-1].bias.data.fill_(float(y_train.mean()))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_std = standardizer.transform(x_train)
    X_val_std = standardizer.transform(x_val)
    y_val = y_val.reshape(-1, 1)

    history = []
    best_val_mae = float("inf")
    best_state = None
    last_state = None
    n_train = x_train.shape[0]

    for epoch in range(epochs):
        # ── Training phase ──
        model.train()
        perm = torch.randperm(n_train, device=device, generator=generator)
        train_pred_all = []

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train_std[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            train_pred_all.append(model(X_train_std[idx]).detach())

        train_pred = torch.cat(train_pred_all, dim=0).squeeze(1)
        train_error = train_pred - y_train.squeeze(1)
        train_mse = (train_error ** 2).mean().item()
        train_mae = train_error.abs().mean().item()

        # ── Validation phase (eval mode, no gradients) ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_std).squeeze(1)
        val_error = val_pred - y_val.squeeze(1)
        val_mse = (val_error ** 2).mean().item()
        val_mae = val_error.abs().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "val_mse": val_mse,
            "val_mae": val_mae,
        })

        print(f"  Epoch {epoch + 1:3d}/{epochs}  "
              f"train_MSE={train_mse:.4f} MAE={train_mae:.4f}  "
              f"val_MSE={val_mse:.4f} MAE={val_mae:.4f}", end="\r")

        # ── Checkpointing ──
        last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print()  # newline after \r

    model_config = {
        "hidden_layers": list(hidden_layers),
        "input_dim": input_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
    }

    return history, best_state, epoch, last_state, standardizer, model_config


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_training_history(history, output_dir: str):
    """Save training/val MSE and MAE curves."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [h["train_mse"] for h in history], label="Train MSE")
    axes[0].plot(epochs, [h["val_mse"] for h in history], label="Val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE over epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h["train_mae"] for h in history], label="Train MAE")
    axes[1].plot(epochs, [h["val_mae"] for h in history], label="Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE over epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_true_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
    """Scatter plot of true vs predicted distances with y=x diagonal."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=1, alpha=0.15, color="steelblue")

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")

    ax.set_xlabel("True distance")
    ax.set_ylabel("Predicted distance")
    ax.set_title("True vs Predicted Euclidean Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    path = os.path.join(output_dir, "true_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_checkpoint_and_eval(state_dict, standardizer, x_data, y_data):
    """Reconstruct model from checkpoint state, apply standardizer, evaluate."""
    input_dim = state_dict["0.weight"].shape[1]
    layers_cfg = standardizer._model_config["hidden_layers"] if hasattr(standardizer, "_model_config") else [128, 64]
    # Reconstruct from saved config
    return None  # placeholder — caller builds model


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Focused MLP training for JEPA Euclidean distance prediction"
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to source .pkl or existing parquet file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/mlp_distance_training",
        help="Output directory for checkpoints, plots, and results"
    )
    parser.add_argument(
        "--tune", type=str, default=None,
        choices=[None, "grid", "cem", "all"],
        help="Hyperparameter search: None (fixed), 'grid', 'cem', or 'all' (both, pick best by val MAE)"
    )
    parser.add_argument(
        "--hidden-layers", type=int, nargs="*", default=[128, 64],
        help="Hidden layer sizes (used when --tune is None)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (used when --tune is None)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (used when --tune is None)"
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
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Training device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    # CEM-specific
    parser.add_argument(
        "--cem-generations", type=int, default=10,
        help="Number of CEM generations"
    )
    parser.add_argument(
        "--cem-batch-size", type=int, default=10,
        help="Candidates per CEM generation"
    )
    parser.add_argument(
        "--cem-elite-fraction", type=float, default=0.2,
        help="Elite fraction for CEM"
    )

    args = parser.parse_args()

    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be < 1.0")

    # ── Setup ──
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("=" * 60)
    print("MLP Distance Training (focused)")
    print("=" * 60)
    print(f"Device:         {device}")
    print(f"Train size:     {args.train_size}")
    print(f"Val size:       {args.val_size}")
    print(f"Test size:      {1.0 - args.train_size - args.val_size}")
    print(f"Tuning:         {args.tune}")
    print(f"Output dir:     {args.output_dir}")
    print()
    print("Seeds:")
    print(f"  Split seed:       {args.seed}  (deterministic 70/15/15 split)")
    print(f"  Training seed:    {TRAINING_SEED}  (fixed for all candidates)")
    if args.tune in ("cem", "all"):
        print(f"  CEM seed:         {args.seed}  (independent optimizer RNG)")
    print()
    if args.tune is None:
        print(f"Hidden layers:  {args.hidden_layers}")
        print(f"Epochs:         {args.epochs}")
        print(f"Batch size:     {args.batch_size}")
        print(f"Learning rate:  {args.lr}")
    else:
        print(f"CEM generations:  {args.cem_generations}")
        print(f"CEM batch size:   {args.cem_batch_size}")
        print(f"CEM elite frac:   {args.cem_elite_fraction}")
    print()

    # ── 1. Load data ──
    parquet_path = resolve_input_path(args.input)
    if os.path.exists(parquet_path):
        print(f"Using existing parquet: {parquet_path}")
    else:
        parquet_path = convert_pickle_to_parquet(args.input)
        print(f"Converted and saved to: {parquet_path}")

    print("Loading features...")
    X, y = load_features(parquet_path, target_col="heuristic", device=device)
    print(f"Features: {tuple(X.shape)} "
          f"({X.numel() * X.element_size() / 1e6:.1f} MB on {X.device})")

    # ── 2. 70/15/15 split ──
    print("Splitting data (70/15/15)...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_three_way(
        X, y, train_size=args.train_size, val_size=args.val_size, seed=args.seed
    )
    del X, y
    print(f"Train: {len(x_train):,}  Val: {len(x_val):,}  Test: {len(x_test):,}")
    print()

    # ── 3. Search or fixed training ──
    best_state = None
    best_params = None
    best_val_mae = float("inf")
    search_results = []
    search_elapsed = 0.0
    standardizer = None
    model_config = None
    history = None
    last_state = None
    best_epoch = 0

    if args.tune == "grid":
        # ── Grid Search ──
        print("\n" + "=" * 60)
        print("MLP Hyperparameter Tuning (Grid Search)")
        print("=" * 60)
        t_search = time.time()

        MLP_PARAM_GRIDS = [
            {"hidden_layers": [(32, 16), (64, 32), (128, 64)],
             "epochs": [50, 100, 200], "lr": [1e-4, 1e-3, 5e-3]},
            {"hidden_layers": [(64, 64, 32), (128, 128, 64)],
             "epochs": [100, 200], "lr": [1e-3, 5e-3]},
        ]

        best_model, best_params, search_results, best_history = grid_search_mlp(
            x_train, y_train, x_val, y_val, MLP_PARAM_GRIDS
        )
        search_elapsed = time.time() - t_search

        # Use the best checkpoint from the search directly
        best_state = {k: v.cpu().clone() for k, v in best_model.state_dict().items()}
        best_val_mae = min(h["val_mae"] for h in best_history)
        best_epoch = min(range(len(best_history)), key=lambda e: best_history[e]["val_mae"])
        last_state = best_state  # same as best for search winner
        standardizer = Standardizer()
        standardizer.mean = best_model.standardizer_mean
        standardizer.std = best_model.standardizer_std
        bl = best_params.get("hidden_layers", (128, 64))
        model_config = {
            "hidden_layers": list(bl) if isinstance(bl, list) else list(bl),
            "input_dim": x_train.shape[1],
            "epochs": best_params.get("epochs", 200),
            "batch_size": args.batch_size,
            "lr": best_params.get("lr", 1e-3),
            "seed": args.seed,
        }
        history = best_history

        print(f"\nGrid search elapsed: {search_elapsed:.1f}s")
        print(f"Best grid params: {best_params}")
        print(f"Best grid val MAE: {best_val_mae:.4f}")

    elif args.tune == "cem":
        # ── CEM Search ──
        print("\n" + "=" * 60)
        print("MLP Hyperparameter Tuning (CEM)")
        print("=" * 60)
        t_search = time.time()

        cem_result = cem_search_mlp(
            x_train, y_train, x_val, y_val,
            generations=args.cem_generations,
            batch_size=args.cem_batch_size,
            elite_fraction=args.cem_elite_fraction,
            base_epochs=args.epochs,
            base_batch_size=args.batch_size,
            seed=args.seed,
        )
        search_elapsed = time.time() - t_search

        best_model = cem_result["best_model"]
        best_params = cem_result["best_params"]
        search_results = cem_result["all_results"]
        best_history = cem_result["best_history"]
        best_val_mae = cem_result["best_mae"]

        # Use the best checkpoint from the search directly
        best_state = {k: v.cpu().clone() for k, v in best_model.state_dict().items()}
        best_epoch = min(range(len(best_history)), key=lambda e: best_history[e]["val_mae"])
        last_state = best_state
        standardizer = Standardizer()
        standardizer.mean = best_model.standardizer_mean
        standardizer.std = best_model.standardizer_std
        bl = best_params.get("hidden_layers", (128, 64))
        model_config = {
            "hidden_layers": list(bl) if isinstance(bl, list) else list(bl),
            "input_dim": x_train.shape[1],
            "epochs": best_params.get("epochs", args.epochs),
            "batch_size": args.batch_size,
            "lr": best_params.get("lr", 1e-3),
            "seed": args.seed,
        }
        history = best_history

        print(f"\nCEM search elapsed: {search_elapsed:.1f}s")
        print(f"Best CEM params: {best_params}")
        print(f"Best CEM val MAE: {best_val_mae:.4f}")

    elif args.tune == "all":
        # ── Run both grid and CEM, pick winner by val MAE ──
        print("\n" + "=" * 60)
        print("MLP Hyperparameter Tuning (Grid + CEM, compare winners)")
        print("=" * 60)

        # --- Grid ---
        print("\n--- Grid Search ---")
        t_grid = time.time()
        MLP_PARAM_GRIDS = [
            {"hidden_layers": [(32, 16), (64, 32), (128, 64)],
             "epochs": [50, 100, 200], "lr": [1e-4, 1e-3, 5e-3]},
            {"hidden_layers": [(64, 64, 32), (128, 128, 64)],
             "epochs": [100, 200], "lr": [1e-3, 5e-3]},
        ]
        grid_model, grid_params, grid_results, grid_history = grid_search_mlp(
            x_train, y_train, x_val, y_val, MLP_PARAM_GRIDS
        )
        grid_elapsed = time.time() - t_grid
        grid_val_mae = min(h["val_mae"] for h in grid_history)
        print(f"Grid elapsed: {grid_elapsed:.1f}s  Best val MAE: {grid_val_mae:.4f}")

        # --- CEM ---
        print("\n--- CEM Search ---")
        t_cem = time.time()
        cem_result = cem_search_mlp(
            x_train, y_train, x_val, y_val,
            generations=args.cem_generations,
            batch_size=args.cem_batch_size,
            elite_fraction=args.cem_elite_fraction,
            base_epochs=args.epochs,
            base_batch_size=args.batch_size,
            seed=args.seed,
        )
        cem_elapsed = time.time() - t_cem
        cem_model = cem_result["best_model"]
        cem_params = cem_result["best_params"]
        cem_history = cem_result["best_history"]
        cem_val_mae = cem_result["best_mae"]
        print(f"CEM elapsed: {cem_elapsed:.1f}s  Best val MAE: {cem_val_mae:.4f}")

        # --- Compare and pick winner by val MAE ---
        search_results = []
        search_elapsed = grid_elapsed + cem_elapsed
        if grid_val_mae <= cem_val_mae:
            print(f"\nWinner: Grid (val MAE {grid_val_mae:.4f} <= CEM {cem_val_mae:.4f})")
            best_model = grid_model
            best_params = grid_params
            best_history = grid_history
            search_results = grid_results
            best_val_mae = grid_val_mae
        else:
            print(f"\nWinner: CEM (val MAE {cem_val_mae:.4f} < Grid {grid_val_mae:.4f})")
            best_model = cem_model
            best_params = cem_params
            best_history = cem_history
            search_results = cem_result["all_results"]
            best_val_mae = cem_val_mae

        # Build artifacts from winner
        best_state = {k: v.cpu().clone() for k, v in best_model.state_dict().items()}
        best_epoch = min(range(len(best_history)), key=lambda e: best_history[e]["val_mae"])
        last_state = best_state
        standardizer = Standardizer()
        standardizer.mean = best_model.standardizer_mean
        standardizer.std = best_model.standardizer_std
        bl = best_params.get("hidden_layers", (128, 64))
        model_config = {
            "hidden_layers": list(bl) if isinstance(bl, list) else list(bl),
            "input_dim": x_train.shape[1],
            "epochs": best_params.get("epochs", args.epochs),
            "batch_size": args.batch_size,
            "lr": best_params.get("lr", 1e-3),
            "seed": args.seed,
        }
        history = best_history
        # Record both methods in search results for reference
        search_results.insert(0, {
            "params": {"_note": "grid_winner_val_mae", "grid": grid_params, "cem": cem_params},
            "MAE": None,
            "grid_val_mae": grid_val_mae,
            "cem_val_mae": cem_val_mae,
            "winner": "grid" if grid_val_mae <= cem_val_mae else "cem",
            "grid_elapsed_s": grid_elapsed,
            "cem_elapsed_s": cem_elapsed,
        })

    else:
        # ── Fixed (no search) ──
        print("Training MLP (fixed hyperparameters)...")
        history, best_state, best_epoch, last_state, standardizer, model_config = (
            train_mlp_with_tracking(
                x_train, y_train, x_val, y_val,
                hidden_layers=tuple(args.hidden_layers),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            )
        )
        best_params = {
            "hidden_layers": list(args.hidden_layers),
            "epochs": args.epochs,
            "lr": args.lr,
        }
        best_val_mae = min(h["val_mae"] for h in history)

    # ── 4. Save checkpoints & artifacts ──
    plot_training_history(history, args.output_dir)
    print(f"Training history plot saved.")

    # Save best and last checkpoint
    best_path = os.path.join(args.output_dir, f"{timestamp}_best_checkpoint.pth")
    last_path = os.path.join(args.output_dir, f"{timestamp}_last_checkpoint.pth")
    torch.save(best_state, best_path)
    torch.save(last_state, last_path)
    print(f"Best checkpoint  (epoch {best_epoch + 1}): {best_path}")
    print(f"Last checkpoint: {last_path}")

    # Save standardizer
    std_path = os.path.join(args.output_dir, f"{timestamp}_standardizer.npz")
    np.savez(std_path, mean=standardizer.mean, std=standardizer.std)
    print(f"Standardizer saved: {std_path}")

    # Save model config
    config_save = {
        **model_config,
        "timestamp": timestamp,
        "device": device,
        "output_dir": args.output_dir,
        "tune": args.tune,
        "best_params": best_params,
        "best_val_mae": best_val_mae,
        "search_elapsed_s": search_elapsed,
    }
    config_path = os.path.join(args.output_dir, f"{timestamp}_config.json")
    with open(config_path, "w") as f:
        json.dump(config_save, f, indent=2)
    print(f"Config saved: {config_path}")

    # Save search history
    history_path = os.path.join(args.output_dir, f"{timestamp}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {history_path}")

    # Save search results
    if search_results:
        search_path = os.path.join(args.output_dir, f"{timestamp}_search_results.json")
        search_save = {
            "method": args.tune,
            "elapsed_s": search_elapsed,
            "candidates": search_results,
        }
        with open(search_path, "w") as f:
            json.dump(search_save, f, indent=2)
        print(f"Search results saved: {search_path}")
    print()

    # ── 5. Evaluate best checkpoint on train-full and test ──
    print("Evaluating best checkpoint...")

    def reconstruct_model(state_dict):
        """Rebuild model architecture from saved state dict.

        Handles both formats:
        - train_mlp format: keys 0.mean, 0.std, 1.weight, 1.bias, ...
        - train_mlp_with_tracking format: keys 0.weight, 0.bias, 2.weight, ...
        """
        layers_cfg = model_config["hidden_layers"]
        if "0.mean" in state_dict:
            # train_mlp format: _Standardize + Linear layers
            input_dim = state_dict["1.weight"].shape[1]
            from utils.euclid_distance_ml_model_training_utils import _Standardize
            std_mean = state_dict["0.mean"]
            std_std = state_dict["0.std"]
            layers = [_Standardize(std_mean, std_std)]
            prev_dim = input_dim
            for h_dim in layers_cfg:
                layers.append(torch.nn.Linear(prev_dim, h_dim))
                layers.append(torch.nn.ReLU())
                prev_dim = h_dim
            layers.append(torch.nn.Linear(prev_dim, 1))
        else:
            # train_mlp_with_tracking format: Linear layers only
            input_dim = state_dict["0.weight"].shape[1]
            layers = []
            layers.append(torch.nn.Linear(input_dim, layers_cfg[0]))
            layers.append(torch.nn.ReLU())
            for i in range(len(layers_cfg) - 1):
                layers.append(torch.nn.Linear(layers_cfg[i], layers_cfg[i + 1]))
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(layers_cfg[-1], 1))
        model = torch.nn.Sequential(*layers).to(x_train.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    best_model_eval = reconstruct_model(best_state)

    # Evaluate on full training set
    print(f"\n  Train-full metrics:")
    train_metrics = evaluate_model(best_model_eval, x_train, y_train)
    for k, v in train_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Evaluate on untouched test set
    print(f"\n  Test metrics (untouched):")
    test_metrics = evaluate_model(best_model_eval, x_test, y_test)
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # ── 6. Save predictions & true-vs-predicted plot ──
    print("\nSaving test predictions...")
    with torch.no_grad():
        y_pred_test = best_model_eval(standardizer.transform(x_test)).cpu().numpy().reshape(-1)
    y_true_test = y_test.cpu().numpy().reshape(-1)

    # CSV
    csv_path = os.path.join(args.output_dir, f"{timestamp}_test_predictions.csv")
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["true_distance", "predicted_distance"])
        for true, pred in zip(y_true_test, y_pred_test):
            writer.writerow([f"{true:.6f}", f"{pred:.6f}"])
    print(f"Test predictions CSV: {csv_path}")

    # True vs predicted plot
    plot_true_vs_predicted(y_true_test, y_pred_test, args.output_dir)
    print("True vs predicted plot saved.")

    # ── 7. Save final results summary ──
    final_results = {
        "timestamp": timestamp,
        "device": device,
        "seed": args.seed,
        "split": {
            "train": len(x_train),
            "val": len(x_val),
            "test": len(x_test),
        },
        "tuning": {
            "method": args.tune,
            "best_params": best_params,
            "best_val_mae": best_val_mae,
            "search_elapsed_s": search_elapsed,
        },
        "model_config": model_config,
        "best_epoch": best_epoch + 1,
        "train_full_metrics": {k: float(v) for k, v in train_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    results_path = os.path.join(args.output_dir, f"{timestamp}_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal results saved: {results_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Test  MAE={test_metrics['MAE']:.4f}  R2={test_metrics['R2']:.4f}  "
          f"Pearson={test_metrics['Pearson']:.4f}  Spearman={test_metrics['Spearman']:.4f}")
    print(f"Train MAE={train_metrics['MAE']:.4f}  R2={train_metrics['R2']:.4f}")
    if search_results:
        print(f"Search candidates: {len(search_results)}  Elapsed: {search_elapsed:.1f}s")


if __name__ == "__main__":
    main()
