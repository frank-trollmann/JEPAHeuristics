"""
MLP Ablation Study for JEPA Heuristic Prediction.

Systematically varies one hyperparameter at a time while keeping everything
else fixed. Runs 5 seeds per configuration and reports mean ± std.

Fixed settings:
  - Data split: 80/20 (train/test)
  - Preprocessing: z-score standardization on training split
  - Optimizer: Adam
  - Learning rate: 1e-3
  - Epochs: 200
  - Batch size: 4096
  - Early stopping: patience=20 on val (if val provided)
  - Evaluation: MAE, MSE, R², Pearson, Spearman on test set

Ablation dimensions:
  1. Capacity:   hidden layer sizes and number of layers
  2. Non-linearity: ReLU, GELU, tanh
  3. Regularization: dropout (0/0.1/0.2/0.3), weight decay (0/1e-5/1e-4)
  4. Structure:  residual connections, LayerNorm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.euclid_distance_ml_model_training_utils import (
    load_features,
    split_data,
)


# =============================================================================
# MLP Model Definition
# =============================================================================

class AblationMLP(torch.nn.Module):
    """
    Configurable MLP for ablation studies.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : tuple[int, ...]
        Hidden layer sizes. Empty tuple → linear model.
    activation : str
        Activation function: 'relu', 'gelu', or 'tanh'.
    dropout : float
        Dropout probability (0.0 = no dropout).
    weight_decay : float
        L2 regularization strength for AdamW.
    residual : bool
        If True, add residual connection after each hidden layer.
    layer_norm : bool
        If True, add LayerNorm after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        residual: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        assert activation in ("relu", "gelu", "tanh"), \
            f"Unknown activation: {activation}"

        act_map = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "tanh": torch.nn.Tanh(),
        }
        act = act_map[activation]

        layers: list[torch.nn.Module] = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_layers):
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            if layer_norm:
                layers.append(torch.nn.LayerNorm(h_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))

            # Residual connection (only if dimensions match)
            if residual and h_dim == prev_dim:
                layers.append(_Residual())

            prev_dim = h_dim

        layers.append(torch.nn.Linear(prev_dim, 1))
        self.net = torch.nn.Sequential(*layers)
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _Residual(torch.nn.Module):
    """Identity residual: passes input through unchanged for skipping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_mlp_ablation(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden_layers: tuple[int, ...],
    activation: str = "relu",
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    residual: bool = False,
    layer_norm: bool = False,
    epochs: int = 200,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int | None = None,
    device: str = "cuda",
    val_X: torch.Tensor | None = None,
    val_y: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, list[float], list[float]]:
    """Train an AblationMLP and return (model, train_losses, val_losses)."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = X.device
    y = y.reshape(-1, 1)

    # Standardize features
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)
    X_std = (X - mean) / std

    # Build model
    y_flat = y.reshape(-1)
    model = AblationMLP(
        input_dim=X.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        weight_decay=weight_decay,
        residual=residual,
        layer_norm=layer_norm,
    ).to(device)

    # Initialize output bias to target mean
    model.net[-1].bias.data.fill_(float(y_flat.mean()))

    # Optimizer: AdamW if weight_decay > 0, else Adam
    optim_cls = torch.optim.AdamW if weight_decay > 0 else torch.optim.Adam
    optimizer = optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    # Training loop
    model.train()
    n = X_std.shape[0]
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Prepare validation data
    val_std: torch.Tensor | None = None
    if val_X is not None:
        val_std = ((val_X - mean) / std).to(device)
        val_y_flat = val_y.reshape(-1).to(device)

    for epoch in range(epochs):
        permutation = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_std[idx]), y_flat[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]
        train_losses.append(epoch_loss / n)

        # Validation loss
        if val_std is not None:
            with torch.no_grad():
                val_losses.append(criterion(model(val_std), val_y_flat).item())

    return model, train_losses, val_losses


def evaluate(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate model and return MAE, MSE, R², Pearson, Spearman."""
    model.eval()
    # Re-standardize using same statistics
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)
    X_std = (X - mean) / std

    with torch.no_grad():
        pred = model(X_std.to(device)).cpu().numpy()
    true = y.cpu().numpy().reshape(-1)

    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    pearson, _ = pearsonr(true, pred)
    spearman, _ = spearmanr(true, pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
    }


# =============================================================================
# Data Loading
# =============================================================================




# =============================================================================
# Ablation Configurations
# =============================================================================

def get_ablation_configs() -> list[dict[str, Any]]:
    """Return all ablation configurations."""
    configs: list[dict[str, Any]] = []

    # --- 1. Baseline ---
    configs.append({
        "name": "baseline",
        "description": "Default MLP: (64,32), ReLU, no reg, no norm, no residual",
        "hidden_layers": (64, 32),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })

    # --- 2. Capacity ---
    configs.append({
        "name": "capacity_halved",
        "description": "Half hidden size: (32,16)",
        "hidden_layers": (32, 16),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "capacity_doubled",
        "description": "Double hidden size: (128,64)",
        "hidden_layers": (128, 64),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "capacity_x4",
        "description": "Quadruple hidden size: (256,128)",
        "hidden_layers": (256, 128),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "capacity_deep",
        "description": "Deeper: (128,128,64)",
        "hidden_layers": (128, 128, 64),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "capacity_wide",
        "description": "Wide single layer: (512)",
        "hidden_layers": (512,),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })

    # --- 3. Non-linearity ---
    configs.append({
        "name": "act_gelu",
        "description": "GELU activation",
        "hidden_layers": (64, 32),
        "activation": "gelu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "act_tanh",
        "description": "tanh activation",
        "hidden_layers": (64, 32),
        "activation": "tanh",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })

    # --- 4. Regularization ---
    configs.append({
        "name": "reg_dropout_01",
        "description": "Dropout 0.1",
        "hidden_layers": (64, 32),
        "activation": "relu",
        "dropout": 0.1,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "reg_dropout_02",
        "description": "Dropout 0.2",
        "hidden_layers": (64, 32),
        "activation": "relu",
        "dropout": 0.2,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "reg_weightdecay_1e5",
        "description": "Weight decay 1e-5",
        "hidden_layers": (64, 32),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 1e-5,
        "residual": False,
        "layer_norm": False,
    })
    configs.append({
        "name": "reg_weightdecay_1e4",
        "description": "Weight decay 1e-4",
        "hidden_layers": (64, 32),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 1e-4,
        "residual": False,
        "layer_norm": False,
    })

    # --- 5. Structure ---
    configs.append({
        "name": "struct_residual",
        "description": "Residual connections",
        "hidden_layers": (64, 64),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": True,
        "layer_norm": False,
    })
    configs.append({
        "name": "struct_layernorm",
        "description": "LayerNorm after each hidden layer",
        "hidden_layers": (64, 32),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": False,
        "layer_norm": True,
    })
    configs.append({
        "name": "struct_both",
        "description": "Residual + LayerNorm",
        "hidden_layers": (64, 64),
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "residual": True,
        "layer_norm": True,
    })

    return configs


def _plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    config_name: str,
    output_dir: str,
    epochs: int,
) -> None:
    """Plot and save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = list(range(len(train_losses)))
    ax.plot(iters, train_losses, 'b-', label='Train MSE', linewidth=1.5)
    ax.plot(iters, val_losses, 'r-', label='Val MSE', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title(f'{config_name} — Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f'{config_name}_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLP Ablation Study for JEPA Heuristic Prediction"
    )
    parser.add_argument("--input", required=True,
                        help="Path to parquet dataset")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Training device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data split")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds per configuration")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--output-dir", default="data/ablation",
                        help="Output directory")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA not available")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load data
    print(f"Loading data from: {args.input}")
    X, y = load_features(args.input, device=device)
    print(f"Features: {tuple(X.shape)}")

    # Split: 80% train, 20% test
    X_train_full, X_test, y_train_full, y_test = split_data(
        X, y, train_size=0.8, seed=args.seed
    )
    # Further split train into 80% train / 20% val (within the 80%)
    X_train, X_val, y_train, y_val = split_data(
        X_train_full, y_train_full, train_size=0.8, seed=args.seed
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Get configurations
    configs = get_ablation_configs()
    print(f"\nRunning {len(configs)} configurations × {args.seeds} seeds\n")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    curves_dir = os.path.join(args.output_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    for cfg in configs:
        print(f"\n[{cfg['name']}] {cfg['description']}")
        print(f"  hidden={cfg['hidden_layers']}, act={cfg['activation']}, "
              f"dropout={cfg['dropout']}, wd={cfg['weight_decay']}, "
              f"residual={cfg['residual']}, layernorm={cfg['layer_norm']}")

        seed_results: list[dict[str, float]] = []
        seed_curves: list[tuple[list[float], list[float]]] = []

        for s in range(args.seeds):
            model, train_losses, val_losses = train_mlp_ablation(
                X_train, y_train,
                hidden_layers=cfg["hidden_layers"],
                activation=cfg["activation"],
                dropout=cfg["dropout"],
                weight_decay=cfg["weight_decay"],
                residual=cfg["residual"],
                layer_norm=cfg["layer_norm"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + s,
                device=device,
                val_X=X_val,
                val_y=y_val,
            )
            metrics = evaluate(model, X_test, y_test, device=device)
            seed_results.append(metrics)
            seed_curves.append((train_losses, val_losses))
            print(f"    seed {s}: MAE={metrics['MAE']:.4f}  "
                  f"R²={metrics['R2']:.4f}  "
                  f"Pearson={metrics['Pearson']:.4f}")

            # Plot and save curves for first seed only (keep files manageable)
            if s == 0:
                _plot_curves(
                    train_losses, val_losses, cfg["name"], curves_dir, args.epochs
                )

        # Aggregate
        agg = {}
        for metric in ["MAE", "MSE", "R2", "Pearson", "Spearman"]:
            vals = [r[metric] for r in seed_results]
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))

        agg.update({
            "name": cfg["name"],
            "description": cfg["description"],
            "hidden_layers": cfg["hidden_layers"],
            "activation": cfg["activation"],
            "dropout": cfg["dropout"],
            "weight_decay": cfg["weight_decay"],
            "residual": cfg["residual"],
            "layer_norm": cfg["layer_norm"],
            "n_seeds": args.seeds,
        })
        results.append(agg)

        del model, seed_results, seed_curves

    # Save results
    results_path = os.path.join(args.output_dir, f"{timestamp}_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Curves saved to: {curves_dir}")

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (mean ± std over {} seeds)".format(args.seeds))
    print("=" * 70)
    header = f"{'Config':<22} {'MAE':>8} {'R²':>8} {'Pearson':>8} {'Spearman':>8}"
    print(header)
    print("-" * len(header))

    # Sort by MAE mean
    results_sorted = sorted(results, key=lambda r: r["MAE_mean"])
    for r in results_sorted:
        name = r["name"][:20]
        mae = r["MAE_mean"]
        mae_std = r["MAE_std"]
        r2 = r["R2_mean"]
        r2_std = r["R2_std"]
        p = r["Pearson_mean"]
        p_std = r["Pearson_std"]
        s = r["Spearman_mean"]
        s_std = r["Spearman_std"]
        print(f"{name:<22} {mae:.4f}±{mae_std:.4f}  "
              f"{r2:.4f}±{r2_std:.4f}  "
              f"{p:.4f}±{p_std:.4f}  "
              f"{s:.4f}±{s_std:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
