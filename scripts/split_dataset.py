"""
Split a full dataset into train / val / test splits.

Takes a parquet file (output of ``generate_jepa_heuristic_dataset`` or the
batch version) and writes three separate parquet files plus a small index JSON
so downstream training scripts can load them without re-shuffling.

Usage::

    python -m scripts.split_dataset \\
        --input data/2026-11-10\\ 17-44-33_10000_heuristic_dataset.pkl \\
        --train-size 0.7 --val-size 0.15 --seed 42

This produces::

    data/splits/2026-11-10_17-44-33_10000/
        train.parquet
        val.parquet
        test.parquet
        split_config.json
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch

from utils.euclid_distance_ml_model_training_utils import (
    convert_pickle_to_parquet,
    load_features,
    split_three_way,
)


def main():
    parser = argparse.ArgumentParser(
        description="Split a JEPA heuristic dataset into train / val / test"
    )
    parser.add_argument("--input", type=str, required=True, help="Source .pkl or .parquet file")
    parser.add_argument("--train-size", type=float, default=0.7, help="Training proportion (default 0.7)")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation proportion (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/splits/<timestamp>_<name>)")
    args = parser.parse_args()

    if args.train_size + args.val_size >= 1.0:
        parser.error("--train-size + --val-size must be < 1.0")

    # Resolve input
    base = args.input
    if base.endswith(".pkl"):
        parquet_path = convert_pickle_to_parquet(base)
    else:
        parquet_path = base
    print(f"Loading: {parquet_path}")

    # Load
    X, y = load_features(parquet_path, target_col="heuristic", device="cpu")
    n = X.shape[0]
    print(f"Total samples: {n}")

    # Split
    x_train, x_val, x_test, y_train, y_val, y_test = split_three_way(
        X, y, train_size=args.train_size, val_size=args.val_size, seed=args.seed
    )
    del X, y

    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # Determine output path
    stem = os.path.basename(parquet_path).replace(".parquet", "")
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = f"data/splits/{ts}_{stem}"
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save splits as parquet (stack X and y back into a DataFrame)
    def _save_split(x, y, path: str):
        df = pd.DataFrame(x.detach().cpu().numpy(),
                          columns=[f"feat_{i}" for i in range(x.shape[1])])
        df["heuristic"] = y.detach().cpu().numpy()
        df.to_parquet(path, index=False)

    _save_split(x_train, y_train, os.path.join(out_dir, "train.parquet"))
    _save_split(x_val,   y_val,   os.path.join(out_dir, "val.parquet"))
    _save_split(x_test,  y_test,  os.path.join(out_dir, "test.parquet"))

    # Save config for downstream scripts
    config = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": 1.0 - args.train_size - args.val_size,
        "seed": args.seed,
        "n_total": n,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "n_features": x_train.shape[1],
        "source_parquet": parquet_path,
    }
    with open(os.path.join(out_dir, "split_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to: {out_dir}")
    print(f"  train.parquet  ({len(x_train):,} samples)")
    print(f"  val.parquet    ({len(x_val):,} samples)")
    print(f"  test.parquet   ({len(x_test):,} samples)")
    print(f"  split_config.json")


if __name__ == "__main__":
    main()
