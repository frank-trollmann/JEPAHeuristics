import argparse
import glob
import os
from datetime import datetime

import pandas as pd

from utils.pair_dataset import load_position_dataset
from utils.position_probe import fit_position_probe, position_errors


def main():
    parser = argparse.ArgumentParser(description="Check how well (x, y) can be read from a single encoding")
    parser.add_argument("--dataset", type=str, default=None, help="Position dataset .npz. Defaults to the newest in data/")
    args = parser.parse_args()

    path = args.dataset or sorted(glob.glob("data/*_position_dataset_*.npz"))[-1]
    positions, encodings, splits = load_position_dataset(path)
    train = splits == "train"
    print(f"Dataset: {path}")

    rows = []
    for name, hidden in [("linear", None), ("MLP 256-256", (256, 256))]:
        model = fit_position_probe(encodings[train], positions[train], hidden)
        for split in ("val", "test"):
            mask = splits == split
            rows.append({"model": name, "split": split, **position_errors(model, encodings[mask], positions[mask])})

    results = pd.DataFrame(rows)
    os.makedirs("data/results", exist_ok=True)
    out = f"data/results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_position_probe.csv"
    results.to_csv(out, index=False)
    print(results.round(3).to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
