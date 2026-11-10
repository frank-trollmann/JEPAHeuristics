"""
Merges legacy per-model result files into one file per training run.

The old pipeline wrote data/MLresults/<timestamp>_<ModelName>_results.json for
every model separately. Within one run the models always finish in the same
order (Linear -> XGBoost -> MLP), so the i-th file of each model type belongs
to the i-th run. Timestamps alone are not reliable for grouping: on big
datasets XGBoost/MLP can finish many minutes after the linear model.

Usage:
    python -m scripts.merge_ml_results --dry-run     # show the grouping first
    python -m scripts.merge_ml_results               # write, keep originals in legacy/
    python -m scripts.merge_ml_results --delete      # write and remove originals
"""

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

RESULTS_DIR = "data/MLresults"
SUFFIX = "_results.json"

# Model class name in the old filenames -> key used in the merged summary
MODEL_KEYS = {
    "LinearRegressor": "Linear",
    "XGBRegressor": "XGBoost",
    "Sequential": "MLP",
}


def parse_filename(path: str) -> tuple[str, str]:
    """'.../2026-10-23_15-55-17_LinearRegressor_results.json' -> (timestamp, model)."""
    name = os.path.basename(path)
    if not name.endswith(SUFFIX):
        raise ValueError(f"unexpected file name: {name}")
    timestamp, _, model = name[:-len(SUFFIX)].rpartition("_")
    return timestamp, model


def main():
    parser = argparse.ArgumentParser(description="Merge per-model ML result JSONs")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print the grouping, do not write anything")
    parser.add_argument("--delete", action="store_true",
                        help="Delete the source files instead of moving them to legacy/")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, f"*{SUFFIX}")))
    if not paths:
        print(f"No files matching *{SUFFIX} in {args.results_dir}")
        return

    by_model = defaultdict(list)
    for path in paths:
        timestamp, model = parse_filename(path)
        if model not in MODEL_KEYS:
            print(f"Skipping unknown model file: {os.path.basename(path)}")
            continue
        by_model[model].append((timestamp, path))

    for model in MODEL_KEYS:
        by_model[model].sort()

    counts = {model: len(files) for model, files in by_model.items()}
    print(f"Found per model: {counts}")
    if len(set(counts.values())) != 1:
        print("WARNING: model counts differ - a run is missing a model, "
              "the pairing below may be wrong.")

    num_runs = min(counts.values()) if counts else 0
    ordered_models = [m for m in MODEL_KEYS if by_model[m]]

    legacy_dir = os.path.join(args.results_dir, "legacy")
    for run_idx in range(num_runs):
        entries = {m: by_model[m][run_idx] for m in ordered_models}
        timestamps = [ts for ts, _ in entries.values()]
        run_timestamp = min(timestamps)
        out_path = os.path.join(args.results_dir, f"{run_timestamp}{SUFFIX}")

        merged = {}
        for model, (_, path) in entries.items():
            with open(path) as f:
                merged[MODEL_KEYS[model]] = json.load(f)

        span = f"{run_timestamp} .. {max(timestamps)}"
        print(f"\nRun {run_idx + 1}: {span}")
        for model, (ts, path) in entries.items():
            print(f"  {MODEL_KEYS[model]:9s} <- {os.path.basename(path)}")
        print(f"  => {os.path.basename(out_path)}")

        if args.dry_run:
            continue

        with open(out_path, "w") as f:
            json.dump(merged, f, indent=4)

        for _, path in entries.values():
            if args.delete:
                os.remove(path)
            else:
                os.makedirs(legacy_dir, exist_ok=True)
                shutil.move(path, os.path.join(legacy_dir, os.path.basename(path)))

    if args.dry_run:
        print("\nDry run - nothing written.")
    else:
        print(f"\nWrote {num_runs} merged file(s) to {args.results_dir}")


if __name__ == "__main__":
    main()
