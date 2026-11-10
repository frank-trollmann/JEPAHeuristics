#!/bin/bash
set -e

echo "=== JEPA Heuristics ML Pipeline ==="
echo ""

# 1. Generate dataset with ImageNet normalization
echo "Step 1: Generating 10k dataset with ImageNet normalization..."
python -m scripts.generate_jepa_dataset --sizes 10000 --model tworoom

# 2. Find the latest dataset
DATASET=$(ls -t data/*heuristic_dataset.pkl | head -1)
echo "Using dataset: $DATASET"
echo ""

# 3. Run ML training with hyperparameter tuning
echo "Step 2: Running ML training with hyperparameter tuning..."
python -m scripts.euclid_distance_ml_model_training \
  --input "$DATASET" \
  --model all \
  --seed 42 \
  --tune

echo ""
echo "=== Pipeline Complete ==="
echo "Results saved to: data/MLresults/"
