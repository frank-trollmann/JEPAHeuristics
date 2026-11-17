"""
CEM-based path sampling for JEPA planning in TwoRoomEnv.

Uses the Cross-Entropy Method to optimise action sequences that minimise the
JEPA prediction-cost against a goal state.  The best sequence is the one whose
rollout predicted embedding is closest to the goal embedding.

Usage::

    # Basic: plan from (50,50) to (180,180), horizon 5
    python -m scripts.cem_path_sampling \\
        --agent-pos 50 50 --goal-pos 180 180 \\
        --horizon 5

    # More generations, larger sample pool
    python -m scripts.cem_path_sampling \\
        --agent-pos 50 50 --goal-pos 180 180 \\
        --horizon 5 --generations 10 --batch-size 30 \\
        --device cpu

    # Use GPU if available
    python -m scripts.cem_path_sampling \\
        --agent-pos 50 50 --goal-pos 180 180 \\
        --horizon 5 --device cuda
"""

import argparse
import sys
from datetime import datetime

import torch

from utils.cem_path_sampler import CEMPathSampler, build_info_from_positions
from utils.load_fix_pretrained import load_fix_pretrained
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="CEM Path Sampling for JEPA Planning"
    )

    # Position arguments
    parser.add_argument(
        "--agent-pos", nargs=2, type=float, default=[50.0, 50.0],
        help="Agent starting position [x y]"
    )
    parser.add_argument(
        "--goal-pos", nargs=2, type=float, default=[180.0, 180.0],
        help="Goal position [x y]"
    )

    # CEM parameters
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Action horizon T (default 5)"
    )
    parser.add_argument(
        "--generations", type=int, default=8,
        help="Number of CEM generations (default 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Candidates per generation (default 20)"
    )
    parser.add_argument(
        "--elite-fraction", type=float, default=0.2,
        help="Elite fraction (default 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default 42)"
    )

    # Model / device
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Local checkpoint path (default: download from HF)"
    )

    args = parser.parse_args()

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    print("=" * 60)
    print("CEM Path Sampling")
    print("=" * 60)
    print(f"Device:          {device}")
    print(f"Agent pos:       {args.agent_pos}")
    print(f"Goal pos:        {args.goal_pos}")
    print(f"Horizon:         {args.horizon}")
    print(f"Generations:     {args.generations}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Elite fraction:  {args.elite_fraction}")
    print(f"Seed:            {args.seed}")

    # Load model
    print("\nLoading JEPA model...")
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = snapshot_download(
            repo_id="quentinll/lewm-tworooms",
            revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
        )
    model = load_fix_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)
    print("Model loaded.")

    # Build info
    print("\nRendering agent and goal images...")
    agent_img, goal_img = build_info_from_positions(
        model, args.agent_pos, args.goal_pos, device=device
    )
    print(f"Agent image shape: {agent_img.shape}")
    print(f"Goal image shape:  {goal_img.shape}")

    # Run CEM
    sampler = CEMPathSampler(
        model=model,
        history_size=3,
        action_dim=10,
        action_low=-1.0,
        action_high=1.0,
        elite_fraction=args.elite_fraction,
        rng=torch.Generator(device=device).manual_seed(args.seed) if args.seed else None,
    )
    # Override rng with numpy generator
    import numpy as np
    sampler.rng = np.random.default_rng(args.seed)

    print(f"\nRunning CEM ({args.generations} generations, {args.batch_size} candidates/gen)...")
    best_path = sampler.sample_path(
        agent_image=agent_img,
        goal_image=goal_img,
        horizon=args.horizon,
        num_generations=args.generations,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    # Output
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Best cost:  {sampler.best_cost():.4f}")
    print(f"Best path shape: {best_path.shape}")
    print("\nAction sequence (each row = one timestep, 10 values):")
    for t in range(args.horizon):
        vals = [f"{v:+.3f}" for v in best_path[t].tolist()]
        print(f"  t={t}:  {vals}")

    # Save
    out_dir = "data/path_planning"
    import os
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(
        out_dir,
        f"{ts}_agent{int(args.agent_pos[0])}_{int(args.agent_pos[1])}"
        f"_goal{int(args.goal_pos[0])}_{int(args.goal_pos[1])}"
        f"_horizon{args.horizon}.pt",
    )
    torch.save({
        "path": best_path,
        "best_cost": sampler.best_cost(),
        "agent_pos": args.agent_pos,
        "goal_pos": args.goal_pos,
        "horizon": args.horizon,
        "generations": args.generations,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }, out_path)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
