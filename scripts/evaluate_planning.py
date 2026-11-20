"""
Planning evaluation script.

Compares four planning strategies on random TwoRoomEnv start-goal pairs:
  1. greedy_mlp     — Greedy action selection guided by the pretrained MLP heuristic
  2. greedy_jepa    — Greedy action selection using raw JEPA prediction cost
  3. cem            — Cross-Entropy Method with raw JEPA cost (baseline)
  4. cem_mlp        — Cross-Entropy Method with MLP heuristic as cost

Metrics:
  - latent_mse: MSE between predicted final embedding and goal embedding
  - latent_dist: Euclidean distance in latent space (lower=better)
  - env_final_distance: Final distance in the real TwoRoomEnv state space
  - planning_time_s: Wall-clock time for planning

Usage:
    python -m scripts.evaluate_planning --num-problems 20 --horizon 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from utils.heuristic_planner import (
    GreedyHeuristicPlanner,
    greedy_jepa_cost_planner,
    evaluate_plan,
    load_pretrained_mlp,
    CEMPathSamplerMLP,
    receding_horizon_plan,
)
from utils.cem_path_sampler import CEMPathSampler
from utils.load_fix_pretrained import load_fix_pretrained
from utils.image_preprocessing import normalise_images
from huggingface_hub import snapshot_download
from stable_worldmodel.envs.two_room import TwoRoomEnv


def generate_problem(rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Random start and goal within the safe content area of TwoRoomEnv."""
    env = TwoRoomEnv()
    lo, hi = env.BORDER_SIZE, env.IMG_SIZE - env.BORDER_SIZE
    agent = [int(rng.integers(lo, hi)), int(rng.integers(lo, hi))]
    goal = [int(rng.integers(lo, hi)), int(rng.integers(lo, hi))]
    return agent, goal


def render_images(agent_pos, goal_pos, device):
    """Render agent and goal images, return normalized tensors (1,1,C,H,W)."""
    env = TwoRoomEnv()
    env.reset(options={"state": agent_pos})
    env.unwrapped._set_goal_state(goal_pos)
    img_a = env.render()

    env.reset(options={"state": goal_pos})
    env.unwrapped._set_goal_state(goal_pos)
    img_g = env.render()

    t_a = normalise_images(
        torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(device)
    )
    t_g = normalise_images(
        torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(device)
    )
    return t_a, t_g


def run_greedy_mlp(model, mlp, agent_img, goal_img, horizon, device, rng):
    planner = GreedyHeuristicPlanner(
        model=model, mlp_heuristic=mlp, device=device, rng=rng,
    )
    t0 = time.time()
    plan = planner.plan(agent_img, goal_img, horizon)
    return plan, time.time() - t0


def run_greedy_jepa(model, agent_img, goal_img, horizon, device, rng):
    t0 = time.time()
    plan = greedy_jepa_cost_planner(
        model=model, agent_image=agent_img, goal_image=goal_img,
        horizon=horizon, rng=rng,
    )
    return plan, time.time() - t0


def run_cem(model, agent_img, goal_img, horizon, device, rng):
    sampler = CEMPathSampler(
        model=model, history_size=3, action_dim=10,
        action_low=-1.0, action_high=1.0, rng=rng,
    )
    t0 = time.time()
    plan = sampler.sample_path(
        agent_image=agent_img, goal_image=goal_img,
        horizon=horizon, num_generations=5, batch_size=16, device=device,
    )
    return plan.cpu().numpy(), time.time() - t0


def run_cem_mlp(model, mlp, agent_img, goal_img, horizon, device, rng):
    sampler = CEMPathSamplerMLP(
        model=model, mlp_heuristic=mlp,
        history_size=3, action_dim=10,
        action_low=-1.0, action_high=1.0, rng=rng,
    )
    t0 = time.time()
    plan = sampler.sample_path(
        agent_image=agent_img, goal_image=goal_img,
        horizon=horizon, num_generations=5, batch_size=16, device=device,
    )
    return plan.cpu().numpy(), time.time() - t0


def evaluate_latent(plan, model, agent_img, goal_img, horizon):
    """Roll out plan through JEPA model, return latent-space metrics."""
    device = next(model.parameters()).device
    H = 3

    # Encode start and goal
    def _enc(img):
        if img.dim() == 3:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 6:
            img = img[:, :, :1]
        return model.encode({"pixels": img})["emb"]

    emb = _enc(agent_img)  # (1,1,D)
    emb_g = _enc(goal_img)  # (1,1,D)

    history: list[np.ndarray] = []
    for action_10d in plan:
        A = 10
        n_hist = min(len(history), H)
        hist_arr = np.zeros((0, A), dtype=np.float32) if len(history) == 0 else np.array(history[-n_hist:], dtype=np.float32)
        if len(history) < H:
            hist_arr = np.concatenate([
                np.zeros((H - len(history), A), dtype=np.float32),
                hist_arr,
            ], axis=0)
        hist_tensor = torch.from_numpy(hist_arr).float().to(device)
        cand = torch.from_numpy(action_10d).float().to(device).unsqueeze(0)
        history_for_pred = torch.cat([
            hist_tensor.unsqueeze(0)[:, :-1, :],
            cand.unsqueeze(1),
        ], dim=1)  # (1, H, A)

        with torch.no_grad():
            act_emb = model.action_encoder(history_for_pred)
            emb = model.predict(emb, act_emb)[:, -1:, :]  # (1,1,D)

    pred_final = emb.squeeze()  # (D,)
    goal_final = emb_g.squeeze()  # (D,)
    latent_mse = float((pred_final - goal_final).pow(2).mean().item())
    latent_dist = float((pred_final - goal_final).norm().item())
    return {"latent_mse": latent_mse, "latent_dist": latent_dist}


def run_receding(model, mlp, agent_pos, goal_pos, plan_horizon, max_steps, method, device, rng):
    """Run receding-horizon planning and return full action sequence + stats."""
    t0 = time.time()
    plan, info = receding_horizon_plan(
        model=model, mlp_heuristic=mlp if method in ("greedy_mlp", "cem_mlp") else None,
        agent_pos=agent_pos, goal_pos=goal_pos,
        plan_horizon=plan_horizon, max_steps=max_steps,
        method=method, device=device, rng=rng,
    )
    return plan, time.time() - t0, info


def main():
    parser = argparse.ArgumentParser(description="Evaluate JEPA planning strategies")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5, help="Planning horizon per re-plan")
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps (default: 3x horizon)")
    parser.add_argument("--receding", action="store_true", help="Use receding-horizon control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--mlp-ckpt",
        type=str,
        default="/Users/richard/Documents/Obsidian Vault/05 - CODE (Uni)/Projects/JEPA - Frank/mlp_euclid_distance_1m_all/2026-11-18_16-47-37_best_checkpoint.pth",
    )
    parser.add_argument("--output-dir", type=str, default="data/planning_benchmark")
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = args.horizon * 3

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    rng = np.random.default_rng(args.seed)

    # Load models
    print("=" * 60)
    print("Loading models...")
    if args.checkpoint:
        cp_path = args.checkpoint
    else:
        cp_path = snapshot_download(
            repo_id="quentinll/lewm-tworooms",
            revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec",
        )
    model = load_fix_pretrained(cp_path)
    model.eval().to(device)
    print(f"  JEPA model loaded on {device}")

    mlp = load_pretrained_mlp(args.mlp_ckpt, device=device)
    print(f"  MLP heuristic loaded on {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results: list[dict] = []

    for prob_i in range(args.num_problems):
        agent_pos, goal_pos = generate_problem(rng)
        initial_dist = float(np.linalg.norm(np.array(agent_pos) - np.array(goal_pos)))

        t_img_a, t_img_g = render_images(agent_pos, goal_pos, device)

        print(f"\n--- Problem {prob_i + 1}/{args.num_problems} ---")
        print(f"  Agent: {agent_pos}, Goal: {goal_pos}, Dist: {initial_dist:.1f}")

        if args.receding:
            methods = [
                ("rh_greedy_jepa", lambda m="greedy_jepa": run_receding(model, None, agent_pos, goal_pos, args.horizon, args.max_steps, m, device, rng)),
                ("rh_greedy_mlp", lambda m="greedy_mlp": run_receding(model, mlp, agent_pos, goal_pos, args.horizon, args.max_steps, m, device, rng)),
                ("rh_cem_mlp", lambda m="cem_mlp": run_receding(model, mlp, agent_pos, goal_pos, args.horizon, args.max_steps, m, device, rng)),
            ]
        else:
            methods = [
                ("greedy_mlp", lambda: run_greedy_mlp(model, mlp, t_img_a, t_img_g, args.horizon, device, rng)),
                ("greedy_jepa", lambda: run_greedy_jepa(model, t_img_a, t_img_g, args.horizon, device, rng)),
                ("cem", lambda: run_cem(model, t_img_a, t_img_g, args.horizon, device, rng)),
                ("cem_mlp", lambda: run_cem_mlp(model, mlp, t_img_a, t_img_g, args.horizon, device, rng)),
            ]

        for method, runner in methods:
            if args.receding:
                plan, plan_time, rh_info = runner()
                latent = evaluate_latent(plan, model, t_img_a, t_img_g, rh_info["steps_taken"])
                env_result = evaluate_plan(plan, agent_pos, goal_pos, rh_info["steps_taken"], method, plan_time)
                row = {
                    "problem": prob_i + 1,
                    "method": method,
                    "agent_pos": agent_pos,
                    "goal_pos": goal_pos,
                    "initial_distance": round(initial_dist, 2),
                    "latent_mse": round(latent["latent_mse"], 4),
                    "latent_dist": round(latent["latent_dist"], 4),
                    "env_final_distance": round(env_result.final_distance, 2),
                    "env_success": env_result.success,
                    "steps_taken": rh_info["steps_taken"],
                    "re_plans": rh_info["re_plans"],
                    "planning_time_s": round(plan_time, 4),
                }
            else:
                plan, plan_time = runner()
                latent = evaluate_latent(plan, model, t_img_a, t_img_g, args.horizon)
                env_result = evaluate_plan(plan, agent_pos, goal_pos, args.horizon, method, plan_time)

                row = {
                    "problem": prob_i + 1,
                    "method": method,
                    "agent_pos": agent_pos,
                    "goal_pos": goal_pos,
                "initial_distance": round(initial_dist, 2),
                "latent_mse": round(latent["latent_mse"], 4),
                "latent_dist": round(latent["latent_dist"], 4),
                "env_final_distance": round(env_result.final_distance, 2),
                "env_success": env_result.success,
                "steps_taken": env_result.steps_taken,
                "planning_time_s": round(plan_time, 4),
            }
            all_results.append(row)

            print(
                f"  {method:15s}  latent_dist={latent['latent_dist']:7.2f}  "
                f"latent_mse={latent['latent_mse']:7.4f}  "
                f"env_dist={env_result.final_distance:6.1f}  "
                f"time={plan_time:.3f}s"
            )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    methods = ["rh_greedy_jepa", "rh_greedy_mlp", "rh_cem_mlp"] if args.receding else ["greedy_mlp", "greedy_jepa", "cem", "cem_mlp"]
    summary_rows = []
    for method in methods:
        subset = [r for r in all_results if r["method"] == method]
        avg_latent_dist = np.mean([r["latent_dist"] for r in subset])
        avg_latent_mse = np.mean([r["latent_mse"] for r in subset])
        avg_env_dist = np.mean([r["env_final_distance"] for r in subset])
        avg_time = np.mean([r["planning_time_s"] for r in subset])
        env_success_rate = sum(1 for r in subset if r["env_success"]) / len(subset) * 100
        avg_steps = np.mean([r.get("steps_taken", args.horizon) for r in subset])
        avg_replans = np.mean([r.get("re_plans", 1) for r in subset]) if args.receding else 0
        steps_label = f"avg_steps={avg_steps:.1f}  replans={avg_replans:.1f}  " if args.receding else ""
        print(
            f"  {method:15s}  avg_latent_dist={avg_latent_dist:7.2f}  "
            f"avg_latent_mse={avg_latent_mse:7.4f}  "
            f"avg_env_dist={avg_env_dist:6.1f}  "
            f"env_success={env_success_rate:.0f}%  "
            f"{steps_label}"
            f"avg_time={avg_time:.3f}s"
        )
        summary_rows.append({
            "method": method,
            "avg_latent_dist": round(avg_latent_dist, 4),
            "avg_latent_mse": round(avg_latent_mse, 4),
            "avg_env_final_distance": round(avg_env_dist, 2),
            "env_success_rate_pct": round(env_success_rate, 1),
            "avg_planning_time_s": round(avg_time, 4),
            "avg_steps_taken": round(avg_steps, 1),
            "avg_re_plans": round(avg_replans, 1) if args.receding else None,
            "n_problems": len(subset),
        })

    # Save
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"{ts}_planning_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "summary": summary_rows, "config": vars(args)}, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
