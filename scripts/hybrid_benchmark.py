"""Benchmark greedy-initialized CEM vs standard CEM."""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from utils.heuristic_planner import GreedyInitCEM, load_pretrained_mlp
from utils.cem_path_sampler import CEMPathSampler
from utils.image_preprocessing import normalise_images
from stable_worldmodel.envs.two_room import TwoRoomEnv
from huggingface_hub import snapshot_download
from utils.load_fix_pretrained import load_fix_pretrained


def generate_problem(rng):
    env = TwoRoomEnv()
    agent_pos = rng.integers(10, 200, size=2).tolist()
    goal_pos = rng.integers(10, 200, size=2).tolist()
    while abs(agent_pos[0] - goal_pos[0]) < 20 and abs(agent_pos[1] - goal_pos[1]) < 20:
        goal_pos = rng.integers(10, 200, size=2).tolist()
    env.reset(options={"state": agent_pos})
    initial_dist = float(torch.norm(torch.tensor(agent_pos, dtype=torch.float) - torch.tensor(goal_pos, dtype=torch.float)).item())
    return agent_pos, goal_pos, initial_dist


def make_images(agent_pos, goal_pos, device):
    env = TwoRoomEnv()
    env.reset(options={"state": agent_pos})
    img_a = env.render()
    t_img_a = normalise_images(torch.from_numpy(img_a).permute(2, 0, 1).float().unsqueeze(0)).squeeze(0).to(device)
    env.reset(options={"state": goal_pos})
    img_g = env.render()
    t_img_g = normalise_images(torch.from_numpy(img_g).permute(2, 0, 1).float().unsqueeze(0)).squeeze(0).to(device)
    return t_img_a, t_img_g


def evaluate_latent(plan, model, t_img_a, t_img_g, horizon):
    device = next(model.parameters()).device
    H = 3
    emb = model.encode({"pixels": t_img_a.unsqueeze(0).unsqueeze(0)})["emb"]
    emb_g = model.encode({"pixels": t_img_g.unsqueeze(0).unsqueeze(0)})["emb"]
    history = []
    for action_10d in plan:
        n_hist = min(len(history), H)
        hist_arr = np.zeros((0, 10), dtype=np.float32) if not history else np.array(history[-n_hist:], dtype=np.float32)
        if len(history) < H:
            hist_arr = np.concatenate([np.zeros((H - len(history), 10), dtype=np.float32), hist_arr], axis=0)
        hist_tensor = torch.from_numpy(hist_arr).float().to(device)
        cand = torch.from_numpy(action_10d).float().to(device).unsqueeze(0)
        history_for_pred = torch.cat([hist_tensor.unsqueeze(0)[:, :-1, :], cand.unsqueeze(1)], dim=1)
        with torch.no_grad():
            act_emb = model.action_encoder(history_for_pred)
            emb = model.predict(emb, act_emb)[:, -1:, :]
    pred_final = emb.squeeze()
    goal_final = emb_g.squeeze()
    latent_mse = float((pred_final - goal_final).pow(2).mean().item())
    latent_dist = float((pred_final - goal_final).norm().item())
    return latent_dist, latent_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/planning_benchmark")
    args = parser.parse_args()

    print("Loading models...")
    cp_path = snapshot_download(repo_id="quentinll/lewm-tworooms", revision="77adaae0bc31deab21c93740d1f8bb947cd0bdec")
    model = load_fix_pretrained(cp_path)
    model = model.to(args.device).eval()
    print(f"  JEPA loaded on {args.device}")

    rng = np.random.default_rng(args.seed)
    results = {"standard_cem": [], "hybrid_cem": []}

    for pi in range(args.num_problems):
        agent_pos, goal_pos, initial_dist = generate_problem(rng)
        t_img_a, t_img_g = make_images(agent_pos, goal_pos, args.device)

        # Standard CEM
        sampler_std = CEMPathSampler(
            model=model, history_size=3, action_dim=10,
            action_low=-1.0, action_high=1.0, elite_fraction=0.2, rng=rng,
        )
        t0 = time.time()
        plan_std = sampler_std.sample_path(
            agent_image=t_img_a, goal_image=t_img_g,
            horizon=args.horizon, num_generations=5, batch_size=16, device=args.device,
        )
        time_std = time.time() - t0
        ld_std, lm_std = evaluate_latent(plan_std.cpu().numpy(), model, t_img_a, t_img_g, args.horizon)

        # Hybrid CEM (greedy-init)
        sampler_hyb = GreedyInitCEM(
            model=model, history_size=3, action_dim=10,
            action_low=-1.0, action_high=1.0, elite_fraction=0.2, rng=rng,
        )
        t0 = time.time()
        plan_hyb = sampler_hyb.sample_path(
            agent_image=t_img_a, goal_image=t_img_g,
            horizon=args.horizon, num_generations=5, batch_size=16, device=args.device,
        )
        time_hyb = time.time() - t0
        ld_hyb, lm_hyb = evaluate_latent(plan_hyb.cpu().numpy(), model, t_img_a, t_img_g, args.horizon)

        results["standard_cem"].append({
            "problem": pi + 1,
            "initial_dist": round(initial_dist, 2),
            "latent_dist": round(ld_std, 4),
            "latent_mse": round(lm_std, 4),
            "plan_time": round(time_std, 4),
        })
        results["hybrid_cem"].append({
            "problem": pi + 1,
            "initial_dist": round(initial_dist, 2),
            "latent_dist": round(ld_hyb, 4),
            "latent_mse": round(lm_hyb, 4),
            "plan_time": round(time_hyb, 4),
        })

        if (pi + 1) % 5 == 0:
            print(f"  Problem {pi+1}/{args.num_problems}: std={ld_std:.2f}  hybrid={ld_hyb:.2f}  Δ={ld_hyb-ld_std:+.2f}")

    # Summary
    std_avg = np.mean([r["latent_dist"] for r in results["standard_cem"]])
    hyb_avg = np.mean([r["latent_dist"] for r in results["hybrid_cem"]])
    std_t = np.mean([r["plan_time"] for r in results["standard_cem"]])
    hyb_t = np.mean([r["plan_time"] for r in results["hybrid_cem"]])

    print(f"\n{'='*50}")
    print(f"SUMMARY (h={args.horizon}, {args.num_problems} problems)")
    print(f"{'='*50}")
    print(f"  standard_cem: latent={std_avg:.2f}  time={std_t:.2f}s")
    print(f"  hybrid_cem:   latent={hyb_avg:.2f}  time={hyb_t:.2f}s")
    print(f"  Δ (hybrid - std): {hyb_avg - std_avg:+.2f}")
    if hyb_avg < std_avg:
        print(f"  ✓ Hybrid IMPROVES by {std_avg - hyb_avg:.2f}")
    else:
        print(f"  ✗ Hybrid WORSENS by {hyb_avg - std_avg:.2f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"{ts}_hybrid_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "config": vars(args),
            "summary": {
                "standard_cem_avg_latent": round(std_avg, 4),
                "hybrid_cem_avg_latent": round(hyb_avg, 4),
                "delta": round(hyb_avg - std_avg, 4),
                "standard_cem_avg_time": round(std_t, 4),
                "hybrid_cem_avg_time": round(hyb_t, 4),
            }
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
