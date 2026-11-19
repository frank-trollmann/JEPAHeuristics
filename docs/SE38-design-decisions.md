# SE 38 - Design Decision Discussion - Planning with a frozen JEPA world model

**Student:** [your name] · **Project:** "Planning for JEPA" (CODE, supervisor Frank Trollmann) · **Code:** branch `adi/planning` of `github.com/frank-trollmann/JEPAHeuristics`

Every result below points to the code that produced it and to its entry in the experiment record `docs/research-log.md` (written "log, entry X"). The log has the full setup, raw-result files and the limits of each experiment.

---

## My Problem

### Motivation
A robot that only has a camera has to plan using images. A **world model** learns from video how the scene changes when the robot acts, so the robot can imagine the result of a plan before it moves. LeWorldModel (LeWM, a JEPA world model) does this in a learned 192-dimensional **latent space**: an encoder turns an image into a latent vector *z*, and a predictor maps (*z*, action) to the next *z*. Its planner is CEM with a fixed cost: the squared distance between the imagined final latent and the goal image's latent.

The test case is **TwoRoom**: two rooms separated by a wall with a single door. When the goal is behind the wall, the straight line to it is blocked. A cost that measures "how close am I in a straight line" then points the agent into the wall. That is a classic **local minimum** for planning, and it is the problem this project studies.

**Research questions.** RQ1: can a better heuristic *h*(z_agent, z_goal) be learned on top of the frozen latent space? RQ2: how do planning algorithms (sampling: CEM, MPPI, iCEM, …; search: greedy, A*, weighted A*, beam, MCTS) compare under different heuristics, including JEPA's own cost?

### Formal problem

| Element | Real environment (used for execution and scoring) | What the planner works with |
|---|---|---|
| **State** | Agent position (x, y) in a 224 × 224 px image. Fixed layout: wall at x = 112 (10 px thick), door centred at y = 49 (half-width 14), agent radius 7. 29,025 valid integer positions. | A window of latents from the frozen encoder (192-d each). The planner **never sees x, y**: it only gets the current image and the goal image. |
| **Actions** | Continuous 2-D action in [-1, 1]², applied for 5 environment steps (one *action block*, about 25 px at full speed). | Sampling planners: a sequence of 5 blocks in the continuous action space (z-scored with the dataset mean and std). Search planners: 8 full-speed compass directions per block (a discretisation I chose, see My Solution). |
| **Transition model** | Deterministic physics: the agent moves by the action, and the wall and borders stop it. | The **frozen LeWM predictor** P(latent window, action block) → next latent. It is learned, so it can be wrong. I checked that it knows the wall: a move into the wall is predicted as about 8 px versus 26 px through the door (log, entry D1). |
| **Goal** | Agent within 16 px of the goal position (the environment's own test). The goal states are all valid positions within that disc. | The goal latent from the goal image. A plan scores well when the heuristic says the imagined final latent is close to it. Search stops when the estimate is below 10 px. |

**Execution (receding horizon).** The planner plans 5 blocks, all 5 run in the real simulator, and then it replans from the new real image. This is LeWM's own evaluation protocol (`utils/cem_planner.py`, `run_episode`). An episode fails if the goal is not reached within 150 environment steps.

**Task sets.** Starts and goals are drawn from held-out positions (the test split), at least 60 px apart, in three kinds that separate the difficulty (`scripts/run_planning_eval.py`, `sample_tasks`):
- `same_room`: start and goal in the same room;
- `cross_near`: across the wall, with the shortest path at most 15% longer than the straight line;
- `cross_far`: across the wall, with the shortest path at least 50% longer. This is **the hard case**, where the wall blocks the straight line.

### Quality criteria and why

| Criterion | Measure | Why it follows from the application |
|---|---|---|
| **1. Success** (primary) | Fraction of episodes that reach the goal, with a 95% Wilson interval | A navigation plan is worthless if it doesn't arrive. Reported per task kind, because averaging would hide the hard case. |
| **2. Compute per decision** | Predictor calls per episode (exact, the same on any hardware). Wall-clock seconds as a secondary measure. | On a robot the planner runs inside the control loop, and every imagined step is a forward pass of the predictor network. |
| **3. Path quality** | Steps to the goal; on the exact grid, path cost divided by the optimal cost | A detour costs time and energy. It is secondary because the environment rewards only arrival. |
| **4. Robustness** | Change in success when settings (merge distance, budget, beam width, replanning rate) change, and on a second task draw | A planner whose success depends on a lucky setting is hard to use on a new problem. |

---

## My Options

**Sampling planners.** These optimise a whole action sequence by sampling, imagining each sample with the predictor and scoring it. They are typical for continuous-control model-predictive control.
- **CEM (Cross-Entropy Method).** Sample 300 plans from a Gaussian, keep the best 30, refit the Gaussian to them, and repeat 30 times. This is LeWM's own planner and the baseline.
- **MPPI.** Like CEM, but every elite plan is weighted by a softmax of its cost and the mean is the weighted average; the spread stays fixed. Common in robotics and autonomous driving.
- **iCEM.** CEM plus time-correlated ("coloured") noise, reuse of the best plans across iterations, and clipping to the action bounds. A more sample-efficient CEM for robot control.
- *Also implemented, as controls:* random shooting (the floor), gradient descent on the plan through the predictor (a local method), and **snapped CEM**, which restricts CEM to the same 8 directions search uses so the action set is not a confound.

**Search planners through the predictor.** These build a tree whose nodes are *imagined* latents. Each node has 8 children (one compass block each), predicted in one batch, and nodes that predict nearly the same latent are merged. They are typical for discrete path planning.
- **Greedy best-first.** Always expand the node with the smallest heuristic h. Fast, but it can be misled.
- **A\* and weighted A\*.** Expand by g + h (g = pixels walked) or g + w·h. A* is optimal with an admissible heuristic; w > 1 trades optimality for speed.
- **Beam search.** Expand level by level and keep only the best k nodes per level. Cheap, with limited memory.
- **MCTS (UCT).** Balance exploiting good branches against exploring rarely visited ones. Used for large branching factors and in games.
- *Classical reference:* DFS, greedy, A* and weighted A* on the **exact** position grid, with true transitions. This separates "the heuristic is good" from "the predictor is good".

**Heuristics (costs).** JEPA's own cost; a **learned heuristic** (an MLP on [z_agent, z_goal] trained on true wall-aware distances); a **frame-order heuristic** (an MLP trained only on how many steps apart two frames are in recorded trajectories, so it uses no coordinates); and **decoded** (read (x, y) from the latent, then the exact wall-aware distance), as a check.

---

## My Solution

### Chosen: A* through the predictor with the learned heuristic

Beam search (width 5) is the cheaper alternative with the same success. I chose A* for three reasons, each backed by results.

**1. Out-of-the-box planning fails on this problem (why a design was needed).** With JEPA's own cost, **every** sampling planner fails on the hard tasks: CEM 0.02, iCEM 0.06, MPPI 0.02, snapped CEM 0.06, random 0.06, gradient descent 0.00 (log, entries P1 and P2). Analysis of the failed episodes shows them stuck at the wall (log, entry F1). The JEPA cost is also a poor distance in general: it explains only R² = 0.10 of the wall-aware distance, worse than the straight line at 0.40 (log, entries E2 and E3).

**2. The two design decisions, and what each fixes:**

| Decision | Problem property it answers | Evidence (cross_far success) |
|---|---|---|
| **Replace the cost with a learned heuristic** (RQ1) | The JEPA cost points into the wall | CEM 0.02 → 0.48; iCEM 0.06 → 0.96; snapped CEM 0.06 → 1.00 |
| **Search the predictor's tree instead of sampling** (RQ2) | A local minimum: a planner that refits one distribution to the best samples gets trapped | Even with the bad JEPA cost: A*, greedy, weighted A* 0.94-0.96, beam 0.92. Snapped CEM with the same 8 actions: 0.06, so the action set is not the reason. |

**3. Why A* specifically:**

| Planner + heuristic | far success (50 / 200 episodes) | predictor calls per episode | seconds per episode | robust to settings? |
|---|---|---|---|---|
| **A\*, learned** | 0.94 / **0.95** | ~1.9k | 1.0 | yes |
| beam (w = 5), learned | 0.92 / 0.96 | ~0.8k | 0.2 | yes |
| snapped CEM, learned | 1.00 / *pending* | ~100-150k | 4.0 | worse when replanning every 2 blocks (0.86) |
| iCEM, learned | 0.96 / 0.98 | ~100k | 4.2 | worse when replanning every 2 blocks (0.72) |
| MCTS + merging, learned | 0.92 / – | ~1.0k | – | merging is essential (0.54 without it) |
| CEM (LeWM default), learned | 0.48 / *pending* | ~270k | 8.3 | – |

(log, entries P1, P2 and P3; `docs/monday-prep/evidence/planner_comparison_far.png`.) Snapped CEM and iCEM succeed slightly more often, but the difference is within the statistical tie range at 50 episodes and costs 50-100 times more predictor calls. A* is the best overall trade-off, and it is the only family that **keeps working when the heuristic is bad** (0.94-0.96 with the JEPA cost). That matters because a learned heuristic can be wrong in a place you haven't tested.

### How the heuristic was built and selected

- **Data.** Every valid position is rendered and encoded once, together with its coordinates (`utils/position_dataset.py`). The split is by **position**, 70/15/15, and pairs are built inside one split only (`utils/pair_dataset.py`), so test positions are never seen in training. Images are scaled to [0, 1] and ImageNet-normalised as the encoder expects. Without that step the position error is 14.3 px instead of 0.9 px (log, entry PP1; fixed for the team in PR #6).
- **Label.** I compared Euclidean, grid Dijkstra, and the continuous door formula (shortest path around a wall grown by the agent radius) (`utils/reference_distance.py`; log, entry L1). I chose the **formula** label because it is closest to the real geometry: the grid overestimates by about 6% because of its 8-direction steps.
- **Model.** An MLP with layers 384-256-256-1 on [z_agent, z_goal], trained on 1M pairs (`utils/heuristic_models.py`, `PairMLP`, `train_pair_model`). Selection on validation pairs (log, entries E1-E3):

| model (test, formula label) | MAE px | overestimates > 2 px |
|---|---|---|
| mean guess / JEPA-cost proxy / linear regression | 60.3 / 56.2 / 54.5 | ~50% |
| metric model ‖f(z_a) − f(z_g)‖ | 1.70 | 18.9% |
| MLP + difference input | 0.64 | 1.3% |
| **MLP (chosen)** | **0.61** | **1.4%** |

  The error keeps falling with more data (5.3 px at 10k pairs, 0.49 px at 4M). I chose 1M pairs as the trade-off between accuracy and training time. The MLP rarely overestimates, so it is close to admissible for A* in practice; W1 showed this is because the formula label is shorter than the grid cost, not a guarantee.
- **Without privileged labels.** The frame-order heuristic, trained without coordinates, matches the learned one with CEM (far 0.48 vs 0.49, pooled over two seeds) and under search (A* 0.94) (log, entries T1, CEM2 and P1).

### How the planner parameters were selected

| Parameter | Value | How it was chosen |
|---|---|---|
| Sampler budget | 300 samples, 30 iterations, top 30, horizon 5 blocks, execute 5 blocks | Taken from LeWM's config so the baseline is the published one. My CEM was calibrated against the official `eval.py` on the same checkpoint: 84% (200 episodes) vs 87.6% (500), within noise (log, entries C1 and C2). |
| Search actions | 8 full-speed compass blocks | Most recorded actions are ±1 per component (63%). Snapped CEM uses the same set as a control. |
| Merge distance | 12 latent units | Swept 6 / 12 / 20. At 20, distinct positions merge (0.00); at 6, the JEPA cost drops to 0.20 (log, entry P2). |
| Expansions / depth | 100 / 12 blocks | Swept 25 / 100 / 200. The learned heuristic is flat (0.94 at each); the JEPA cost needs ≥ 100 (0.60 at 25). |
| Beam width | 5 | Width 1: learned 0.98 but JEPA 0.14. Width 5 works for both. |
| MCTS c, merging | 0.7, merge 12 | c swept 0.3 / 0.7 / 1.5. Merging raised far success from 0.54 to 0.92 (log, entry P3). |
| Pixel scale for non-pixel costs | straight-line fit on validation pairs | Search adds g (px) to h, so h must be in px. Exposed bug: early stop at the root; fixed with `--stop-at 0` for the frame-order heuristic (log, entry P1). |
| MPPI temperature | 0.5 × elite cost spread | Makes the planner independent of cost scale; 0.1 was also tried. |

---

## My Evaluation

### Part 1: Correctness and completeness (does it find an action sequence that reaches the goal?)

**Correctness of the implementation.**
- **82 unit tests** (`.venv/Scripts/python.exe -m pytest utils/tests -q`, all pass). Search and geometry are checked against hand-computed cases: a U-shaped corridor, a corner-cutting case, a case where A* must re-open a node, and the door formula against independent bounds (228.6 px between 214.2 and 235.1).
- **Mutation checks.** For each new planner I broke the code on purpose and confirmed a test fails. This found and fixed two weak tests (log, entry P2).
- **Calibration against the original.** My CEM vs the official LeWM evaluation on the same checkpoint: 84% vs 87.6% (log, entries C1 and C2). Rendered frames are pixel-identical to the dataset frames.
- **Reproducibility.** A rerun on 2026-11-19 reproduced an earlier cell exactly (decoded CEM: 0.62 = 0.62), and a second draw of tasks and heuristic (seed 4) matches seed 2 within 0.04 on every cell (log, entries P2 and P3).
- **Mistakes found and reported:** the encoder preprocessing missing in the team's data, a calibration bug in the early stop, two weak tests, and two geometry bugs caught by hand checks (log, entries PP1, P1, P2 and L1).

**Completeness and optimality.**
- *On the exact grid* (true transitions), A* is complete and finds a path on every task. With the learned heuristic it is optimal on 97-100% of tasks, and weighted A* (w = 1.5) never exceeds its w·optimal bound while using about 25 times fewer expansions (log, entries S1 and W1). Trained on the grid label itself, the MLP overestimates at 42% of nodes and A* loses optimality on 9% of far tasks. Its admissibility is therefore empirical, not guaranteed.
- *Through the predictor*, the planner is **not complete by construction**: the tree has a budget (100 expansions, depth 12), the action set is discretised, and the transition model is learned and can be wrong. Receding-horizon replanning corrects the predictor's errors with real observations, so completeness is measured empirically as success rate.
- **Where the remaining failures come from.** With an exact-geometry heuristic (decoded), search still reaches only 0.92 on far tasks, no better than the learned 0.94-0.96. So the last ~5% of failures come from the search budget, the predictor's imagined transitions, or the end game near the goal, not from the heuristic (log, entry P3). For CEM, the failures split into "stuck at the wall" and "crossed but hovering 24-32 px from the goal" (log, entry F1).

### Part 2: Quality criteria

**1. Success.** Full grid, hard tasks, 50 episodes per cell (log, entries P1-P3):

| Planner | JEPA cost | Learned heuristic |
|---|---|---|
| A*, greedy, weighted A* | 0.94-0.96 | 0.94-0.96 |
| Beam (w = 5) | 0.92 | 0.92 |
| MCTS (with merging) | 0.72 | 0.92 |
| iCEM | 0.06 | 0.96 |
| Snapped CEM | 0.06 | 1.00 |
| CEM | 0.02 | 0.48 |
| Gradient descent | 0.00 | 0.40 |
| Random shooting | 0.06 | 0.16 |
| MPPI (fixed / adapted spread) | 0.02 / 0.04 | 0.00 / 0.18 |

Same-room and near tasks are easy for almost every planner with the learned heuristic (0.88-1.00; the exception is MPPI with a fixed spread on near tasks, 0.66; random shooting was run on far tasks only). **Replication with 200 new hard tasks** (interval about ±3 points): A* 0.95 / 0.96, greedy 0.95 / 0.96, beam 0.96 / 0.93 (learned / JEPA), and iCEM learned 0.98. The 50-episode conclusions hold. *Samplers with the JEPA cost on 200 tasks: running at the time of writing.*

**2. Compute.** Search uses about 1-2k predictor calls per episode versus 100-270k for the samplers, 50-250 times fewer. In seconds the advantage is smaller (4-40 times: beam 0.2 s, A* 1.0 s, iCEM 4.2 s, CEM 8.3 s per episode) because samplers score 300 plans in one GPU batch, while search expands 8 children at a time.

**3. Path quality.** On the exact grid, greedy with the learned heuristic needs 24 times fewer expansions than with Euclidean on far goals for the same path quality (1.11 vs 1.12 times optimal). Weighted A* with w = 1.5 gives paths 2.2% longer than optimal (log, entries S1 and W1). In the same room the learned heuristic is slightly worse than Euclidean (A* needs 1.6 times the expansions), because its small noise costs more than its wall knowledge gains there.

**4. Robustness.** With the learned heuristic, A* is insensitive to its settings (0.88-0.98 across merge distance 6-12, 25-200 expansions, and beam width 1-5). With the JEPA cost the same changes drop it to 0.14-0.60. **A good heuristic buys robustness, not only accuracy.** Replanning more often does not help search, and it hurts the samplers (iCEM 0.96 → 0.72).

### What the results say
1. **The cost is the main obstacle, but the planner decides whether it can be overcome.** A frontier of alternatives (A*, beam, MCTS with merging) escapes the wall's local minimum even with JEPA's cost. Refitting one distribution to the best samples does not.
2. **The wall knowledge lives in the predictor's dynamics, not in latent distance** (R² 0.10). That is why searching through the predictor works without a wall-aware heuristic.
3. **A heuristic can be learned on the frozen latent space (RQ1: yes)**, to about 0.6 px. It can also be learned without coordinates, from frame order alone. It turns failing samplers into working ones.
4. **With a good heuristic the choice of planner matters less, but not nothing.** MPPI, plain CEM and local gradient descent still fail on the hard tasks, and the reason sits in the algorithm (diagnosed for MCTS, partly diagnosed for MPPI).

### Limits (what is not shown)
One map and one checkpoint; 50 episodes per cell for most comparisons, so only gaps of about 20 points are claims (200 for the headline cells). The learned heuristic uses privileged labels (the frame-order one does not, but its demonstrations may already be door-aware). Converting non-pixel costs to pixels for search uses a fit on true distances. Search settings were tuned on this map. Why snapping helps CEM and why MPPI fails on far tasks are hypotheses. Full list: `docs/writeup/planning-experiments-writeup.md` §8 and the "not shown" item of each log entry.

### How to reproduce
```
.venv/Scripts/python.exe -m pytest utils/tests -q
.venv/Scripts/python.exe -m scripts.run_planner_comparison --planners astar icem --heuristics learned jepa --kinds cross_far --tag _demo
.venv/Scripts/python.exe -m scripts.summarize_planner_comparison
```
