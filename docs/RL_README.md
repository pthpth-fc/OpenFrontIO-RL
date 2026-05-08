# Reinforcement Learning — OpenFrontIO Bot

Documentation for the RL training pipeline that learns to play OpenFrontIO
against AlgoBot opponents.

## Quick start

```bash
# Install (Node side)
npm run inst

# Install (Python side)
pip install -r rl/requirements.txt

# Train (auto-detects cuda > mps > cpu; override with TRAIN_DEVICE=cpu)
python rl/train.py

# Evaluate a checkpoint on world map vs 200 opponents
python rl/eval_world.py rl/runs/<run-name>/checkpoints/<ckpt>.zip

# Inspect the action distribution of a learned policy
python rl/action_distribution.py <ckpt> 10
```

## Document index

- **`RL_ARCHITECTURE.md`** — high-level architecture: Node ↔ Python pipe,
  PPO setup, custom CNN+MLP feature extractor.
- **`RL_OBSERVATION.md`** — exhaustive spec of obs (81-dim flat + 5×32×32
  spatial), action space (Discrete 46), and action mask. Read this first
  if you're trying to understand a specific policy decision.
- **`RL_RUNS.md`** — full chronological history of runs v1 → v10 with
  outcomes, action distributions, and the reasoning behind each pivot.

## At-a-glance pipeline

```
              Game (TypeScript, lockstep deterministic)
                          │
                          ▼
       runner.ts — extracts obs, decodes actions, JSON-RPC
                          │
                          ▼  stdin/stdout
         env.py (gymnasium.Env, one per parallel env)
                          │
                          ▼
   MaskablePPO (sb3-contrib) on MPS / CUDA / CPU (env-controlled)
            │                        │
   ┌────────┴────────┐       ┌───────┴────────┐
   │  CNN+MLP        │       │  Discrete(46)  │
   │  feature        │ ───►  │  action head   │ ──► back to runner
   │  extractor      │       │  + masking     │
   └─────────────────┘       └────────────────┘
```

## Versions at a glance

| Version | Status | Best reward |
|---------|--------|-------------|
| v1–v3 | superseded | <+1 |
| v4 | superseded | +6 |
| v5–v7 | superseded | +6.4 |
| **v8** | best policy on big_plains+3 AlgoBots | **+27** |
| v9 | adapting v8 to world+10 AlgoBots | +13 (still climbing when stopped) |
| **v10** | under construction — Dict obs, CNN, K=10, boats | TBD |

See `RL_RUNS.md` for what's behind each number.

## Where to look in the code

| What | File |
|------|------|
| Game-side obs/action plumbing | `src/rl/headless/` |
| Python env wrapper | `rl/env.py` |
| Training driver | `rl/train.py` |
| BC pipeline (demos + supervised pre-training) | `rl/collect_demos.py`, `rl/bc_train.py` |
| Eval / rendering | `rl/eval_world.py`, `rl/render_game.py`, `rl/generate_episode.py` |
| Diagnostics | `rl/action_distribution.py` |
