"""
Train an OpenFrontIO RL agent using MaskablePPO (sb3-contrib).

Usage:
    pip install -r rl/requirements.txt
    python rl/train.py

Checkpoints and logs are saved to rl/runs/<run_name>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

from env import OpenFrontEnv  # noqa: E402
from render_callback import RenderCallback  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────

RUN_NAME = "ppo_v9_world_10algobots_parallel"
N_ENVS = 4  # parallel envs (each spawns its own runner subprocess)
RESUME_FROM = str(
    Path(__file__).parent
    / "runs/ppo_v8_bc_warmstart/checkpoints/rl_model_150000_steps.zip"
)
BC_INIT_FROM = None  # not needed when resuming
TOTAL_TIMESTEPS = 5_000_000
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 3e-4
GAMMA = 0.995
ENT_COEF = 0.01
CLIP_RANGE = 0.2

def _select_device() -> str:
    """Pick training device. TRAIN_DEVICE env var ('cuda'/'mps'/'cpu') overrides
    auto-detection — useful for forcing CPU on servers with shared GPUs, or
    pinning to a specific backend."""
    import os as _os
    forced = _os.environ.get("TRAIN_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()

RUNS_DIR = Path(__file__).parent / "runs" / RUN_NAME
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def mask_fn(env: OpenFrontEnv):
    return env.action_masks()


def make_env_factory(rank: int, runs_dir: Path):
    """Returns a zero-arg callable that creates a fresh wrapped env.

    Used by SubprocVecEnv: each subprocess pickles + calls one factory.
    """
    def _init() -> Monitor:
        env = OpenFrontEnv(project_root=PROJECT_ROOT)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env, str(runs_dir / f"monitor_{rank}"))
        return env

    return _init


# ── Train ────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Training on device: {DEVICE}  (override with TRAIN_DEVICE=cuda|mps|cpu)")
    print(f"Run dir: {RUNS_DIR}")
    print(f"Parallel envs: {N_ENVS}")

    print(f"Spawning {N_ENVS} parallel training envs...")
    train_env = SubprocVecEnv(
        [make_env_factory(i, RUNS_DIR) for i in range(N_ENVS)],
        start_method="spawn",
    )
    # Eval env stays single (sequential) — keeps eval deterministic and avoids
    # extra runner subprocesses competing with training.
    eval_env = DummyVecEnv([make_env_factory(99, RUNS_DIR)])

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(RUNS_DIR / "checkpoints"),
        name_prefix="rl_model",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(RUNS_DIR / "best_model"),
        log_path=str(RUNS_DIR / "eval_logs"),
        eval_freq=100_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    render_cb = RenderCallback(RUNS_DIR, render_every=200)

    if RESUME_FROM and Path(RESUME_FROM).exists():
        print(f"Resuming policy weights from: {RESUME_FROM}")
        model = MaskablePPO.load(
            RESUME_FROM,
            env=train_env,
            device=DEVICE,
            # Hyperparameters can be overridden when loading; keep current settings
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            verbose=1,
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            device=DEVICE,
            verbose=1,
        )

        # BC warm-start: copy BC-trained weights into the policy network
        if BC_INIT_FROM and Path(BC_INIT_FROM).exists():
            print(f"Warm-starting actor from BC model: {BC_INIT_FROM}")
            bc_state = torch.load(BC_INIT_FROM, map_location=DEVICE)
            pol = model.policy
            with torch.no_grad():
                # MaskablePPO MlpPolicy uses mlp_extractor.policy_net (Sequential)
                # composed of Linear-Tanh-Linear-Tanh; weights live at indices 0 and 2.
                pol.mlp_extractor.policy_net[0].weight.copy_(bc_state["fc1.weight"])
                pol.mlp_extractor.policy_net[0].bias.copy_(bc_state["fc1.bias"])
                pol.mlp_extractor.policy_net[2].weight.copy_(bc_state["fc2.weight"])
                pol.mlp_extractor.policy_net[2].bias.copy_(bc_state["fc2.bias"])
                pol.action_net.weight.copy_(bc_state["action_head.weight"])
                pol.action_net.bias.copy_(bc_state["action_head.bias"])
            print("  ✓ BC weights loaded into policy actor (critic stays random)")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, render_cb],
        progress_bar=False,
        reset_num_timesteps=True,  # fresh counter for the new run/log dir
    )

    final_path = str(RUNS_DIR / "final_model")
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
