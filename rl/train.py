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

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(__file__).parent))

from cnn_policy import COMBINED, OpenFrontExtractor  # noqa: E402
from env import OpenFrontEnv  # noqa: E402
from render_callback import RenderCallback  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────────

RUN_NAME = "ppo_openfront_v2_cnn"
TOTAL_TIMESTEPS = 5_000_000
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 3e-4
GAMMA = 0.995
ENT_COEF = 0.01
CLIP_RANGE = 0.2

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

RUNS_DIR = Path(__file__).parent / "runs" / RUN_NAME
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_env() -> OpenFrontEnv:
    return OpenFrontEnv(project_root=PROJECT_ROOT)


def mask_fn(env: OpenFrontEnv):
    return env.action_masks()


# ── Train ────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Training on device: {DEVICE}")
    print(f"Run dir: {RUNS_DIR}")

    train_env = ActionMasker(make_env(), mask_fn)
    train_env = Monitor(train_env, str(RUNS_DIR / "monitor"))

    eval_env = ActionMasker(make_env(), mask_fn)
    eval_env = Monitor(eval_env, str(RUNS_DIR / "eval_monitor"))

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
    render_cb = RenderCallback(RUNS_DIR, render_every=25)

    policy_kwargs = dict(
        features_extractor_class=OpenFrontExtractor,
        # Actor/critic nets after the combined extractor (COMBINED = 384)
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, render_cb],
        progress_bar=False,
    )

    final_path = str(RUNS_DIR / "final_model")
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
