"""
Run sample episodes with a saved checkpoint and save the first one that
reaches a target reward as a GIF.

Usage:
    python rl/generate_episode.py [checkpoint.zip] [target_reward]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from env import OBS_SIZE  # noqa: E402
from render import render_episode  # noqa: E402

CHECKPOINT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(PROJECT_ROOT / "rl/runs/ppo_v4_richer_obs_rewards/checkpoints/rl_model_200000_steps.zip")
)
TARGET_REWARD = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
MAX_TRIES = 12


def main() -> None:
    print(f"Loading {CHECKPOINT}")
    import os, torch
    device = os.environ.get("TRAIN_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    model = MaskablePPO.load(CHECKPOINT, device=device)

    best = None  # (reward, snapshots)

    for attempt in range(1, MAX_TRIES + 1):
        proc = subprocess.Popen(
            ["npm", "run", "--silent", "rl:runner"],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        def send(msg: dict) -> dict:
            assert proc.stdin and proc.stdout
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            return json.loads(proc.stdout.readline())

        try:
            resp = send({"cmd": "reset"})
            obs = np.array(resp["vec"], dtype=np.float32)
            mask = np.array(resp["mask"], dtype=bool)

            snapshots = [send({"cmd": "snapshot"})]
            total_reward = 0.0
            done = False

            while not done:
                action, _ = model.predict(
                    obs.reshape(1, -1),
                    action_masks=mask.reshape(1, -1),
                    deterministic=False,  # stochastic — different rollouts each attempt
                )
                action_int = int(action[0]) if hasattr(action, "__len__") else int(action)

                resp = send({"cmd": "step", "action": action_int})
                obs = np.array(resp["vec"], dtype=np.float32)
                mask = np.array(resp["mask"], dtype=bool)
                total_reward += float(resp["reward"])
                done = bool(resp["done"])

                snapshots.append(send({"cmd": "snapshot"}))

        finally:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
            proc.terminate()

        print(f"  attempt {attempt}: reward={total_reward:+.3f} ({len(snapshots)} frames)")

        if best is None or total_reward > best[0]:
            best = (total_reward, snapshots)

        if total_reward >= TARGET_REWARD:
            print(f"  ✓ hit target {TARGET_REWARD:.1f}, stopping")
            break

    out = PROJECT_ROOT / "rl/runs/ppo_v4_richer_obs_rewards/sample_episode.gif"
    print(f"\nSaving best episode (reward={best[0]:+.3f}) → {out}")
    render_episode(best[1], out, scale=3, frame_duration_ms=120)


if __name__ == "__main__":
    main()
