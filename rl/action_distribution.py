"""
Run N rollouts with a saved checkpoint and report the action distribution.

Tells us whether the policy actually uses attack/ally/build actions,
or whether it's stuck in a turtle (mostly expand + noop).

Usage:
    python rl/action_distribution.py [checkpoint.zip] [n_episodes]
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

CHECKPOINT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(PROJECT_ROOT / "rl/runs/ppo_v4_richer_obs_rewards/checkpoints/rl_model_200000_steps.zip")
)
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Action labels matching ActionDecoder.ts (K_NEIGHBORS=8, total=28)
K = 8

def label(a: int) -> str:
    if a == 0:
        return "noop"
    if 1 <= a <= K:
        return f"attack[{a-1}]"
    if K + 1 <= a <= 2 * K:
        return f"ally_req[{a-K-1}]"
    if 2 * K + 1 <= a <= 3 * K:
        return f"break_ally[{a-2*K-1}]"
    if a == 3 * K + 1:
        return "build_city"
    if a == 3 * K + 2:
        return "build_defpost"
    if a == 3 * K + 3:
        return "expand"
    return f"unknown[{a}]"


def category(a: int) -> str:
    """Roll up to 6 macro categories."""
    if a == 0:
        return "noop"
    if 1 <= a <= K:
        return "attack"
    if K + 1 <= a <= 2 * K:
        return "ally_req"
    if 2 * K + 1 <= a <= 3 * K:
        return "break_ally"
    if a in (3 * K + 1, 3 * K + 2):
        return "build"
    if a == 3 * K + 3:
        return "expand"
    return "unknown"


def main() -> None:
    print(f"Loading {CHECKPOINT}")
    import os, torch
    device = os.environ.get("TRAIN_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    model = MaskablePPO.load(CHECKPOINT, device=device)

    fine = Counter()
    coarse = Counter()
    rewards = []

    for ep in range(N_EPISODES):
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
            total_r = 0.0
            ep_actions = []
            done = False

            while not done:
                action, _ = model.predict(
                    obs.reshape(1, -1),
                    action_masks=mask.reshape(1, -1),
                    deterministic=False,
                )
                a = int(action[0]) if hasattr(action, "__len__") else int(action)
                ep_actions.append(a)
                fine[a] += 1
                coarse[category(a)] += 1

                resp = send({"cmd": "step", "action": a})
                obs = np.array(resp["vec"], dtype=np.float32)
                mask = np.array(resp["mask"], dtype=bool)
                total_r += float(resp["reward"])
                done = bool(resp["done"])
        finally:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
            proc.terminate()

        rewards.append(total_r)
        print(f"  ep {ep+1}: reward={total_r:+.2f}, len={len(ep_actions)}")

    total_actions = sum(fine.values())

    print(f"\nMean reward: {np.mean(rewards):+.3f}  (over {N_EPISODES} eps, total {total_actions} actions)")

    print("\n── Coarse distribution (macro categories) ──")
    for cat, count in sorted(coarse.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_actions
        bar = "█" * int(pct / 2)
        print(f"  {cat:12s} {count:5d}  {pct:5.1f}%  {bar}")

    print("\n── Fine distribution (top 12 actions) ──")
    for a, count in fine.most_common(12):
        pct = 100 * count / total_actions
        print(f"  {label(a):20s} {count:5d}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
