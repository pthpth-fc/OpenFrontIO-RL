"""
Collect demonstration trajectories from AlgoBots for behavioral cloning.

Each demo_episode runs a full game with N AlgoBots (no RL agent) and streams
(obs, mask, action) samples to stdout. We accumulate them into a JSONL file.

Usage:
    python rl/collect_demos.py [n_episodes] [out_path] [num_bots]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT_PATH = Path(sys.argv[2] if len(sys.argv) > 2 else "rl/demos.jsonl")
NUM_BOTS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Collecting {N_EPISODES} episodes × {NUM_BOTS} AlgoBots → {OUT_PATH}")
total_samples = 0
t0 = time.time()

with open(OUT_PATH, "w") as out_f:
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
        try:
            assert proc.stdin and proc.stdout
            proc.stdin.write(
                json.dumps({"cmd": "demo_episode", "num_bots": NUM_BOTS}) + "\n"
            )
            proc.stdin.flush()

            ep_samples = 0
            ticks = 0
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                d = json.loads(line)
                if "sample" in d:
                    out_f.write(json.dumps(d["sample"]) + "\n")
                    ep_samples += 1
                elif "done" in d:
                    ticks = d.get("ticks", 0)
                    break

            total_samples += ep_samples
            dt = time.time() - t0
            print(
                f"  ep {ep+1:3d}/{N_EPISODES}: {ep_samples:4d} samples "
                f"({ticks} ticks)  total={total_samples:6d}  elapsed={dt:.0f}s"
            )

        finally:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
            proc.terminate()

print(f"\n✓ Wrote {total_samples} samples to {OUT_PATH}")
