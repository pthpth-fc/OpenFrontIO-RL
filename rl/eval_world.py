"""
Run a single episode with the v8 BC-warmstart agent on the world map
against 200 opponents (10 AlgoBots + 190 tribes).

Saves frames live to rl/runs/eval_world_200/:
  - latest.png       (overwritten each snapshot — open this in VS Code's image
                      preview for live updates while the game runs)
  - frame_NNNN.png   (every snapshot, in order)
  - episode.gif      (full replay, written at end)

Usage:
    python rl/eval_world.py [checkpoint.zip]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from render_game import render_game_frame, render_game_episode  # noqa: E402


CHECKPOINT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(
        PROJECT_ROOT
        / "rl/runs/ppo_v8_bc_warmstart/checkpoints/rl_model_150000_steps.zip"
    )
)

# Episode config
NUM_ALGOBOTS = 10
NUM_TRIBES = 190
MAX_TICKS = 25_000      # ~6× normal
SPAWN_BUFFER = 400      # generous — 200 spawns need time
RENDER_SCALE = 2        # game-style render upscales nearest-neighbor
SNAPSHOT_EVERY = 1      # take a snapshot every decision step

OUT_DIR = PROJECT_ROOT / "rl/runs/eval_world_200"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEST = OUT_DIR / "latest.png"
FRAMES_DIR = OUT_DIR / "frames"
if FRAMES_DIR.exists():
    shutil.rmtree(FRAMES_DIR)
FRAMES_DIR.mkdir(parents=True)

print(f"Loading checkpoint: {CHECKPOINT}")
import os, torch
DEVICE = os.environ.get("TRAIN_DEVICE")
if not DEVICE:
    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
model = MaskablePPO.load(CHECKPOINT, device=DEVICE)

print(f"Output dir: {OUT_DIR}")
print(f"  Open {LATEST} in VS Code image preview for live updates.")
print(f"\nLaunching runner...")
proc = subprocess.Popen(
    ["npm", "run", "--silent", "rl:runner"],
    cwd=str(PROJECT_ROOT),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)

assert proc.stdin and proc.stdout

def send(msg: dict) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


terrain_data: dict = {}  # populated after reset


def save_latest(snapshot: dict) -> Path:
    img = render_game_frame(snapshot, terrain_data, scale=RENDER_SCALE)
    # Save numbered frame
    n = len(list(FRAMES_DIR.glob("frame_*.png")))
    frame_path = FRAMES_DIR / f"frame_{n:05d}.png"
    img.save(frame_path)
    # Update latest.png atomically (write tmp then rename)
    tmp = LATEST.with_suffix(".tmp.png")
    img.save(tmp)
    tmp.replace(LATEST)
    return frame_path


try:
    print(
        f"\nStarting episode: {NUM_ALGOBOTS} AlgoBots + {NUM_TRIBES} tribes "
        f"on world map, max {MAX_TICKS} ticks"
    )
    t0 = time.time()
    resp = send({
        "cmd": "reset",
        "map": "world",
        "num_algobots": NUM_ALGOBOTS,
        "num_tribes": NUM_TRIBES,
        "max_ticks": MAX_TICKS,
        "spawn_buffer": SPAWN_BUFFER,
    })
    obs = np.array(resp["vec"], dtype=np.float32)
    mask = np.array(resp["mask"], dtype=bool)
    print(f"  reset done in {time.time()-t0:.1f}s; fetching terrain...")

    # Fetch the static terrain once — it doesn't change during the episode
    terrain_resp = send({"cmd": "get_terrain"})
    terrain_data.update(terrain_resp)
    print(
        f"  terrain: {terrain_data['width']}×{terrain_data['height']} tiles "
        f"({len(base64_decoded := terrain_resp['terrain_b64'])//1024}KB b64)"
    )

    snapshots: list[dict] = []
    snap = send({"cmd": "snapshot"})
    snapshots.append(snap)
    save_latest(snap)
    print(f"  initial snapshot saved → {LATEST.name} (t={snap['ticks']})")

    total_reward = 0.0
    decision = 0
    done = False
    while not done:
        decision += 1
        action_arr, _ = model.predict(
            obs.reshape(1, -1),
            action_masks=mask.reshape(1, -1),
            deterministic=False,
        )
        a = int(action_arr[0])

        resp = send({"cmd": "step", "action": a})
        obs = np.array(resp["vec"], dtype=np.float32)
        mask = np.array(resp["mask"], dtype=bool)
        total_reward += float(resp["reward"])
        done = bool(resp["done"])

        if decision % SNAPSHOT_EVERY == 0 or done:
            snap = send({"cmd": "snapshot"})
            snapshots.append(snap)
            save_latest(snap)
            agent_idx = next(
                (p["idx"] for p in snap["players"] if p.get("isAgent")),
                None,
            )
            if agent_idx is None:
                agent_tiles = 0
            else:
                owned = snap["owned"]
                agent_tiles = sum(
                    1 for i in range(0, len(owned), 2) if owned[i + 1] == agent_idx
                )
            elapsed = time.time() - t0
            print(
                f"  d={decision:3d}  t={snap['ticks']:5d}  agent_tiles={agent_tiles:6d}  "
                f"reward_so_far={total_reward:+.2f}  elapsed={elapsed:.0f}s"
            )

    print(f"\n✓ Episode done (decisions={decision})")
    print(f"  Final reward: {total_reward:+.3f}")
    print(f"  Wall time: {time.time() - t0:.0f}s")

    print(f"\nBuilding GIF from {len(snapshots)} snapshots...")
    render_game_episode(
        snapshots, terrain_data, OUT_DIR / "episode.gif",
        scale=RENDER_SCALE, frame_duration_ms=120,
    )

    # Build MP4 from saved frame PNGs (smaller, smoother than GIF)
    mp4_path = OUT_DIR / "episode.mp4"
    print(f"\nBuilding MP4 → {mp4_path}")
    ffmpeg_result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "10",
            "-i", str(FRAMES_DIR / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # ensure even dimensions
            str(mp4_path),
        ],
        capture_output=True,
    )
    if ffmpeg_result.returncode == 0:
        print(f"  ✓ saved → {mp4_path}")
    else:
        print(f"  ✗ ffmpeg failed: {ffmpeg_result.stderr.decode()[:200]}")

finally:
    try:
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
    except Exception:
        pass
    proc.terminate()
