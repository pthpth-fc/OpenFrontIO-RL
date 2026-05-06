"""
SB3 callback: after every RENDER_EVERY completed episodes, run one eval episode
with the current policy and save an animated GIF to rl/runs/<name>/renders/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

sys.path.insert(0, str(Path(__file__).parent))
from render import render_episode, PIL_AVAILABLE  # noqa: E402

PROJECT_ROOT = str(Path(__file__).parent.parent)
RENDER_EVERY = 200  # episodes

def _parse_obs(resp: dict) -> np.ndarray:
    return np.array(resp["vec"], dtype=np.float32)


class RenderCallback(BaseCallback):
    """Saves a GIF render every RENDER_EVERY completed episodes."""

    def __init__(self, runs_dir: Path, render_every: int = RENDER_EVERY, verbose: int = 0):
        super().__init__(verbose)
        self.runs_dir = runs_dir
        self.render_every = render_every
        self._ep_count = 0
        self._last_rendered = -1

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        self._ep_count += int(np.sum(dones))

        milestone = (self._ep_count // self.render_every) * self.render_every
        if milestone > self._last_rendered and milestone > 0:
            self._last_rendered = milestone
            self._generate_render(milestone)
        return True

    def _generate_render(self, ep_num: int) -> None:
        if not PIL_AVAILABLE:
            print(f"[RenderCallback] Pillow not installed — skipping render at ep {ep_num}")
            return

        print(f"\n[RenderCallback] Generating render at episode {ep_num}...")
        snapshots = self._run_render_episode()
        if not snapshots:
            print("[RenderCallback] No snapshots collected — skipping")
            return

        out_dir = self.runs_dir / "renders"
        out_path = out_dir / f"ep_{ep_num:06d}.gif"
        render_episode(snapshots, out_path, scale=3, frame_duration_ms=150)

    def _run_render_episode(self) -> list[dict]:
        """Run one full episode using the current policy, collecting snapshots."""
        proc = subprocess.Popen(
            ["npm", "run", "--silent", "rl:runner"],
            cwd=PROJECT_ROOT,
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

        snapshots: list[dict] = []
        try:
            resp = send({"cmd": "reset"})
            obs = _parse_obs(resp)
            mask = np.array(resp["mask"], dtype=bool)

            # Grab initial snapshot
            snap = send({"cmd": "snapshot"})
            snapshots.append(snap)

            done = False
            while not done:
                action, _ = self.model.predict(
                    obs.reshape(1, -1),
                    action_masks=mask.reshape(1, -1),
                    deterministic=True,
                )
                action_int = int(action[0]) if hasattr(action, "__len__") else int(action)

                resp = send({"cmd": "step", "action": action_int})
                obs = _parse_obs(resp)
                mask = np.array(resp["mask"], dtype=bool)
                done = resp["done"]

                snap = send({"cmd": "snapshot"})
                snapshots.append(snap)

        finally:
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass
            proc.terminate()

        return snapshots
