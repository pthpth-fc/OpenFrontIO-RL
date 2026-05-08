"""
OpenFrontIO gymnasium environment.

Spawns a persistent `npm run rl:runner` process and communicates via
newline-delimited JSON over stdin/stdout.
"""

from __future__ import annotations

import json
import os
import subprocess

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Must match RLConfig.ts
K_NEIGHBORS = 8
OBS_SIZE = 10 + K_NEIGHBORS * 7        # 66 (10 self + 7×K neighbor features)
ACTION_SIZE = 1 + K_NEIGHBORS * 3 + 3  # 28 (adds expand=attack TerraNullius)
PATCH_SIZE = 32
PATCH_CHANNELS = 4                      # land, self, enemy, ally


class OpenFrontEnv(gym.Env):
    """Single-agent OpenFront.io environment with Dict observations."""

    metadata = {"render_modes": []}

    def __init__(self, project_root: str | None = None) -> None:
        super().__init__()
        self._root = project_root or os.path.dirname(os.path.dirname(__file__))
        self._proc: subprocess.Popen | None = None

        # v1 baseline: flat MLP over 62-dim vec only (CNN/map are ignored)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self._mask: np.ndarray = np.ones(ACTION_SIZE, dtype=bool)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            ["npm", "run", "--silent", "rl:runner"],
            cwd=self._root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _send(self, msg: dict) -> dict:
        assert self._proc and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        return json.loads(line)

    def _parse_obs(self, resp: dict) -> np.ndarray:
        return np.array(resp["vec"], dtype=np.float32)

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._send({"cmd": "quit"})
            except Exception:
                pass
            self._proc.terminate()
            self._proc = None

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_proc()
        resp = self._send({"cmd": "reset"})
        self._mask = np.array(resp["mask"], dtype=bool)
        return self._parse_obs(resp), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        resp = self._send({"cmd": "step", "action": int(action)})
        self._mask = np.array(resp["mask"], dtype=bool)
        obs = self._parse_obs(resp)
        reward = float(resp["reward"])
        terminated = bool(resp["done"])
        info = resp.get("info", {})
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Called by sb3-contrib MaskablePPO at each step."""
        return self._mask
