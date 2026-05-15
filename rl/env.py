"""
OpenFrontIO gymnasium environment.

Spawns a persistent `npm run rl:runner` process and communicates via
newline-delimited JSON over stdin/stdout.

Observation is a Dict:
    "vec": flat 81-dim feature vector (11 self + 7 × K=10 neighbor features)
    "map": 5-channel × 32 × 32 spatial patch around the agent's centroid
           (CHW order — channels: land, self, enemy, ally, own_border)

Action space is Discrete(46):
    0          noop
    1..K       attack[k]              (land, must border)
    K+1..2K    ally_req[k]
    2K+1..3K   break_ally[k]
    3K+1       build_city
    3K+2       build_defpost
    3K+3       expand                 (land — attack TerraNullius)
    3K+4       build_port
    3K+5..4K+4 boat_attack[k]         (ship-based attack from a port)
    4K+5       boat_expand            (ship to unclaimed shore)
"""

from __future__ import annotations

import json
import os
import subprocess

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Must match RLConfig.ts
K_NEIGHBORS = 10
OBS_SIZE = 11 + K_NEIGHBORS * 7         # 81 (11 self + 7 × K neighbor features)
ACTION_SIZE = 1 + K_NEIGHBORS * 4 + 5   # 46 (with K=10 boat actions)
PATCH_SIZE = 32
PATCH_CHANNELS = 5                       # land, self, enemy, ally, own_border


class OpenFrontEnv(gym.Env):
    """Single-agent OpenFront.io environment with Dict observation
    (flat features + spatial CNN patch)."""

    metadata = {"render_modes": []}

    def __init__(self, project_root: str | None = None) -> None:
        super().__init__()
        self._root = project_root or os.path.dirname(os.path.dirname(__file__))
        self._proc: subprocess.Popen | None = None

        self.observation_space = spaces.Dict({
            "vec": spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32),
            "map": spaces.Box(
                0.0, 1.0,
                shape=(PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE),
                dtype=np.float32,
            ),
        })
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

    def _parse_obs(self, resp: dict) -> dict[str, np.ndarray]:
        vec = np.array(resp["vec"], dtype=np.float32)
        map_flat = np.array(resp["map"], dtype=np.float32)
        map_chw = map_flat.reshape(PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        return {"vec": vec, "map": map_chw}

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
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self._ensure_proc()
        resp = self._send({"cmd": "reset"})
        self._mask = np.array(resp["mask"], dtype=bool)
        return self._parse_obs(resp), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
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
