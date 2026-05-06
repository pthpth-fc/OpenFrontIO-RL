"""
Render a recorded episode (list of snapshots) as a GIF.

Each snapshot comes from runner.ts `{"cmd":"snapshot"}` and contains:
  width, height, owned ([ref, playerIdx, ...]), players, ticks
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Distinct player colours (index 1..N); index 0 = unclaimed land
PLAYER_COLORS = [
    (40, 40, 40),      # 0: unclaimed land (dark grey)
    (70, 130, 255),    # 1: blue  (agent)
    (255, 80, 80),     # 2: red
    (80, 200, 80),     # 3: green
    (255, 200, 50),    # 4: yellow
    (200, 80, 255),    # 5: purple
    (255, 140, 0),     # 6: orange
    (0, 200, 200),     # 7: cyan
    (255, 100, 180),   # 8: pink
]
OCEAN_COLOR = (20, 30, 60)
AGENT_BORDER = (255, 255, 255)


def _color_for(player_idx: int, is_agent: bool) -> tuple[int, int, int]:
    if player_idx == 0:
        return PLAYER_COLORS[0]
    idx = player_idx % len(PLAYER_COLORS)
    return PLAYER_COLORS[idx]


def render_snapshot(
    snapshot: dict,
    scale: int = 3,
) -> "Image.Image":
    """Render one snapshot to a PIL Image."""
    assert PIL_AVAILABLE, "pip install Pillow"

    w, h = snapshot["width"], snapshot["height"]
    owned_flat: list[int] = snapshot["owned"]
    players: list[dict] = snapshot["players"]

    # Build a player_idx → colour map
    agent_idx: int | None = None
    for i, p in enumerate(players):
        if p["isAgent"]:
            agent_idx = i + 1
            break

    # Build grid: default OCEAN for all, will fill land below
    grid = [[None] * w for _ in range(h)]  # None = ocean

    # We need to know which tiles are land but unowned — we don't have that here
    # directly. We mark owned tiles; unowned land tiles we approximate as unclaimed.
    # For a clean render we just show owned vs unowned (both land-ish colour).
    ownership: dict[int, int] = {}  # ref → playerIdx
    for i in range(0, len(owned_flat), 2):
        ref, pidx = owned_flat[i], owned_flat[i + 1]
        ownership[ref] = pidx

    img = Image.new("RGB", (w * scale, h * scale), OCEAN_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw all owned tiles
    for ref, pidx in ownership.items():
        # ref encodes position: ref = y * width + x  (standard GameMap layout)
        x = ref % w
        y = ref // w
        is_agent = (pidx == agent_idx)
        color = _color_for(pidx, is_agent)
        px, py = x * scale, y * scale
        draw.rectangle([px, py, px + scale - 1, py + scale - 1], fill=color)

    # Tick counter label
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    label = f"t={snapshot['ticks']}"
    draw.text((4, 4), label, fill=(255, 255, 255), font=font)

    return img


def render_episode(
    snapshots: list[dict],
    out_path: Path,
    scale: int = 3,
    frame_duration_ms: int = 200,
) -> None:
    """Render a list of snapshots to an animated GIF."""
    assert PIL_AVAILABLE, "pip install Pillow"
    frames = [render_snapshot(s, scale=scale) for s in snapshots]
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved render → {out_path}")
