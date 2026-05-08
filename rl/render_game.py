"""
Game-style renderer that draws terrain (ocean, plains, highland, mountain,
lake, shoreline) with player territories overlaid as semi-transparent
colored regions, plus thin black borders between different owners.

Public API:
    render_game_frame(snapshot, terrain, scale=2)  -> PIL.Image

Snapshot format (from runner.ts snapshotGame):
    width, height, owned (flat [ref, playerIdx, ref, playerIdx, ...]),
    players (list of {id, name, isAgent}), ticks

Terrain format (from runner.ts getTerrain):
    width, height, terrain_b64 (base64 of byte-per-tile;
       low 3 bits = TerrainType, bit 3 = isShoreline)
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Terrain colours ─────────────────────────────────────────────────────────
# Approximate the in-game palette (greens for land, blues for water).
T_PLAINS = (140, 175, 90)        # 0 — pale green
T_HIGHLAND = (160, 145, 90)      # 1 — tan/olive
T_MOUNTAIN = (110, 100, 95)      # 2 — gray-brown
T_LAKE = (90, 130, 180)          # 3 — medium blue
T_OCEAN = (35, 60, 110)          # 4 — deep blue
T_SHORE = (210, 200, 160)        # shoreline highlight

TERRAIN_COLOURS = [T_PLAINS, T_HIGHLAND, T_MOUNTAIN, T_LAKE, T_OCEAN]

# ── Player colours ──────────────────────────────────────────────────────────
# Agent uses bright yellow to stand out against blue ocean and green plains.
AGENT_COLOR = (255, 235, 0)      # vivid yellow
AGENT_OUTLINE = (255, 255, 255)  # white halo for max visibility
OPPONENT_COLORS = [
    (220, 50, 50),    # red
    (60, 200, 80),    # green
    (130, 80, 220),   # purple
    (255, 140, 0),    # orange
    (40, 200, 200),   # cyan
    (255, 100, 180),  # pink
    (180, 120, 60),   # brown
    (120, 180, 220),  # sky
    (220, 220, 220),  # silver
    (160, 80, 40),    # dark brown
]

# Translucency for player overlay (0-255). Higher = more saturated player colour.
OVERLAY_ALPHA = 180


def _decode_terrain(terrain: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Decode the base64 terrain bytes → (terrain_type[h,w], shoreline[h,w]).

    terrain_type values are 0..4 matching TerrainType.
    """
    w = terrain["width"]
    h = terrain["height"]
    data = base64.b64decode(terrain["terrain_b64"])
    arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
    t_type = arr & 0x07
    shore = (arr & 0x08).astype(bool)
    return t_type, shore, w, h


def _player_color(player_idx: int, is_agent: bool) -> tuple[int, int, int]:
    if is_agent:
        return AGENT_COLOR
    return OPPONENT_COLORS[(player_idx - 1) % len(OPPONENT_COLORS)]


def render_game_frame(
    snapshot: dict,
    terrain: dict,
    scale: int = 2,
) -> Image.Image:
    """Render one frame in a game-like style."""
    t_type, shore, w, h = _decode_terrain(terrain)

    # ── Step 1: terrain background ──────────────────────────────────────────
    # Build an HxWx3 RGB image from the terrain grid using vectorised ops.
    bg = np.empty((h, w, 3), dtype=np.uint8)
    for tt_value, color in enumerate(TERRAIN_COLOURS):
        mask = t_type == tt_value
        if mask.any():
            bg[mask] = color
    # Shoreline highlight (a touch lighter on water side)
    bg[shore] = T_SHORE

    # ── Step 2: ownership overlay ───────────────────────────────────────────
    # Build an owner_idx array (uint16, 0 = unclaimed) and an agent mask.
    owner_idx = np.zeros((h, w), dtype=np.uint16)
    owned_flat = snapshot["owned"]
    if owned_flat:
        # owned_flat = [ref, idx, ref, idx, ...]
        flat = np.asarray(owned_flat, dtype=np.int64)
        refs = flat[0::2]
        idxs = flat[1::2]
        ys = (refs // w).astype(np.int64)
        xs = (refs % w).astype(np.int64)
        owner_idx[ys, xs] = idxs.astype(np.uint16)

    # Agent index — use the stable smallID-based idx from the snapshot.
    # Fallback to position-in-list for older snapshot formats without "idx".
    players = snapshot.get("players", [])
    agent_pidx = 0
    for i, p in enumerate(players):
        if p.get("isAgent"):
            agent_pidx = int(p.get("idx", i + 1))
            break

    # Build per-pixel colour overlay where owner_idx > 0
    overlay = bg.copy()
    occupied_mask = owner_idx > 0
    if occupied_mask.any():
        # Per-tile blend: bg * (1-a) + player * a
        a = OVERLAY_ALPHA / 255.0
        for unique_idx in np.unique(owner_idx[occupied_mask]):
            tile_mask = owner_idx == unique_idx
            is_agent = (int(unique_idx) == agent_pidx)
            color = np.array(_player_color(int(unique_idx), is_agent), dtype=np.float32)
            overlay[tile_mask] = (
                overlay[tile_mask].astype(np.float32) * (1 - a) + color * a
            ).astype(np.uint8)

    # ── Step 3: borders between different owners ───────────────────────────
    # Black 1-px line where current pixel's owner differs from right or down neighbor.
    if occupied_mask.any():
        right = np.zeros_like(owner_idx)
        right[:, :-1] = owner_idx[:, 1:]
        down = np.zeros_like(owner_idx)
        down[:-1, :] = owner_idx[1:, :]
        diff_right = (owner_idx != right) & (occupied_mask | (right > 0))
        diff_down = (owner_idx != down) & (occupied_mask | (down > 0))
        border_mask = diff_right | diff_down
        overlay[border_mask] = (15, 15, 15)

    # ── Step 3.5: thick white halo around the agent's territory ────────────
    # Make the agent unmistakable on a busy map. Dilate the agent mask 2x and
    # paint everything outside the agent itself with white.
    agent_mask = (owner_idx == agent_pidx) if agent_pidx > 0 else None
    agent_centroid: tuple[int, int] | None = None
    if agent_mask is not None and agent_mask.any():
        # Dilate using shifted ORs
        dilated = agent_mask.copy()
        for _ in range(2):
            d = dilated.copy()
            d[1:] |= dilated[:-1]
            d[:-1] |= dilated[1:]
            d[:, 1:] |= dilated[:, :-1]
            d[:, :-1] |= dilated[:, 1:]
            dilated = d
        halo = dilated & ~agent_mask
        overlay[halo] = AGENT_OUTLINE

        ys, xs = np.where(agent_mask)
        agent_centroid = (int(xs.mean()), int(ys.mean()))

    # ── Step 4: scale + label ───────────────────────────────────────────────
    img = Image.fromarray(overlay, mode="RGB")
    if scale != 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    n_alive = sum(1 for _ in players)
    agent_status = "ALIVE" if agent_centroid is not None else "DEAD"
    label = (
        f"t={snapshot.get('ticks', 0)}  alive={n_alive}  agent=YELLOW ({agent_status})"
    )
    # Background for legibility
    draw.rectangle([0, 0, len(label) * 7 + 8, 16], fill=(0, 0, 0, 180))
    draw.text((4, 2), label, fill=(255, 255, 255), font=font)

    # Big circle around the agent's centroid so it pops on huge maps
    if agent_centroid is not None:
        cx = agent_centroid[0] * scale
        cy = agent_centroid[1] * scale
        r = max(20, scale * 12)
        # outer dark ring
        draw.ellipse(
            [cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2],
            outline=(0, 0, 0), width=3,
        )
        # bright yellow inner ring
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=AGENT_COLOR, width=3,
        )
        # "AGENT" label above the circle
        tag = "AGENT"
        draw.rectangle(
            [cx - 22, cy - r - 18, cx + 22, cy - r - 4],
            fill=(0, 0, 0, 220),
        )
        draw.text((cx - 18, cy - r - 16), tag, fill=AGENT_COLOR, font=font)

    return img


def render_game_episode(
    snapshots: list[dict],
    terrain: dict,
    out_path: Path,
    scale: int = 2,
    frame_duration_ms: int = 100,
) -> None:
    """Render a list of snapshots to an animated GIF using game-style rendering."""
    frames = [render_game_frame(s, terrain, scale=scale) for s in snapshots]
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
    print(f"Saved game-style render → {out_path}")
