# RL Observation Space — Complete Specification

> **Purpose.** Exhaustive description of what the policy network sees at
> every decision step. If you're trying to understand *why* the agent
> makes a particular choice, the answer is always in here.

---

## Top-level shape

The policy receives a Python dict per step:

```python
obs = {
    "vec": np.ndarray,  # shape (81,), float32, all in [0, 1]
    "map": np.ndarray,  # shape (5, 32, 32), float32, all in {0, 1}
}
```

Plus a **legal-action mask** of shape `(46,)` returned alongside the obs.
The mask is *not* part of the observation in the gym sense — it's consumed
by `MaskablePPO` directly via `env.action_masks()`.

---

## Part 1 — `obs["vec"]`: Flat 81-dim vector

Concatenation of 11 self features and 7 features × K=10 stable opponent
slots = 11 + 70 = 81.

### Self block (indices 0..10)

| idx | Name | Computation | Bounds | Why |
|----:|------|-------------|--------|-----|
| 0 | `troops_ratio` | `min(troops / config.maxTroops(agent), 1)` | [0, 1] | Capacity for attacks. Max troops scales with player size. |
| 1 | `gold_ratio` | `min(gold, 10000) / 10000` | [0, 1] | Buying power (clipped at 10k since most builds cost <2k). |
| 2 | `tiles_pct` | `numTilesOwned() / totalLandTiles` | [0, 1] | Map share. Tells the agent how dominant it is. |
| 3 | `border_exposure` | `min(borderTiles.size / numTilesOwned, 1)` | [0, 1] | High = thin / fragmented territory; low = compact. |
| 4 | `alliance_density` | `alliances.length / (alive − 1)` | [0, 1] | Cooperation level. 0=going solo, 1=allied with all. |
| 5 | `rank` | `rank / (alive − 1)` (0=biggest) | [0, 1] | Relative standing. Useful for endgame strategy. |
| 6 | `game_phase` | `min(ticks / MAX_EPISODE_TICKS, 1)` | [0, 1] | Time pressure. Mid-game = 0.5, endgame = ~1.0. |
| 7 | `num_cities_log` | `min(log1p(cities) / log(11), 1)` | [0, 1] | Income infrastructure. log scaling so 0→0, 10→~1. |
| 8 | `num_defposts_log` | same scaling, defense posts | [0, 1] | Defensive infrastructure. |
| 9 | `num_ports_log` | same scaling, ports | [0, 1] | Naval infrastructure (gates boat actions). |
| 10 | `tile_delta_recent` | `clip((cur − prev) / max(cur,1), −1, 1)` | [−1, 1] | Are we growing or shrinking right now? |

### Neighbor block (indices 11..80)

For each of K=10 stable opponent slots:

| local idx | Name | Computation | Why |
|----------:|------|-------------|-----|
| 0 | `valid` | 1 if slot occupied, else 0 | Empty slots zero out the rest |
| 1 | `troop_ratio` | `min(their_troops / our_troops, 5) / 5` | Power asymmetry, clipped at 5× |
| 2 | `tiles_pct` | `their_tiles / totalLandTiles` | Their map share |
| 3 | `is_allied` | 1 if `agent.allianceWith(n) !== null` else 0 | Cooperation status |
| 4 | `relation` | `agent.relation(n) / 3` | Hostile=0, Distrustful=1/3, Neutral=2/3, Friendly=1 |
| 5 | `is_bordering` | 1 if `agent.nearby()` includes them | Tells: can we land-attack them? |
| 6 | `has_incoming_attack` | 1 if any of their attacks is targeting us | Are they aggressing right now? |

The slot-to-opponent binding is **fixed at episode start** (sorted by
`smallID()` — a permanent player ID — descending). When a slot's opponent
dies, the slot stays empty (zeros) for the rest of the episode rather than
reshuffling. This is what gives the policy stable per-slot identity.

#### Why K=10?

v1-v9 used K=8. On `big_plains` against 3 AlgoBots there was always slack
— only 3 opponents existed, so 5 slots were always empty. On the world map
with 10 AlgoBots all of K=10 fits exactly; on world+200 the agent still
only sees the first 10 (sorted by smallID).

K=10 is large enough for "normal" game sizes; large maps with 50+ players
need different machinery (see `docs/RL_RUNS.md` § "Decision: alternative
path vs architecture overhaul").

#### Why `smallID()` for ordering?

`smallID()` is a permanent 1-based integer assigned when a player joins
the game. Unlike `id()` (a hash string) it's order-stable. Sorting by it
gives a deterministic, immutable mapping from player to slot for the entire
episode.

Earlier runs sorted by current tile count, which meant the policy had to
relearn "neighbor 3 is the threat" on every step because the ordering kept
shifting. v4 introduced `smallID()` ordering and immediately produced
coherent strategic behavior (e.g., v8 used `attack[6]` as 9.6% of all
actions — a stable target preference).

---

## Part 2 — `obs["map"]`: Spatial CNN patch

A 32×32 patch of the map centered on the **centroid of the agent's owned
tiles**, encoded as 5 binary channels in CHW order.

```
shape: (5, 32, 32), dtype=float32
channels-first: [land, self, enemy, ally, own_border]
```

### Centroid logic

```python
sumX = sum(game.x(t) for t in agent.tiles())
sumY = sum(game.y(t) for t in agent.tiles())
cx = round(sumX / len(tiles)) if tiles else mapW // 2
cy = round(sumY / len(tiles)) if tiles else mapH // 2

# Top-left corner of patch (clamped so the 32×32 stays in-bounds)
x0 = max(0, min(mapW - 32, cx - 16))
y0 = max(0, min(mapH - 32, cy - 16))
```

Patch covers a fixed 32×32 region around `cx, cy`. **Patch size doesn't
scale with agent territory** — a 1-tile spawn and a 5000-tile empire both
get a 32×32 view. The intuition: tactical decisions are local; strategic
state lives in the flat features (`tiles_pct`, `rank`, etc.).

### Channel definitions

| ch | Mask | What it means |
|---:|------|---------------|
| 0 | `is_land` | 1 if the tile is land (Plains/Highland/Mountain/Lake), 0 if Ocean |
| 1 | `is_self` | 1 if owned by agent, 0 otherwise |
| 2 | `is_enemy` | 1 if owned by a non-allied other player |
| 3 | `is_ally` | 1 if owned by a player the agent has an alliance with |
| 4 | `is_own_border` | 1 if **agent's tile AND** in `agent.borderTiles()` |

> **Channel 4 is a novel addition in v10.** It's redundant in principle
> (could be derived from channels 1+2+3) but providing it directly saves
> the CNN from learning a "find boundaries" filter. Cheap on the producer
> side (`agent.borderTiles()` is precomputed in the engine) and high
> information density.

### Caveats

1. **Centroid is fragile when the agent has split territory.** A blue
   blob in the north-west and another in the south-east → centroid in the
   middle of the ocean → 32×32 patch is mostly empty. Mitigation candidates
   (not yet implemented):
   - Pick the largest cluster's centroid
   - Use multi-scale patches (small zoom + global thumbnail)
   - Use multiple patches centered on each cluster
2. **Empty agent (just spawned, 0 tiles)** falls back to map center. The
   first decision sees no useful spatial info; from the second decision on
   it's anchored on the spawn area.
3. **Resolution.** 32 pixels per side means a `world` map (2000×1000 tiles)
   is sampled at ~1.6% coverage. Tactical resolution is fine; strategic
   reasoning has to come from the flat features.
4. **Big agents see only their tactical core, not their full territory.**
   Once the agent grows past ~32 tiles in any dimension, the outer
   borders fall outside the patch. The CNN can still reason about what's
   near the centroid (tactical fights), but cannot see distant borders or
   far flanks. Mitigated for now by the flat features (`tiles_pct`,
   `border_exposure`, `tile_delta_recent`); the structural fix is
   multi-scale obs — see `RL_RUNS.md` § "Decisions still on the table"
   item 0. Deferred until v10 results justify the extra complexity.

---

## Action mask

Returned alongside obs at every step:

```python
mask = np.ndarray(shape=(46,), dtype=bool)
```

For each action slot, `mask[i] = True` if that action is currently legal.

| Action | Mask condition |
|--------|----------------|
| `noop` (0) | always True |
| `attack[k]` (1..K) | `agent.nearby()` contains neighbor[k] AND `not agent.isFriendly(neighbor[k])` |
| `ally_req[k]` (K+1..2K) | `agent.allianceWith(neighbor[k])` is null AND slot occupied |
| `break_ally[k]` (2K+1..3K) | `agent.allianceWith(neighbor[k])` is not null |
| `build_city` (3K+1) | `agent.numTilesOwned() > 5` |
| `build_defpost` (3K+2) | `agent.borderTiles().size > 0` |
| `expand` (3K+3) | `agent.numTilesOwned() > 0` |
| `build_port` (3K+4) | any tile in `agent.borderTiles()` is on shoreline |
| `boat_attack[k]` (3K+5..4K+4) | `agent.units(Port).length > 0` AND `not agent.isFriendly(neighbor[k])` |
| `boat_expand` (4K+5) | `agent.units(Port).length > 0` |

(K = K_NEIGHBORS = 10.)

### Why masking matters

Without masking, ~60% of policy outputs are illegal in any given state (8 of
10 attack slots typically have empty / friendly / non-bordering targets).
The agent would waste rollout samples sampling impossible actions, then
get gradient signal from log-probs over a malformed distribution.

`MaskablePPO` (sb3-contrib) handles this end-to-end:
- During sampling: zero probability for masked actions
- During rollout: action mask stored alongside obs
- During PPO update: log-prob computed over the masked distribution

---

## Pipe overhead

Per step, the runner emits a JSON line containing:
- `vec`: 81 floats (~600 chars JSON serialized)
- `map`: 5120 floats (~30 KB JSON serialized — this is the dominant cost)
- `mask`: 46 booleans (~200 chars)
- `reward`, `done`, `info`: ~50 chars

Total: ~30 KB per step over stdin/stdout. On big_plains+3 the CPU sim is
fast enough that JSON encoding becomes a measurable fraction of total
time. On world map the sim dominates.

If throughput becomes critical, the path forward is binary serialization
(msgpack) or shared memory between the runner and Python. Not yet
implemented — game sim is the bigger lever to attack first.

---

## Notes for re-implementation

If you ever need to re-derive these features, the canonical sources are:

- **Self / neighbor flat features:** `src/rl/headless/Observation.ts` →
  `extractObs()`
- **Spatial map:** `src/rl/headless/SpatialObs.ts` → `extractSpatialObs()`
- **Action mask:** `src/rl/headless/Observation.ts` → `computeActionMask()`
- **Constants (K, OBS_SIZE, ACTION_SIZE, PATCH_*):** `src/rl/headless/RLConfig.ts`

Python side mirrors these in `rl/env.py` and must stay in sync with the
TypeScript constants. A mismatch means silent JSON parse misalignment —
debugging is painful, so update both files together.
