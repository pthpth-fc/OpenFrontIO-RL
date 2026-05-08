# RL Architecture — Observation, Action, Reward

This document describes the agent's interface to the OpenFrontIO simulation.
It is the source of truth for what the policy network sees, what choices it
can make, and how reward is computed.

> **Versioning.** This doc reflects **v10** (CNN + Dict observation + boat
> attacks, K=10 neighbors, 46-action space). Earlier runs (v1–v9) used
> simpler shapes documented in `RUNS_LOG.md`.

---

## Overview

```
                  ┌────────────────────────────────────────┐
                  │  OpenFrontIO Game (Web Worker / Node)  │
                  │  - Lockstep deterministic simulation   │
                  │  - 11 players: 1 RL agent + 10 AlgoBots│
                  └─────────────┬──────────────────────────┘
                                │
                  every DECISION_INTERVAL=50 ticks
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │  src/rl/headless/runner.ts           │
              │  - extracts obs from agent's POV     │
              │  - computes legal-action mask        │
              │  - decodes action → game Executions  │
              │  - JSON-RPC over stdin/stdout        │
              └─────────────┬────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────────────┐
              │  rl/env.py    (gymnasium.Env)        │
              │  Dict obs:                           │
              │    "vec":  (81,)  flat features      │
              │    "map":  (5, 32, 32)  CNN patch    │
              └─────────────┬────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────────────┐
              │  MaskablePPO (sb3-contrib)           │
              │  Custom feature extractor:           │
              │    CNN(5×32×32 → 256)                │
              │    MLP(81 → 128)                     │
              │    concat → 384 → actor [256, 128]   │
              │                  → critic [256, 128] │
              │  Discrete(46) action head            │
              └──────────────────────────────────────┘
```

---

## Observation

The policy receives a **Dict** observation per decision step. Two parts:

### 1. Flat vector — `obs["vec"]` shape `(81,)`

#### Self block (11 features)

All values clipped to [0, 1].

| # | Feature | Computation | Why |
|---|---------|-------------|-----|
| 0 | `troops_ratio` | `min(troops / max_troops, 1)` | Capacity for attacks |
| 1 | `gold_ratio` | `min(gold, 10000) / 10000` | Buying power for buildings |
| 2 | `tiles_pct` | `tiles / total_land_tiles` | Map share — how big are we |
| 3 | `border_exposure` | `min(border_tiles / tiles, 1)` | Defensive exposure |
| 4 | `alliance_density` | `alliances / (alive − 1)` | Cooperation level |
| 5 | `rank` | `rank / (alive − 1)` (0=biggest) | Relative standing |
| 6 | `game_phase` | `min(ticks / MAX_EPISODE_TICKS, 1)` | Time pressure |
| 7 | `num_cities_log` | `min(log1p(cities) / log(11), 1)` | Income infrastructure |
| 8 | `num_defposts_log` | same scaling, defense posts | Defensive infrastructure |
| 9 | `num_ports_log` | same scaling, ports | Naval infrastructure (gates boat actions) |
| 10 | `tile_delta_recent` | `clip((cur − prev) / max(cur,1), −1, 1)` | Are we growing or shrinking |

#### Neighbor block — 7 features × K=10 = 70 features

For each of the K=10 stable opponent slots (sorted by `smallID()` once at episode
start, **slot k stays bound to the same opponent for the entire episode**, even
when others die — dead slots get null and zeros).

| # | Feature | Computation | Why |
|---|---------|-------------|-----|
| 0 | `valid` | `1 if slot occupied, else 0` | Is this a real opponent |
| 1 | `troop_ratio` | `min(their_troops / our_troops, 5) / 5` | Power asymmetry |
| 2 | `tiles_pct` | `their_tiles / total_land_tiles` | Their map share |
| 3 | `is_allied` | 0 or 1 | Cooperation status |
| 4 | `relation` | `relation / 3` (Hostile=0 → Friendly=1) | Diplomatic stance |
| 5 | `is_bordering` | 0 or 1 | Can we land-attack? |
| 6 | `has_incoming_attack` | 0 or 1 | Are they attacking us right now |

> **Stable identity** is critical. Without it, the model would have to relearn
> "neighbor 3 is the threat" on every step because the slot-to-opponent
> mapping kept shifting. v4 introduced `buildStableNeighborOrder()` and
> immediately unlocked stable target preferences (e.g., attack[1] became
> 9% of all v4-250k actions — a coherent strategic choice).

### 2. Spatial patch — `obs["map"]` shape `(5, 32, 32)`

Channels-first (CHW) for PyTorch. Centered on the **centroid of the agent's
owned tiles**, clamped to map bounds. Each channel is a binary mask:

| Channel | Meaning |
|---------|---------|
| 0 | **land mask** — 1 if land tile, 0 if ocean |
| 1 | **self tiles** — 1 if owned by agent |
| 2 | **enemy tiles** — 1 if owned by a non-allied player |
| 3 | **ally tiles** — 1 if owned by an allied player |
| 4 | **own border** — 1 if agent's tile AND on the border (adjacent to a non-self tile) |

> **Channel 4 is the key spatial cue.** It directly tells the policy "you have a
> thin spike here that's vulnerable" without forcing the CNN to derive it from
> the self/enemy masks. Cheap to compute, big information density.

The patch is 32×32 = 1024 pixels, so for a world map (2000×1000 tiles) the
agent only sees 1.6% of the map at a time. That's intentional — local
spatial reasoning is what matters most for tactical decisions.

#### Caveats

- **Centroid is unstable when the agent has split territory.** Mitigation
  candidates: pick largest cluster's centroid, or multi-scale patches. Not
  yet implemented.
- **Empty agent** (just spawned, 0 tiles) → patch centered on map middle.
  Useless for the first decision but harmless.
- **Pipe overhead.** 5 × 32 × 32 = 5120 floats × 4 bytes ≈ 20 KB per step
  through stdin/stdout JSON. Throughput cost is real — see `RUNS_LOG.md`.

---

## Action space — `Discrete(46)`

Slots are laid out so that the v8 sub-action set (slots 0–27) keeps its
indices, allowing partial weight transfer when needed.

```
slot   action               valid when
─────  ────────────────     ─────────────────────────────────────────────
0      noop                 always
1..K   attack[k]            bordering enemy[k] (k = 0..9, K=10)
K+1..2K   ally_req[k]       not already allied with neighbor[k]
2K+1..3K  break_ally[k]     currently allied with neighbor[k]
3K+1   build_city           agent has > 5 tiles
3K+2   build_defpost        agent has any border tile
3K+3   expand               agent alive (attack TerraNullius from land)
3K+4   build_port           agent has any shoreline border tile     ← new in v10
3K+5..4K+4   boat_attack[k] agent has a port AND target alive       ← new in v10
4K+5   boat_expand          agent has a port                        ← new in v10
                                                                     (K=10 → 46 total)
```

### Decoder mapping

The runner consumes an action index and emits the corresponding game
`Execution` objects:

| Action category | Execution emitted |
|-----------------|-------------------|
| attack[k] | `AttackExecution(null, agent, neighbor[k].id(), null)` |
| ally_req[k] | `AllianceRequestExecution(agent, neighbor[k].id())` |
| break_ally[k] | `BreakAllianceExecution(agent, neighbor[k].id())` |
| build_city | `ConstructionExecution(agent, UnitType.City, firstBorderTile)` |
| build_defpost | `ConstructionExecution(agent, UnitType.DefensePost, firstBorderTile)` |
| build_port | `ConstructionExecution(agent, UnitType.Port, firstShorelineBorderTile)` |
| expand | `AttackExecution(null, agent, null, null)` (null target = TerraNullius) |
| boat_attack[k] | `AttackExecution(null, agent, neighbor[k].id(), portTile)` |
| boat_expand | `AttackExecution(null, agent, null, portTile)` |

> **Build placement is deterministic, not learned.** The first border tile
> from `agent.borderTiles().values()` is always picked. This means the
> model decides *whether* to build, not *where*. Could be improved with a
> spatial output head (a pointer net), but that's a substantial change.

### Action mask

The mask is a `bool[46]` returned alongside the obs at every step. MaskablePPO
uses it both for sampling (impossible actions get zero probability) and for
log-prob computation during PPO updates.

Without masking, ~80% of policy outputs would be illegal in any given state
(8 attack slots but only 2-3 visible enemies bordering us). Masking is
essential — without it, training is dramatically slower and noisier.

---

## Reward function

Per decision step, applied at the end of the 50-tick advance:

```
reward = REWARD_TICK × DECISION_INTERVAL                    # always present
       + REWARD_KILL × (opponents_eliminated_this_step)     # kill bonus
       + REWARD_DEATH                                       # if agent died
       + REWARD_TILE_GAIN × (cur_tiles − prev_tiles)        # if alive
       + REWARD_WIN                                         # if agent won
```

### Constants (from `RLConfig.ts`)

| Constant | Value | Rationale |
|----------|-------|-----------|
| `REWARD_WIN` | +15.0 | Decisive — winning is worth ~2× best timeout outcome, pushes off the turtle equilibrium |
| `REWARD_DEATH` | −5.0 | Strictly dominates max possible expansion shaping (+2.5) so "expand wildly then die" is net-negative |
| `REWARD_TILE_GAIN` | 0.0005 / tile | Symmetric (gains and losses both counted via delta — agent can't farm tile-loss reward by attacking allies) |
| `REWARD_TICK` | −0.00005 / tick | Mild time pressure; nudges agent toward decisive games |
| `REWARD_KILL` | +0.5 / kill | Mid-episode signal that strategic eliminations are valuable |

### Why the asymmetry between WIN (+15) and DEATH (−5)?

A "best-case timeout" episode (survive to MAX_EPISODE_TICKS, accumulate +2.5
shaping, +1-2 kill rewards) sums to ~+5. If we set `REWARD_WIN = +5` (matching
death's magnitude), winning would equal turtling — no incentive to push for
elimination. With `+15`, winning pays roughly 2-3× turtling, which empirically
broke the v5 plateau.

The death penalty doesn't need to be that large. v4 used −5 throughout and
the agent didn't get reward-hacked into suicide. The asymmetry reflects that
positive outcomes are rare and need a large bonus to dominate the gradient.

---

## Implementation notes

### Stable per-episode opponent IDs

`buildStableNeighborOrder(game, agent)` is called **once per episode** at
reset time. It sorts all current opponents by `smallID()` (which is permanent
for the lifetime of a player) and freezes the resulting list as
`neighborOrder: string[]`.

`getNeighbors(game, neighborOrder)` is then called every step to resolve the
order to current Player references. Dead opponents become `null` — we
never re-pack the array. This is what gives the model stable per-slot
identity.

### Kill detection

Each step, runner.ts diffs `aliveOpponentIDs` (Set<PlayerID>) against the
current alive set. The size of the symmetric difference = number of kills,
which directly multiplies into `REWARD_KILL`.

The kill is awarded to the agent **regardless of who landed the killing
blow** — a kill could be the result of an AlgoBot eliminating another
AlgoBot. This is fine: the agent is rewarded for the world becoming simpler
(fewer threats), not specifically for personally dispatching opponents. If
this incentivizes "let everyone fight while I sit", the death penalty + win
bonus dominate to push for active play.

### Spawn phase buffer

`SPAWN_PHASE_BUFFER = 250` ticks (was 115 for big_plains). The game has its
own spawn phase during which players are placed on the map. The agent's
first observation must come *after* this phase or it sees a half-empty
world. World map with 11 players needs ~250 ticks to stably place
everyone.

### Decision interval

`DECISION_INTERVAL = 50` ticks. The agent acts once every 50 game ticks.
This is roughly 2.5 seconds of in-game time. AlgoBots act every 40-60
ticks (per-bot randomized), close to the same cadence.

---

## File map

| File | Role |
|------|------|
| `src/rl/headless/RLConfig.ts` | Constants: K, OBS_SIZE, ACTION_SIZE, rewards |
| `src/rl/headless/Observation.ts` | `extractObs`, `getNeighbors`, `computeActionMask` |
| `src/rl/headless/SpatialObs.ts` | 5-channel spatial patch extractor |
| `src/rl/headless/ActionDecoder.ts` | action index → game Executions |
| `src/rl/headless/DemoCollector.ts` | Wraps `addExecution` to record AlgoBot trajectories |
| `src/rl/headless/runner.ts` | Persistent process; JSON-RPC protocol |
| `rl/env.py` | gymnasium.Env wrapping the runner subprocess |
| `rl/cnn_policy.py` | CNN+MLP feature extractor for MaskablePPO MultiInputPolicy |
| `rl/train.py` | MaskablePPO training driver |
| `rl/collect_demos.py` | Drives `runner demo_episode` to gather AlgoBot demos |
| `rl/bc_train.py` | Behavioral cloning trainer (supervised cross-entropy) |
| `rl/action_distribution.py` | Diagnostic — runs N rollouts and prints action histogram |
| `rl/eval_world.py` | Run a trained policy on world map vs many opponents, render to GIF/MP4 |
| `rl/render_game.py` | Game-style terrain-aware renderer |
| `rl/generate_episode.py` | Stochastic-sampling utility — retries until a target reward is hit |
