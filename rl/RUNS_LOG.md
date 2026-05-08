# RL Run Log — to sync to Notion

(Notion MCP was transiently unavailable when these were written;
this file is the source of truth until synced.)

---

## Run history (chronological)

| Run | Setup | Outcome | Lesson |
|-----|-------|---------|--------|
| **v1** | plains, 4 AlgoBots, 27 actions, MLP | Plateau at **−0.04** | Action space was missing TerraNullius expand → agent literally could not grow. |
| **v1_expand** | + action 27 (expand) | Plateau at **+2.60** in 12k steps | Single-action fix unlocked aggressive expansion. Easy win on small map. |
| **v2 bigplains_8tribes** | big_plains, 8 weak Bots | **+7.4** plateau | Tribes are pushovers — exposed need for stronger opponents. |
| **v3 bigplains_8algobots** | + 8 AlgoBots | Climb to **+0.5** in 49k steps; 10× slower | Real difficulty. Renders show agent dying mid-game; not attacking enough. |
| **v4 richer_obs_rewards** | + 4 new obs features (game_phase, num_cities, num_defposts, tile_delta) + stable neighbor IDs + REWARD_DEATH=−5/REWARD_TILE_GAIN=0.0005/REWARD_KILL=0.5 | Climb to **+6.05** in 285k steps | Bigger gradient + identity stability + reward variety unlocked real strategic play. |
| **v5 win15_resume** | resumed v4 250k + REWARD_WIN 5→15 | Plateau at **+6.0–6.4**; abandoned at 40k | Win bonus alone didn't push past v4 — wins still rare. Value head spiked (expected). |
| **v6 win15_explore** | fresh start + ent_coef 0.01→0.05 | 3.45M steps → **+4.5–5.4** | High exploration found a *different* local optimum (more building, fewer expansions, different attack target) but did not exceed v4. ent_coef=0.05 too high. |
| **v7 3algobots** | resumed v4 250k + NUM_OPPONENTS 8→3 + ent_coef back to 0.01 | Plateau **+3.2** | Smaller bot count = fewer kill rewards available; agent didn't improve, just survived more often. |
| **v8 bc_warmstart** | fresh start + BC pre-trained on 100 AlgoBot episodes (7424 samples, 88% val acc) + 3 AlgoBots | **+27.3 at 164k steps** | **Best policy by a huge margin.** BC warm-start gave the policy aggressive habits that PPO refined. Climbed from −2.3 (initial) to +27 in 164k steps. |
| **v9 world_10algobots** | resumed v8 150k + map=world + NUM_OPPONENTS=10 + MAX_EPISODE_TICKS=12000 + SPAWN_PHASE_BUFFER=250 | **RUNNING** | Test whether v8's policy generalizes to bigger maps with more opponents *before* committing to architecture rewrite. |

---

## Scaling to large maps & many opponents — the path forward

Tested **v8 (best policy, +27 reward on big_plains+3 AlgoBots) on world map + 200 opponents**:
agent peaked at 2040 tiles around d=40 then collapsed at d=42. Final reward −5.08.

### Why v8 fails on world+200

1. **Trained only on big_plains × 3 AlgoBots** — never saw the dynamics of huge map / many opponents.
2. **Observation only sees K=8 nearest opponents** — 192 are invisible.
3. **No spatial info at all** — can't reason about "I have a long thin border on my east that's exposed."
4. **Policy was tuned for short ~80-decision episodes** — on world map, episodes are 500+ decisions, very different game phase distribution.

### Two paths forward

**Alternative path (cheaper, taken first — v9):**
- Resume v8's 150k checkpoint, change map to `world`, opponents to **10 AlgoBots** (no tribes), bump `MAX_EPISODE_TICKS` 4000 → 12000, `SPAWN_PHASE_BUFFER` 115 → 250.
- No architecture changes. Same MLP policy, same K=8, same 28 actions.
- **Goal:** confirm whether v8's *strategy* generalizes to bigger maps *before* throwing out the architecture.
- **Decision criteria:** if v9 plateaus low (e.g., −2 or worse) within ~300k steps, we move on to the bigger architecture change.

**Bigger architecture overhaul (the CNN+MLP idea, queued for v10+):**

If v9 plateaus, this is the next move:

#### Architecture
- **Bigger flat K**: `K_NEIGHBORS = 8 → 10`. `OBS_SIZE` 66 → 76, `ACTION_SIZE` 28 → 30 (one extra slot per direction in attack/ally/break_ally).
- **CNN spatial head**: 5 channels × 32×32 patch around agent's centroid:
  - 0: land mask
  - 1: self tiles
  - 2: enemy tiles
  - 3: ally tiles
  - 4: own borders (binary mask of agent's exposed border tiles)
  - The own-borders channel directly answers "where am I exposed" without the model having to derive it from raw masks.
- **MultiInputPolicy** with custom feature extractor: MLP(76 → 128) + CNN(5×32×32 → 256) → concat 384 → actor [256, 128] / critic [256, 128].

#### Training pipeline
1. **Collect demos** on world+200 with AlgoBots in dict-obs format
2. **Train BC** on dict obs with 5-channel CNN + flat MLP
3. **Warm-start PPO** with BC weights
4. **PPO fine-tune** on world+200 from the BC start

#### Cost
- v8 weights are completely incompatible (different obs shape, action count, architecture). Must start over. BC warm-start is critical for fast bootstrap.
- Throughput: world+200 simulation is ~20–50× slower than big_plains+3. Expect ~5–15 fps instead of 125 fps.
- Pipe overhead: 5×32×32 = 5120 floats per step on top of the existing flat obs.

#### Pre-existing pieces (mostly already written)
- `src/rl/headless/SpatialObs.ts` — 4-channel patch extractor (need 5th channel)
- `rl/cnn_policy.py` — CNN feature extractor (input shape tweak)
- `rl/env.py` — needs Dict obs space (was reverted for v1)
- `rl/train.py` — needs `MultiInputPolicy` + `policy_kwargs`

About 1–2 hours of focused work plus collection (~30 min) and BC training (~10 min) before PPO can start.
