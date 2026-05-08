# RL Run History — Results, Decisions, Lessons

Detailed log of training runs, what worked, what didn't, and **why**. The
chronological table below is the source of truth; the post-table sections
explain the bigger architectural pivots.

---

## Run-by-run results

| Run | Setup | Outcome | Key lesson |
|-----|-------|---------|------------|
| **v1** | plains, 4 AlgoBots, 27 actions, MLP (62-dim flat) | Plateau at **−0.04** after ~200k steps | Action space was missing TerraNullius expand → agent literally could not grow. |
| **v1_expand** | + action 27 (expand to TerraNullius) | Plateau at **+2.60** in 12k steps | Single-action fix unlocked aggressive expansion. Easy win on small map. |
| **v2 bigplains_8tribes** | big_plains, 8 weak tribe Bots | Plateau at **+7.4** | Tribes are pushovers — exposed need for stronger opponents. |
| **v3 bigplains_8algobots** | big_plains, 8 AlgoBots | Climb to **+0.5** in 49k steps; ~10× slower than v1 | Real difficulty. Renders show agent dying mid-game; not attacking enough. |
| **v4 richer_obs_rewards** | + 4 new self features (game_phase, num_cities, num_defposts, tile_delta_recent) + stable neighbor IDs + REWARD_DEATH=−5 / REWARD_TILE_GAIN=0.0005 / REWARD_KILL=0.5 | Climb to **+6.05** at 285k steps | Bigger gradient + stable identity + reward variety unlocked real strategic play. |
| **v5 win15_resume** | resumed v4 250k + REWARD_WIN 5→15 | Plateau at **+6.0–6.4**; stopped at 40k | Win bonus alone didn't push past v4 — wins still rare so the bigger bonus rarely fires. Value head spiked (expected). |
| **v6 win15_explore** | fresh start + ent_coef 0.01→0.05 + REWARD_WIN=15 | 3.45M steps → **+4.5–5.4** | High exploration found a *different* local optimum (more building, fewer expansions, different attack target) but didn't exceed v4. ent_coef=0.05 too high. |
| **v7 3algobots** | resumed v4 250k + NUM_OPPONENTS 8→3 + ent_coef back to 0.01 | Plateau **+3.2** | Smaller bot count = fewer kill rewards available. Agent didn't improve, just survived more often. |
| **v8 bc_warmstart** | fresh start + BC pre-trained on 100 AlgoBot episodes (7424 samples, 88% val acc) + 3 AlgoBots | **+27.3 at 164k steps** | **Best policy by a huge margin.** BC warm-start gave aggressive habits that PPO refined. Climbed from −2.3 (initial) to +27 in 164k steps. |
| **v9 world_10algobots** | resumed v8 150k + map=world + NUM_OPPONENTS=10 + MAX_EPISODE_TICKS=12000 + SPAWN_PHASE_BUFFER=250 + SubprocVecEnv N_ENVS=4 | **+13.1 at 180k steps** (still climbing when stopped) | v8 generalizes to bigger maps but slowly — fps dropped from 125 → 22 (single-env had 21, parallel 4-env got 22). Apparent MPS bottleneck on larger sim. |
| **v10 cnn_dictobs** | NEW architecture: K=8→10, 5-channel CNN spatial patch, MultiInputPolicy, boat actions, 81-dim flat + 5×32×32 spatial, 46-action space | **TODO — under construction** | Designed for world+200 scenarios where K=8 was too narrow and lack of spatial info crippled v8. |

---

## Eval against v8: world map + 200 opponents

Tested **v8 (best, +27 reward on big_plains+3 AlgoBots)** on the world map
with 200 opponents (10 AlgoBots + 190 weak tribes):

- **t=400 (post-spawn):** agent visible, 949 tiles
- **t=600:** agent peaked at **1494 tiles** (third attempt got to 2040 tiles by t=2400)
- **t=2450 (decision 41):** agent collapsed from 2040 → 102 tiles in one step
- **t=2500:** **eliminated**, final reward **−5.08**

What this told us:
1. v8's *strategy* generalized partially — it could expand into a meaningful
   territory (1500-2000 tiles) before being overrun.
2. v8's *limitations* are in observation, not strategy:
   - Only sees K=8 nearest opponents; on world+200 there are 192 invisible.
   - No spatial info → can't reason about thin spike borders, exposed flanks,
     or which neighbors share an ocean.
3. Without boat attacks the agent is stuck on whatever landmass it spawns on.

This eval informed v10's design: bigger K, CNN spatial head, boat actions.

---

## v8: action distribution at +27 plateau

10 stochastic rollouts of v8 250k checkpoint:

```
── Coarse distribution ──
  build          300   38.5%   ███████████████████
  expand         171   21.9%   ██████████
  noop           153   19.6%   █████████
  attack          96   12.3%   ██████
  ally_req        60    7.7%   ███
  break_ally       0    0.0%

── Fine distribution (top 12) ──
  build_city             172   22.1%
  expand                 171   21.9%
  noop                   153   19.6%
  build_defpost          128   16.4%
  attack[6]               75    9.6%   ← consistent target slot
  attack[1]               21    2.7%
  ally_req[3]             13    1.7%
  ally_req[1]             13    1.7%
  ally_req[7]             12    1.5%
  ally_req[4]             10    1.3%
```

**Reading the policy:**
- 38% build is high — agent invests in city/defpost infrastructure
- 22% expand — steady territorial growth
- 12% attack — real combat, not turtle
- **`attack[6]` 9.6% — single-slot dominance is the stable-neighbor-ID
  payoff.** Without v4's stable identity, this concentration would be
  impossible because slot 6 would point to different opponents each step.
- **`break_ally` 0%** — once allied, never breaks. Could be a problem in
  longer games; not yet exploited.

---

## Decision: alternative path (v9) vs architecture overhaul (v10)

v8 worked great on big_plains+3 (+27) but failed on world+200 (−5). The
question was: is this a **policy** problem (not enough training on the new
distribution) or an **architecture** problem (the model can't represent
the necessary reasoning)?

We took the **alternative path first** (v9): keep the architecture, just
resume v8 on a harder env (world map + 10 AlgoBots) and let PPO adapt.

### Why the alternative was worth trying first

- Cheap: ~30 min of config edits, no infrastructure change
- High signal: if v8 adapts smoothly, we learn that the architecture is fine
  and the issue was distribution shift. Saves us a multi-hour rewrite.
- Reusable: even if it doesn't solve world+200, the v9 checkpoint becomes a
  better starting point for v10 (knows the world map dynamics).

### What v9 told us

By 180k steps (~30k after resume), v9 climbed from initial −3.76 (loss of
v8's competence on the new env) to +13 (recovering competence). Episodes
ran 191/240 max decisions, mostly timing out positively.

Crucially: **fps stayed at 22 even with 4 parallel envs.** Sequential v9 was
21 fps; parallel v9 was 22 fps. The parallel speedup didn't materialize
because MPS GPU is the bottleneck, not CPU sim. This is important context
for moving to a CUDA server: there we expect ~3-4× from parallelism.

### What still pushed us to v10

Even though v9 was learning, it couldn't really go past the K=8 ceiling on
world+200. With 200 opponents only 8 visible at any time, the policy is
permanently undersighted. The right move was the architecture overhaul,
which v10 is.

---

## v10 design summary

| Aspect | v8 | v10 |
|--------|----|-----|
| Observation type | flat 66-dim | Dict: 81 flat + 5×32×32 spatial |
| Self features | 10 | 11 (added `num_ports_log`) |
| K (visible opponents) | 8 | **10** |
| Action space | 28 | **46** (added build_port, 10 boat_attack[k], boat_expand) |
| Architecture | MlpPolicy | **MultiInputPolicy + custom CNN+MLP extractor** |
| Spatial info | none | **5-channel patch around centroid** |
| Boat attacks | none | **yes** (across-ocean engagement) |

See `docs/RL_ARCHITECTURE.md` for the full obs/action specification.

### v10 training plan

1. **Recollect demos** on world+10 AlgoBots in dict obs format (covers
   AlgoBot boat behavior — they already use ships)
2. **Train BC** on dict obs with the new CNN+MLP architecture
3. **Warm-start PPO** with BC weights into MultiInputPolicy
4. **PPO fine-tune** on world+10 → eventually world+50 → world+200

### Known costs

- v8 weights are completely incompatible (different obs shape, action count,
  architecture). Must start over. BC warm-start is critical.
- World+10 sim throughput on Mac MPS is ~22 fps. CUDA server should be ~3-5×.
- Pipe overhead: 5×32×32 = 5120 floats per step on top of the 81-dim flat.
- Demo collection from 4-bot games on big_plains gave ~75 samples/episode.
  World+10 will produce ~200-500 samples/episode but episodes are 2-5×
  longer (real time). Expect ~1-2 minutes per episode for demos.

---

## Decisions still on the table

0. **Multi-scale spatial obs (deferred — wait for v10 signal first).** The
   v10 CNN sees a 32×32 patch around the agent's centroid. For agents with
   territories larger than ~32 tiles in any dimension, the outer borders
   are *outside* the patch — the CNN can't see them. We're starting v10
   with the single-scale patch and reading what it learns; if performance
   plateaus and renders show the agent failing on its outer rim (where the
   patch can't see), we'll add a second channel set: a downsampled global
   view of the whole map (e.g. 64×64 covering everything at coarse
   resolution). The CNN would then have both local detail and global
   layout. Cheap to add (~30 lines TS) but only worth doing if we observe
   the limitation hurting in practice.

1. **Is build placement worth learning?** Currently fixed (first border tile).
   Pointer-net output head would let the model pick *where* to build, but it's
   substantial new infra. Punt until we see whether build location is
   actually limiting policy quality.

2. **Should `break_ally` be removed?** v8 used it 0% of the time. Removing
   it would shrink the action space slightly. Probably fine to leave — the
   mask handles validity and the model learns to ignore it.

3. **Self-play.** Once v10 is solid against AlgoBots, replace some opponents
   with frozen-self copies. Not yet implemented. Requires either ONNX in
   Node (sync inference is the hard part — ORT is async) or another runner
   protocol that queries Python for opponent actions.

4. **Multi-map training.** All training so far has been single-map. Adding
   a "map name one-hot" to the obs and rotating maps each episode would
   be a small obs change but could give us a much more robust policy.

5. **CUDA server.** All v1-v9 trained on Mac MPS. Moving to a CUDA box
   should give ~3-5× throughput, especially with parallel envs. The
   `TRAIN_DEVICE` env var lets the same scripts run anywhere.
