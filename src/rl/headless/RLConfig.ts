export const K_NEIGHBORS = 8;

// Self features: 10
//   Original 6: troops, gold, tiles%, border_exposure, alliance_density, rank
//   Added 4:    game_phase, num_cities (log), num_defposts (log), tile_delta_recent
// Per-neighbor features: 7 each × K
export const OBS_SIZE = 10 + K_NEIGHBORS * 7;

// 0=noop, 1..K=attack[i], K+1..2K=ally_req[i], 2K+1..3K=break_ally[i], 3K+1=city, 3K+2=defpost, 3K+3=expand (attack TerraNullius)
export const ACTION_SIZE = 1 + K_NEIGHBORS * 3 + 3;

// Spatial patch (channels-first, CHW)
// Channels: 0=land, 1=self, 2=enemy, 3=ally
export const PATCH_SIZE = 32;
export const PATCH_CHANNELS = 4;

export const DECISION_INTERVAL = 50; // ticks between agent decisions
export const SPAWN_PHASE_BUFFER = 250; // longer to allow ~10 players to spawn on bigger maps
export const MAX_EPISODE_TICKS = 12000; // world map games need much more time to play out

// Reward shaping — death dominates expansion, winning dominates timeout.
//   max plausible cumulative tile shaping in an episode: ~+2.5 (5000 × 0.0005)
//   death penalty: -5.0 → "expand wildly then die" is strictly net-negative
//   win bonus: +15.0 → winning is ~2× best timeout outcome, pushes the policy
//                      off "turtle to MAX_EPISODE_TICKS" and toward decisive play
export const REWARD_WIN = 15.0;
export const REWARD_DEATH = -5.0;
export const REWARD_TILE_GAIN = 0.0005; // per tile delta per step
export const REWARD_TICK = -0.00005; // time penalty per tick
export const REWARD_KILL = 0.5; // bonus per opponent eliminated

export const TRAINING_MAP = "world";
export const NUM_OPPONENTS = 10;
