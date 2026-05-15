export const K_NEIGHBORS = 10;

// Self features: 11
//   Original 6: troops, gold, tiles%, border_exposure, alliance_density, rank
//   Added 4:    game_phase, num_cities (log), num_defposts (log), tile_delta_recent
//   Added 1:    num_ports (log) — ports gate boat-based actions
// Per-neighbor features: 7 each × K
export const OBS_SIZE = 11 + K_NEIGHBORS * 7;

// Action layout (slots 0..27 unchanged from v4 so v8 weights can transfer):
//   0                                    noop
//   1..K                                 attack[k]                (land, must border)
//   K+1..2K                              ally_req[k]
//   2K+1..3K                             break_ally[k]
//   3K+1                                 build_city
//   3K+2                                 build_defpost
//   3K+3                                 expand                   (land, attack TerraNullius)
//   3K+4                                 build_port               (NEW)
//   3K+5..4K+4                           boat_attack[k]           (NEW; attack via ship from a port)
//   4K+5                                 boat_expand              (NEW; ship to unclaimed shore)
export const ACTION_SIZE = 1 + K_NEIGHBORS * 4 + 5;

// Spatial patch (channels-first, CHW)
// Channels: 0=land, 1=self, 2=enemy, 3=ally, 4=own_border (exposed agent tiles)
export const PATCH_SIZE = 32;
export const PATCH_CHANNELS = 5;

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
