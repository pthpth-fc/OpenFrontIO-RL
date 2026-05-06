export const K_NEIGHBORS = 8;

// Self features: 6
// Per-neighbor features: 7 each × K
export const OBS_SIZE = 6 + K_NEIGHBORS * 7;

// 0=noop, 1..K=attack[i], K+1..2K=ally_req[i], 2K+1..3K=break_ally[i], 3K+1=city, 3K+2=defpost, 3K+3=expand (attack TerraNullius)
export const ACTION_SIZE = 1 + K_NEIGHBORS * 3 + 3;

// Spatial patch (channels-first, CHW)
// Channels: 0=land, 1=self, 2=enemy, 3=ally
export const PATCH_SIZE = 32;
export const PATCH_CHANNELS = 4;

export const DECISION_INTERVAL = 50; // ticks between agent decisions
export const SPAWN_PHASE_BUFFER = 115; // ticks to run before first observation
export const MAX_EPISODE_TICKS = 4000;

export const REWARD_WIN = 1.0;
export const REWARD_DEATH = -1.0;
export const REWARD_TILE_GAIN = 0.0002; // per tile gained per step
export const REWARD_TICK = -0.00005; // time penalty per tick

export const TRAINING_MAP = "plains";
export const NUM_OPPONENTS = 4;
