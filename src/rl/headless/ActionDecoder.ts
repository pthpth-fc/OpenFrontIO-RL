import { Execution, Game, Player, UnitType } from "../../core/game/Game";
import { TileRef } from "../../core/game/GameMap";
import { AttackExecution } from "../../core/execution/AttackExecution";
import { AllianceRequestExecution } from "../../core/execution/alliance/AllianceRequestExecution";
import { BreakAllianceExecution } from "../../core/execution/alliance/BreakAllianceExecution";
import { ConstructionExecution } from "../../core/execution/ConstructionExecution";
import { K_NEIGHBORS } from "./RLConfig";
import { Neighbors } from "./Observation";

// Action layout offsets — keep in sync with RLConfig.ts
const BUILD_CITY = K_NEIGHBORS * 3 + 1;
const BUILD_DEFPOST = K_NEIGHBORS * 3 + 2;
const EXPAND_LAND = K_NEIGHBORS * 3 + 3;
const BUILD_PORT = K_NEIGHBORS * 3 + 4;
const BOAT_ATTACK_BASE = K_NEIGHBORS * 3 + 5;            // boat_attack[0] starts here
const BOAT_EXPAND = K_NEIGHBORS * 4 + 5;                 // last action

// Returns the Execution(s) to apply for the given action index, or [] for no-op / invalid.
export function decodeAction(
  action: number,
  game: Game,
  agent: Player,
  neighbors: Neighbors,
): Execution[] {
  // 0: no-op
  if (action === 0) return [];

  // 1..K: attack neighbor[k]
  if (action >= 1 && action <= K_NEIGHBORS) {
    const k = action - 1;
    const target = neighbors.list[k];
    if (!target) return [];
    return [new AttackExecution(null, agent, target.id(), null)];
  }

  // K+1..2K: request alliance with neighbor[k]
  if (action >= K_NEIGHBORS + 1 && action <= K_NEIGHBORS * 2) {
    const k = action - K_NEIGHBORS - 1;
    const target = neighbors.list[k];
    if (!target) return [];
    return [new AllianceRequestExecution(agent, target.id())];
  }

  // 2K+1..3K: break alliance with neighbor[k]
  if (action >= K_NEIGHBORS * 2 + 1 && action <= K_NEIGHBORS * 3) {
    const k = action - K_NEIGHBORS * 2 - 1;
    const target = neighbors.list[k];
    if (!target) return [];
    return [new BreakAllianceExecution(agent, target.id())];
  }

  // 3K+1: build city
  if (action === BUILD_CITY) {
    const tile = pickRandomBorderTile(agent);
    if (!tile) return [];
    return [new ConstructionExecution(agent, UnitType.City, tile)];
  }

  // 3K+2: build defense post
  if (action === BUILD_DEFPOST) {
    const tile = pickRandomBorderTile(agent);
    if (!tile) return [];
    return [new ConstructionExecution(agent, UnitType.DefensePost, tile)];
  }

  // 3K+3: land expand into TerraNullius
  if (action === EXPAND_LAND) {
    return [new AttackExecution(null, agent, null, null)];
  }

  // 3K+4: build port (must be on a shoreline tile of the agent's territory)
  if (action === BUILD_PORT) {
    const tile = pickShorelineTile(game, agent);
    if (!tile) return [];
    return [new ConstructionExecution(agent, UnitType.Port, tile)];
  }

  // 3K+5..4K+4: boat_attack[k] — ship-based attack on neighbor[k] launched from a port
  if (action >= BOAT_ATTACK_BASE && action < BOAT_ATTACK_BASE + K_NEIGHBORS) {
    const k = action - BOAT_ATTACK_BASE;
    const target = neighbors.list[k];
    if (!target) return [];
    const port = pickPortTile(agent);
    if (!port) return [];
    return [new AttackExecution(null, agent, target.id(), port)];
  }

  // 4K+5: boat_expand — ship to unclaimed shore (null target with sourceTile = port)
  if (action === BOAT_EXPAND) {
    const port = pickPortTile(agent);
    if (!port) return [];
    return [new AttackExecution(null, agent, null, port)];
  }

  return [];
}

function pickRandomBorderTile(agent: Player): TileRef | null {
  const tiles = agent.borderTiles();
  if (tiles.size === 0) return null;
  return tiles.values().next().value as TileRef;
}

function pickShorelineTile(game: Game, agent: Player): TileRef | null {
  // First shoreline border tile we encounter — deterministic.
  for (const t of agent.borderTiles()) {
    if (game.isShoreline(t)) return t;
  }
  return null;
}

function pickPortTile(agent: Player): TileRef | null {
  const ports = agent.units(UnitType.Port);
  if (ports.length === 0) return null;
  return ports[0].tile();
}
