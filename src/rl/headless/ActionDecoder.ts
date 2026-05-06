import { Execution, Game, Player, UnitType } from "../../core/game/Game";
import { TileRef } from "../../core/game/GameMap";
import { AttackExecution } from "../../core/execution/AttackExecution";
import { AllianceRequestExecution } from "../../core/execution/alliance/AllianceRequestExecution";
import { BreakAllianceExecution } from "../../core/execution/alliance/BreakAllianceExecution";
import { ConstructionExecution } from "../../core/execution/ConstructionExecution";
import { K_NEIGHBORS } from "./RLConfig";
import { Neighbors } from "./Observation";

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
  if (action === K_NEIGHBORS * 3 + 1) {
    const tile = pickRandomBorderTile(agent);
    if (!tile) return [];
    return [new ConstructionExecution(agent, UnitType.City, tile)];
  }

  // 3K+2: build defense post
  if (action === K_NEIGHBORS * 3 + 2) {
    const tile = pickRandomBorderTile(agent);
    if (!tile) return [];
    return [new ConstructionExecution(agent, UnitType.DefensePost, tile)];
  }

  return [];
}

function pickRandomBorderTile(agent: Player): TileRef | null {
  const tiles = agent.borderTiles();
  if (tiles.size === 0) return null;
  return tiles.values().next().value as TileRef;
}
