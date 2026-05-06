import { Game, Player, Relation } from "../../core/game/Game";
import { K_NEIGHBORS, OBS_SIZE } from "./RLConfig";

export interface Neighbors {
  list: (Player | null)[];
}

export function getNeighbors(game: Game, agent: Player): Neighbors {
  const alive = game.players().filter((p) => p.isAlive() && p.id() !== agent.id());
  const bordering = new Set<string>(
    agent
      .nearby()
      .filter((n) => n.isPlayer())
      .map((n) => (n as Player).id()),
  );

  const sorted = alive.sort((a, b) => {
    const ab = bordering.has(a.id()) ? 1 : 0;
    const bb = bordering.has(b.id()) ? 1 : 0;
    if (ab !== bb) return bb - ab;
    return b.numTilesOwned() - a.numTilesOwned();
  });

  const list: (Player | null)[] = [];
  for (let k = 0; k < K_NEIGHBORS; k++) {
    list.push(sorted[k] ?? null);
  }
  return { list };
}

export function extractObs(
  game: Game,
  agent: Player,
  totalLandTiles: number,
  neighbors: Neighbors,
): Float32Array {
  const obs = new Float32Array(OBS_SIZE);
  let i = 0;

  const alive = game.players().filter((p) => p.isAlive());
  const maxTroops = game.config().maxTroops(agent);
  const sortedByTiles = [...alive].sort(
    (a, b) => b.numTilesOwned() - a.numTilesOwned(),
  );
  const rank = sortedByTiles.findIndex((p) => p.id() === agent.id());

  // Self (6)
  obs[i++] = maxTroops > 0 ? Math.min(agent.troops() / maxTroops, 1) : 0;
  obs[i++] = Math.min(Number(agent.gold()), 10000) / 10000;
  obs[i++] =
    totalLandTiles > 0 ? agent.numTilesOwned() / totalLandTiles : 0;
  obs[i++] =
    agent.numTilesOwned() > 0
      ? Math.min(agent.borderTiles().size / agent.numTilesOwned(), 1)
      : 0;
  obs[i++] =
    alive.length > 1 ? agent.alliances().length / (alive.length - 1) : 0;
  obs[i++] = alive.length > 1 ? rank / (alive.length - 1) : 0;

  // Neighbors (K × 7)
  for (const n of neighbors.list) {
    if (n === null) {
      i += 7;
      continue;
    }
    obs[i++] = 1; // valid slot
    obs[i++] =
      agent.troops() > 0
        ? Math.min(n.troops() / agent.troops(), 5) / 5
        : 0;
    obs[i++] = totalLandTiles > 0 ? n.numTilesOwned() / totalLandTiles : 0;
    obs[i++] = agent.allianceWith(n) !== null ? 1 : 0;
    obs[i++] = agent.relation(n) / 3;
    obs[i++] = agent.nearby().some((nb) => nb.isPlayer() && (nb as Player).id() === n.id()) ? 1 : 0;
    obs[i++] = agent.incomingAttacks().some((a) => a.attacker().id() === n.id()) ? 1 : 0;
  }

  return obs;
}

export function computeActionMask(
  agent: Player,
  neighbors: Neighbors,
): boolean[] {
  const mask: boolean[] = new Array(1 + K_NEIGHBORS * 3 + 3).fill(false);
  mask[0] = true; // no-op always valid

  for (let k = 0; k < K_NEIGHBORS; k++) {
    const n = neighbors.list[k];
    if (!n) continue;
    const bordering = agent
      .nearby()
      .some((nb) => nb.isPlayer() && (nb as Player).id() === n.id());

    // Attack: valid if bordering enemy
    if (bordering && !agent.isFriendly(n)) {
      mask[1 + k] = true;
    }
    // Alliance request: valid if not already allied
    if (agent.allianceWith(n) === null) {
      mask[1 + K_NEIGHBORS + k] = true;
    }
    // Break alliance: valid if allied
    if (agent.allianceWith(n) !== null) {
      mask[1 + K_NEIGHBORS * 2 + k] = true;
    }
  }

  // Build actions always available (ConstructionExecution ignores gold silently)
  mask[1 + K_NEIGHBORS * 3] = agent.numTilesOwned() > 5; // city
  mask[1 + K_NEIGHBORS * 3 + 1] = agent.borderTiles().size > 0; // defpost

  // Expand (attack TerraNullius): valid whenever agent has tiles to attack from
  mask[1 + K_NEIGHBORS * 3 + 2] = agent.numTilesOwned() > 0;

  return mask;
}
