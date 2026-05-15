import { Game, Player, UnitType } from "../../core/game/Game";
import { K_NEIGHBORS, MAX_EPISODE_TICKS, OBS_SIZE } from "./RLConfig";

export interface Neighbors {
  list: (Player | null)[];
}

/**
 * Build a stable opponent ordering for the episode. Slot k is bound to a
 * specific opponent ID for the entire episode — when that opponent dies,
 * the slot stays empty (null) instead of shifting other opponents up.
 *
 * Ordering: sort opponents by smallID() so the assignment is deterministic
 * within a game (independent of who's bordering whom at any moment).
 */
export function buildStableNeighborOrder(game: Game, agent: Player): string[] {
  const opponents = game
    .players()
    .filter((p) => p.id() !== agent.id())
    .sort((a, b) => a.smallID() - b.smallID());
  return opponents.slice(0, K_NEIGHBORS).map((p) => p.id());
}

/**
 * Resolve the stable order to current Player references. Dead opponents
 * become null in their slot so the model sees "this slot is empty now".
 */
export function getNeighbors(
  game: Game,
  stableOrder: string[],
): Neighbors {
  const list: (Player | null)[] = [];
  for (let k = 0; k < K_NEIGHBORS; k++) {
    const id = stableOrder[k];
    if (!id || !game.hasPlayer(id)) {
      list.push(null);
      continue;
    }
    const p = game.player(id);
    list.push(p.isAlive() ? p : null);
  }
  return { list };
}

export function extractObs(
  game: Game,
  agent: Player,
  totalLandTiles: number,
  neighbors: Neighbors,
  ticks: number,
  prevTiles: number,
): Float32Array {
  const obs = new Float32Array(OBS_SIZE);
  let i = 0;

  const alive = game.players().filter((p) => p.isAlive());
  const maxTroops = game.config().maxTroops(agent);
  const sortedByTiles = [...alive].sort(
    (a, b) => b.numTilesOwned() - a.numTilesOwned(),
  );
  const rank = sortedByTiles.findIndex((p) => p.id() === agent.id());

  // Self (10)
  obs[i++] = maxTroops > 0 ? Math.min(agent.troops() / maxTroops, 1) : 0;
  obs[i++] = Math.min(Number(agent.gold()), 10000) / 10000;
  obs[i++] = totalLandTiles > 0 ? agent.numTilesOwned() / totalLandTiles : 0;
  obs[i++] =
    agent.numTilesOwned() > 0
      ? Math.min(agent.borderTiles().size / agent.numTilesOwned(), 1)
      : 0;
  obs[i++] =
    alive.length > 1 ? agent.alliances().length / (alive.length - 1) : 0;
  obs[i++] = alive.length > 1 ? rank / (alive.length - 1) : 0;

  // Game phase (1)
  obs[i++] = Math.min(ticks / MAX_EPISODE_TICKS, 1);

  // Structure counts, log-scaled to keep small numbers separable (3)
  // log(1+count)/log(11) ≈ maps 0→0, 10→1, saturates above
  const numCities = agent.units(UnitType.City).length;
  const numDefposts = agent.units(UnitType.DefensePost).length;
  const numPorts = agent.units(UnitType.Port).length;
  obs[i++] = Math.min(Math.log1p(numCities) / Math.log(11), 1);
  obs[i++] = Math.min(Math.log1p(numDefposts) / Math.log(11), 1);
  obs[i++] = Math.min(Math.log1p(numPorts) / Math.log(11), 1);

  // Recent tile delta as a fraction of current territory (1)
  // Positive = growing, negative = losing tiles to attackers
  const cur = agent.numTilesOwned();
  const denom = Math.max(cur, 1);
  obs[i++] = Math.max(-1, Math.min(1, (cur - prevTiles) / denom));

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
    obs[i++] = agent
      .nearby()
      .some((nb) => nb.isPlayer() && (nb as Player).id() === n.id())
      ? 1
      : 0;
    obs[i++] = agent.incomingAttacks().some((a) => a.attacker().id() === n.id())
      ? 1
      : 0;
  }

  return obs;
}

export function computeActionMask(
  agent: Player,
  neighbors: Neighbors,
  game?: Game,
): boolean[] {
  const mask: boolean[] = new Array(1 + K_NEIGHBORS * 4 + 5).fill(false);
  mask[0] = true; // no-op always valid

  const hasPort = agent.units(UnitType.Port).length > 0;

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
    // Boat attack: valid if we have a port and target is alive (engine handles
    // the rest — if there's no viable sea route, the AttackExecution rejects).
    if (hasPort && !agent.isFriendly(n)) {
      mask[K_NEIGHBORS * 3 + 5 + k] = true;
    }
  }

  // Build actions
  mask[K_NEIGHBORS * 3 + 1] = agent.numTilesOwned() > 5; // city
  mask[K_NEIGHBORS * 3 + 2] = agent.borderTiles().size > 0; // defpost

  // Expand (attack TerraNullius): valid whenever agent has tiles to attack from
  mask[K_NEIGHBORS * 3 + 3] = agent.numTilesOwned() > 0;

  // Build port: requires a shoreline border tile
  if (game) {
    let hasShoreBorder = false;
    for (const t of agent.borderTiles()) {
      if (game.isShoreline(t)) {
        hasShoreBorder = true;
        break;
      }
    }
    mask[K_NEIGHBORS * 3 + 4] = hasShoreBorder;
  }

  // Boat expand: requires at least one port
  mask[K_NEIGHBORS * 4 + 5] = hasPort;

  return mask;
}
