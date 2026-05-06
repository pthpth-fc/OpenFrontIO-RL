/**
 * Headless RL runner — persistent Node.js process.
 * Communicates with the Python gymnasium.Env via newline-delimited JSON on stdin/stdout.
 *
 * Protocol:
 *   reset  → {"cmd":"reset"}
 *   step   → {"cmd":"step","action":<int>}
 *   quit   → {"cmd":"quit"}
 *
 * Responses:
 *   reset  ← {"obs":[...float32],"mask":[...bool]}
 *   step   ← {"obs":[...float32],"mask":[...bool],"reward":<float>,"done":<bool>,"info":{}}
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import * as readline from "readline";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Redirect all console output to stderr so stdout stays clean for JSON protocol
const toStderr =
  (...args: unknown[]) =>
    process.stderr.write(args.join(" ") + "\n");
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;
console.warn = toStderr;
// Keep console.error as-is (already goes to stderr)

import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
  TerraNullius,
} from "../../core/game/Game";
import { createGame } from "../../core/game/GameImpl";
import {
  genTerrainFromBin,
  MapManifest,
} from "../../core/game/TerrainMapLoader";
import { UserSettings } from "../../core/game/UserSettings";
import { GameConfig } from "../../core/Schemas";
import { DefaultConfig } from "../../core/configuration/DefaultConfig";
import { AlgoBotExecution } from "../../core/execution/AlgoBotExecution";
import { SpawnExecution } from "../../core/execution/SpawnExecution";
import { WinCheckExecution } from "../../core/execution/WinCheckExecution";
import { PseudoRandom } from "../../core/PseudoRandom";
import { simpleHash } from "../../core/Util";
import { TestServerConfig } from "../../../tests/util/TestServerConfig";
import {
  DECISION_INTERVAL,
  MAX_EPISODE_TICKS,
  NUM_OPPONENTS,
  REWARD_DEATH,
  REWARD_TICK,
  REWARD_TILE_GAIN,
  REWARD_WIN,
  SPAWN_PHASE_BUFFER,
  TRAINING_MAP,
} from "./RLConfig";
import { computeActionMask, extractObs, getNeighbors } from "./Observation";
import { decodeAction } from "./ActionDecoder";
import { extractSpatialObs } from "./SpatialObs";

const MAPS_DIR = path.resolve(
  __dirname,
  "../../../tests/testdata/maps",
);
const GAME_ID = "rl-headless";

interface ResetResponse {
  vec: number[];   // flat obs vector (OBS_SIZE)
  map: number[];   // spatial patch (PATCH_CHANNELS * PATCH_SIZE * PATCH_SIZE), CHW
  mask: boolean[];
}

interface StepResponse {
  vec: number[];
  map: number[];
  mask: boolean[];
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

interface SnapshotResponse {
  width: number;
  height: number;
  // Flat array of player-index per tile (land tiles only stored as [tileRef, playerIdx] pairs
  // to keep payload small). playerIdx 0 = unclaimed, 1..N = players in order of first spawn.
  // Format: [tileRef, playerIdx, tileRef, playerIdx, ...] for owned tiles only.
  owned: number[];
  // Player list in index order (1-based): [{id, name, isAgent}]
  players: { id: string; name: string; isAgent: boolean }[];
  ticks: number;
}

async function loadMap(mapName: string) {
  const dir = path.join(MAPS_DIR, mapName);
  const mapBin = fs.readFileSync(path.join(dir, "map.bin"));
  const miniMapBin = fs.readFileSync(path.join(dir, "map4x.bin"));
  const manifest = JSON.parse(
    fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
  ) as MapManifest;

  const gameMap = await genTerrainFromBin(manifest.map, mapBin);
  const miniGameMap = await genTerrainFromBin(manifest.map4x, miniMapBin);
  return { gameMap, miniGameMap };
}

function makeConfig(gameConfig: GameConfig): DefaultConfig {
  return new DefaultConfig(new TestServerConfig(), gameConfig, new UserSettings(), false);
}

// ── Episode state ────────────────────────────────────────────────────────────

let game: Game | null = null;
let agentPlayer: Player | null = null;
let ticks = 0;
let prevTiles = 0;

async function resetEpisode(): Promise<ResetResponse> {
  const { gameMap, miniGameMap } = await loadMap(TRAINING_MAP);

  const agentID = "rl-agent";
  const agentInfo = new PlayerInfo("RLAgent", PlayerType.Human, null, agentID);

  const gameConfig: GameConfig = {
    gameMap: GameMapType.World,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: NUM_OPPONENTS,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: true,
    algoBots: 0,
  };

  const config = makeConfig(gameConfig);
  game = createGame([agentInfo], [], gameMap, miniGameMap, config);

  // Spawn regular tribe bots (PlayerType.Bot — weaker stats than Human; SpawnExecution
  // auto-attaches TribeExecution which gives them their AI behavior)
  const random = new PseudoRandom(simpleHash(GAME_ID) + 2);
  for (let i = 0; i < NUM_OPPONENTS; i++) {
    const botInfo = new PlayerInfo(
      `Tribe${i + 1}`,
      PlayerType.Bot,
      null,
      random.nextID(),
    );
    game.addExecution(new SpawnExecution(GAME_ID, botInfo));
  }
  game.addExecution(new WinCheckExecution());

  // Spawn agent
  game.addExecution(new SpawnExecution(GAME_ID, agentInfo));

  ticks = 0;
  prevTiles = 0;

  // Run through spawn phase buffer before first observation
  for (let t = 0; t < SPAWN_PHASE_BUFFER; t++) {
    game.executeNextTick();
    ticks++;
  }

  agentPlayer = game.player(agentID);
  prevTiles = agentPlayer.numTilesOwned();

  return buildResetResponse();
}

function buildResetResponse(): ResetResponse {
  const g = game!;
  const agent = agentPlayer!;
  const neighbors = getNeighbors(g, agent);
  const vec = Array.from(extractObs(g, agent, g.numLandTiles(), neighbors));
  const map = Array.from(extractSpatialObs(g, agent));
  const mask = computeActionMask(agent, neighbors);
  return { vec, map, mask };
}

function stepEpisode(action: number): StepResponse {
  const g = game!;
  const agent = agentPlayer!;

  // Apply action
  const executions = decodeAction(action, g, agent, getNeighbors(g, agent));
  if (executions.length > 0) {
    g.addExecution(...executions);
  }

  // Advance DECISION_INTERVAL ticks
  let done = false;
  for (let t = 0; t < DECISION_INTERVAL; t++) {
    g.executeNextTick();
    ticks++;
    if (g.getWinner() !== null || ticks >= MAX_EPISODE_TICKS) {
      done = true;
      break;
    }
  }

  // Compute reward
  let reward = REWARD_TICK * DECISION_INTERVAL;

  if (!agent.isAlive()) {
    reward += REWARD_DEATH;
    done = true;
  } else {
    const curTiles = agent.numTilesOwned();
    reward += (curTiles - prevTiles) * REWARD_TILE_GAIN;
    prevTiles = curTiles;

    const winner = g.getWinner();
    if (winner !== null) {
      if (
        typeof winner === "object" &&
        "id" in winner &&
        winner.id() === agent.id()
      ) {
        reward += REWARD_WIN;
      }
      done = true;
    }
  }

  const neighbors = getNeighbors(g, agent);
  const vec = Array.from(extractObs(g, agent, g.numLandTiles(), neighbors));
  const map = Array.from(extractSpatialObs(g, agent));
  const mask = computeActionMask(agent, neighbors);

  return { vec, map, mask, reward, done, info: { ticks } };
}

function snapshotGame(): SnapshotResponse {
  const g = game!;
  const agent = agentPlayer!;

  // Build player index map (1-based; 0 = unclaimed)
  const alivePlayers = g.players().filter((p) => p.isAlive());
  const playerIndex = new Map<string, number>();
  alivePlayers.forEach((p, i) => playerIndex.set(p.id(), i + 1));

  const players = alivePlayers.map((p, i) => ({
    id: p.id(),
    name: p.displayName(),
    isAgent: p.id() === agent.id(),
  }));

  const owned: number[] = [];
  g.forEachTile((ref) => {
    if (!g.isLand(ref)) return;
    const o = g.owner(ref);
    if (o.isPlayer()) {
      const idx = playerIndex.get((o as Player).id()) ?? 0;
      if (idx > 0) {
        owned.push(ref, idx);
      }
    }
  });

  return {
    width: g.width(),
    height: g.height(),
    owned,
    players,
    ticks,
  };
}

// ── Main loop ────────────────────────────────────────────────────────────────

async function main() {
  const rl = readline.createInterface({ input: process.stdin });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    let msg: { cmd: string; action?: number; actions?: number[] };
    try {
      msg = JSON.parse(trimmed);
    } catch {
      process.stdout.write(
        JSON.stringify({ error: "invalid json" }) + "\n",
      );
      continue;
    }

    if (msg.cmd === "reset") {
      const resp = await resetEpisode();
      process.stdout.write(JSON.stringify(resp) + "\n");
    } else if (msg.cmd === "step") {
      if (game === null || agentPlayer === null) {
        process.stdout.write(
          JSON.stringify({ error: "call reset first" }) + "\n",
        );
        continue;
      }
      const resp = stepEpisode(msg.action ?? 0);
      process.stdout.write(JSON.stringify(resp) + "\n");
    } else if (msg.cmd === "snapshot") {
      if (game === null || agentPlayer === null) {
        process.stdout.write(
          JSON.stringify({ error: "call reset first" }) + "\n",
        );
        continue;
      }
      process.stdout.write(JSON.stringify(snapshotGame()) + "\n");
    } else if (msg.cmd === "quit") {
      break;
    } else {
      process.stdout.write(
        JSON.stringify({ error: `unknown cmd: ${msg.cmd}` }) + "\n",
      );
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
