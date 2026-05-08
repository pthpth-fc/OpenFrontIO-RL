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
import { TribeSpawner } from "../../core/execution/TribeSpawner";
import { WinCheckExecution } from "../../core/execution/WinCheckExecution";
import { PseudoRandom } from "../../core/PseudoRandom";
import { simpleHash } from "../../core/Util";
import { TestServerConfig } from "../../../tests/util/TestServerConfig";
import {
  DECISION_INTERVAL,
  MAX_EPISODE_TICKS,
  NUM_OPPONENTS,
  REWARD_DEATH,
  REWARD_KILL,
  REWARD_TICK,
  REWARD_TILE_GAIN,
  REWARD_WIN,
  SPAWN_PHASE_BUFFER,
  TRAINING_MAP,
} from "./RLConfig";
import {
  buildStableNeighborOrder,
  computeActionMask,
  extractObs,
  getNeighbors,
} from "./Observation";
import { decodeAction } from "./ActionDecoder";
import { extractSpatialObs } from "./SpatialObs";
import { attachDemoHook } from "./DemoCollector";

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
let neighborOrder: string[] = []; // stable opponent IDs (slot k bound for the episode)
let aliveOpponentIDs = new Set<string>(); // for kill-reward detection
let currentMaxTicks = MAX_EPISODE_TICKS;

interface ResetOverrides {
  map?: string;
  numTribes?: number;
  numAlgoBots?: number;
  maxTicks?: number;
  spawnBuffer?: number;
}

async function resetEpisode(overrides: ResetOverrides = {}): Promise<ResetResponse> {
  const mapName = overrides.map ?? TRAINING_MAP;
  const numAlgoBots = overrides.numAlgoBots ?? NUM_OPPONENTS;
  const numTribes = overrides.numTribes ?? 0;
  currentMaxTicks = overrides.maxTicks ?? MAX_EPISODE_TICKS;
  const spawnBuffer = overrides.spawnBuffer ?? SPAWN_PHASE_BUFFER;

  const { gameMap, miniGameMap } = await loadMap(mapName);

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
    bots: numTribes,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: true,
    algoBots: numAlgoBots,
  };

  const config = makeConfig(gameConfig);
  game = createGame([agentInfo], [], gameMap, miniGameMap, config);

  // Spawn AlgoBots — full Human stats + nation-style attack/alliance/structure AI
  const random = new PseudoRandom(simpleHash(GAME_ID) + 3);
  for (let i = 0; i < numAlgoBots; i++) {
    const botInfo = new PlayerInfo(
      `AlgoBot${i + 1}`,
      PlayerType.Human,
      null,
      random.nextID(),
    );
    game.addExecution(new AlgoBotExecution(GAME_ID, botInfo));
  }

  // Spawn tribe bots (PlayerType.Bot, weak stats, simple AI auto-attached on spawn)
  if (numTribes > 0) {
    const tribeSpawner = new TribeSpawner(game, GAME_ID);
    game.addExecution(...tribeSpawner.spawnTribes(numTribes));
  }

  game.addExecution(new WinCheckExecution());

  // Spawn agent
  game.addExecution(new SpawnExecution(GAME_ID, agentInfo));

  ticks = 0;
  prevTiles = 0;

  // Run through spawn phase buffer before first observation
  for (let t = 0; t < spawnBuffer; t++) {
    game.executeNextTick();
    ticks++;
  }

  agentPlayer = game.player(agentID);
  prevTiles = agentPlayer.numTilesOwned();

  // Build stable neighbor ordering (slot k bound to specific opponent for the whole episode)
  neighborOrder = buildStableNeighborOrder(game, agentPlayer);
  aliveOpponentIDs = new Set(
    neighborOrder.filter((id) => game!.hasPlayer(id) && game!.player(id).isAlive()),
  );

  return buildResetResponse();
}

function buildResetResponse(): ResetResponse {
  const g = game!;
  const agent = agentPlayer!;
  const neighbors = getNeighbors(g, neighborOrder);
  const vec = Array.from(
    extractObs(g, agent, g.numLandTiles(), neighbors, ticks, prevTiles),
  );
  const map = Array.from(extractSpatialObs(g, agent));
  const mask = computeActionMask(agent, neighbors);
  return { vec, map, mask };
}

function stepEpisode(action: number): StepResponse {
  const g = game!;
  const agent = agentPlayer!;

  // Apply action — use the stable per-episode neighbor order
  const executions = decodeAction(
    action,
    g,
    agent,
    getNeighbors(g, neighborOrder),
  );
  if (executions.length > 0) {
    g.addExecution(...executions);
  }

  // Advance DECISION_INTERVAL ticks
  let done = false;
  for (let t = 0; t < DECISION_INTERVAL; t++) {
    g.executeNextTick();
    ticks++;
    if (g.getWinner() !== null || ticks >= currentMaxTicks) {
      done = true;
      break;
    }
  }

  // Compute reward
  let reward = REWARD_TICK * DECISION_INTERVAL;

  // Kill detection — opponents that were alive last step but aren't now
  let kills = 0;
  const stillAlive = new Set<string>();
  for (const id of aliveOpponentIDs) {
    if (g.hasPlayer(id) && g.player(id).isAlive()) {
      stillAlive.add(id);
    } else {
      kills++;
    }
  }
  aliveOpponentIDs = stillAlive;
  reward += kills * REWARD_KILL;

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

  const neighbors = getNeighbors(g, neighborOrder);
  const vec = Array.from(
    extractObs(g, agent, g.numLandTiles(), neighbors, ticks, prevTiles),
  );
  const map = Array.from(extractSpatialObs(g, agent));
  const mask = computeActionMask(agent, neighbors);

  return { vec, map, mask, reward, done, info: { ticks, kills } };
}

/**
 * Encode the static terrain of the current map as a base64 byte string.
 * Each byte holds one tile in the order y*width + x:
 *   bits 0..2  → TerrainType (0=Plains, 1=Highland, 2=Mountain, 3=Lake, 4=Ocean)
 *   bit  3     → isShoreline
 */
function getTerrain(): {
  width: number;
  height: number;
  terrain_b64: string;
} {
  const g = game!;
  const w = g.width();
  const h = g.height();
  const buf = Buffer.alloc(w * h);
  let i = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ref = g.ref(x, y);
      const t = g.terrainType(ref) & 0x07;
      const shore = g.isShoreline(ref) ? 0x08 : 0;
      buf[i++] = t | shore;
    }
  }
  return { width: w, height: h, terrain_b64: buf.toString("base64") };
}

function snapshotGame(): SnapshotResponse {
  const g = game!;
  const agent = agentPlayer!;

  // Use smallID() as a stable, permanent index — never changes when other
  // players die, so colours stay consistent across frames.
  const alivePlayers = g.players().filter((p) => p.isAlive());
  const players = alivePlayers.map((p) => ({
    id: p.id(),
    name: p.displayName(),
    isAgent: p.id() === agent.id(),
    idx: p.smallID(),
  }));

  const owned: number[] = [];
  g.forEachTile((ref) => {
    if (!g.isLand(ref)) return;
    const o = g.owner(ref);
    if (o.isPlayer()) {
      owned.push(ref, (o as Player).smallID());
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

// ── Demo episode (record AlgoBot trajectories for behavioral cloning) ──────

/**
 * Run one episode with N AlgoBots only (no RL agent), and stream
 * (obs, mask, action) samples to stdout — one per line — for every
 * AlgoBot decision (attack/ally/break_ally/build/expand) the bot makes.
 *
 * Output format (newline-delimited JSON):
 *   {"sample": {"vec": [...], "mask": [...], "action": <int>, "player_id": "..."}}
 *   {"sample": ...}
 *   ...
 *   {"done": true, "n_samples": <int>, "ticks": <int>}
 */
async function runDemoEpisode(numBots: number): Promise<void> {
  const { gameMap, miniGameMap } = await loadMap(TRAINING_MAP);

  const gameConfig: GameConfig = {
    gameMap: GameMapType.World,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: true,
    algoBots: numBots,
  };

  const config = makeConfig(gameConfig);
  const localGame = createGame([], [], gameMap, miniGameMap, config);

  // Use a time-varying seed so each demo episode is different
  const random = new PseudoRandom(simpleHash(GAME_ID + Date.now()) + 3);
  const botInfos: PlayerInfo[] = [];
  for (let i = 0; i < numBots; i++) {
    const info = new PlayerInfo(
      `Bot${i + 1}`,
      PlayerType.Human,
      null,
      random.nextID(),
    );
    botInfos.push(info);
    localGame.addExecution(new AlgoBotExecution(GAME_ID, info));
  }
  localGame.addExecution(new WinCheckExecution());

  // Run spawn phase
  let demoTicks = 0;
  for (let t = 0; t < SPAWN_PHASE_BUFFER; t++) {
    localGame.executeNextTick();
    demoTicks++;
  }

  // After spawn phase, build each bot's stable neighbor order
  const neighborOrders = new Map<string, string[]>();
  const prevTilesByBot = new Map<string, number>();
  for (const info of botInfos) {
    if (!localGame.hasPlayer(info.id)) continue;
    const p = localGame.player(info.id);
    if (!p.isAlive()) continue;
    neighborOrders.set(info.id, buildStableNeighborOrder(localGame, p));
    prevTilesByBot.set(info.id, p.numTilesOwned());
  }

  let nSamples = 0;

  // Hook addExecution: when a tracked bot adds a recognized execution,
  // snapshot its current obs and emit a sample.
  const unhook = attachDemoHook(localGame, neighborOrders, (playerID, action) => {
    if (!localGame.hasPlayer(playerID)) return;
    const player = localGame.player(playerID);
    if (!player.isAlive()) return;
    const order = neighborOrders.get(playerID);
    if (!order) return;

    const neighbors = getNeighbors(localGame, order);
    const prevTiles = prevTilesByBot.get(playerID) ?? player.numTilesOwned();
    const vec = Array.from(
      extractObs(
        localGame,
        player,
        localGame.numLandTiles(),
        neighbors,
        demoTicks,
        prevTiles,
      ),
    );
    const mask = computeActionMask(player, neighbors);

    process.stdout.write(
      JSON.stringify({
        sample: { vec, mask, action, player_id: playerID },
      }) + "\n",
    );
    nSamples++;
  });

  // Run game
  while (
    demoTicks < MAX_EPISODE_TICKS &&
    localGame.getWinner() === null
  ) {
    localGame.executeNextTick();
    demoTicks++;

    // Update prevTiles per bot every DECISION_INTERVAL ticks
    if (demoTicks % DECISION_INTERVAL === 0) {
      for (const info of botInfos) {
        if (!localGame.hasPlayer(info.id)) continue;
        const p = localGame.player(info.id);
        if (p.isAlive()) {
          prevTilesByBot.set(info.id, p.numTilesOwned());
        }
      }
    }
  }

  unhook();

  process.stdout.write(
    JSON.stringify({ done: true, n_samples: nSamples, ticks: demoTicks }) + "\n",
  );
}

// ── Main loop ────────────────────────────────────────────────────────────────

async function main() {
  const rl = readline.createInterface({ input: process.stdin });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    let msg: {
      cmd: string;
      action?: number;
      actions?: number[];
      num_bots?: number;
      // Optional reset overrides
      map?: string;
      num_tribes?: number;
      num_algobots?: number;
      max_ticks?: number;
      spawn_buffer?: number;
    };
    try {
      msg = JSON.parse(trimmed);
    } catch {
      process.stdout.write(
        JSON.stringify({ error: "invalid json" }) + "\n",
      );
      continue;
    }

    if (msg.cmd === "reset") {
      const resp = await resetEpisode({
        map: msg.map,
        numTribes: msg.num_tribes,
        numAlgoBots: msg.num_algobots,
        maxTicks: msg.max_ticks,
        spawnBuffer: msg.spawn_buffer,
      });
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
    } else if (msg.cmd === "demo_episode") {
      const numBots = msg.num_bots ?? 4;
      await runDemoEpisode(numBots);
    } else if (msg.cmd === "snapshot") {
      if (game === null || agentPlayer === null) {
        process.stdout.write(
          JSON.stringify({ error: "call reset first" }) + "\n",
        );
        continue;
      }
      process.stdout.write(JSON.stringify(snapshotGame()) + "\n");
    } else if (msg.cmd === "get_terrain") {
      if (game === null) {
        process.stdout.write(
          JSON.stringify({ error: "call reset first" }) + "\n",
        );
        continue;
      }
      process.stdout.write(JSON.stringify(getTerrain()) + "\n");
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
