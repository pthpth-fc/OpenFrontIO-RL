import { Execution, Game, Player, PlayerInfo } from "../game/Game";
import { PseudoRandom } from "../PseudoRandom";
import { GameID } from "../Schemas";
import { simpleHash } from "../Util";
import { NationAllianceBehavior } from "./nation/NationAllianceBehavior";
import { NationEmojiBehavior } from "./nation/NationEmojiBehavior";
import { NationStructureBehavior } from "./nation/NationStructureBehavior";
import { SpawnExecution } from "./SpawnExecution";
import { AiAttackBehavior } from "./utils/AiAttackBehavior";

// In-process algorithmic bot. Runs with human-level stats and uses the same
// attack/alliance/structure behaviors as NationExecution, but spawns randomly
// like a tribe. Inject via GameRunner.init() by setting algoBots > 0 in GameConfig.
export class AlgoBotExecution implements Execution {
  private active = true;
  private mg!: Game;
  private player: Player | null = null;
  private behaviorsInitialized = false;
  private attackBehavior!: AiAttackBehavior;
  private allianceBehavior!: NationAllianceBehavior;
  private structureBehavior!: NationStructureBehavior;
  private emojiBehavior!: NationEmojiBehavior;
  private spawnScheduled = false;

  private readonly random: PseudoRandom;
  private readonly attackRate: number;
  private readonly attackTick: number;
  private readonly triggerRatio: number;
  private readonly reserveRatio: number;
  private readonly expandRatio: number;

  constructor(
    private readonly gameID: GameID,
    private readonly playerInfo: PlayerInfo,
  ) {
    this.random = new PseudoRandom(
      simpleHash(playerInfo.id) + simpleHash(gameID),
    );
    this.attackRate = this.random.nextInt(40, 60);
    this.attackTick = this.random.nextInt(0, this.attackRate);
    this.triggerRatio = this.random.nextInt(50, 65) / 100;
    this.reserveRatio = this.random.nextInt(25, 35) / 100;
    this.expandRatio = this.random.nextInt(15, 25) / 100;
  }

  activeDuringSpawnPhase(): boolean {
    return true;
  }

  init(mg: Game) {
    this.mg = mg;
    if (!mg.hasPlayer(this.playerInfo.id)) {
      this.player = mg.addPlayer(this.playerInfo);
    } else {
      this.player = mg.player(this.playerInfo.id);
    }
  }

  tick(ticks: number) {
    if (!this.player) return;

    if (this.mg.inSpawnPhase()) {
      if (!this.spawnScheduled) {
        this.mg.addExecution(new SpawnExecution(this.gameID, this.playerInfo));
        this.spawnScheduled = true;
      }
      return;
    }

    if (!this.player.isAlive()) {
      this.active = false;
      return;
    }

    if (!this.player.hasSpawned()) return;

    if (ticks % this.attackRate !== this.attackTick) {
      if (this.behaviorsInitialized && this.player.isAlive()) {
        const offset = ticks % this.attackRate;
        const oneThird =
          (this.attackTick + Math.floor(this.attackRate / 3)) %
          this.attackRate;
        const twoThirds =
          (this.attackTick + Math.floor((this.attackRate * 2) / 3)) %
          this.attackRate;
        if (offset === oneThird || offset === twoThirds) {
          this.structureBehavior.handleStructures();
        }
      }
      return;
    }

    if (!this.behaviorsInitialized) {
      this.initBehaviors();
      this.attackBehavior.forceSendAttack(this.mg.terraNullius());
      return;
    }

    this.allianceBehavior.handleAllianceRequests();
    this.allianceBehavior.handleAllianceExtensionRequests();
    this.structureBehavior.handleStructures();
    this.attackBehavior.maybeAttack();
  }

  private initBehaviors() {
    if (!this.player) throw new Error("player not initialized");
    this.emojiBehavior = new NationEmojiBehavior(
      this.random,
      this.mg,
      this.player,
    );
    this.allianceBehavior = new NationAllianceBehavior(
      this.random,
      this.mg,
      this.player,
      this.emojiBehavior,
    );
    this.structureBehavior = new NationStructureBehavior(
      this.random,
      this.mg,
      this.player,
    );
    this.attackBehavior = new AiAttackBehavior(
      this.random,
      this.mg,
      this.player,
      this.triggerRatio,
      this.reserveRatio,
      this.expandRatio,
      this.allianceBehavior,
      this.emojiBehavior,
    );
    this.behaviorsInitialized = true;
  }

  isActive(): boolean {
    return this.active;
  }
}
