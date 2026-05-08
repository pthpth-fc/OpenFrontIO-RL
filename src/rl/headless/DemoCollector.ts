import { Execution, Game, Player, UnitType } from "../../core/game/Game";
import { AttackExecution } from "../../core/execution/AttackExecution";
import { AllianceRequestExecution } from "../../core/execution/alliance/AllianceRequestExecution";
import { BreakAllianceExecution } from "../../core/execution/alliance/BreakAllianceExecution";
import { ConstructionExecution } from "../../core/execution/ConstructionExecution";
import { K_NEIGHBORS } from "./RLConfig";

// Action layout (must match ActionDecoder.ts):
// 0=noop, 1..K=attack[k], K+1..2K=ally_req[k], 2K+1..3K=break_ally[k],
// 3K+1=city, 3K+2=defpost, 3K+3=expand
const ATTACK_BASE = 1;
const ALLY_REQ_BASE = 1 + K_NEIGHBORS;
const BREAK_ALLY_BASE = 1 + K_NEIGHBORS * 2;
const BUILD_CITY = 1 + K_NEIGHBORS * 3;
const BUILD_DEFPOST = 1 + K_NEIGHBORS * 3 + 1;
const EXPAND = 1 + K_NEIGHBORS * 3 + 2;

/**
 * Wraps `game.addExecution` so that whenever a tracked player adds an
 * AttackExecution / AllianceRequestExecution / BreakAllianceExecution /
 * ConstructionExecution, we receive a callback with the corresponding
 * RL action index (mapped via that player's stable neighbor ordering).
 *
 * Returns an unhook function that restores the original addExecution.
 */
export function attachDemoHook(
  game: Game,
  neighborOrders: Map<string, string[]>,
  onAction: (playerID: string, action: number) => void,
): () => void {
  const original = game.addExecution.bind(game);

  function classify(e: Execution): { playerID: string; action: number } | null {
    // AttackExecution: target=null → expand, target=Player → attack[k]
    if (e instanceof AttackExecution) {
      const owner = (e as unknown as { _owner: Player })._owner;
      const targetID = (e as unknown as { _targetID: string | null })._targetID;
      if (!owner) return null;
      const order = neighborOrders.get(owner.id());
      if (!order) return null;

      if (targetID === null) {
        return { playerID: owner.id(), action: EXPAND };
      }
      const k = order.indexOf(targetID);
      if (k >= 0 && k < K_NEIGHBORS) {
        return { playerID: owner.id(), action: ATTACK_BASE + k };
      }
      return null;
    }

    if (e instanceof AllianceRequestExecution) {
      const requestor = (e as unknown as { requestor: Player }).requestor;
      const recipientID = (e as unknown as { recipientID: string }).recipientID;
      if (!requestor) return null;
      const order = neighborOrders.get(requestor.id());
      if (!order) return null;
      const k = order.indexOf(recipientID);
      if (k >= 0 && k < K_NEIGHBORS) {
        return { playerID: requestor.id(), action: ALLY_REQ_BASE + k };
      }
      return null;
    }

    if (e instanceof BreakAllianceExecution) {
      const requestor = (e as unknown as { requestor: Player }).requestor;
      const recipientID = (e as unknown as { recipientID: string }).recipientID;
      if (!requestor) return null;
      const order = neighborOrders.get(requestor.id());
      if (!order) return null;
      const k = order.indexOf(recipientID);
      if (k >= 0 && k < K_NEIGHBORS) {
        return { playerID: requestor.id(), action: BREAK_ALLY_BASE + k };
      }
      return null;
    }

    if (e instanceof ConstructionExecution) {
      const player = (e as unknown as { player: Player }).player;
      const constructionType = (
        e as unknown as { constructionType: UnitType }
      ).constructionType;
      if (!player) return null;
      if (!neighborOrders.has(player.id())) return null;
      if (constructionType === UnitType.City) {
        return { playerID: player.id(), action: BUILD_CITY };
      }
      if (constructionType === UnitType.DefensePost) {
        return { playerID: player.id(), action: BUILD_DEFPOST };
      }
      return null;
    }

    return null;
  }

  (game as unknown as { addExecution: (...e: Execution[]) => void }).addExecution =
    function (...execs: Execution[]) {
      for (const e of execs) {
        try {
          const classified = classify(e);
          if (classified) {
            onAction(classified.playerID, classified.action);
          }
        } catch (_) {
          // Skip — introspection failed for this execution type
        }
      }
      return original(...execs);
    };

  return () => {
    (game as unknown as { addExecution: (...e: Execution[]) => void }).addExecution =
      original;
  };
}
