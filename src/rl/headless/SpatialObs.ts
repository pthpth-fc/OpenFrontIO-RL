import { Game, Player } from "../../core/game/Game";
import { PATCH_CHANNELS, PATCH_SIZE } from "./RLConfig";

const HALF = PATCH_SIZE >> 1;

/**
 * Extract a PATCH_SIZE×PATCH_SIZE spatial observation centered on the agent's
 * territory centroid, in channels-first (CHW) order for PyTorch.
 *
 * Channels:
 *   0 – land mask       (1 = land, 0 = ocean)
 *   1 – agent tiles     (1 = owned by agent)
 *   2 – enemy tiles     (1 = owned by a non-allied player)
 *   3 – ally tiles      (1 = owned by an allied player)
 */
export function extractSpatialObs(game: Game, agent: Player): Float32Array {
  const mapW = game.width();
  const mapH = game.height();
  const out = new Float32Array(PATCH_CHANNELS * PATCH_SIZE * PATCH_SIZE);

  // ── Compute centroid of agent's tiles ────────────────────────────────────
  let sumX = 0;
  let sumY = 0;
  let count = 0;
  for (const ref of agent.tiles()) {
    sumX += game.x(ref);
    sumY += game.y(ref);
    count++;
  }

  // If agent has no tiles yet (just spawned), center on map
  const cx = count > 0 ? Math.round(sumX / count) : mapW >> 1;
  const cy = count > 0 ? Math.round(sumY / count) : mapH >> 1;

  // Top-left corner of the patch (clamped to map bounds)
  const x0 = Math.max(0, Math.min(mapW - PATCH_SIZE, cx - HALF));
  const y0 = Math.max(0, Math.min(mapH - PATCH_SIZE, cy - HALF));

  // Build a lookup: tileRef → channel value (2=enemy,3=ally) for quick fill
  // We only need enemy/ally for alive players other than agent
  const allyIDs = new Set<string>(
    agent.alliances().map((a) => {
      const other =
        a.requestor().id() === agent.id() ? a.recipient() : a.requestor();
      return other.id();
    }),
  );

  // ── Fill channels ─────────────────────────────────────────────────────────
  const P = PATCH_SIZE;
  const C = PATCH_CHANNELS;

  for (let dy = 0; dy < P; dy++) {
    const gy = y0 + dy;
    if (gy >= mapH) break;

    for (let dx = 0; dx < P; dx++) {
      const gx = x0 + dx;
      if (gx >= mapW) break;

      const ref = game.ref(gx, gy);
      const pixelBase = dy * P + dx;

      // Ch0: land
      const isLand = game.isLand(ref) ? 1 : 0;
      out[0 * P * P + pixelBase] = isLand;

      if (!isLand) continue;

      const owner = game.owner(ref);
      if (!owner.isPlayer()) continue; // unclaimed land — channels stay 0

      const ownerID = (owner as Player).id();

      if (ownerID === agent.id()) {
        // Ch1: self
        out[1 * P * P + pixelBase] = 1;
      } else if (allyIDs.has(ownerID)) {
        // Ch3: ally
        out[3 * P * P + pixelBase] = 1;
      } else {
        // Ch2: enemy
        out[2 * P * P + pixelBase] = 1;
      }
    }
  }

  return out;
}
