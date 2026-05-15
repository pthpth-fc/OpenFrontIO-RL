"""
Train a CNN+MLP policy via behavioral cloning on AlgoBot demonstrations.

Architecture mirrors `cnn_policy.OpenFrontExtractor` + a MaskablePPO
MultiInputPolicy actor head (256 → 128 → action_size with Tanh) so weights
transfer cleanly into PPO at warm-start time.

Inputs (per sample, in JSONL):
    {"vec": [81 floats], "map": [5*32*32 = 5120 floats], "mask": [46 bools],
     "action": <int>, "player_id": "..."}

Output: bc_model.pt — state_dict with keys matching what train.py copies
into MaskablePPO's MultiInputPolicy.

Usage:
    python rl/bc_train.py [demos.jsonl] [out.pt] [epochs]
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEMOS_PATH = sys.argv[1] if len(sys.argv) > 1 else "rl/demos.jsonl"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "rl/bc_model.pt"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

# Must match RLConfig.ts / env.py
K_NEIGHBORS = 10
OBS_SIZE = 11 + K_NEIGHBORS * 7         # 81
ACTION_SIZE = 1 + K_NEIGHBORS * 4 + 5   # 46
PATCH_SIZE = 32
PATCH_CHANNELS = 5

BATCH = 256
LR = 1e-3

# Match cnn_policy.OpenFrontExtractor sizes
CNN_OUT = 256
MLP_OUT = 128
COMBINED = CNN_OUT + MLP_OUT             # 384


def _select_device() -> str:
    forced = os.environ.get("TRAIN_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _select_device()


# ── Architecture ─────────────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    """CNN+MLP feature extractor + Tanh-MLP actor head, matching how
    MaskablePPO's MultiInputPolicy will use the weights when warm-started.

    Layer naming follows what stable-baselines3 expects on the policy side:
        features_extractor.cnn.0/2/4              (Conv2d layers)
        features_extractor.cnn_head.0             (Linear cnn_flat -> 256)
        features_extractor.vec_head.0             (Linear 81 -> 128)
        mlp_extractor.policy_net.0                (Linear 384 -> 256)
        mlp_extractor.policy_net.2                (Linear 256 -> 128)
        action_net                                (Linear 128 -> 46)
    """

    def __init__(self) -> None:
        super().__init__()
        # CNN branch — same as cnn_policy.OpenFrontExtractor
        self.cnn = nn.Sequential(
            nn.Conv2d(PATCH_CHANNELS, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE)
            cnn_flat = self.cnn(dummy).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(cnn_flat, CNN_OUT), nn.ReLU())

        # MLP branch
        self.vec_head = nn.Sequential(nn.Linear(OBS_SIZE, MLP_OUT), nn.ReLU())

        # Actor head (MaskablePPO MlpExtractor with net_arch=[256, 128] + Tanh)
        self.policy_net_0 = nn.Linear(COMBINED, 256)
        self.policy_net_2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, ACTION_SIZE)

    def forward(self, vec: torch.Tensor, map_: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_head(self.cnn(map_))
        vec_feat = self.vec_head(vec)
        h = torch.cat([cnn_feat, vec_feat], dim=1)
        h = torch.tanh(self.policy_net_0(h))
        h = torch.tanh(self.policy_net_2(h))
        return self.action_head(h)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_demos(path: str):
    """Load demo samples, filtering out any whose action is masked-out by
    its own recorded mask. AlgoBots sometimes take actions our mask
    considers invalid (e.g., a "bordering" check that disagrees with the
    engine's actual adjacency definition); training on those produces an
    exploding masked-cross-entropy loss.
    """
    raw_samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_samples.append(json.loads(line))

    # Filter to "consistent" samples — mask[action] must be True.
    samples = [s for s in raw_samples if s["mask"][s["action"]]]
    n_raw = len(raw_samples)
    n = len(samples)
    if n < n_raw:
        print(f"  Filtered {n_raw - n} / {n_raw} samples whose action is mask-invalid")

    vecs = np.empty((n, OBS_SIZE), dtype=np.float32)
    maps = np.empty((n, PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    masks = np.empty((n, ACTION_SIZE), dtype=bool)
    actions = np.empty((n,), dtype=np.int64)

    for i, s in enumerate(samples):
        vecs[i] = s["vec"]
        maps[i] = np.asarray(s["map"], dtype=np.float32).reshape(
            PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE
        )
        masks[i] = s["mask"]
        actions[i] = s["action"]

    return vecs, maps, masks, actions


# ── Training ─────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Loading demos from {DEMOS_PATH}")
    vecs, maps, masks, actions = load_demos(DEMOS_PATH)
    n = len(actions)
    print(
        f"  {n} samples, vec={vecs.shape[1]}, map={maps.shape[1:]}, "
        f"action_dim={masks.shape[1]}"
    )
    print(f"  device: {DEVICE}")

    # Action distribution
    counter = Counter(actions.tolist())
    print("\nAction distribution in demos:")
    for a, c in sorted(counter.items()):
        print(f"  action {a:2d}: {c:5d}  {100*c/n:5.1f}%")

    # Train/val split
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(n // 10, 1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    Vt = torch.from_numpy(vecs)
    Mp = torch.from_numpy(maps)
    Mk = torch.from_numpy(masks)
    Yt = torch.from_numpy(actions)

    Vtr = Vt[train_idx].to(DEVICE)
    Mptr = Mp[train_idx].to(DEVICE)
    Mktr = Mk[train_idx].to(DEVICE)
    Ytr = Yt[train_idx].to(DEVICE)
    Vv = Vt[val_idx].to(DEVICE)
    Mpv = Mp[val_idx].to(DEVICE)
    Mkv = Mk[val_idx].to(DEVICE)
    Yv = Yt[val_idx].to(DEVICE)

    model = BCPolicy().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {EPOCHS} epochs (batch={BATCH}, lr={LR})")
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    best_val_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm_tr = torch.randperm(len(Vtr), device=DEVICE)
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for i in range(0, len(Vtr), BATCH):
            idx = perm_tr[i : i + BATCH]
            v = Vtr[idx]
            m = Mptr[idx]
            mk = Mktr[idx]
            y = Ytr[idx]

            logits = model(v, m)
            masked = logits.masked_fill(~mk, -1e9)
            loss = F.cross_entropy(masked, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(y)
            total_correct += int((masked.argmax(1) == y).sum().item())
            total_n += len(y)

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        model.eval()
        with torch.no_grad():
            logits_v = model(Vv, Mpv).masked_fill(~Mkv, -1e9)
            val_loss = float(F.cross_entropy(logits_v, Yv).item())
            val_acc = float((logits_v.argmax(1) == Yv).float().mean().item())

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_PATH)
            marker = "  ✓ best"
        print(
            f"  ep {ep:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}"
        )

    print(f"\n✓ Best val_acc={best_val_acc:.3f}, saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
