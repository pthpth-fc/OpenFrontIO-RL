"""
Train an MLP via behavioral cloning on AlgoBot demonstrations.

The architecture matches MaskablePPO's MlpPolicy default:
    Linear(obs_size, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, action_size)

After training, weights can be copied into a MaskablePPO policy as a warm-start.

Usage:
    python rl/bc_train.py [demos.jsonl] [out.pt] [epochs]
"""

from __future__ import annotations

import json
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

OBS_SIZE = 66
ACTION_SIZE = 28
BATCH = 256
LR = 1e-3

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ── Architecture (matches MaskablePPO MlpPolicy default) ─────────────────────


class BCPolicy(nn.Module):
    """MLP policy net matching MaskablePPO's default MlpPolicy architecture.

    Layer naming follows what stable-baselines3 expects on the policy side:
        mlp_extractor.policy_net.0  (Linear obs→64)
        mlp_extractor.policy_net.2  (Linear 64→64)
        action_net                  (Linear 64→action_size)
    Activations are Tanh between layers.
    """

    def __init__(self, obs_size: int = OBS_SIZE, action_size: int = ACTION_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.action_head(h)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_demos(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    X = np.array([s["vec"] for s in samples], dtype=np.float32)
    M = np.array([s["mask"] for s in samples], dtype=bool)
    Y = np.array([s["action"] for s in samples], dtype=np.int64)
    return X, M, Y


# ── Training ─────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Loading demos from {DEMOS_PATH}")
    X, M, Y = load_demos(DEMOS_PATH)
    n = len(Y)
    print(f"  {n} samples, obs_dim={X.shape[1]}, action_dim={M.shape[1]}")
    print(f"  device: {DEVICE}")

    # Action distribution
    counter = Counter(Y.tolist())
    print("\nAction distribution in demos:")
    for a, c in sorted(counter.items()):
        print(f"  action {a:2d}: {c:5d}  {100*c/n:5.1f}%")

    # Train/val split (random)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(n // 10, 1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    Xt = torch.from_numpy(X)
    Mt = torch.from_numpy(M)
    Yt = torch.from_numpy(Y)
    Xtr = Xt[train_idx].to(DEVICE)
    Mtr = Mt[train_idx].to(DEVICE)
    Ytr = Yt[train_idx].to(DEVICE)
    Xv = Xt[val_idx].to(DEVICE)
    Mv = Mt[val_idx].to(DEVICE)
    Yv = Yt[val_idx].to(DEVICE)

    model = BCPolicy().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {EPOCHS} epochs (batch={BATCH}, lr={LR})")
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    best_val_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm_tr = torch.randperm(len(Xtr), device=DEVICE)
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for i in range(0, len(Xtr), BATCH):
            idx = perm_tr[i : i + BATCH]
            x = Xtr[idx]
            m = Mtr[idx]
            y = Ytr[idx]

            logits = model(x)
            # Mask out invalid actions: cross-entropy over masked logits
            masked = logits.masked_fill(~m, -1e9)
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
            logits_v = model(Xv).masked_fill(~Mv, -1e9)
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
