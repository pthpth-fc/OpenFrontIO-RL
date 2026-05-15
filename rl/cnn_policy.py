"""
Custom feature extractor combining:
  - Small CNN for the spatial map patch (5 × 32 × 32)
  - MLP for the flat feature vector (81-dim)

Used as policy_kwargs["features_extractor_class"] in MaskablePPO.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Must match env.py constants
PATCH_CHANNELS = 5
PATCH_SIZE = 32
VEC_SIZE = 81

CNN_OUT = 256   # flattened CNN output features
MLP_OUT = 128   # flat-vec embedding size
COMBINED = CNN_OUT + MLP_OUT  # total features fed to actor/critic MLP


class OpenFrontExtractor(BaseFeaturesExtractor):
    """
    CNN over map patch + MLP over flat vec, concatenated.

    Input:  obs dict {"vec": (B, 62), "map": (B, 4, 32, 32)}
    Output: (B, COMBINED)
    """

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=COMBINED)

        # ── CNN branch ───────────────────────────────────────────────────────
        # 4 × 32 × 32
        # Conv1: 4  → 32, k=3 → 30×30
        # Conv2: 32 → 64, k=3 → 28×28
        # Conv3: 64 → 64, k=3, stride=2 → 13×13
        # Flatten → 64 × 13 × 13 = 10816 → Linear → 256
        self.cnn = nn.Sequential(
            nn.Conv2d(PATCH_CHANNELS, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE)
            cnn_flat = self.cnn(dummy).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_flat, CNN_OUT),
            nn.ReLU(),
        )

        # ── MLP branch ───────────────────────────────────────────────────────
        self.vec_head = nn.Sequential(
            nn.Linear(VEC_SIZE, MLP_OUT),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        map_feat = self.cnn_head(self.cnn(obs["map"]))
        vec_feat = self.vec_head(obs["vec"])
        return torch.cat([map_feat, vec_feat], dim=1)
