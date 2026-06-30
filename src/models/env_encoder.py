"""Environment encoder E_env: 3D voxels -> condition vector."""
import torch
import torch.nn as nn
from config import VOXEL_RES, CONDITION_DIM


class EnvEncoder2D(nn.Module):
    """2D variant for Phase 1 validation. Processes (1, H, W) voxel grid."""

    def __init__(self, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        # After 4 stride-2 convs: 32 -> 16 -> 8 -> 4 -> 2
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 + 4, cond_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, voxels, start, goal):
        f = self.conv(voxels).flatten(1)  # (B, 256*2*2)
        sg = torch.cat([start, goal], dim=1) * 2 - 1  # normalize to [-1,1]
        return self.fc(torch.cat([f, sg], dim=1))


class EnvEncoder3D(nn.Module):
    """3D variant for Phase 2+. Processes (1, D, H, W) voxel grid."""

    def __init__(self, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
        )
        # 32 -> 16 -> 8 -> 4 -> 2
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 * 2 + 6, cond_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, voxels, start, goal):
        f = self.conv(voxels).flatten(1)
        sg = torch.cat([start, goal], dim=1) * 2 - 1
        return self.fc(torch.cat([f, sg], dim=1))
