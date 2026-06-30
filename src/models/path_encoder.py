"""Path encoder E_path(tau, c): trajectory -> latent code."""
import torch
import torch.nn as nn
from config import LATENT_DIM, CONDITION_DIM


class PathEncoder2D(nn.Module):
    def __init__(self, dim_in=2, latent_dim=LATENT_DIM, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        # 100 -> 50 -> 25 -> 13 -> 7
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, trajectory, condition):
        tau = trajectory.transpose(1, 2)  # (B, dim, N)
        f = self.conv(tau).flatten(1)
        return self.fc(torch.cat([f, condition], dim=1))


class PathEncoder3D(nn.Module):
    """3D variant for Phase 2+."""

    def __init__(self, dim_in=3, latent_dim=LATENT_DIM, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, trajectory, condition):
        tau = trajectory.transpose(1, 2)
        f = self.conv(tau).flatten(1)
        return self.fc(torch.cat([f, condition], dim=1))
