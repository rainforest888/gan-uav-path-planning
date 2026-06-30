"""Critic D(tau, c): WGAN-GP critic, outputs scalar score."""
import torch
import torch.nn as nn
from config import CONDITION_DIM, N_TRAJECTORY


class Critic(nn.Module):
    def __init__(self, dim_in=2, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        # 100 -> 50 -> 25 -> 13 -> 7 (after 4 stride-2 convs)
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, trajectory, condition):
        # trajectory: (B, N, dim) -> (B, dim, N) for Conv1d
        tau = trajectory.transpose(1, 2)
        f = self.conv(tau).flatten(1)
        x = torch.cat([f, condition], dim=1)
        return self.fc(x)  # no sigmoid (WGAN-GP)
