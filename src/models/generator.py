"""Generator G(z, c): latent vector + condition -> key waypoints."""
import torch
import torch.nn as nn
from config import LATENT_DIM, CONDITION_DIM, K_WAYPOINTS


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, cond_dim=CONDITION_DIM,
                 k_waypoints=K_WAYPOINTS, dim_out=2):
        super().__init__()
        self.k = k_waypoints
        self.dim_out = dim_out
        input_dim = latent_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(0.2),
            nn.Linear(2048, k_waypoints * dim_out),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        out = self.net(x)
        out = out.view(-1, self.k, self.dim_out)
        out = torch.tanh(out)  # [-1, 1], denormalize later
        return out
