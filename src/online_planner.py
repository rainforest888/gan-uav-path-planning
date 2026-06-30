"""Online planner with latent space gradient optimization for dynamic replanning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    LATENT_DIM, CONDITION_DIM, K_WAYPOINTS, N_TRAJECTORY,
    LAMBDA_CONTINUITY, ONLINE_LR, ONLINE_STEPS,
)
from models.env_encoder import EnvEncoder2D, EnvEncoder3D
from models.generator import Generator
from models.path_encoder import PathEncoder2D, PathEncoder3D
from interpolation import cubic_spline_trajectory, minimum_snap_trajectory
from losses import collision_loss


class OnlinePlanner:
    """Latent-space continuous evolution for dynamic path replanning.

    Instead of discarding and regenerating paths, this continuously
    moves the latent code in the latent space, producing smoothly
    deforming trajectories.
    """

    def __init__(self, dim=2, checkpoint_path=None, device='cuda'):
        self.dim = dim
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else device
        self._build_models()
        if checkpoint_path:
            self.load(checkpoint_path)

    def _build_models(self):
        if self.dim == 2:
            self.E_env = EnvEncoder2D().to(self.device).eval()
            self.G = Generator(dim_out=2).to(self.device).eval()
            self.E_path = PathEncoder2D(dim_in=2).to(self.device).eval()
        else:
            self.E_env = EnvEncoder3D().to(self.device).eval()
            self.G = Generator(dim_out=3).to(self.device).eval()
            self.E_path = PathEncoder3D(dim_in=3).to(self.device).eval()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.E_env.load_state_dict(ckpt['E_env'])
        self.G.load_state_dict(ckpt['G'])
        self.E_path.load_state_dict(ckpt['E_path'])

    def initialize(self, init_path=None, condition=None):
        """Initialize latent code z_0.

        Args:
            init_path: optional (1, N, dim) initial path for E_path encoding
            condition: (1, CONDITION_DIM) environment condition
        Returns:
            z: (1, LATENT_DIM) initial latent code
        """
        if init_path is not None and condition is not None:
            with torch.no_grad():
                z = self.E_path(init_path.to(self.device), condition.to(self.device))
        else:
            z = torch.randn(1, LATENT_DIM, device=self.device)
        return z

    def replan_step(self, z_prev, condition, voxels):
        """One replanning cycle: optimize latent code, return new trajectory.

        Args:
            z_prev:   (1, LATENT_DIM) previous latent code
            condition: (1, CONDITION_DIM) current environment condition
            voxels:   (1, 1|C, H, W) or (1, 1|C, D, H, W) current voxel map
        Returns:
            z_new:     (1, LATENT_DIM) updated latent code
            trajectory: (N, dim) numpy array, final smoothed trajectory
        """
        z_prev = z_prev.to(self.device)
        z_cur = z_prev.clone().detach()
        z_cur.requires_grad_(True)

        mode = '2d' if self.dim == 2 else '3d'
        condition = condition.to(self.device)
        voxels = voxels.to(self.device)

        optimizer = torch.optim.SGD([z_cur], lr=ONLINE_LR)

        for _ in range(ONLINE_STEPS):
            optimizer.zero_grad()
            waypoints = self.G(z_cur, condition)  # (1, K, dim) in [0,1]
            trajectory = cubic_spline_trajectory(waypoints.squeeze(0),
                                                 N_TRAJECTORY).unsqueeze(0)

            # Collision avoidance
            col = collision_loss(trajectory, voxels, mode=mode)

            # Continuity: stay close to previous latent code
            continuity = ((z_cur - z_prev) ** 2).sum()

            # Goal attraction: encourage endpoint near goal
            # Goal is encoded in condition; we extract from E_env conv output
            # Instead, use trajectory endpoint distance from last segment
            traj_len = (trajectory[:, 1:, :] -
                        trajectory[:, :-1, :]).norm(2, dim=-1).sum()

            loss = col + LAMBDA_CONTINUITY * continuity + 0.01 * traj_len
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            waypoints = self.G(z_cur, condition)
            traj_np = minimum_snap_trajectory(waypoints.squeeze(0), N_TRAJECTORY)

        return z_cur.detach(), traj_np

    def generate_path(self, z, condition):
        """Generate a single path from latent code (no optimization)."""
        with torch.no_grad():
            waypoints = self.G(z.to(self.device), condition.to(self.device))
            traj_np = minimum_snap_trajectory(waypoints.squeeze(0), N_TRAJECTORY)
        return traj_np
