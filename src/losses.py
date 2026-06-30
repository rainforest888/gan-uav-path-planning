"""Loss functions for CWGAN-GP + path constraints."""
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from config import LAMBDA_GP, LAMBDA_COL, LAMBDA_LEN, LAMBDA_SMOOTH, LAMBDA_RECON, LAMBDA_CONVEXITY


def critic_loss(D, tau_real, tau_fake, c):
    """WGAN-GP critic loss (to minimize).
    L_D = E[D(G(z))] - E[D(real)] + lambda * GP
    """
    d_real = D(tau_real, c)
    d_fake = D(tau_fake, c)
    return d_fake.mean() - d_real.mean()


def generator_adv_loss(D, tau_fake, c):
    """WGAN-GP generator adversarial loss (to minimize).
    L_adv_G = -E[D(G(z))]
    """
    return -D(tau_fake, c).mean()


def gradient_penalty(D, tau_real, tau_fake, c):
    """WGAN-GP gradient penalty."""
    B = tau_real.shape[0]
    eps = torch.rand(B, 1, 1, device=tau_real.device)
    x_hat = eps * tau_real + (1 - eps) * tau_fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat, c)
    grads = autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grads.reshape(B, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def collision_loss(trajectory, voxels, mode='2d'):
    """Soft collision loss via grid sampling.
    Args:
        trajectory: (B, N, dim) in [0,1] normalized coords
        voxels:     (B, 1, H, W) or (B, 1, D, H, W)
        mode:       '2d' or '3d'
    """
    B, N, dim = trajectory.shape
    if mode == '2d':
        grid = trajectory[:, :, :2] * 2 - 1  # (B, N, 2) in [-1,1]
        grid = grid.view(B, N, 1, 2)
        sampled = F.grid_sample(voxels.float(), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    else:
        grid = trajectory * 2 - 1
        grid = grid.view(B, N, 1, 1, 3)
        sampled = F.grid_sample(voxels.float(), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    return sampled.mean()


def path_length_loss(trajectory):
    """Mean path length: E[sum_i ||tau_i - tau_{i-1}||]."""
    diffs = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    return diffs.norm(2, dim=-1).sum(dim=1).mean()


def smoothness_loss(trajectory):
    """Acceleration minimization: E[||tau_i - 2*tau_{i-1} + tau_{i-2}||^2]."""
    acc = trajectory[:, 2:, :] - 2 * trajectory[:, 1:-1, :] + trajectory[:, :-2, :]
    return (acc ** 2).sum(dim=-1).mean()


def reconstruction_loss(G, E_path, tau_real, c):
    """L_recon = ||G(E(tau_real, c), c) - tau_real||^2."""
    z_hat = E_path(tau_real, c)
    tau_recon_waypoints = G(z_hat, c)  # (B, K, dim) waypoints
    # Compare waypoints with subsampled real path
    B, N, dim = tau_real.shape
    K = tau_recon_waypoints.shape[1]
    indices = torch.linspace(0, N-1, K, device=tau_real.device).long()
    tau_real_sub = tau_real[:, indices, :]
    return F.mse_loss(tau_recon_waypoints, tau_real_sub)


def endpoint_loss(trajectory, goal):
    """L_endpoint = ||trajectory_last - goal||^2.
    Encourages the generated path to end at the goal.
    Args:
        trajectory: (B, N, dim) in [0,1] coords
        goal: (B, dim) in [0,1] coords
    """
    return F.mse_loss(trajectory[:, -1, :], goal)


def convexity_loss(G, z_a, z_b, c_a, voxels_a, interpolation, mode='2d'):
    """L_convexity: interpolated latent should produce collision-free paths.
    Args:
        G: generator
        z_a, z_b: (B, latent_dim) two sets of latent codes
        c_a: (B, cond_dim) conditions for scene a
        voxels_a: voxel map for scene a
        interpolation: callable(G, z, c) -> trajectory
    """
    alpha = torch.rand(z_a.shape[0], 1, device=z_a.device)
    z_interp = (1 - alpha) * z_a + alpha * z_b
    tau_waypoints = G(z_interp, c_a)  # (B, K, dim)
    tau_full = interpolation(tau_waypoints)  # (B, N, dim), in [0,1] space
    return collision_loss(tau_full, voxels_a, mode=mode)
