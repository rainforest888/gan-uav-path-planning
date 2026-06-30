"""Tests for loss functions."""
import torch
import sys
sys.path.insert(0, 'G:/claude code_workspace/gan-uav-path-planning/src')
from losses import (
    critic_loss, generator_adv_loss, gradient_penalty,
    collision_loss, path_length_loss, smoothness_loss,
    reconstruction_loss, convexity_loss,
)
from config import LAMBDA_GP


def test_gradient_penalty_shape():
    """GP should return a non-negative scalar."""

    # Simple critic that accepts (trajectory, condition) like the real D
    class SimpleCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(100 * 2 + 256, 1)

        def forward(self, tau, c):
            return self.fc(torch.cat([tau.flatten(1), c], dim=1))

    D = SimpleCritic()
    tau_real = torch.randn(4, 100, 2, requires_grad=True)
    tau_fake = torch.randn(4, 100, 2, requires_grad=True)
    c = torch.randn(4, 256)
    gp = gradient_penalty(D, tau_real, tau_fake, c)
    assert gp.shape == ()
    assert gp.item() >= 0


def test_collision_loss_zero_for_free_space():
    """Collision loss should be 0 when all voxels are free."""
    traj = torch.ones(4, 100, 2) * 0.5  # center, away from edges
    voxels = torch.zeros(4, 1, 32, 32)  # all free
    loss = collision_loss(traj, voxels, mode='2d')
    assert loss.item() == 0.0


def test_collision_loss_positive_for_obstacle():
    """Collision loss should be >0 when all voxels are obstacles."""
    traj = torch.ones(4, 100, 2) * 0.5
    voxels = torch.ones(4, 1, 32, 32)  # all obstacle
    loss = collision_loss(traj, voxels, mode='2d')
    assert loss.item() > 0.0


def test_path_length_loss():
    """Path length of a 3-4-5 triangle edge should be 5.0."""
    traj = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])  # 1 batch, 2 pts
    loss = path_length_loss(traj)
    assert abs(loss.item() - 5.0) < 0.01


def test_smoothness_loss_zero_for_straight_line():
    """Smoothness loss for a straight line should be ~0."""
    t = torch.linspace(0, 1, 100).view(-1, 1)
    traj = torch.cat([t, t], dim=1).unsqueeze(0)  # (1, 100, 2) straight line
    loss = smoothness_loss(traj)
    assert loss.item() < 0.01
