"""Tests for all neural network model modules."""
import torch
import sys
sys.path.insert(0, 'G:/claude code_workspace/gan-uav-path-planning/src')
from config import LATENT_DIM, CONDITION_DIM, K_WAYPOINTS, N_TRAJECTORY
from models.env_encoder import EnvEncoder2D, EnvEncoder3D
from models.generator import Generator
from models.critic import Critic
from models.path_encoder import PathEncoder2D, PathEncoder3D

B = 4  # tiny batch


def test_env_encoder_2d_output_shape():
    """E_env 2D should output (B, 256)."""
    voxels = torch.zeros(B, 1, 32, 32)
    start = torch.zeros(B, 2)
    goal = torch.ones(B, 2)
    enc = EnvEncoder2D()
    c = enc(voxels, start, goal)
    assert c.shape == (B, CONDITION_DIM)


def test_generator_output_shape():
    """G should output (B, K, dim) with values in [-1,1]."""
    z = torch.randn(B, LATENT_DIM)
    c = torch.randn(B, CONDITION_DIM)
    g = Generator(dim_out=2)
    waypoints = g(z, c)
    assert waypoints.shape == (B, K_WAYPOINTS, 2)
    assert waypoints.min() >= 0.0 and waypoints.max() <= 1.0


def test_critic_output_shape():
    """D should output (B, 1) scalar."""
    tau = torch.randn(B, N_TRAJECTORY, 2)
    c = torch.randn(B, CONDITION_DIM)
    d = Critic(dim_in=2)
    score = d(tau, c)
    assert score.shape == (B, 1)


def test_path_encoder_2d_output_shape():
    """E_path 2D should output (B, 128)."""
    tau = torch.randn(B, N_TRAJECTORY, 2)
    c = torch.randn(B, CONDITION_DIM)
    enc = PathEncoder2D()
    z_hat = enc(tau, c)
    assert z_hat.shape == (B, LATENT_DIM)


def test_reconstruction_cycle():
    """G(E(tau, c), c) shapes should be consistent."""
    tau = torch.randn(B, N_TRAJECTORY, 2)
    c = torch.randn(B, CONDITION_DIM)
    enc = PathEncoder2D()
    gen = Generator(dim_out=2)
    z_hat = enc(tau, c)
    tau_recon = gen(z_hat, c)  # (B, K, 2) waypoints
    assert tau_recon.shape == (B, K_WAYPOINTS, 2)
