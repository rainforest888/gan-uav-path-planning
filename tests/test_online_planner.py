"""Tests for online planner."""
import torch
import sys
sys.path.insert(0, 'G:/claude code_workspace/gan-uav-path-planning/src')
from online_planner import OnlinePlanner
from config import LATENT_DIM, CONDITION_DIM


def test_online_planner_initialization():
    """initialize() should return (1, LATENT_DIM)."""
    planner = OnlinePlanner(dim=2)
    z = planner.initialize()
    assert z.shape == (1, LATENT_DIM)


def test_online_planner_replan_step():
    """replan_step should return updated z and trajectory."""
    planner = OnlinePlanner(dim=2)
    z_prev = torch.randn(1, LATENT_DIM)
    c = torch.randn(1, CONDITION_DIM)
    voxels = torch.zeros(1, 1, 32, 32)
    z_new, traj = planner.replan_step(z_prev, c, voxels)
    assert z_new.shape == (1, LATENT_DIM)
    assert traj.shape[0] > 0  # trajectory returned
