"""Tests for environment generation."""
import torch
import sys
sys.path.insert(0, 'G:/claude code_workspace/gan-uav-path-planning/src')
from environment import generate_random_scene_2d, sample_trajectory_on_voxels


def test_generate_random_scene_2d_shape():
    """Voxel map should have correct shape."""
    from config import VOXEL_RES
    voxels, start, goal, obs_list = generate_random_scene_2d()
    assert voxels.shape == (1, VOXEL_RES, VOXEL_RES)
    assert start.shape == (2,)
    assert goal.shape == (2,)
    assert torch.all(voxels >= 0) and torch.all(voxels <= 1)


def test_start_goal_not_in_obstacle():
    """Start and goal should be in free space."""
    voxels, start, goal, _ = generate_random_scene_2d()
    V = voxels.squeeze(0)
    si, sj = int(start[1] * 31), int(start[0] * 31)
    gi, gj = int(goal[1] * 31), int(goal[0] * 31)
    assert V[si, sj] == 0, f"Start at ({si},{sj}) is obstructed"
    assert V[gi, gj] == 0, f"Goal at ({gi},{gj}) is obstructed"


def test_sample_trajectory_no_crash():
    """Trajectory sampling should work for valid trajectory."""
    voxels, _, _, _ = generate_random_scene_2d()
    traj = torch.rand(100, 2)  # random [0,1] trajectory
    collisions = sample_trajectory_on_voxels(voxels, traj, mode='2d')
    assert collisions.shape == (100,)
    assert torch.all(collisions >= 0) and torch.all(collisions <= 1)
