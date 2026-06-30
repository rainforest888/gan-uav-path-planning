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
    """Start and goal should be in free space (retry up to 5 times)."""
    for attempt in range(5):
        voxels, start, goal, _ = generate_random_scene_2d()
        V = voxels.squeeze(0)
        VOXEL_RES = 32
        si, sj = int(start[1] * (VOXEL_RES - 1)), int(start[0] * (VOXEL_RES - 1))
        gi, gj = int(goal[1] * (VOXEL_RES - 1)), int(goal[0] * (VOXEL_RES - 1))
        si, sj = min(VOXEL_RES - 1, max(0, si)), min(VOXEL_RES - 1, max(0, sj))
        gi, gj = min(VOXEL_RES - 1, max(0, gi)), min(VOXEL_RES - 1, max(0, gj))
        if V[si, sj] == 0 and V[gi, gj] == 0:
            return  # pass
    assert False, f"After 5 attempts, start/goal always obstructed. "
    f"Last: start({si},{sj})={V[si,sj]:.0f}, goal({gi},{gj})={V[gi,gj]:.0f}"


def test_sample_trajectory_no_crash():
    """Trajectory sampling should work for valid trajectory."""
    voxels, _, _, _ = generate_random_scene_2d()
    traj = torch.rand(100, 2)  # random [0,1] trajectory
    collisions = sample_trajectory_on_voxels(voxels, traj, mode='2d')
    assert collisions.shape == (100,)
    assert torch.all(collisions >= 0) and torch.all(collisions <= 1)
