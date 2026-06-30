"""Tests for A* planner."""
import torch
import sys
sys.path.insert(0, 'G:/claude code_workspace/gan-uav-path-planning/src')
from environment import generate_random_scene_2d
from astar import astar_2d


def test_astar_finds_path():
    """A* should find a path in random free space (retry up to 5 times)."""
    for attempt in range(5):
        voxels, start, goal, _ = generate_random_scene_2d(voxel_res=32)
        path = astar_2d(voxels.squeeze(0), start, goal, voxel_res=32)
        if path is not None:
            assert path.shape[1] == 2, "Path should be (N, 2)"
            assert path.shape[0] >= 2, "Path should have at least start and goal"
            return  # pass
    assert False, "A* failed to find a path after 5 attempts"


def test_astar_path_start_goal_match():
    """Start and goal of found path should match input (retry up to 5 times)."""
    for attempt in range(5):
        voxels, start, goal, _ = generate_random_scene_2d(voxel_res=32)
        path = astar_2d(voxels.squeeze(0), start, goal, voxel_res=32)
        if path is not None:
            assert torch.allclose(path[0], start, atol=1 / 32)
            assert torch.allclose(path[-1], goal, atol=1 / 32)
            return  # pass
    assert False, "A* failed to find a path after 5 attempts"
