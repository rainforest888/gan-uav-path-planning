"""Generate training datasets: random scenes + A* ground-truth paths."""
import torch
from tqdm import tqdm
from environment import generate_random_scene_2d, generate_random_scene_3d
from astar import astar_2d, astar_3d
from interpolation import cubic_spline_trajectory
from config import VOXEL_RES, N_TRAJECTORY


def generate_dataset_2d(num_samples=5000, voxel_res=32):
    """Generate 2D training dataset.

    Returns:
        list of dicts: [{voxels, start, goal, path_real}]
    """
    dataset = []
    failed = 0
    for _ in tqdm(range(num_samples), desc="Generating 2D data"):
        voxels, start, goal, _ = generate_random_scene_2d(voxel_res=voxel_res)
        raw_path = astar_2d(voxels.squeeze(0), start, goal, voxel_res=voxel_res)
        if raw_path is None:
            failed += 1
            continue
        # Resample to fixed N_TRAJECTORY points via cubic spline
        path_real = cubic_spline_trajectory(raw_path, N_TRAJECTORY).detach()
        dataset.append({
            'voxels': voxels,
            'start': start,
            'goal': goal,
            'path_real': path_real,
        })
    print(f"Generated {len(dataset)} samples ({failed} A* failures)")
    return dataset


def generate_dataset_3d(num_samples=5000, voxel_res=32):
    """Generate 3D training dataset."""
    dataset = []
    failed = 0
    for _ in tqdm(range(num_samples), desc="Generating 3D data"):
        voxels, start, goal, _ = generate_random_scene_3d(voxel_res=voxel_res)
        raw_path = astar_3d(voxels.squeeze(0), start, goal, voxel_res=voxel_res)
        if raw_path is None:
            failed += 1
            continue
        path_real = cubic_spline_trajectory(raw_path, N_TRAJECTORY).detach()
        dataset.append({
            'voxels': voxels,
            'start': start,
            'goal': goal,
            'path_real': path_real,
        })
    print(f"Generated {len(dataset)} samples ({failed} A* failures)")
    return dataset
