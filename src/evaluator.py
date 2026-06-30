"""Evaluation metrics and baseline comparison for path planning methods."""
import time
import torch
import numpy as np


def compute_metrics(paths, voxels_sequence, start, goal):
    """Compute all evaluation metrics for a planning run.

    Args:
        paths: list of (N, dim) numpy arrays, one per timestep
        voxels_sequence: list of (1, H, W) or (1, D, H, W) voxel maps
        start: (dim,) start position
        goal: (dim,) goal position
    Returns:
        dict of metric name -> float value
    """
    metrics = {}

    # Success rate: reached goal without collision
    if isinstance(start, torch.Tensor):
        start = start.numpy()
    if isinstance(goal, torch.Tensor):
        goal = goal.numpy()

    final_pos = paths[-1][-1] if paths else start
    reached_goal = np.linalg.norm(final_pos - goal) < 0.1
    had_collision = False
    for path, voxels in zip(paths, voxels_sequence):
        V = voxels.squeeze().numpy() if isinstance(voxels, torch.Tensor) else voxels
        if V.ndim == 3:
            H, W = V.shape[-2:]
            for pt in path:
                pi, pj = int(pt[1] * H), int(pt[0] * W)
                pik = int(pt[2] * V.shape[0])
                pi, pj = max(0, min(H-1, pi)), max(0, min(W-1, pj))
                pik = max(0, min(V.shape[0]-1, pik))
                if V[pik, pi, pj] > 0:
                    had_collision = True
                    break
        else:
            H, W = V.shape[-2:]
            for pt in path:
                pi, pj = int(pt[1] * H), int(pt[0] * W)
                pi, pj = max(0, min(H-1, pi)), max(0, min(W-1, pj))
                if V[pi, pj] > 0:
                    had_collision = True
                    break
        if had_collision:
            break
    metrics['success'] = float(reached_goal and not had_collision)

    # Path length
    total_length = 0
    for path in paths:
        diffs = np.diff(path, axis=0)
        total_length += np.sum(np.linalg.norm(diffs, axis=1))
    metrics['path_length'] = total_length

    # Path consistency: Hausdorff distance between consecutive paths
    hausdorffs = []
    for i in range(1, len(paths)):
        d1 = np.min(np.linalg.norm(
            paths[i][:, None] - paths[i-1][None, :], axis=2), axis=1).max()
        d2 = np.min(np.linalg.norm(
            paths[i-1][:, None] - paths[i][None, :], axis=2), axis=1).max()
        hausdorffs.append(max(d1, d2))
    metrics['mean_hausdorff'] = np.mean(hausdorffs) if hausdorffs else 0.0

    # Smoothness: mean jerk (3rd derivative)
    jerks = []
    for path in paths:
        if len(path) > 3:
            j = np.diff(path, n=3, axis=0)
            jerks.append(np.mean(np.linalg.norm(j, axis=1)))
    metrics['mean_jerk'] = np.mean(jerks) if jerks else 0.0

    return metrics


def run_astar_baseline(voxels, start, goal, voxel_res=32):
    """Run A* baseline and return path + timing."""
    from astar import astar_2d
    t0 = time.time()
    if start.shape[0] == 2:
        path = astar_2d(voxels.squeeze(0), start, goal, voxel_res)
    else:
        from astar import astar_3d
        path = astar_3d(voxels.squeeze(0), start, goal, voxel_res)
    elapsed = (time.time() - t0) * 1000  # ms
    return path.numpy() if path is not None else None, elapsed
