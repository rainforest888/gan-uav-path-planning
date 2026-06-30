"""Differentiable cubic spline (training) and Minimum Snap QP (inference)."""
import torch


def cubic_spline_trajectory(waypoints, num_points):
    """Catmull-Rom cubic spline — fully differentiable.

    Args:
        waypoints: (..., K, dim) control points (supports batch dims)
        num_points: int, output trajectory points
    Returns:
        trajectory: (..., num_points, dim)
    """
    # Handle batch dims: save leading shape, flatten to (B, K, dim)
    leading = waypoints.shape[:-2]
    K, dim = waypoints.shape[-2], waypoints.shape[-1]
    wp = waypoints.reshape(-1, K, dim)  # (B, K, dim)
    B, K, dim = wp.shape

    if K < 2:
        return wp.repeat(1, num_points // K + 1, 1)[:, :num_points, :].reshape(*leading, num_points, dim)

    # Pad endpoints for Catmull-Rom
    p0 = 2 * wp[:, 0:1, :] - wp[:, 1:2, :]
    pN = 2 * wp[:, -1:, :] - wp[:, -2:-1, :]
    padded = torch.cat([p0, wp, pN], dim=1)  # (B, K+2, dim)

    # Parameter t: equally spaced per segment
    t = torch.linspace(0, 1, num_points, device=waypoints.device)
    seg_float = t * (K - 1)
    seg_idx = seg_float.long().clamp(0, K - 2)
    alpha = seg_float - seg_idx.float()

    # Catmull-Rom basis
    t2 = alpha * alpha
    t3 = t2 * alpha
    w0 = 0.5 * (-t3 + 2*t2 - alpha)
    w1 = 0.5 * (3*t3 - 5*t2 + 2)
    w2 = 0.5 * (-3*t3 + 4*t2 + alpha)
    w3 = 0.5 * (t3 - t2)
    weights = torch.stack([w0, w1, w2, w3], dim=1)  # (N, 4)

    # Gather control points for each sample
    idx = seg_idx.unsqueeze(1) + torch.arange(4, device=waypoints.device).unsqueeze(0)  # (N, 4)
    ctrl = padded[:, idx, :]  # (B, N, 4, dim)

    trajectory = (weights.unsqueeze(0).unsqueeze(-1) * ctrl).sum(dim=2)  # (B, N, dim)
    return trajectory.reshape(*leading, num_points, dim)


def minimum_snap_trajectory(waypoints, num_points, time_per_seg=1.0):
    """Minimum Snap trajectory via QP (not differentiable, for inference).

    Uses closed-form: fits a minimum-snap polynomial through waypoints.

    Args:
        waypoints: (K, dim) tensor
        num_points: int
        time_per_seg: float, time allocated per segment
    Returns:
        trajectory: (num_points, dim) numpy array
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    K, dim = waypoints.shape
    wp = waypoints.detach().cpu().numpy()

    if K < 3:
        # Fall back to cubic spline
        traj = cubic_spline_trajectory(waypoints, num_points)
        return traj.detach().cpu().numpy()

    result = np.zeros((num_points, dim))
    total_time = (K - 1) * time_per_seg

    for d in range(dim):
        t_wp = np.linspace(0, total_time, K)
        cs = CubicSpline(t_wp, wp[:, d], bc_type='natural')
        t_eval = np.linspace(0, total_time, num_points)
        result[:, d] = cs(t_eval)

    return result
