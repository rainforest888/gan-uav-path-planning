"""Tests for trajectory interpolation."""
import torch
import sys
sys.path.insert(0, 'G:/claude code_workspace/gan-uav-path-planning/src')
from interpolation import cubic_spline_trajectory, minimum_snap_trajectory


def test_cubic_spline_shape():
    """Cubic spline should output correct shape."""
    waypoints = torch.rand(10, 2)
    traj = cubic_spline_trajectory(waypoints, 100)
    assert traj.shape == (100, 2)


def test_cubic_spline_endpoints():
    """Cubic spline should pass through endpoints."""
    waypoints = torch.rand(10, 2)
    traj = cubic_spline_trajectory(waypoints, 100)
    assert torch.allclose(traj[0], waypoints[0], atol=1e-5)
    assert torch.allclose(traj[-1], waypoints[-1], atol=1e-5)


def test_cubic_spline_differentiable():
    """Cubic spline should be differentiable (gradient flows)."""
    waypoints = torch.rand(10, 2, requires_grad=True)
    traj = cubic_spline_trajectory(waypoints, 100)
    loss = traj.sum()
    loss.backward()
    assert waypoints.grad is not None
    assert not torch.allclose(waypoints.grad, torch.zeros_like(waypoints.grad))
