"""3D/2D voxel environment generation and collision querying."""
import torch
import torch.nn.functional as F


def generate_random_scene_2d(voxel_res=32, space_size=10.0,
                              min_obs=3, max_obs=10,
                              min_r=0.5, max_r=1.5):
    """Generate a random 2D voxel scene with circular obstacles.

    Returns:
        voxels:   Tensor (1, H, W) — 1=obstacle, 0=free
        start:    Tensor (2,) — normalized start coords [0,1]
        goal:     Tensor (2,) — normalized goal coords [0,1]
        obs_list: list of (cx, cy, r_norm) obstacle params
    """
    import random
    n_obs = random.randint(min_obs, max_obs)
    voxels = torch.zeros(voxel_res, voxel_res)
    obs_list = []
    ys, xs = torch.meshgrid(torch.arange(voxel_res, dtype=torch.float32),
                            torch.arange(voxel_res, dtype=torch.float32),
                            indexing='ij')
    for _ in range(n_obs):
        cx, cy = random.random(), random.random()
        r = random.uniform(min_r / space_size, max_r / space_size)
        dist = torch.sqrt((xs / voxel_res - cx) ** 2 + (ys / voxel_res - cy) ** 2)
        voxels[dist < r] = 1.0
        obs_list.append((cx, cy, r))

    # Place start and goal in free space
    free_cells = (voxels == 0).nonzero(as_tuple=False)
    if free_cells.shape[0] < 2:
        return generate_random_scene_2d(voxel_res, space_size, min_obs, max_obs, min_r, max_r)
    idx = torch.randperm(free_cells.shape[0])[:2]
    sy, sx = free_cells[idx[0]].float()
    gy, gx = free_cells[idx[1]].float()
    start = torch.tensor([sx / voxel_res, sy / voxel_res])
    goal = torch.tensor([gx / voxel_res, gy / voxel_res])
    return voxels.unsqueeze(0), start, goal, obs_list


def sample_trajectory_on_voxels(voxels, trajectory, mode='2d'):
    """Sample trajectory points on voxel grid, return collision scores.

    Args:
        voxels:     (1|C, D, H, W) voxel grid
        trajectory: (N, dim) points in normalized [0,1] coords
        mode:       '2d' or '3d'
    Returns:
        collisions: (N,) float tensor, 1=collision, 0=free
                     Uses bilinear (2d) or trilinear (3d) interpolation
                     for soft collision values.
    """
    N = trajectory.shape[0]
    # Scale normalized coords to voxel indices [-1, 1]
    grid_coords = trajectory[:, :2] * 2 - 1  # (N, 2) in [-1,1]
    # grid_sample expects (N, C, H_out, W_out) with coords (N, H_out, W_out, 2)
    grid = grid_coords.view(1, N, 1, 2)  # (1, N, 1, 2)
    voxels_batch = voxels.unsqueeze(0).float()  # (1, 1, H, W)
    sampled = F.grid_sample(voxels_batch, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return sampled.squeeze()  # (N,)


def generate_random_scene_3d(voxel_res=32, space_size=10.0,
                              min_obs=3, max_obs=10,
                              min_r=0.5, max_r=1.5):
    """Generate a random 3D voxel scene with spherical obstacles.

    Returns:
        voxels:   Tensor (1, D, H, W)
        start:    Tensor (3,)
        goal:     Tensor (3,)
        obs_list: list of (cx, cy, cz, r_norm)
    """
    import random
    n_obs = random.randint(min_obs, max_obs)
    voxels = torch.zeros(voxel_res, voxel_res, voxel_res)
    obs_list = []
    zs, ys, xs = torch.meshgrid(torch.arange(voxel_res, dtype=torch.float32),
                                 torch.arange(voxel_res, dtype=torch.float32),
                                 torch.arange(voxel_res, dtype=torch.float32),
                                 indexing='ij')
    for _ in range(n_obs):
        cx, cy, cz = random.random(), random.random(), random.random()
        r = random.uniform(min_r / space_size, max_r / space_size)
        dist = torch.sqrt((xs / voxel_res - cx) ** 2 +
                          (ys / voxel_res - cy) ** 2 +
                          (zs / voxel_res - cz) ** 2)
        voxels[dist < r] = 1.0
        obs_list.append((cx, cy, cz, r))

    free_cells = (voxels == 0).nonzero(as_tuple=False)
    if free_cells.shape[0] < 2:
        return generate_random_scene_3d(voxel_res, space_size, min_obs, max_obs, min_r, max_r)
    idx = torch.randperm(free_cells.shape[0])[:2]
    sz, sy, sx = free_cells[idx[0]].float()
    gz, gy, gx = free_cells[idx[1]].float()
    start = torch.tensor([sx / voxel_res, sy / voxel_res, sz / voxel_res])
    goal = torch.tensor([gx / voxel_res, gy / voxel_res, gz / voxel_res])
    return voxels.unsqueeze(0), start, goal, obs_list


def sample_trajectory_on_voxels_3d(voxels, trajectory):
    """3D trilinear sampling of trajectory on voxel grid."""
    N = trajectory.shape[0]
    grid_coords = trajectory * 2 - 1  # (N, 3) in [-1,1]
    grid = grid_coords.view(1, N, 1, 1, 3)
    voxels_batch = voxels.unsqueeze(0).unsqueeze(0).float()
    sampled = F.grid_sample(voxels_batch, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return sampled.squeeze()
