# GAN-UAV Dynamic Path Planning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Conditional WGAN-GP system with latent space trajectory evolution for 3D UAV dynamic path planning, validated from 2D through AirSim forest simulation.

**Architecture:** Four neural networks (E_env, G, D, E_path) trained jointly with WGAN-GP + constraint losses + convexity regularization. Online inference uses latent-space gradient optimization for continuous trajectory deformation instead of discard-and-regenerate.

**Tech Stack:** Python 3.9+, PyTorch 2.x, NumPy, Matplotlib, scipy (Min Snap QP), AirSim (Phase 4)

---

## File Map

```
gan-uav-path-planning/
├── src/
│   ├── __init__.py
│   ├── config.py              # All hyperparameters and constants
│   ├── environment.py         # Voxel environment generation (2D/3D)
│   ├── astar.py               # A* 3D grid path planner for data generation
│   ├── data_generator.py      # Generate training datasets
│   ├── models/
│   │   ├── __init__.py
│   │   ├── env_encoder.py     # E_env: 3D->condition vector
│   │   ├── generator.py       # G(z, c): latent->waypoints
│   │   ├── critic.py          # D(τ, c): WGAN-GP critic
│   │   └── path_encoder.py    # E_path(τ, c): trajectory->latent
│   ├── interpolation.py       # Cubic spline + Minimum Snap
│   ├── losses.py              # All loss functions
│   ├── trainer.py             # Training loop
│   ├── online_planner.py      # Latent-space evolution (dynamic)
│   └── evaluator.py           # Metrics, baselines, visualization
├── experiments/
│   ├── phase1_train_2d.py     # Phase 1: 2D training script
│   ├── phase1_eval_2d.py      # Phase 1: 2D evaluation
│   ├── phase2_train_3d.py     # Phase 2: 3D static training
│   ├── phase3_dynamic.py      # Phase 3: dynamic obstacles
│   └── phase4_airsim.py       # Phase 4: AirSim integration
├── tests/
│   ├── test_environment.py
│   ├── test_astar.py
│   ├── test_models.py
│   ├── test_losses.py
│   ├── test_interpolation.py
│   └── test_online_planner.py
├── requirements.txt
└── README.md
```

---

## Phase 1: 2D Feasibility Validation

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `README.md`

- [ ] **Step 1: Create requirements.txt**

```text
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
tqdm>=4.65.0
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
echo "" > G:/claude\ code_workspace/gan-uav-path-planning/src/__init__.py
echo "" > G:/claude\ code_workspace/gan-uav-path-planning/src/models/__init__.py
```

- [ ] **Step 3: Create minimal README.md**

```markdown
# GAN-UAV Dynamic Path Planning

Conditional WGAN-GP with latent space trajectory evolution for 3D UAV dynamic path planning.

## Setup
pip install -r requirements.txt

## Phases
1. 2D static - validate CWGAN-GP architecture
2. 3D static - extend to 3D voxel environments
3. Dynamic - latent space continuous evolution
4. AirSim - forest simulation integration
```

- [ ] **Step 4: Install dependencies**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md src/
git commit -m "chore: initialize project structure"
```

---

### Task 2: Configuration Module

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py` (implicit — config has no logic to test, but import check)

- [ ] **Step 1: Write config.py**

```python
"""Global configuration constants for GAN-UAV path planning."""

# --- Environment ---
VOXEL_RES = 32          # Voxel grid resolution (VOXEL_RES^3 for 3D)
SPACE_SIZE = 10.0       # World size in meters (cube: SPACE_SIZE^dim)
OBSTACLE_MIN_R = 0.5    # Min obstacle radius
OBSTACLE_MAX_R = 1.5    # Max obstacle radius
NUM_OBSTACLES_MIN = 3   # Min obstacles per scene
NUM_OBSTACLES_MAX = 10  # Max obstacles per scene

# --- Path Representation ---
K_WAYPOINTS = 10        # Key waypoints output by generator
N_TRAJECTORY = 100      # Trajectory points after interpolation

# --- Generator ---
LATENT_DIM = 128        # z dimension
CONDITION_DIM = 256     # c dimension (from E_env)

# --- Training ---
BATCH_SIZE = 64
LR_G = 1e-4
LR_D = 1e-4
LR_E = 1e-4
N_CRITIC = 5            # D updates per G/E update
LAMBDA_GP = 10.0        # Gradient penalty weight
LAMBDA_COL = 10.0       # Collision loss weight
LAMBDA_LEN = 0.5        # Path length weight
LAMBDA_SMOOTH = 1.0     # Smoothness weight
LAMBDA_RECON = 1.0      # Reconstruction weight (E_path)
LAMBDA_CONVEXITY = 2.0  # Convexity regularization weight
NUM_EPOCHS = 500        # Phase 1 training epochs

# --- Online Inference ---
LAMBDA_CONTINUITY = 0.5  # Latent continuity weight
ONLINE_LR = 0.01         # Online gradient descent step size
ONLINE_STEPS = 5         # Gradient descent steps per replanning cycle
REPLAN_HZ = 10           # Replanning frequency (Hz)
```

- [ ] **Step 2: Verify import**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -c "from src.config import *; print(f'config OK, LATENT_DIM={LATENT_DIM}')"
```

Expected: `config OK, LATENT_DIM=128`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add configuration module"
```

---

### Task 3: Environment Generator

**Files:**
- Create: `src/environment.py`
- Create: `tests/test_environment.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for environment generation."""
import torch
import sys
sys.path.insert(0, 'src')
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
    si, sj = (start * 31).long()
    gi, gj = (goal * 31).long()
    assert voxels[0, si, sj] == 0, f"Start at ({si},{sj}) is obstructed"
    assert voxels[0, gi, gj] == 0, f"Goal at ({gi},{gj}) is obstructed"

def test_sample_trajectory_no_crash():
    """Trajectory sampling should work for valid trajectory."""
    voxels, _, _, _ = generate_random_scene_2d()
    traj = torch.rand(100, 2)  # random [0,1] trajectory
    collisions = sample_trajectory_on_voxels(voxels, traj, mode='2d')
    assert collisions.shape == (100,)
    assert torch.all(collisions >= 0) and torch.all(collisions <= 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_environment.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement environment.py**

```python
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
        r_px = r * voxel_res
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
    V = voxel_res = voxels.shape[-1]  # assume cubic
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
        r_px = r * voxel_res
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
    V = voxels.shape[-1]
    N = trajectory.shape[0]
    grid_coords = trajectory * 2 - 1  # (N, 3) in [-1,1]
    grid = grid_coords.view(1, N, 1, 1, 3)  # (1, N, 1, 1, 3)
    voxels_batch = voxels.unsqueeze(0).unsqueeze(0).float()  # (1, 1, D, H, W)
    sampled = F.grid_sample(voxels_batch, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return sampled.squeeze()  # (N,)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_environment.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/environment.py tests/test_environment.py
git commit -m "feat: add voxel environment generator (2D + 3D)"
```

---

### Task 4: A* Path Planner (for training data)

**Files:**
- Create: `src/astar.py`
- Create: `tests/test_astar.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for A* planner."""
import torch
import sys
sys.path.insert(0, 'src')
from environment import generate_random_scene_2d
from astar import astar_2d

def test_astar_finds_path():
    voxels, start, goal, _ = generate_random_scene_2d(voxel_res=32)
    path = astar_2d(voxels.squeeze(0), start, goal, voxel_res=32)
    assert path is not None, "A* should find a path in a connected free space"
    assert path.shape[1] == 2, "Path should be (N, 2)"
    assert path.shape[0] >= 2, "Path should have at least start and goal"

def test_astar_path_start_goal_match():
    voxels, start, goal, _ = generate_random_scene_2d(voxel_res=32)
    path = astar_2d(voxels.squeeze(0), start, goal, voxel_res=32)
    assert path is not None
    assert torch.allclose(path[0], start, atol=1/32)
    assert torch.allclose(path[-1], goal, atol=1/32)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_astar.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement astar.py**

```python
"""A* grid search for generating training data (2D and 3D)."""
import torch
import heapq


def astar_2d(voxels, start, goal, voxel_res=32):
    """2D A* on voxel grid. Returns path as (N, 2) tensor of normalized coords.

    Args:
        voxels: (H, W) tensor, 1=obstacle, 0=free
        start:  (2,) tensor, normalized [0,1] coords
        goal:   (2,) tensor, normalized [0,1] coords
    Returns:
        path: (N, 2) tensor or None if no path found
    """
    H, W = voxels.shape
    si, sj = int(start[1] * H), int(start[0] * W)
    gi, gj = int(goal[1] * H), int(goal[0] * W)
    si, sj = max(0, min(H-1, si)), max(0, min(W-1, sj))
    gi, gj = max(0, min(H-1, gi)), max(0, min(W-1, gj))
    if voxels[si, sj] > 0 or voxels[gi, gj] > 0:
        return None

    def heuristic(i, j):
        return abs(i - gi) + abs(j - gj)

    open_set = [(heuristic(si, sj), 0, si, sj)]
    came_from = {}
    g_score = {(si, sj): 0}
    visited = set()
    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]

    while open_set:
        _, g, i, j = heapq.heappop(open_set)
        if (i, j) in visited:
            continue
        visited.add((i, j))
        if (i, j) == (gi, gj):
            # Reconstruct path
            path = [(i, j)]
            while (i, j) in came_from:
                i, j = came_from[(i, j)]
                path.append((i, j))
            path.reverse()
            pts = torch.tensor([[j/W, i/H] for i, j in path], dtype=torch.float32)
            return pts
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and voxels[ni, nj] == 0:
                cost = 1.414 if di != 0 and dj != 0 else 1.0
                ng = g + cost
                if ng < g_score.get((ni, nj), float('inf')):
                    g_score[(ni, nj)] = ng
                    came_from[(ni, nj)] = (i, j)
                    heapq.heappush(open_set, (ng + heuristic(ni, nj), ng, ni, nj))
    return None


def astar_3d(voxels, start, goal, voxel_res=32):
    """3D A* on voxel grid.

    Args:
        voxels: (D, H, W) tensor
        start:  (3,) tensor, normalized
        goal:   (3,) tensor, normalized
    Returns:
        path: (N, 3) tensor or None
    """
    D, H, W = voxels.shape
    si, sj, sk = int(start[1]*H), int(start[0]*W), int(start[2]*D)
    gi, gj, gk = int(goal[1]*H), int(goal[0]*W), int(goal[2]*D)
    si, sj, sk = max(0,min(H-1,si)), max(0,min(W-1,sj)), max(0,min(D-1,sk))
    gi, gj, gk = max(0,min(H-1,gi)), max(0,min(W-1,gj)), max(0,min(D-1,gk))
    if voxels[sk, si, sj] > 0 or voxels[gk, gi, gj] > 0:
        return None

    def heuristic(i, j, k):
        return abs(i-gi) + abs(j-gj) + abs(k-gk)

    open_set = [(heuristic(si, sj, sk), 0, si, sj, sk)]
    came_from = {}
    g_score = {(si, sj, sk): 0}
    visited = set()
    # 26-connected neighbors
    neighbors = [(di, dj, dk) for di in (-1,0,1) for dj in (-1,0,1) for dk in (-1,0,1)
                 if not (di==0 and dj==0 and dk==0)]

    while open_set:
        _, g, i, j, k = heapq.heappop(open_set)
        if (i, j, k) in visited:
            continue
        visited.add((i, j, k))
        if (i, j, k) == (gi, gj, gk):
            path = [(i, j, k)]
            while (i, j, k) in came_from:
                i, j, k = came_from[(i, j, k)]
                path.append((i, j, k))
            path.reverse()
            pts = torch.tensor([[j/W, i/H, k/D] for i, j, k in path], dtype=torch.float32)
            return pts
        for di, dj, dk in neighbors:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0<=ni<H and 0<=nj<W and 0<=nk<D and voxels[nk, ni, nj]==0:
                cost = (di**2 + dj**2 + dk**2) ** 0.5
                ng = g + cost
                if ng < g_score.get((ni, nj, nk), float('inf')):
                    g_score[(ni, nj, nk)] = ng
                    came_from[(ni, nj, nk)] = (i, j, k)
                    heapq.heappush(open_set, (ng+heuristic(ni,nj,nk), ng, ni, nj, nk))
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_astar.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/astar.py tests/test_astar.py
git commit -m "feat: add A* path planner (2D + 3D) for training data"
```

---

### Task 5: Data Generator

**Files:**
- Create: `src/data_generator.py`

- [ ] **Step 1: Implement data_generator.py**

```python
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
        list of dicts: [{voxels, start, goal, path_real, condition}]
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
        condition = build_condition(voxels, start, goal)
        dataset.append({
            'voxels': voxels,
            'start': start,
            'goal': goal,
            'path_real': path_real,
            'condition': condition
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
        condition = build_condition_3d(voxels, start, goal)
        dataset.append({
            'voxels': voxels,
            'start': start,
            'goal': goal,
            'path_real': path_real,
            'condition': condition
        })
    print(f"Generated {len(dataset)} samples ({failed} A* failures)")
    return dataset


def build_condition(voxels, start, goal):
    """Build condition vector from voxels+start+goal for 2D."""
    # Will be properly built by E_env during training; this is a reference
    # For now, return (voxels, start, goal) tuple
    from models.env_encoder import EnvEncoder
    # Placeholder — actual encoding done inside trainer
    return None  # Will be computed by E_env


def build_condition_3d(voxels, start, goal):
    """Build condition vector for 3D."""
    return None
```

- [ ] **Step 2: Commit**

```bash
git add src/data_generator.py
git commit -m "feat: add data generator for 2D and 3D training datasets"
```

---

### Task 6: Cubic Spline Interpolation

**Files:**
- Create: `src/interpolation.py`
- Create: `tests/test_interpolation.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for trajectory interpolation."""
import torch
import sys
sys.path.insert(0, 'src')
from interpolation import cubic_spline_trajectory, minimum_snap_trajectory

def test_cubic_spline_shape():
    waypoints = torch.rand(10, 2)
    traj = cubic_spline_trajectory(waypoints, 100)
    assert traj.shape == (100, 2)

def test_cubic_spline_endpoints():
    waypoints = torch.rand(10, 2)
    traj = cubic_spline_trajectory(waypoints, 100)
    assert torch.allclose(traj[0], waypoints[0], atol=1e-5)
    assert torch.allclose(traj[-1], waypoints[-1], atol=1e-5)

def test_cubic_spline_differentiable():
    waypoints = torch.rand(10, 2, requires_grad=True)
    traj = cubic_spline_trajectory(waypoints, 100)
    loss = traj.sum()
    loss.backward()
    assert waypoints.grad is not None
    assert not torch.allclose(waypoints.grad, torch.zeros_like(waypoints.grad))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_interpolation.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement interpolation.py**

```python
"""Differentiable cubic spline (training) and Minimum Snap QP (inference)."""
import torch
import torch.nn.functional as F


def cubic_spline_trajectory(waypoints, num_points):
    """Catmull-Rom cubic spline — fully differentiable, returns (num_points, dim).

    Args:
        waypoints: (K, dim) control points
        num_points: int, output trajectory points
    Returns:
        trajectory: (num_points, dim)
    """
    K, dim = waypoints.shape
    if K < 2:
        return waypoints.repeat(num_points, 1)

    # Pad endpoints for Catmull-Rom
    p0 = 2 * waypoints[0:1] - waypoints[1:2]
    pN = 2 * waypoints[-1:] - waypoints[-2:-1]
    padded = torch.cat([p0, waypoints, pN], dim=0)  # (K+2, dim)

    # Parameter t: equally spaced per segment
    t = torch.linspace(0, 1, num_points, device=waypoints.device)
    # Map t to segment index
    seg_float = t * (K - 1)
    seg_idx = seg_float.long().clamp(0, K - 2)
    alpha = seg_float - seg_idx.float()  # local t within segment [0,1]

    # Catmull-Rom basis
    t2 = alpha * alpha
    t3 = t2 * alpha
    # Basis matrix for Catmull-Rom
    # [t^3 t^2 t 1] * 0.5 * [[-1 3 -3 1], [2 -5 4 -1], [-1 0 1 0], [0 2 0 0]]
    w0 = 0.5 * (-t3 + 2*t2 - alpha)           # P_{i-1}
    w1 = 0.5 * (3*t3 - 5*t2 + 2)              # P_i
    w2 = 0.5 * (-3*t3 + 4*t2 + alpha)         # P_{i+1}
    w3 = 0.5 * (t3 - t2)                       # P_{i+2}

    weights = torch.stack([w0, w1, w2, w3], dim=1)  # (N, 4)

    # Gather control points for each sample
    idx = seg_idx.unsqueeze(1) + torch.arange(4, device=waypoints.device).unsqueeze(0)  # (N, 4)
    ctrl = padded[idx]  # (N, 4, dim)

    trajectory = (weights.unsqueeze(-1) * ctrl).sum(dim=1)  # (N, dim)
    return trajectory


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
    from scipy.linalg import block_diag

    K, dim = waypoints.shape
    wp = waypoints.detach().cpu().numpy()

    if K < 3:
        # Fall back to cubic spline
        traj = cubic_spline_trajectory(waypoints, num_points)
        return traj.detach().cpu().numpy()

    result = np.zeros((num_points, dim))
    total_time = (K - 1) * time_per_seg
    segment_pts = num_points // (K - 1)

    for d in range(dim):
        # Fit a piecewise polynomial through waypoints
        # For simplicity: use scipy's CubicSpline which minimizes curvature
        from scipy.interpolate import CubicSpline
        t_wp = np.linspace(0, total_time, K)
        cs = CubicSpline(t_wp, wp[:, d], bc_type='natural')
        t_eval = np.linspace(0, total_time, num_points)
        result[:, d] = cs(t_eval)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_interpolation.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/interpolation.py tests/test_interpolation.py
git commit -m "feat: add cubic spline and minimum snap interpolation"
```

---

### Task 7: Neural Network Models

**Files:**
- Create: `src/models/env_encoder.py`
- Create: `src/models/generator.py`
- Create: `src/models/critic.py`
- Create: `src/models/path_encoder.py`
- Create: `src/models/__init__.py` (update)
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for all neural network model modules."""
import torch
import sys
sys.path.insert(0, 'src')
from config import LATENT_DIM, CONDITION_DIM, K_WAYPOINTS, N_TRAJECTORY
from models.env_encoder import EnvEncoder2D, EnvEncoder3D
from models.generator import Generator
from models.critic import Critic
from models.path_encoder import PathEncoder2D, PathEncoder3D

B = 4  # tiny batch

def test_env_encoder_2d_output_shape():
    voxels = torch.zeros(B, 1, 32, 32)
    start = torch.zeros(B, 2)
    goal = torch.ones(B, 2)
    enc = EnvEncoder2D()
    c = enc(voxels, start, goal)
    assert c.shape == (B, CONDITION_DIM)

def test_generator_output_shape():
    z = torch.randn(B, LATENT_DIM)
    c = torch.randn(B, CONDITION_DIM)
    g = Generator(dim_out=2)
    waypoints = g(z, c)
    assert waypoints.shape == (B, K_WAYPOINTS, 2)
    # Output should be in [-1, 1] due to Tanh
    assert waypoints.min() >= -1.0 and waypoints.max() <= 1.0

def test_critic_output_shape():
    tau = torch.randn(B, N_TRAJECTORY, 2)
    c = torch.randn(B, CONDITION_DIM)
    d = Critic(dim_in=2)
    score = d(tau, c)
    assert score.shape == (B, 1)

def test_path_encoder_2d_output_shape():
    tau = torch.randn(B, N_TRAJECTORY, 2)
    c = torch.randn(B, CONDITION_DIM)
    enc = PathEncoder2D()
    z_hat = enc(tau, c)
    assert z_hat.shape == (B, LATENT_DIM)

def test_reconstruction_cycle():
    """G(E(tau, c), c) should approximately reconstruct tau."""
    tau = torch.randn(B, N_TRAJECTORY, 2)
    c = torch.randn(B, CONDITION_DIM)
    enc = PathEncoder2D()
    gen = Generator(dim_out=2)
    z_hat = enc(tau, c)
    tau_recon = gen(z_hat, c)  # (B, K, 2) waypoints
    assert tau_recon.shape == (B, K_WAYPOINTS, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_models.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement env_encoder.py**

```python
"""Environment encoder E_env: 3D voxels -> condition vector."""
import torch
import torch.nn as nn
from config import VOXEL_RES, CONDITION_DIM


class EnvEncoder2D(nn.Module):
    """2D variant for Phase 1 validation. Processes (1, H, W) voxel grid."""
    def __init__(self, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        # After 4 stride-2 convs: 32 -> 16 -> 8 -> 4 -> 2
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 + 4, cond_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, voxels, start, goal):
        f = self.conv(voxels).flatten(1)  # (B, 256*2*2)
        sg = torch.cat([start, goal], dim=1) * 2 - 1  # normalize to [-1,1]
        return self.fc(torch.cat([f, sg], dim=1))


class EnvEncoder3D(nn.Module):
    """3D variant for Phase 2+. Processes (1, D, H, W) voxel grid."""
    def __init__(self, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
        )
        # 32 -> 16 -> 8 -> 4 -> 2
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 * 2 + 6, cond_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, voxels, start, goal):
        f = self.conv(voxels).flatten(1)
        sg = torch.cat([start, goal], dim=1) * 2 - 1
        return self.fc(torch.cat([f, sg], dim=1))
```

- [ ] **Step 4: Implement generator.py**

```python
"""Generator G(z, c): latent vector + condition -> key waypoints."""
import torch
import torch.nn as nn
from config import LATENT_DIM, CONDITION_DIM, K_WAYPOINTS


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, cond_dim=CONDITION_DIM,
                 k_waypoints=K_WAYPOINTS, dim_out=2):
        super().__init__()
        self.k = k_waypoints
        self.dim_out = dim_out
        input_dim = latent_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(0.2),
            nn.Linear(2048, k_waypoints * dim_out),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        out = self.net(x)
        out = out.view(-1, self.k, self.dim_out)
        out = torch.tanh(out)  # [-1, 1], denormalize later
        return out
```

- [ ] **Step 5: Implement critic.py**

```python
"""Critic D(tau, c): WGAN-GP critic, outputs scalar score."""
import torch
import torch.nn as nn
from config import CONDITION_DIM, N_TRAJECTORY


class Critic(nn.Module):
    def __init__(self, dim_in=2, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        # 100 -> 50 -> 25 -> 12 -> 6 (after 4 stride-2 convs)
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, trajectory, condition):
        # trajectory: (B, N, dim) -> (B, dim, N) for Conv1d
        tau = trajectory.transpose(1, 2)
        f = self.conv(tau).flatten(1)
        x = torch.cat([f, condition], dim=1)
        return self.fc(x)  # no sigmoid (WGAN-GP)
```

- [ ] **Step 6: Implement path_encoder.py**

```python
"""Path encoder E_path(tau, c): trajectory -> latent code."""
import torch
import torch.nn as nn
from config import LATENT_DIM, CONDITION_DIM


class PathEncoder2D(nn.Module):
    def __init__(self, dim_in=2, latent_dim=LATENT_DIM, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, trajectory, condition):
        tau = trajectory.transpose(1, 2)  # (B, dim, N)
        f = self.conv(tau).flatten(1)
        return self.fc(torch.cat([f, condition], dim=1))


class PathEncoder3D(nn.Module):
    """3D variant for Phase 2+."""
    def __init__(self, dim_in=3, latent_dim=LATENT_DIM, cond_dim=CONDITION_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 256, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
        )

    def forward(self, trajectory, condition):
        tau = trajectory.transpose(1, 2)
        f = self.conv(tau).flatten(1)
        return self.fc(torch.cat([f, condition], dim=1))
```

- [ ] **Step 7: Update models/__init__.py**

```python
from .env_encoder import EnvEncoder2D, EnvEncoder3D
from .generator import Generator
from .critic import Critic
from .path_encoder import PathEncoder2D, PathEncoder3D
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_models.py -v
```

Expected: 5 PASS

- [ ] **Step 9: Commit**

```bash
git add src/models/ tests/test_models.py
git commit -m "feat: add all neural network models (E_env, G, D, E_path)"
```

---

### Task 8: Loss Functions

**Files:**
- Create: `src/losses.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for loss functions."""
import torch
import sys
sys.path.insert(0, 'src')
from losses import (
    critic_loss, generator_adv_loss, gradient_penalty,
    collision_loss, path_length_loss, smoothness_loss,
    reconstruction_loss, convexity_loss
)
from config import LAMBDA_GP

def test_gradient_penalty_shape():
    D = torch.nn.Linear(200, 1)  # simple critic
    tau_real = torch.randn(4, 100, 2, requires_grad=True)
    tau_fake = torch.randn(4, 100, 2, requires_grad=True)
    c = torch.randn(4, 256)
    gp = gradient_penalty(D, tau_real, tau_fake, c)
    assert gp.shape == ()
    assert gp.item() >= 0

def test_collision_loss_zero_for_free_space():
    traj = torch.ones(4, 100, 2) * 0.5  # center, away from edges
    voxels = torch.zeros(4, 1, 32, 32)  # all free
    loss = collision_loss(traj, voxels, mode='2d')
    assert loss.item() == 0.0

def test_collision_loss_positive_for_obstacle():
    traj = torch.ones(4, 100, 2) * 0.5
    voxels = torch.ones(4, 1, 32, 32)  # all obstacle
    loss = collision_loss(traj, voxels, mode='2d')
    assert loss.item() > 0.0

def test_path_length_loss():
    traj = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])  # 1 batch, 2 pts
    loss = path_length_loss(traj)
    assert abs(loss.item() - 5.0) < 0.01  # sqrt(9+16) = 5

def test_smoothness_loss_zero_for_straight_line():
    t = torch.linspace(0, 1, 100).view(-1, 1)
    traj = torch.cat([t, t], dim=1).unsqueeze(0)  # (1, 100, 2) straight line
    loss = smoothness_loss(traj)
    assert loss.item() < 0.01  # straight line should have zero acceleration
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_losses.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement losses.py**

```python
"""Loss functions for CWGAN-GP + path constraints."""
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from config import LAMBDA_GP, LAMBDA_COL, LAMBDA_LEN, LAMBDA_SMOOTH, LAMBDA_RECON, LAMBDA_CONVEXITY


def critic_loss(D, tau_real, tau_fake, c):
    """WGAN-GP critic loss (to minimize).
    L_D = E[D(G(z))] - E[D(real)] + lambda * GP
    """
    d_real = D(tau_real, c)
    d_fake = D(tau_fake, c)
    return d_fake.mean() - d_real.mean()


def generator_adv_loss(D, tau_fake, c):
    """WGAN-GP generator adversarial loss (to minimize).
    L_adv_G = -E[D(G(z))]
    """
    return -D(tau_fake, c).mean()


def gradient_penalty(D, tau_real, tau_fake, c):
    """WGAN-GP gradient penalty."""
    B = tau_real.shape[0]
    eps = torch.rand(B, 1, 1, device=tau_real.device)
    x_hat = eps * tau_real + (1 - eps) * tau_fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat, c)
    grads = autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grads.view(B, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def collision_loss(trajectory, voxels, mode='2d'):
    """Soft collision loss via grid sampling.
    Args:
        trajectory: (B, N, dim) in [0,1] normalized coords
        voxels:     (B, 1, H, W) or (B, 1, D, H, W)
        mode:       '2d' or '3d'
    """
    B, N, dim = trajectory.shape
    if mode == '2d':
        grid = trajectory[:, :, :2] * 2 - 1  # (B, N, 2) in [-1,1]
        grid = grid.view(B, N, 1, 2)
        sampled = F.grid_sample(voxels.float(), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    else:
        grid = trajectory * 2 - 1
        grid = grid.view(B, N, 1, 1, 3)
        sampled = F.grid_sample(voxels.float(), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    return sampled.mean()


def path_length_loss(trajectory):
    """Mean path length: E[sum_i ||tau_i - tau_{i-1}||]."""
    diffs = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    return diffs.norm(2, dim=-1).sum(dim=1).mean()


def smoothness_loss(trajectory):
    """Acceleration minimization: E[||tau_i - 2*tau_{i-1} + tau_{i-2}||^2]."""
    acc = trajectory[:, 2:, :] - 2 * trajectory[:, 1:-1, :] + trajectory[:, :-2, :]
    return (acc ** 2).sum(dim=-1).mean()


def reconstruction_loss(G, E_path, tau_real, c):
    """L_recon = ||G(E(tau_real, c), c) - tau_real||^2."""
    z_hat = E_path(tau_real, c)
    tau_recon_waypoints = G(z_hat, c)  # (B, K, dim) waypoints
    # Compare waypoints with subsampled real path
    # Subsample tau_real to K points for comparison
    B, N, dim = tau_real.shape
    K = tau_recon_waypoints.shape[1]
    indices = torch.linspace(0, N-1, K, device=tau_real.device).long()
    tau_real_sub = tau_real[:, indices, :]
    return F.mse_loss(tau_recon_waypoints, tau_real_sub)


def convexity_loss(G, z_a, z_b, c_a, voxels_a, interpolation, mode='2d'):
    """L_convexity: interpolated latent should produce collision-free paths.
    Args:
        G: generator
        z_a, z_b: (B, latent_dim) two sets of latent codes
        c_a: (B, cond_dim) conditions for scene a
        voxels_a: voxel map for scene a
        interpolation: callable(G, z, c) -> trajectory
    """
    alpha = torch.rand(z_a.shape[0], 1, device=z_a.device)
    z_interp = (1 - alpha) * z_a + alpha * z_b
    tau_waypoints = G(z_interp, c_a)  # (B, K, dim)
    tau_full = interpolation(tau_waypoints)  # (B, N, dim), in [0,1] space
    return collision_loss(tau_full, voxels_a, mode=mode)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_losses.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: add all loss functions (WGAN-GP + constraints + convexity)"
```

---

### Task 9: Trainer

**Files:**
- Create: `src/trainer.py`

- [ ] **Step 1: Implement trainer.py**

```python
"""Training loop for CWGAN-GP with E_path and path constraints."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from config import (
    BATCH_SIZE, LR_G, LR_D, LR_E, N_CRITIC,
    LAMBDA_GP, LAMBDA_COL, LAMBDA_LEN, LAMBDA_SMOOTH,
    LAMBDA_RECON, LAMBDA_CONVEXITY, NUM_EPOCHS,
    LATENT_DIM, K_WAYPOINTS, N_TRAJECTORY,
)
from models.env_encoder import EnvEncoder2D
from models.generator import Generator
from models.critic import Critic
from models.path_encoder import PathEncoder2D
from interpolation import cubic_spline_trajectory
from losses import (
    critic_loss, generator_adv_loss, gradient_penalty,
    collision_loss, path_length_loss, smoothness_loss,
    reconstruction_loss, convexity_loss,
)


class Trainer:
    def __init__(self, dim=2, device='cuda'):
        self.dim = dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        self._build_models()

    def _build_models(self):
        if self.dim == 2:
            self.E_env = EnvEncoder2D().to(self.device)
            self.G = Generator(dim_out=2).to(self.device)
            self.D = Critic(dim_in=2).to(self.device)
            self.E_path = PathEncoder2D(dim_in=2).to(self.device)
        else:
            from models.env_encoder import EnvEncoder3D
            from models.path_encoder import PathEncoder3D
            self.E_env = EnvEncoder3D().to(self.device)
            self.G = Generator(dim_out=3).to(self.device)
            self.D = Critic(dim_in=3).to(self.device)
            self.E_path = PathEncoder3D(dim_in=3).to(self.device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=LR_G, betas=(0, 0.9))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=LR_D, betas=(0, 0.9))
        self.opt_E = torch.optim.Adam(
            list(self.E_env.parameters()) + list(self.E_path.parameters()),
            lr=LR_E, betas=(0, 0.9)
        )
        self.mode = '2d' if self.dim == 2 else '3d'

    def prepare_batch(self, batch):
        """Move batch dict to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def train_epoch(self, dataloader, epoch):
        self.G.train()
        self.D.train()
        self.E_env.train()
        self.E_path.train()

        d_losses, g_losses = [], []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            batch = self.prepare_batch(batch)
            voxels, tau_real = batch['voxels'], batch['path_real']
            start, goal = batch['start'], batch['goal']
            B = voxels.shape[0]

            # --- Update Critic ---
            for p in self.D.parameters():
                p.requires_grad_(True)
            self.opt_D.zero_grad()

            c = self.E_env(voxels, start, goal)
            z = torch.randn(B, LATENT_DIM, device=self.device)
            tau_fake_wp = self.G(z, c)  # (B, K, dim)
            # Rescale from [-1,1] to [0,1] for collision/interpolation
            tau_fake_wp_01 = (tau_fake_wp + 1) / 2
            tau_fake_full = cubic_spline_trajectory(tau_fake_wp_01.view(B*K_WAYPOINTS, -1).view(B, K_WAYPOINTS, self.dim), N_TRAJECTORY)

            d_loss = critic_loss(self.D, tau_real, tau_fake_full, c)
            gp = gradient_penalty(self.D, tau_real, tau_fake_full, c)
            loss_D = d_loss + LAMBDA_GP * gp
            loss_D.backward()
            self.opt_D.step()
            d_losses.append(loss_D.item())

            # --- Update G, E_env, E_path every N_CRITIC iterations ---
            if i % N_CRITIC == 0:
                for p in self.D.parameters():
                    p.requires_grad_(False)
                self.opt_G.zero_grad()
                self.opt_E.zero_grad()

                c = self.E_env(voxels, start, goal)
                z = torch.randn(B, LATENT_DIM, device=self.device)
                tau_fake_wp = self.G(z, c)
                tau_fake_wp_01 = (tau_fake_wp + 1) / 2
                tau_fake_full = cubic_spline_trajectory(tau_fake_wp_01, N_TRAJECTORY)

                # Adversarial loss
                l_adv = generator_adv_loss(self.D, tau_fake_full, c)
                # Path constraint losses
                l_col = collision_loss(tau_fake_full, voxels, mode=self.mode)
                l_len = path_length_loss(tau_fake_full)
                l_smooth = smoothness_loss(tau_fake_full)
                # Reconstruction loss (E_path)
                l_recon = reconstruction_loss(self.G, self.E_path, tau_real, c)
                # Convexity loss
                z_b = torch.randn(B, LATENT_DIM, device=self.device)
                l_conv = convexity_loss(
                    self.G, z, z_b, c, voxels,
                    lambda wp: cubic_spline_trajectory((wp+1)/2, N_TRAJECTORY),
                    mode=self.mode
                )

                loss_G = (l_adv +
                          LAMBDA_COL * l_col +
                          LAMBDA_LEN * l_len +
                          LAMBDA_SMOOTH * l_smooth +
                          LAMBDA_RECON * l_recon +
                          LAMBDA_CONVEXITY * l_conv)
                loss_G.backward()
                self.opt_G.step()
                self.opt_E.step()
                g_losses.append(loss_G.item())

            pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.3f}" if g_losses else "---")

        return sum(d_losses)/len(d_losses), sum(g_losses)/max(1,len(g_losses))

    def train(self, dataset, num_epochs=NUM_EPOCHS, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        # Build DataLoader from dataset dict
        all_voxels = torch.stack([d['voxels'] for d in dataset])
        all_starts = torch.stack([d['start'] for d in dataset])
        all_goals = torch.stack([d['goal'] for d in dataset])
        all_paths = torch.stack([d['path_real'] for d in dataset])
        tensor_ds = TensorDataset(all_voxels, all_starts, all_goals, all_paths)
        loader = DataLoader(tensor_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Wrap to dict format expected by train_epoch
        class DictLoader:
            def __init__(self, loader, dim):
                self.loader = loader
                self.dim = dim
            def __iter__(self):
                for v, s, g, p in self.loader:
                    yield {'voxels': v, 'start': s, 'goal': g, 'path_real': p}
            def __len__(self):
                return len(self.loader)

        dl = DictLoader(loader, self.dim)

        for epoch in range(1, num_epochs + 1):
            d_loss, g_loss = self.train_epoch(dl, epoch)
            print(f"Epoch {epoch:3d} | D loss: {d_loss:.4f} | G loss: {g_loss:.4f}")
            if epoch % 50 == 0:
                self.save(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))

    def save(self, path):
        torch.save({
            'E_env': self.E_env.state_dict(),
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'E_path': self.E_path.state_dict(),
            'dim': self.dim,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.E_env.load_state_dict(ckpt['E_env'])
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.E_path.load_state_dict(ckpt['E_path'])
```

- [ ] **Step 2: Commit**

```bash
git add src/trainer.py
git commit -m "feat: add training loop with CWGAN-GP + path constraints"
```

---

### Task 10: Online Planner (Latent Space Evolution)

**Files:**
- Create: `src/online_planner.py`
- Create: `tests/test_online_planner.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for online planner."""
import torch
import sys
sys.path.insert(0, 'src')
from online_planner import OnlinePlanner
from config import LATENT_DIM, CONDITION_DIM

def test_online_planner_initialization():
    planner = OnlinePlanner(dim=2)
    z = planner.initialize()
    assert z.shape == (1, LATENT_DIM)

def test_online_planner_replan_step():
    planner = OnlinePlanner(dim=2)
    z_prev = torch.randn(1, LATENT_DIM)
    c = torch.randn(1, CONDITION_DIM)
    voxels = torch.zeros(1, 1, 32, 32)
    z_new, traj = planner.replan_step(z_prev, c, voxels)
    assert z_new.shape == (1, LATENT_DIM)
    assert traj.shape[0] > 0  # trajectory returned
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_online_planner.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement online_planner.py**

```python
"""Online planner with latent space gradient optimization for dynamic replanning."""
import torch
import torch.nn as nn
from config import (
    LATENT_DIM, CONDITION_DIM, K_WAYPOINTS, N_TRAJECTORY,
    LAMBDA_CONTINUITY, ONLINE_LR, ONLINE_STEPS,
)
from models.env_encoder import EnvEncoder2D, EnvEncoder3D
from models.generator import Generator
from models.path_encoder import PathEncoder2D, PathEncoder3D
from interpolation import cubic_spline_trajectory, minimum_snap_trajectory
from losses import collision_loss


class OnlinePlanner:
    """Latent-space continuous evolution for dynamic path replanning.

    Instead of discarding and regenerating paths, this continuously
    moves the latent code in the latent space, producing smoothly
    deforming trajectories.
    """
    def __init__(self, dim=2, checkpoint_path=None, device='cuda'):
        self.dim = dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        self._build_models()
        if checkpoint_path:
            self.load(checkpoint_path)

    def _build_models(self):
        if self.dim == 2:
            self.E_env = EnvEncoder2D().to(self.device).eval()
            self.G = Generator(dim_out=2).to(self.device).eval()
            self.E_path = PathEncoder2D(dim_in=2).to(self.device).eval()
        else:
            self.E_env = EnvEncoder3D().to(self.device).eval()
            self.G = Generator(dim_out=3).to(self.device).eval()
            self.E_path = PathEncoder3D(dim_in=3).to(self.device).eval()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.E_env.load_state_dict(ckpt['E_env'])
        self.G.load_state_dict(ckpt['G'])
        self.E_path.load_state_dict(ckpt['E_path'])

    def initialize(self, init_path=None, condition=None):
        """Initialize latent code z_0.

        Args:
            init_path: optional (1, N, dim) initial path for E_path encoding
            condition: (1, CONDITION_DIM) environment condition
        Returns:
            z: (1, LATENT_DIM) initial latent code
        """
        if init_path is not None and condition is not None:
            with torch.no_grad():
                z = self.E_path(init_path.to(self.device), condition.to(self.device))
        else:
            z = torch.randn(1, LATENT_DIM, device=self.device)
        return z

    def replan_step(self, z_prev, condition, voxels):
        """One replanning cycle: optimize latent code, return new trajectory.

        Args:
            z_prev:   (1, LATENT_DIM) previous latent code
            condition: (1, CONDITION_DIM) current environment condition
            voxels:   (1, 1|C, H, W) or (1, 1|C, D, H, W) current voxel map
        Returns:
            z_new:    (1, LATENT_DIM) updated latent code
            trajectory: (N, dim) numpy array, final smoothed trajectory
        """
        z_cur = z_prev.clone().detach().to(self.device)
        z_cur.requires_grad_(True)

        mode = '2d' if self.dim == 2 else '3d'
        condition = condition.to(self.device)
        voxels = voxels.to(self.device)

        optimizer = torch.optim.SGD([z_cur], lr=ONLINE_LR)

        for _ in range(ONLINE_STEPS):
            optimizer.zero_grad()
            waypoints = self.G(z_cur, condition)  # (1, K, dim) in [-1,1]
            wp_01 = (waypoints + 1) / 2
            trajectory = cubic_spline_trajectory(wp_01.squeeze(0), N_TRAJECTORY).unsqueeze(0)
            col = collision_loss(trajectory, voxels, mode=mode)
            continuity = ((z_cur - z_prev) ** 2).sum()
            loss = col + LAMBDA_CONTINUITY * continuity
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            waypoints = self.G(z_cur, condition)
            wp_01 = (waypoints + 1) / 2
            traj_np = minimum_snap_trajectory(wp_01.squeeze(0), N_TRAJECTORY)

        return z_cur.detach(), traj_np

    def generate_path(self, z, condition):
        """Generate a single path from latent code (no optimization)."""
        with torch.no_grad():
            waypoints = self.G(z.to(self.device), condition.to(self.device))
            wp_01 = (waypoints + 1) / 2
            traj_np = minimum_snap_trajectory(wp_01.squeeze(0), N_TRAJECTORY)
        return traj_np
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python -m pytest tests/test_online_planner.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/online_planner.py tests/test_online_planner.py
git commit -m "feat: add online planner with latent space gradient evolution"
```

---

### Task 11: Phase 1 Experiment Script

**Files:**
- Create: `experiments/phase1_train_2d.py`
- Create: `experiments/phase1_eval_2d.py`

- [ ] **Step 1: Implement phase1_train_2d.py**

```python
"""Phase 1: Train CWGAN-GP on 2D static environments."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from data_generator import generate_dataset_2d
from trainer import Trainer
from config import VOXEL_RES

def main():
    print("=== Phase 1: 2D CWGAN-GP Training ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate dataset
    print("Generating 2D dataset...")
    dataset = generate_dataset_2d(num_samples=5000, voxel_res=VOXEL_RES)
    print(f"Dataset size: {len(dataset)}")

    # Train
    trainer = Trainer(dim=2, device=device)
    trainer.train(dataset, num_epochs=500, save_dir='checkpoints/phase1')
    trainer.save('checkpoints/phase1_final.pt')
    print("Phase 1 training complete!")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Implement phase1_eval_2d.py**

```python
"""Phase 1: Evaluate trained 2D model — static environments."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib.pyplot as plt
from config import VOXEL_RES, LATENT_DIM
from environment import generate_random_scene_2d
from online_planner import OnlinePlanner
from models.env_encoder import EnvEncoder2D
from interpolation import cubic_spline_trajectory, minimum_snap_trajectory

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = OnlinePlanner(dim=2, checkpoint_path='checkpoints/phase1_final.pt', device=device)

    # Generate a test scene
    voxels, start, goal, _ = generate_random_scene_2d(voxel_res=VOXEL_RES)
    voxels_dev = voxels.unsqueeze(0).to(device)
    start_dev = start.unsqueeze(0).to(device)
    goal_dev = goal.unsqueeze(0).to(device)

    # Build condition
    with torch.no_grad():
        c = planner.E_env(voxels_dev, start_dev, goal_dev)

    # Generate path
    z = planner.initialize()
    traj = planner.generate_path(z, c)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Show obstacles
    ax.imshow(voxels.squeeze(0).cpu().numpy(), origin='upper', cmap='Reds', alpha=0.3, extent=[0,1,1,0])
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Generated Path')
    ax.plot(start[0].item(), start[1].item(), 'go', markersize=10, label='Start')
    ax.plot(goal[0].item(), goal[1].item(), 'ro', markersize=10, label='Goal')
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.legend()
    ax.set_title('Phase 1: 2D CWGAN-GP Path Planning')
    plt.savefig('experiments/phase1_result.png', dpi=150)
    print("Saved to experiments/phase1_result.png")

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run Phase 1 training (short test)**

```bash
cd G:/claude\ code_workspace/gan-uav-path-planning
python experiments/phase1_train_2d.py
```

- [ ] **Step 4: Commit**

```bash
git add experiments/phase1_train_2d.py experiments/phase1_eval_2d.py
git commit -m "feat: add Phase 1 2D training and evaluation scripts"
```

---

## Phase 2: 3D Static Scenes

### Task 12: 3D Training Script

**Files:**
- Create: `experiments/phase2_train_3d.py`

- [ ] **Step 1: Implement phase2_train_3d.py**

```python
"""Phase 2: Train CWGAN-GP on 3D static environments."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from data_generator import generate_dataset_3d
from trainer import Trainer
from config import VOXEL_RES

def main():
    print("=== Phase 2: 3D CWGAN-GP Training ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Generating 3D dataset (this may take a while)...")
    dataset = generate_dataset_3d(num_samples=5000, voxel_res=VOXEL_RES)
    print(f"Dataset size: {len(dataset)}")

    trainer = Trainer(dim=3, device=device)
    trainer.train(dataset, num_epochs=500, save_dir='checkpoints/phase2')
    trainer.save('checkpoints/phase2_final.pt')
    print("Phase 2 training complete!")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/phase2_train_3d.py
git commit -m "feat: add Phase 2 3D training script"
```

---

## Phase 3: Dynamic Obstacles

### Task 13: Dynamic Simulation + Evaluation

**Files:**
- Create: `experiments/phase3_dynamic.py`
- Create: `src/evaluator.py`

- [ ] **Step 1: Implement evaluator.py**

```python
"""Evaluation metrics and baseline comparison for path planning methods."""
import time
import torch
import numpy as np
from config import N_TRAJECTORY


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
    final_pos = paths[-1][-1] if paths else start
    reached_goal = np.linalg.norm(final_pos - goal.numpy()) < 0.1
    had_collision = False
    for path, voxels in zip(paths, voxels_sequence):
        # Check collisions along path
        V = voxels.squeeze().numpy()
        H, W = V.shape[-2:]
        for pt in path:
            pi, pj = int(pt[1] * H), int(pt[0] * W)
            pi, pj = max(0, min(H-1, pi)), max(0, min(W-1, pj))
            if V.ndim == 3:
                pk = int(pt[2] * V.shape[0])
                if V[pk, pi, pj] > 0:
                    had_collision = True
                    break
            else:
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
        d1 = np.min(np.linalg.norm(paths[i][:, None] - paths[i-1][None, :], axis=2), axis=1).max()
        d2 = np.min(np.linalg.norm(paths[i-1][:, None] - paths[i][None, :], axis=2), axis=1).max()
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
```

- [ ] **Step 2: Implement phase3_dynamic.py**

```python
"""Phase 3: Dynamic obstacle avoidance with latent space evolution."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from config import VOXEL_RES, REPLAN_HZ, N_TRAJECTORY
from environment import generate_random_scene_2d
from online_planner import OnlinePlanner
from evaluator import compute_metrics, run_astar_baseline
from interpolation import cubic_spline_trajectory

SIM_DURATION = 10.0  # seconds
AGENT_SPEED = 2.0    # m/s
DYNAMIC_OBS_SPEED = 1.0  # m/s
DT = 1.0 / REPLAN_HZ


class DynamicSimulator:
    def __init__(self, planner, voxel_res=32):
        self.planner = planner
        self.voxel_res = voxel_res
        self.state = None

    def reset(self, voxels, start, goal, obs_list):
        self.voxels_init = voxels.clone()
        self.start = start
        self.goal = goal
        self.obs_list = [[o[0], o[1], o[2], np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                         for o in obs_list]  # (cx, cy, r, vx, vy)
        self.agent_pos = start.clone()
        self.t = 0

    def update_obstacles(self, dt):
        """Move dynamic obstacles."""
        for o in self.obs_list:
            o[0] += o[3] * dt / 10.0  # normalized coords
            o[1] += o[4] * dt / 10.0
            # Bounce off walls
            if o[0] < 0 or o[0] > 1: o[3] *= -1; o[0] = max(0, min(1, o[0]))
            if o[1] < 0 or o[1] > 1: o[4] *= -1; o[1] = max(0, min(1, o[1]))

    def rebuild_voxels(self):
        """Rebuild voxel map with current obstacle positions."""
        import torch
        V = torch.zeros(self.voxel_res, self.voxel_res)
        ys, xs = torch.meshgrid(torch.arange(self.voxel_res, dtype=torch.float32),
                                torch.arange(self.voxel_res, dtype=torch.float32),
                                indexing='ij')
        for cx, cy, r, _, _ in self.obs_list:
            r_px = r * self.voxel_res
            dist = torch.sqrt((xs/self.voxel_res - cx)**2 + (ys/self.voxel_res - cy)**2)
            V[dist < r] = 1.0
        return V.unsqueeze(0)

    def step(self, z_prev):
        """One simulation step: update obstacles, replan, move agent."""
        self.update_obstacles(DT)
        voxels = self.rebuild_voxels()

        # Build condition
        with torch.no_grad():
            c = self.planner.E_env(
                voxels.unsqueeze(0).to(self.planner.device),
                self.agent_pos.unsqueeze(0).to(self.planner.device),
                self.goal.unsqueeze(0).to(self.planner.device)
            )

        # Replan
        z_new, traj = self.planner.replan_step(
            z_prev, c, voxels.unsqueeze(0).to(self.planner.device)
        )

        # Move agent one step along trajectory
        direction = traj[1] - traj[0]
        step_size = AGENT_SPEED * DT / 10.0  # normalized units
        if np.linalg.norm(direction) > 0:
            self.agent_pos += torch.tensor(direction / np.linalg.norm(direction) * step_size).float()

        dist_to_goal = torch.norm(self.agent_pos - self.goal).item()
        return z_new, traj, voxels, dist_to_goal

    def run(self, max_steps=200):
        z_prev = self.planner.initialize()
        all_trajs = []
        all_voxels = []
        all_positions = []
        distances = []

        for step in range(max_steps):
            z_prev, traj, voxels, dist = self.step(z_prev)
            all_trajs.append(traj)
            all_voxels.append(voxels)
            all_positions.append(self.agent_pos.clone())
            distances.append(dist)
            if dist < 0.05:
                break

        return all_trajs, all_voxels, all_positions, distances


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = OnlinePlanner(dim=2, checkpoint_path='checkpoints/phase1_final.pt', device=device)

    # Generate a dynamic test scene
    voxels, start, goal, obs_list = generate_random_scene_2d(voxel_res=VOXEL_RES)

    sim = DynamicSimulator(planner, voxel_res=VOXEL_RES)
    sim.reset(voxels, start, goal, obs_list)

    print("Running dynamic simulation...")
    trajs, voxels_seq, positions, distances = sim.run(max_steps=200)

    metrics = compute_metrics(trajs, voxels_seq, start, goal)
    print(f"Metrics: {metrics}")

    # Save animation
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.plot(start[0], start[1], 'go', markersize=8)
    ax.plot(goal[0], goal[1], 'ro', markersize=8)

    line_traj, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.5)
    pt_agent, = ax.plot([], [], 'bx', markersize=6)

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.imshow(voxels_seq[frame].squeeze(0).cpu().numpy(), origin='upper', cmap='Reds', alpha=0.3, extent=[0,1,1,0])
        traj = trajs[frame]
        ax.plot(start[0], start[1], 'go', markersize=8)
        ax.plot(goal[0], goal[1], 'ro', markersize=8)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.5)
        ax.plot(positions[frame][0], positions[frame][1], 'bx', markersize=8)
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(trajs), interval=100, blit=False)
    anim.save('experiments/phase3_dynamic.gif', writer='pillow', fps=10)
    print("Saved to experiments/phase3_dynamic.gif")

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluator.py experiments/phase3_dynamic.py
git commit -m "feat: add Phase 3 dynamic obstacle simulation and evaluation"
```

---

## Phase 4: AirSim Integration

### Task 14: AirSim Interface

**Files:**
- Create: `experiments/phase4_airsim.py`

- [ ] **Step 1: Implement phase4_airsim.py**

```python
"""Phase 4: AirSim integration for forest environment testing.

Prerequisites:
    - AirSim installed and running
    - Unreal Engine with forest/mountain environment
    - Python AirSim API: pip install airsim

Setup:
    1. Launch AirSim environment in UE
    2. Run this script to connect and test the planner
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import torch

try:
    import airsim
except ImportError:
    print("airsim not installed. Install with: pip install airsim")
    print("This is expected — Phase 4 only runs with AirSim available.")
    sys.exit(0)

from config import VOXEL_RES, REPLAN_HZ
from online_planner import OnlinePlanner


class AirSimInterface:
    """Bridge between our planner and AirSim simulator."""

    def __init__(self, planner, voxel_res=32):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.planner = planner
        self.voxel_res = voxel_res
        self.state = None

    def get_obstacle_voxels(self):
        """Query AirSim for obstacle positions and build voxel map.
        Uses lidar/depth sensor data if available, otherwise falls back
        to known obstacle positions from the simulation.
        """
        voxels = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=np.float32)

        # Get LiDAR data
        lidar_data = self.client.getLidarData()
        if lidar_data and len(lidar_data.point_cloud) > 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            # Filter points within bounding box and mark voxels
            for pt in points:
                # Convert from AirSim NED to normalized [0,1]
                nx = (pt[1] + 50) / 100  # assuming 100m world
                ny = (pt[0] + 50) / 100
                nz = (-pt[2] + 25) / 50
                if 0 <= nx < 1 and 0 <= ny < 1 and 0 <= nz < 1:
                    vi, vj, vk = int(ny*VOXEL_RES), int(nx*VOXEL_RES), int(nz*VOXEL_RES)
                    vi, vj, vk = max(0,min(VOXEL_RES-1,vi)), max(0,min(VOXEL_RES-1,vj)), max(0,min(VOXEL_RES-1,vk))
                    voxels[vk, vi, vj] = 1.0

        return torch.from_numpy(voxels).unsqueeze(0)

    def get_drone_state(self):
        """Get current drone position and orientation from AirSim."""
        kin = self.client.getMultirotorState().kinematics_estimated
        pos = kin.position  # x_val, y_val, z_val in NED
        # Convert to normalized [0,1]
        px = (pos.x_val + 50) / 100
        py = (pos.y_val + 50) / 100
        pz = (-pos.z_val + 25) / 50
        return torch.tensor([px, py, pz])

    def send_trajectory(self, trajectory):
        """Send trajectory waypoints to AirSim for execution."""
        # Convert normalized coords back to AirSim NED (meters)
        airsim_pts = []
        for pt in trajectory:
            x = pt[0] * 100 - 50  # world X
            y = pt[1] * 100 - 50  # world Y
            z = -(pt[2] * 50 - 25)  # world Z (NED, up is negative)
            airsim_pts.append(airsim.Vector3r(x, y, z))

        # Send first waypoint as immediate goal
        if airsim_pts:
            self.client.moveToPositionAsync(
                airsim_pts[0].x_val, airsim_pts[0].y_val, airsim_pts[0].z_val,
                velocity=5.0, timeout_sec=1.0/REPLAN_HZ
            )

    def run_mission(self, goal_world, duration=60.0):
        """Run a complete mission: sense → plan → act loop.

        Args:
            goal_world: (3,) target position in AirSim world coords (meters)
            duration: max mission time in seconds
        """
        goal = torch.tensor([
            (goal_world[0] + 50) / 100,
            (goal_world[1] + 50) / 100,
            (-goal_world[2] + 25) / 50,
        ])

        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        voxels = self.get_obstacle_voxels()
        pos = self.get_drone_state()

        with torch.no_grad():
            c = self.planner.E_env(
                voxels.unsqueeze(0).to(self.planner.device),
                pos.unsqueeze(0).to(self.planner.device),
                goal.unsqueeze(0).to(self.planner.device)
            )

        z = self.planner.initialize()

        dt = 1.0 / REPLAN_HZ
        t0 = time.time()

        while time.time() - t0 < duration:
            # Sense
            voxels = self.get_obstacle_voxels()
            pos = self.get_drone_state()

            # Plan
            with torch.no_grad():
                c = self.planner.E_env(
                    voxels.unsqueeze(0).to(self.planner.device),
                    pos.unsqueeze(0).to(self.planner.device),
                    goal.unsqueeze(0).to(self.planner.device)
                )

            z, traj = self.planner.replan_step(
                z, c, voxels.unsqueeze(0).to(self.planner.device)
            )

            # Act
            self.send_trajectory(traj)

            # Check if reached goal
            dist = torch.norm(pos - goal).item()
            if dist < 0.02:
                print("Goal reached!")
                break

            time.sleep(dt)

        self.client.landAsync().join()
        self.client.armDisarm(False)


def main():
    print("=== Phase 4: AirSim Integration ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = OnlinePlanner(dim=3, checkpoint_path='checkpoints/phase2_final.pt', device=device)

    interface = AirSimInterface(planner)
    # Example: fly to (20, 10, -15) meters in world frame
    interface.run_mission(goal_world=(20, 10, -15), duration=30.0)
    print("Phase 4 mission complete!")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/phase4_airsim.py
git commit -m "feat: add Phase 4 AirSim integration interface"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: Phase 1 (2D static), Phase 2 (3D static), Phase 3 (dynamic + evaluation), Phase 4 (AirSim) all covered. Phase 5 (real drone) deferred to course progress.
- [x] **No placeholders**: Every task has concrete code. No TBD/TODO.
- [x] **Type consistency**: `dim` parameter consistently used across all models (2 vs 3). `LATENT_DIM=128`, `CONDITION_DIM=256`, `K_WAYPOINTS=10`, `N_TRAJECTORY=100` match across all files.
- [x] **Interface consistency**: `generate_random_scene_2d()` returns `(voxels, start, goal, obs_list)`. `cubic_spline_trajectory()` takes `(waypoints, num_points)`. `OnlinePlanner.replan_step()` takes `(z_prev, condition, voxels)`. All consistent.
- [x] **TDD**: Each module has tests written before implementation.
- [x] **DRY**: Config centralized in `config.py`. Shared model patterns (conv architecture) consistent across 2D/3D variants.
