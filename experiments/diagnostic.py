"""Quick diagnostic: does the trained G generate collision-free paths?"""
import sys
sys.path.insert(0, 'src')
import torch
import matplotlib.pyplot as plt
from config import VOXEL_RES, LATENT_DIM, N_TRAJECTORY, K_WAYPOINTS
from environment import generate_random_scene_2d
from models.env_encoder import EnvEncoder2D
from models.generator import Generator
from models.path_encoder import PathEncoder2D
from interpolation import cubic_spline_trajectory, minimum_snap_trajectory
from losses import collision_loss
from astar import astar_2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load models
ckpt = torch.load('checkpoints/phase1/checkpoint_epoch_50.pt', map_location=device)
E_env = EnvEncoder2D().to(device).eval()
G = Generator(dim_out=2).to(device).eval()
E_path = PathEncoder2D(dim_in=2).to(device).eval()
E_env.load_state_dict(ckpt['E_env'])
G.load_state_dict(ckpt['G'])
E_path.load_state_dict(ckpt['E_path'])

for i in range(5):
    voxels, start, goal, _ = generate_random_scene_2d(voxel_res=VOXEL_RES)

    # A* ground truth
    astar_path = astar_2d(voxels.squeeze(0), start, goal, voxel_res=VOXEL_RES)

    # Build condition
    voxels_dev = voxels.unsqueeze(0).to(device)
    start_dev = start.unsqueeze(0).to(device)
    goal_dev = goal.unsqueeze(0).to(device)
    with torch.no_grad():
        c = E_env(voxels_dev, start_dev, goal_dev)

    # Generate path from random noise
    z = torch.randn(1, LATENT_DIM, device=device)
    with torch.no_grad():
        wp = G(z, c)  # (1, K, 2)
        traj_raw = cubic_spline_trajectory(wp.squeeze(0), N_TRAJECTORY)  # (N, 2)
        # Min snap
        wp_np = minimum_snap_trajectory(wp.squeeze(0), N_TRAJECTORY)

    col = collision_loss(traj_raw.unsqueeze(0), voxels_dev, mode='2d').item()
    start_err = torch.norm(traj_raw[0] - start.to(device)).item()
    goal_err = torch.norm(traj_raw[-1] - goal.to(device)).item()

    print(f"Scene {i+1}: collision={col:.4f} | start_err={start_err:.4f} | goal_err={goal_err:.4f} "
          f"| A*_len={len(astar_path) if astar_path is not None else 'fail'}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(voxels.squeeze(0).cpu().numpy(), origin='upper', cmap='Reds', alpha=0.3, extent=[0, 1, 1, 0])
    ax.plot(traj_raw[:, 0].cpu(), traj_raw[:, 1].cpu(), 'b-', linewidth=2, label='G(z,c) raw')
    ax.plot(wp_np[:, 0], wp_np[:, 1], 'g--', linewidth=2, alpha=0.7, label='MinSnap')
    if astar_path is not None:
        ax.plot(astar_path[:, 0], astar_path[:, 1], 'k:', linewidth=1, alpha=0.5, label='A*')
    ax.plot(start[0], start[1], 'go', markersize=8)
    ax.plot(goal[0], goal[1], 'ro', markersize=8)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    ax.legend()
    ax.set_title(f'Scene {i+1}: col={col:.3f} start_err={start_err:.3f} goal_err={goal_err:.3f}')
    plt.savefig(f'experiments/diag_scene_{i+1}.png', dpi=120)
    plt.close()

print("Diagnostic plots saved to experiments/diag_scene_*.png")
