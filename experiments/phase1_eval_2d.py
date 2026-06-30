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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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
    ax.imshow(voxels.squeeze(0).cpu().numpy(), origin='upper',
              cmap='Reds', alpha=0.3, extent=[0, 1, 1, 0])
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
