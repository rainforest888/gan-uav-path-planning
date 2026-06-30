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

SIM_DURATION = 15.0  # seconds (longer run)
DYNAMIC_OBS_SPEED = 0.3  # m/s (slower obstacles for debugging)
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
        self.obs_list = [[o[0], o[1], o[2],
                          np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
                         for o in obs_list]  # (cx, cy, r, vx, vy)
        self.agent_pos = start.clone()
        self.t = 0

    def update_obstacles(self, dt):
        """Move dynamic obstacles."""
        for o in self.obs_list:
            o[0] += o[3] * dt / 10.0  # normalized coords
            o[1] += o[4] * dt / 10.0
            # Bounce off walls
            if o[0] < 0 or o[0] > 1:
                o[3] *= -1; o[0] = max(0, min(1, o[0]))
            if o[1] < 0 or o[1] > 1:
                o[4] *= -1; o[1] = max(0, min(1, o[1]))

    def rebuild_voxels(self):
        """Rebuild voxel map with current obstacle positions."""
        V = torch.zeros(self.voxel_res, self.voxel_res)
        ys, xs = torch.meshgrid(torch.arange(self.voxel_res, dtype=torch.float32),
                                torch.arange(self.voxel_res, dtype=torch.float32),
                                indexing='ij')
        for cx, cy, r, _, _ in self.obs_list:
            dist = torch.sqrt((xs / self.voxel_res - cx) ** 2 +
                              (ys / self.voxel_res - cy) ** 2)
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

        # Follow the planned trajectory: move several waypoints along it
        pos_np = self.agent_pos.numpy()
        # Find closest point on trajectory, advance by several steps
        dists = np.linalg.norm(traj - pos_np, axis=1)
        closest_idx = int(np.argmin(dists))
        advance = 10  # jump ahead 10 trajectory points per step (faster)
        target_idx = min(closest_idx + advance, len(traj) - 1)
        target = torch.tensor(traj[target_idx], dtype=torch.float32)
        self.agent_pos = target

        dist_to_goal = torch.norm(self.agent_pos - self.goal).item()
        return z_new, traj, voxels, dist_to_goal

    def run(self, max_steps=150):
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
            if dist < 0.03:  # tighter goal threshold
                break

        return all_trajs, all_voxels, all_positions, distances


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    planner = OnlinePlanner(dim=2, checkpoint_path='checkpoints/phase1_final.pt', device=device)

    # Run multiple scenarios and aggregate
    all_metrics = []
    n_scenarios = 5
    for s in range(n_scenarios):
        voxels, start, goal, obs_list = generate_random_scene_2d(voxel_res=VOXEL_RES)
        sim = DynamicSimulator(planner, voxel_res=VOXEL_RES)
        sim.reset(voxels, start, goal, obs_list)

        trajs, voxels_seq, positions, distances = sim.run(max_steps=200)
        metrics = compute_metrics(trajs, voxels_seq, start, goal)
        all_metrics.append(metrics)
        print(f"  Scenario {s+1}: success={metrics['success']}, "
              f"length={metrics['path_length']:.1f}, "
              f"final_dist={distances[-1]:.3f}")

    # Aggregate
    successes = [m['success'] for m in all_metrics]
    lengths = [m['path_length'] for m in all_metrics]
    hausdorffs = [m['mean_hausdorff'] for m in all_metrics]
    jerks = [m['mean_jerk'] for m in all_metrics]
    print(f"\n=== Aggregate over {n_scenarios} scenarios ===")
    print(f"Success rate: {sum(successes)/len(successes):.1%}")
    print(f"Path length:  {sum(lengths)/len(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Hausdorff:    {sum(hausdorffs)/len(hausdorffs):.5f} ± {np.std(hausdorffs):.5f}")
    print(f"Jerk:         {sum(jerks)/len(jerks):.6f} ± {np.std(jerks):.6f}")

    # Save animation from last scenario
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.imshow(voxels_seq[frame].squeeze(0).cpu().numpy(), origin='upper',
                  cmap='Reds', alpha=0.3, extent=[0, 1, 1, 0])
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
