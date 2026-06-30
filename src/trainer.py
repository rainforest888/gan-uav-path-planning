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
    reconstruction_loss, convexity_loss, endpoint_loss,
)


class Trainer:
    def __init__(self, dim=2, device='cuda'):
        self.dim = dim
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else device
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

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=LR_G, betas=(0.0, 0.9))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=LR_D, betas=(0.0, 0.9))
        self.opt_E = torch.optim.Adam(
            list(self.E_env.parameters()) + list(self.E_path.parameters()),
            lr=LR_E, betas=(0.0, 0.9)
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
            tau_fake_wp = self.G(z, c)  # (B, K, dim) in [-1,1]
            tau_fake_full = cubic_spline_trajectory(tau_fake_wp, N_TRAJECTORY)

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
                tau_fake_full = cubic_spline_trajectory(tau_fake_wp, N_TRAJECTORY)

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
                    lambda wp: cubic_spline_trajectory(wp, N_TRAJECTORY),
                    mode=self.mode
                )
                # Endpoint loss: path must end at goal
                l_endp = endpoint_loss(tau_fake_full, goal)

                loss_G = (l_adv +
                          LAMBDA_COL * l_col +
                          LAMBDA_LEN * l_len +
                          LAMBDA_SMOOTH * l_smooth +
                          LAMBDA_RECON * l_recon +
                          LAMBDA_CONVEXITY * l_conv +
                          10.0 * l_endp)      # λ_endp=10 — must reach goal
                loss_G.backward()
                self.opt_G.step()
                self.opt_E.step()
                g_losses.append(loss_G.item())

            pbar.set_postfix(D=f"{loss_D.item():.3f}",
                             G=f"{g_losses[-1]:.3f}" if g_losses else "---")

        return sum(d_losses) / len(d_losses), sum(g_losses) / max(1, len(g_losses))

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
            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                for v, s, g, p in self.loader:
                    yield {'voxels': v, 'start': s, 'goal': g, 'path_real': p}

            def __len__(self):
                return len(self.loader)

        dl = DictLoader(loader)

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
