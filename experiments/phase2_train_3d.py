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
