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
