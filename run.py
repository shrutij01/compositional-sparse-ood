import numpy as np
import torch
import torch.optim as optim
import argparse

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from src.data import generate_datasets

from src.trainers import train_supervised_coding, train_unsupervised_coding

import wandb

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sparse coding model for OOD detection')
    
    # Data parameters
    parser.add_argument('--n_points', type=int, default=2000, 
                       help='Number of data points (default: 2000)')
    parser.add_argument('--n', type=int, default=100, 
                       help='Number of sources (default: 100)')
    parser.add_argument('--k', type=int, default=10, 
                       help='Sparsity level (default: 10)')
    parser.add_argument('--m', type=int, default=None, 
                       help='Compressed sensing bound (default: computed as ceil(k * log(n/k)))')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=7012025, 
                       help='Random seed (default: 7012025)')
    parser.add_argument('--num_seed', type=int, default=1, 
                       help='Number of random seeds to try (default: 1)')
    parser.add_argument('--lambda_p', type=float, default=0.1, 
                       help='L1 regularization weight (default: 0.1)')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--steps', type=int, default=100000, 
                       help='Number of training steps (default: 100000)')
    
    # Model parameters
    parser.add_argument('--supervised', type=bool, default=False, 
                       help='Supervised or unsupervised dictionary learning (default: False)')

    # Wandb parameters
    parser.add_argument('--project', type=str, default='sparse_ood', 
                       help='Wandb project name (default: sparse_ood)')
    parser.add_argument('--no_wandb', action='store_true', 
                       help='Disable wandb logging')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (default: cuda)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get the name and type of each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_type = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {gpu_name} ({gpu_type[0]}.{gpu_type[1]})")

    if not args.no_wandb:
        wandb.login()

    device = torch.device(args.device)

    if args.m is None:
        args.m = int(np.ceil(args.k * np.log(args.n/args.k) * 1))

    (train_Z_iid, train_Y_iid, train_label_iid), (val_Z_iid, val_Y_iid, val_label_iid), (Z_ood, Y_ood, label_ood), A = generate_datasets(n=args.n, k=args.k, n_samples=args.n_points, m=args.m)

    if not args.no_wandb:
        run = wandb.init(
            project=args.project,
            config={
                "learning_rate": args.lr,
                "steps": args.steps,
                "n": args.n,
                "k": args.k,
                "m": args.m,
                "n_points": args.n_points,
                "lambda_p": args.lambda_p,
                "seed": args.seed,
                "num_seed": args.num_seed,
            },
        )
    else:
        run = None

    inputs = torch.tensor(train_Y_iid, dtype=torch.float32, device=device)

    if args.supervised:
        best_D, best_Z = train_supervised_coding(
                args.seed, args.num_seed, args.lambda_p, args.lr, args.steps, 
                args.n, args.n_points, A, inputs, optim, train_Z_iid, run)
    else:
        best_D, best_Z = train_unsupervised_coding(
            args.seed, args.num_seed, args.lambda_p, args.lr, args.steps, 
            args.n, args.n_points, inputs, optim, train_Z_iid, run, args.m)

if __name__ == "__main__":
    main()