import numpy as np
import torch
import torch.optim as optim
import argparse

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from src.data import generate_datasets

from src.trainers import train_supervised_coding, train_unsupervised_coding
from src.evaluation import downstream_accuracy, accuracy_best_all

import wandb

import pandas as pd

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
    import os
    # List of lambda_p values to try
    # lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    lambda_list = [1e-3]

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_type = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {gpu_name} ({gpu_type[0]}.{gpu_type[1]})")

    device = torch.device(args.device)

    if args.m is None:
        args.m = int(np.ceil(args.k * np.log(args.n/args.k) * 1))

    (train_Z_iid, train_Y_iid, train_label_iid), (val_Z_iid, val_Y_iid, val_label_iid), (Z_ood, Y_ood, label_ood), A = generate_datasets(n=args.n, k=args.k, n_samples=args.n_points, m=args.m)

    for lambda_p in lambda_list:
        print(f"\nRunning with lambda_p={lambda_p}")
        args.lambda_p = lambda_p

        for rep in range(args.num_seed):
            actual_seed = args.seed + rep
            print(f"  Running seed {actual_seed}")

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
                        "seed": actual_seed,
                        "num_seed": 1,
                    },
                    reinit=True
                )
            else:
                run = None

            inputs = torch.tensor(train_Y_iid, dtype=torch.float32, device=device)
            val_inputs = torch.tensor(val_Y_iid, dtype=torch.float32, device=device)
            Y_ood = torch.tensor(Y_ood, dtype=torch.float32, device=device)

            if args.supervised:
                best_D, best_Z, mccs, l0s, losses = train_supervised_coding(
                    actual_seed, 1, args.lambda_p, args.lr, args.steps, 
                    args.n, args.n_points, A, inputs, optim, train_Z_iid, train_label_iid, run, val_inputs=val_inputs, val_Z_iid=val_Z_iid)
                accuracy_dict = None
            else:
                best_D, best_Z, mccs, l0s, losses, accuracy_dict = train_unsupervised_coding(
                    actual_seed, 1, args.lambda_p, args.lr, args.steps, 
                    args.n, args.n_points, A, inputs, optim, train_Z_iid, train_label_iid, run, args.m, Y_ood, label_ood, Z_ood, val_inputs=val_inputs, val_Z_iid=val_Z_iid)

            # Organize results by seed and lambda_p
            lambda_str = f"{lambda_p:.0e}" if lambda_p < 1 else str(lambda_p)
            result_dir = f"results/seed_{actual_seed}/lambda_{lambda_str}"
            os.makedirs(result_dir, exist_ok=True)

            df = pd.DataFrame()
            df["mcc"] = mccs
            df["l0"] = l0s
            df["loss"] = losses
            df.to_csv(f"{result_dir}/results.csv", index=False)
            np.save(f"{result_dir}/z.npy", best_Z)
            np.save(f"{result_dir}/D.npy", best_D)

            if accuracy_dict is not None:
                acc_df = pd.DataFrame([accuracy_dict])
                acc_df.to_csv(f"{result_dir}/accuracy.csv", index=False)

if __name__ == "__main__":
    main()