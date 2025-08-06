import numpy as np

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from tqdm import tqdm

import torch

from src.mcc import mean_corr_coef as mcc

import wandb

wandb.login()

device = torch.device('cuda')

def train_seed(log_Z, D, inputs, optim, steps, lambda_p, train_Z_iid, run):
    best_loss = np.inf
    best_D, best_Z = None, None
    for i in tqdm(range(steps)):
            Z = torch.nn.functional.softplus(log_Z)
            rec = Z @ D
            mse = torch.mean((inputs - rec)**2)
            l1 = torch.mean(torch.abs(Z) * torch.linalg.norm(D, dim=1))
            loss = mse + lambda_p * l1
                
            optim.zero_grad()
            loss.backward()
            optim.step()

            ## log and compute every 100 epochs
            if (i % 100 == 0):
                with torch.no_grad():
                    # print("MCC OOD", mcc(Z.detach().cpu().numpy(), Z_ood))
                    mcc_iid = mcc(Z.detach().cpu().numpy(), train_Z_iid)

                run.log({
                    "loss": loss.item(),
                    "mse": mse.item(),
                    "l1": l1.item(),
                    "step": i,
                    "mcc": mcc_iid
                })

            ## early stopping
            if i > 10000:
                if loss.item() > best_loss * 1.1:
                    break

            if loss.item() < best_loss:
                best_D = D.detach().cpu().numpy()
                best_Z = Z.detach().cpu().numpy()

    return loss, mcc_iid, best_Z, best_D

def train_supervised_coding(seed, num_seed, lambda_p, lr, steps, n, n_points, A, inputs, optim, train_Z_iid, run):
    best_D, best_Z = None, None
    for rep in range(num_seed):
        torch.manual_seed(seed + rep)

        # Initialize with smaller values to prevent gradient explosion
        log_Z = torch.randn(n_points//2, n, dtype=torch.float32, device=device).requires_grad_()
        D = torch.tensor(A.T, dtype=torch.float32, device=device) # supervised
        optim = torch.optim.Adam([log_Z, D], lr=lr)

        loss, mcc_iid, Z, D = train_seed(log_Z, D, inputs, optim, steps, lambda_p, train_Z_iid, run)

    return best_D, best_Z

def train_unsupervised_coding(seed, num_seed, lambda_p, lr, steps, n, n_points, inputs, optim, train_Z_iid, run, m):
    best_loss = np.inf
    best_D, best_Z = None, None
    for rep in range(num_seed):
        torch.manual_seed(seed + rep)

        # Initialize with smaller values to prevent gradient explosion
        log_Z = torch.randn(n_points//2, n, dtype=torch.float32, device=device).requires_grad_()
        D = torch.randn(n, m, dtype=torch.float32, device=device).requires_grad_() # unsupervised
        optim = torch.optim.Adam([log_Z, D], lr=lr)

        loss, mcc_iid, Z, D = train_seed(log_Z, D, inputs, optim, steps, lambda_p, train_Z_iid, run)

    return best_D, best_Z