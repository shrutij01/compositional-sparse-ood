import numpy as np

from scipy.special import comb as n_choose_k
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, spearmanr
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from itertools import combinations
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda')
device

from src.data import generate_datasets

from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import sparse_encode

from src.mcc import mean_corr_coef as mcc

import wandb

wandb.login()

device = torch.device('cuda')

n_points = 2000
n = 100
k = 10
m = int(np.ceil(k * np.log(n/k) * 1))

(train_Z_iid, train_Y_iid, train_label_iid), (val_Z_iid, val_Y_iid, val_label_iid), (Z_ood, Y_ood, label_ood), A = generate_datasets(n=n, k=k, n_samples=n_points, m=m)

seed = 7012025
num_seed = 1
lambda_p = 0.1
lr = 1e-3
steps = 100000

best_loss = np.inf
best_D, best_Z = None, None

run = wandb.init(
    project="sparse_ood",  # Specify your project
    config={                        # Track hyperparameters and metadata
        "learning_rate": lr,
        "steps": steps,
        "n": n,
        "k": k,
        "m": m,
        "n_points": n_points,
        "lambda_p": lambda_p,
        "seed": seed,
        "num_seed": num_seed,
    },
)

# inputs = torch.tensor(Y_ood, dtype=torch.float32, device=device) ## this is giving me numerical errors
inputs = torch.tensor(train_Y_iid, dtype=torch.float32, device=device)

for rep in range(num_seed):
    torch.manual_seed(seed + rep)

    # Initialize with smaller values to prevent gradient explosion
    log_Z = torch.randn(n_points//2, n, dtype=torch.float32, device=device).requires_grad_()
    # D = torch.randn(n, m, dtype=torch.float32, device=device).requires_grad_() # unsupervised
    D = torch.tensor(A.T, dtype=torch.float32, device=device) # supervised
    optim = torch.optim.Adam([log_Z, D], lr=lr)

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