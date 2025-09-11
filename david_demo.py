import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import random
import argparse

from scipy.special import comb as n_choose_k
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import _LRScheduler


### MCC ###
def pad(a, num_col):
    # pad a to have num_col columns
    diff = num_col - a.shape[1]
    noise = np.random.normal(0, 1, (a.shape[0], diff))
    return np.concatenate([a, noise], axis=1)
    
    
def normalize(x):
    # Center the data
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    # Normalize by the L2 norm of the centered data
    x_normed = x_centered / np.linalg.norm(x_centered, axis=0, keepdims=True)
    return x_normed


def compute_correlation(a, b):
    # The dot product of centered and normalized vectors gives the correlation.
    return normalize(a).T @ normalize(b)


def compute_matches(cost):
    matches = [(row_ind, col_ind) for (row_ind, col_ind) in zip(*linear_sum_assignment(cost))]
    return matches


def compute_mcc(a, b, seed=42, return_correlations=False, return_matches=False):
    if type(a) == torch.Tensor:
        a = a.detach().cpu().numpy()
    if type(b) == torch.Tensor:
        b = b.detach().cpu().numpy()
    if a.shape[1] < b.shape[1]:
        np.random.seed(seed)
        a = pad(a, b.shape[1])
    elif b.shape[1] < a.shape[1]:
        np.random.seed(seed)
        b = pad(b, a.shape[1])
    correlations = compute_correlation(a, b)
    bad_ind = np.isinf(correlations)
    correlations[bad_ind] = 0
    bad_ind = np.isnan(correlations)
    correlations[bad_ind] = 0
    cost = - np.abs(correlations)
    matches = compute_matches(cost)
    mcc = np.mean([abs(correlations[i, j]) for i, j in matches])
    output = [mcc]
    if return_correlations:
        output += [correlations]
    if return_matches:
        output += [matches]
    return output


### LR scheduler ###
# unsupervised learning seems ill conditioned, needs greedy lr increases...
class AdaptiveLR(object):
    """
    A learning rate scheduler that increases the learning rate when the loss
    is decreasing and decreases it when the loss is stagnating or increasing.
    """
    def __init__(self, optimizer, mode='min', factor=0.5, increase_factor=1.1, 
                 patience_increase=1, patience_decrease=2, min_lr=1e-6, 
                 max_lr=1e0, verbose=False):
        
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.increase_factor = increase_factor
        self.patience_increase = patience_increase
        self.patience_decrease = patience_decrease
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose

        self.best_loss = float('inf') if self.mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        
        self.last_epoch = 0

    def step(self, metrics):
        current_loss = metrics

        if self.mode == 'min':
            if current_loss < self.best_loss:
                # Loss is improving, reset bad epochs counter and increment good epochs
                self.best_loss = current_loss
                self.num_bad_epochs = 0
                self.num_good_epochs += 1
                
                # Check if we should increase LR
                if self.num_good_epochs >= self.patience_increase:
                    self._adjust_lr(self.increase_factor)
                    self.num_good_epochs = 0 # Reset counter after increase
            else:
                # Loss is not improving, reset good epochs and increment bad epochs
                self.num_good_epochs = 0
                self.num_bad_epochs += 1
                
                # Check if we should decrease LR
                if self.num_bad_epochs >= self.patience_decrease:
                    self._adjust_lr(self.factor)
                    self.num_bad_epochs = 0 # Reset counter after decrease
        
        self.last_epoch += 1

    def _adjust_lr(self, factor):
        """Adjusts the learning rate by a given factor for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * factor
            
            # Clamp the new learning rate within the defined bounds
            new_lr = max(self.min_lr, new_lr)
            new_lr = min(self.max_lr, new_lr)
            
            if new_lr != old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Epoch {self.last_epoch}: adjusting learning rate "
                          f"of group {i} from {old_lr:.6f} to {new_lr:.6f}.")


### Simulation ###
def sample_comb(ind, N, K, power):
    """Given K indices, sample sources."""
    z = np.zeros(N)
    z[ind] = np.random.uniform(0, 1, K) ** power
    return z

def sample_setting_a(N, K, num_ood, power, seed=None):
    """Include the first index and sample randomly from IID sources."""
    if seed is not None:
        np.random.seed(seed)
    # Select K-1 indices from the in-distribution sources (excluding the first one)
    ind_distractors = np.random.choice(
        np.arange(1, N - num_ood), K - 1, replace=False)
    # Add the first source index
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    z = sample_comb(ind, N=N, K=K, power=power)
    return z

def sample_setting_b(N, K, power, seed=None):
    """Sample randomly from all sources except the first one."""
    if seed is not None:
        np.random.seed(seed)
    ind = np.random.choice(np.arange(1, N), K, replace=False)
    z = sample_comb(ind, N=N, K=K, power=power)
    return z

def sample_setting_c(N, K, num_ood, power, seed=None):
    """Include the first index and sample randomly from OOD sources."""
    if seed is not None:
        np.random.seed(seed)
    # Select K-1 indices from the out-of-distribution sources
    ind_distractors = np.random.choice(
        np.arange(N - num_ood, N), K - 1, replace=False)
    # Add the first source index
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    z = sample_comb(ind, N=N, K=K, power=power)
    return z

def sample_iid(N, K, num_ood, power, seed=None):
    """Sample only from IID latent combinations."""
    if seed is not None:
        np.random.seed(seed)
    # Decide whether the variable of interest (the first source) is in the sample
    first_one_in = np.random.binomial(n=1, p=0.5)
    
    if first_one_in:  # Setting (a)
        return sample_setting_a(N=N, K=K, num_ood=num_ood, power=power)
    else:  # Setting (b)
        return sample_setting_b(N=N, K=K, power=power)

def sample_ood(N, K, num_ood, power, seed=None):
    """Sample a combination that includes the first source and OOD sources."""
    return sample_setting_c(N=N, K=K, num_ood=num_ood, power=power, seed=seed)

def sample_all(N, K, power, seed=None):
    """Sample randomly from all N sources without an IID/OOD split."""
    if seed is not None:
        np.random.seed(seed)
    ind = np.random.choice(np.arange(N), K, replace=False)
    z = sample_comb(ind, N=N, K=K, power=power)
    return z


def main():
    # STEP 1: Add argument parsing at the beginning of main()
    parser = argparse.ArgumentParser(
        description='Run sparse coding experiment for a single lambda.')
    parser.add_argument(
        '--lam', type=float, default=0.01668101,
        help='L1 regularization strength lambda.')
    # lams = np.geomspace(1e-4, 1e0, 10)[5:6]
    parser.add_argument(
        '--supervised', action='store_true', 
        help='Flag to run in supervised mode.')
    ### Setting ###
    # Sparse code (supervised) setting:
    # 1) infer Z_iid_ and Z_ood_
    # 2) On Z_iid_, train linear classifier - how good (iid and ood)?
    # Sparse code (unsupervised) setting:
    # 1) Learn matrix A (=dictionary) on Y_iid
    # 2) infer Z_iid_ and Z_ood_ [can be done at same time as step 1) or at the end]
    # 3) On Z_iid_, train linear classifier - how good (iid and ood)?    
    args = parser.parse_args()
    
    # Use the parsed lambda
    lam = args.lam
    supervised = args.supervised

    device = torch.device('cuda')
    print(f'device {device}, lambda={lam:.6e}, supervised={supervised}')
    
    # Parameters
    seed = 7012025
    C = np.inf # no regularisation for lin. regression
    power = 1 # uniform density
    D = 1000 # number of data samples
    N = 100 # number of sources
    K = 10 # sparsity
    num_ood = N // 2 # how many new OOD sources
    M = int(np.ceil(K * np.log(N / K) * 2)) # Compressed Sensing bound times 2
    lr = 1e-2
    
    # lazy example (might not be perfect), just draw random A
    np.random.seed(seed)
    # https://en.wikipedia.org/wiki/Restricted_isometry_property
    A = np.random.normal(0, 1, (M, N)) # random normal has RIP -> CS works =)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    
    # generate data
    np.random.seed(seed)
    Z_iid = np.array([sample_iid(N, K, num_ood, power, seed=None) for _ in range(D)])
    Y_iid = Z_iid @ A.T
    label_iid = Z_iid[:, 0] > .5
    Z_ood = np.array([sample_ood(N, K, num_ood, power, seed=None) for _ in range(D)])
    Y_ood = Z_ood @ A.T
    label_ood = Z_ood[:, 0] > .5
    ind_Z_ood_active = Z_ood.var(0) > 0 # need for mcc calculation
    
    # test linear probe
    print('Linear Probe:')
    clf = LogisticRegression(C=C).fit(Z_iid, label_iid)
    auc_iid = clf.score(Z_iid, label_iid)
    auc_ood = clf.score(Z_ood, label_ood)
    print(f'In source (Z) space: acc IID: {auc_iid:.3f}, acc OOD: {auc_ood:.3f}')
    clf = LogisticRegression(C=C).fit(Y_iid, label_iid)
    auc_iid = clf.score(Y_iid, label_iid)
    auc_ood = clf.score(Y_ood, label_ood)
    print(f'In observation (Y=AZ) space: acc IID: {auc_iid:.3f}, acc OOD: {auc_ood:.3f}')
    
    # get some intution how hard the problem is, starting from the mixed observations
    ind_iid = Z_iid.std(0) > 0
    print('MCC(Z_iid, Y_iid)=', compute_mcc(Y_iid, Z_iid[:, ind_iid]))
    ind_ood = Z_ood.std(0) > 0
    print('MCC(Z_ood, Y_ood)=', compute_mcc(Y_ood, Z_ood[:, ind_ood]))
    
    inputs_iid = torch.tensor(Y_iid, dtype=torch.float32, device=device)
    inputs_ood = torch.tensor(Y_ood, dtype=torch.float32, device=device)
    
    if supervised:
        max_steps = 100000
    else: # unsupervised takes longer
        max_steps = 500000
    
    nonlinearity = torch.nn.functional.softplus
    
    print('lambda=%.6e' % lam)
    torch.manual_seed(seed)

    # "pre_" means before the activation function (nonlinearity)
    pre_Z_iid_ = torch.randn(inputs_iid.shape[0], N, device=device).requires_grad_()
    pre_Z_ood_ = torch.randn(inputs_ood.shape[0], N, device=device).requires_grad_()
    pre_Z_iid_.data -= 10 # good for softplus, small init
    pre_Z_ood_.data -= 10

    params = [pre_Z_iid_, pre_Z_ood_]

    if supervised:
        A_ = torch.tensor(A, dtype=torch.float32, device=device)
        A_norms = torch.linalg.norm(A_, dim=0) # remains fixed
    else:
        A_ = torch.randn(A.shape, device=device).requires_grad_()
        A_.data *= 1e-3 # small init
        params += [A_]

    optim = torch.optim.Adam(params, lr=lr)
    scheduler = AdaptiveLR(optim, verbose=True)
    
    log = {
        'lam': lam, 'i': [], 'lr': [], 'avg_loss': [], 
        'mcc_A': [], 
        # if mcc_A is high but mcc (on latents) low, then we need to run step 2) 
        # again separately after training to see if we can get a better estimate.
        'corr_first_iid': [], 'corr_first_ood': [], 
        'mse_iid': [], 'mse_ood': [], 
        'l1_iid': [], 'l1_ood': [], 
        'mcc_iid': [], 'mcc_ood': [], 
        'auc_iid': [], 'auc_ood': []
    }
    run_loss = []
    for i in tqdm(range(max_steps + 1)):
        Z_iid_ = nonlinearity(pre_Z_iid_)
        Z_ood_ = nonlinearity(pre_Z_ood_)
        rec_iid = Z_iid_ @ A_.T
        rec_ood = Z_ood_ @ A_.T.detach() # A_ is only trained on iid
        mse_iid = torch.mean((inputs_iid - rec_iid)**2)
        mse_ood = torch.mean((inputs_ood - rec_ood)**2)
        if not supervised:
            A_norms = torch.linalg.norm(A_, dim=0)
        l1_iid = torch.mean(torch.abs(Z_iid_) * A_norms)
        l1_ood = torch.mean(torch.abs(Z_ood_) * A_norms.detach()) # A_ is only trained on iid
        loss = mse_iid + mse_ood + lam * (l1_iid + l1_ood)
        optim.zero_grad()
        loss.backward()
        optim.step()
        run_loss.append(loss.item())

        # logging
        if i > 0 and not i % 5000:
            avg_loss = np.mean(run_loss)
            run_loss = []

            # compute MCCs
            Z_iid_ = Z_iid_.detach().cpu().numpy()
            Z_ood_ = Z_ood_.detach().cpu().numpy()
            mcc_iid = compute_mcc(Z_iid[:, ind_iid], Z_iid_)[0]
            # calculate OOD mcc only on the active sources and 
            # matching number of most active learned latents
            ind_Z_ood_active_ = np.argsort(-Z_ood_.var(0))[:np.sum(ind_Z_ood_active)]
            mcc_ood = compute_mcc(
                Z_ood[:, ind_Z_ood_active], 
                Z_ood_[:, ind_Z_ood_active_]
            )[0]
            # shape of A: M x N, we want the mcc over all N columns
            # compute_mcc expects inputs: samples x dimensions
            mcc_A = compute_mcc(A, A_.detach().cpu().numpy())[0]

            # pick best matching dim on iid
            correlations_iid = compute_correlation(Z_iid_, label_iid[:, None])[:, 0]
            ind_fit_first = np.argmax(abs(correlations_iid))
            corr_first_iid = pearsonr(Z_iid[:, 0], Z_iid_[:, ind_fit_first])[0]
            corr_first_ood = pearsonr(Z_ood[:, 0], Z_ood_[:, ind_fit_first])[0]

            # AUC from best latent
            auc_iid = roc_auc_score(label_iid, Z_iid_[:, ind_fit_first])
            auc_ood = roc_auc_score(label_ood, Z_ood_[:, ind_fit_first])

            log['i'].append(i)
            log['lr'].append(optim.param_groups[0]['lr'])
            log['avg_loss'].append(avg_loss)
            log['mcc_A'].append(mcc_A)
            log['corr_first_iid'].append(corr_first_iid)
            log['corr_first_ood'].append(corr_first_ood)
            log['mse_iid'].append(mse_iid.item())
            log['mse_ood'].append(mse_ood.item())
            log['l1_iid'].append(l1_iid.item())
            log['l1_ood'].append(l1_ood.item())
            log['mcc_iid'].append(mcc_iid)
            log['mcc_ood'].append(mcc_ood)
            log['auc_iid'].append(auc_iid)
            log['auc_ood'].append(auc_ood)

            print(
                'i=%s, ' % i +\
                'avg_loss=%4e, ' % avg_loss +\
                'mse_iid=%4e, ' % mse_iid +\
                'mse_ood=%4e, ' % mse_ood +\
                'l1_iid=%4e, ' % l1_iid +\
                'l1_ood=%4e, ' % l1_ood +\
                'mcc_A=%4f, ' % mcc_A +\
                'mcc_iid=%4f, ' % mcc_iid +\
                'mcc_ood=%4f, ' % mcc_ood +\
                'corr_first_iid=%4f, ' % corr_first_iid +\
                'corr_first_ood=%4f, ' % corr_first_ood +\
                'auc_iid=%4f, ' % auc_iid +\
                'auc_ood=%4f, ' % auc_ood
            )

            # Update the scheduler
            scheduler.step(avg_loss)


    # after training
    Z_iid_ = nonlinearity(pre_Z_iid_)
    Z_ood_ = nonlinearity(pre_Z_ood_)
    log['Z_iid_'] = Z_iid_.detach().cpu().numpy()
    log['Z_ood_'] = Z_ood_.detach().cpu().numpy()
    log['A_'] = A_.detach().cpu().numpy()
    
    # Save the single result dictionary to a unique file
    filename = f'results_supervised={supervised}_lambda={lam:.4e}.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()