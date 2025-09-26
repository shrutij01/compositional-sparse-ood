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
import json
import os
import pandas as pd
from time import time

from scipy.special import comb as n_choose_k
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import _LRScheduler



KEYS = [
    '5_hist_fig_ismale', 
    '6_hist_fig_isamerican', 
    '7_hist_fig_ispolitician', 
    '66_living-room',
    '67_social-security', 
    '73_control-group', 
    '87_glue_cola', 
    '90_glue_qnli', 
]


def get_auc_and_optimal_accuracy(y_true: np.ndarray, y_pred_continuous: np.ndarray) -> tuple[float, float]:
    """
    Calculates auc and accuracy at the optimal threshold.

    The optimal threshold is found by maximizing the Youden's J statistic
    (TPR - FPR) on the ROC curve.

    Args:
        y_true: A numpy array of true binary labels (0 or 1).
        y_pred_continuous: A numpy array of continuous prediction scores (e.g., probabilities).

    Returns:
        A tuple containing:
        - auc (float): The Area Under the ROC Curve.
        - optimal_accuracy (float): The accuracy score at the optimal threshold.
    """
    # 1. Calculate auc
    auc = roc_auc_score(y_true, y_pred_continuous)

    # 2. Get ROC curve components
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_continuous)
    
    # 3. Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 4. Calculate accuracy at this threshold
    y_pred_optimal = (y_pred_continuous >= optimal_threshold).astype(int)
    optimal_accuracy = accuracy_score(y_true, y_pred_optimal)
    
    return auc, optimal_accuracy


def load_data(my_home):
    act_dir_ood = f'{my_home}/model_activations_gemma-2-9b_OOD/model_activations_gemma-2-9b_OOD/'
    label_dir_ood = f'{my_home}/OOD data/'
    act_dir_iid = f'{my_home}/model_activations_gemma-2-9b/model_activations_gemma-2-9b/'
    label_dir_iid = f'{my_home}/cleaned_data/'
    
    labels_ood, act_ood = {}, {}
    labels_iid, act_iid = {}, {}
    
    for f in os.listdir(label_dir_ood):
        if 'csv' in f:
            key = f.split('.')[0][:-4]
            df = pd.read_csv(os.path.join(label_dir_ood, f))
            if not 'target' in df:
                continue
            labels_ood[key] = df
    
            for f2 in os.listdir(label_dir_iid):
                if key[:3] == f2[:3]:
                    key2 = f2.split('.')[0]
                    df = pd.read_csv(os.path.join(label_dir_iid, f2))
                    assert 'target' in df
                    labels_iid[key] = df                
    
            for f2 in os.listdir(act_dir_ood):
                if key[:3] == f2[:3]:
                    file_path = os.path.join(act_dir_ood, f2)
                    # Use map_location to force the tensor onto the CPU
                    act_ood[key] = torch.load(file_path, map_location=torch.device('cpu'))
    
            for f2 in os.listdir(act_dir_iid):
                if key[:3] == f2[:3]:
                    file_path = os.path.join(act_dir_iid, f2)
                    # Use map_location to force the tensor onto the CPU
                    act_iid[key] = torch.load(file_path, map_location=torch.device('cpu'))
    return labels_iid, act_iid, labels_ood, act_ood



class LinearGenerativeClassifier:
    def __init__(self):
        pass

    def preprocess(self, X, new=False):
        if new:
            # save preprocessing stats
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            # Add a small epsilon for stability if a feature has zero variance
            self.std[self.std == 0] = 1e-9
        return (X - self.mean) / self.std

    def variance_explained(self, X_):
        X = self.preprocess(X_)
        Z = self.transform(X_)
        
        tot_var = X.var()
        res = X - Z[:, None] @ self.beta[None]
        res_var = res.var()
        return 1 - res_var / tot_var
        
    def fit_transform(self, X_, y):
        X = self.preprocess(X_, new=True)
        
        # regress Y onto X: X ~ Y @ beta
        self.beta = y @ X / (y @ y)
        self.beta /= np.linalg.norm(self.beta)

        # get Z
        Z = self.transform(X, preprocessed=True)

        thresholds = np.linspace(Z.min(), Z.max(), 1000)
        accs = []
        for threshold in thresholds:
            accs.append(np.mean(y == (Z > threshold)))
        self.threshold = thresholds[np.argmax(accs)]

        return Z

    def transform(self, X_, preprocessed=False):
        if preprocessed:
            X = X_
        else:
            X = self.preprocess(X_)
        # regress beta onto X: X.T ~ beta.T @ Z.T
        Z = X @ self.beta
        return Z

    def decision_function(self, X):
        return self.transform(X)
        
    def fit(self, X, y):
        Z = self.fit_transform(X, y)
        return self

    def predict(self, X):
        Z = self.transform(X)
        return (Z > self.threshold).astype(int)

    def score(self, X, y):
        return np.mean(y == self.predict(X))



def matching_pursuit(signals, dictionary, K=20, 
                     force_first_atom_step=None, verbose=False):
    """
    Implements a vectorized Matching Pursuit algorithm with an option to force
    the selection of the first atom by a specific iteration.

    Args:
        signals (np.ndarray): 2D array of input signals (each signal is a column).
        dictionary (np.ndarray): The dictionary of atoms (each column is an atom).
        K (int): The L0 sparsity, i.e., maximum number of iterations.
        force_first_atom_step (int, optional): The iteration at which to force the
                                               selection of the first atom (index 0)
                                               if it hasn't been selected yet. Defaults to None.

    Returns:
        tuple: A tuple containing the sparse coefficients and the reconstructed signals.
    """
    # 1. Initialization
    norms = np.linalg.norm(dictionary, axis=0)
    norms[norms == 0] = 1
    dictionary_normalized = dictionary / norms[:, np.newaxis].T
    residuals = signals.copy()
    num_signals = signals.shape[1]
    num_atoms = dictionary.shape[1]
    coefficients = np.zeros((num_atoms, num_signals))

    # 2. Main Iteration Loop
    if verbose:
        iterator = tqdm(range(K))
    else:
        iterator = range(K)
    # for k in :
    for k in iterator:
        # 3. Find best matching atoms for all signals
        inner_products = np.dot(dictionary_normalized.T, residuals)
        best_atom_indices = np.argmax(np.abs(inner_products), axis=0)
        
        # --- FORCING LOGIC INJECTED HERE ---
        # At the specified step, check which signals haven't used the first atom yet.
        if force_first_atom_step is not None and k == force_first_atom_step:
            # A signal has not used the first atom if its corresponding coefficient is still 0.
            signals_to_force = (coefficients[0, :] == 0)
            
            # For those signals, override the greedy choice with atom 0.
            best_atom_indices[signals_to_force] = 0
            if verbose:
                print(f"\n--- Step {k}: Forcing atom 0 for {np.sum(signals_to_force)} signal(s) ---")
        # --- END OF FORCING LOGIC ---

        # 4. Update coefficients and residuals (Vectorized Update)
        best_coeffs = inner_products[best_atom_indices, np.arange(num_signals)]
        coefficients[best_atom_indices, np.arange(num_signals)] += best_coeffs
        best_atoms_for_batch = dictionary_normalized[:, best_atom_indices]
        residuals -= best_atoms_for_batch * best_coeffs
            
    # Re-scale coefficients and reconstruct
    coefficients = coefficients / norms[:, np.newaxis]
    reconstructed_signals = np.dot(dictionary, coefficients)

    return coefficients.T, reconstructed_signals.T, residuals.T



def variance_explained(true, pred):
    tot_var = true.var()
    res = true - pred
    res_var = res.var()
    return 1 - res_var / tot_var


    
def main(args, run_seed, 
         X_iid_train, X_iid_val, 
         Y_iid_train, Y_iid_val,
         X_ood, Y_ood):

    # Seed everything with the unique seed for this trial
    np.random.seed(run_seed)

    # get dimensions
    num_iid_train, M = X_iid_train.shape
    num_iid_val = X_iid_val.shape[0]
    num_ood = X_ood.shape[0]
    
    # LinReg train on iid
    print('LinReg')
    log_reg = LogisticRegression(
        C=args.C, random_state=run_seed
    ).fit(X_iid_train, Y_iid_train)
    lin_reg_Y_iid_train = log_reg.decision_function(X_iid_train)
    lin_reg_Y_iid_val = log_reg.decision_function(X_iid_val)
    lin_reg_Y_ood = log_reg.decision_function(X_ood)
    lin_reg_auc_iid_train, lin_reg_acc_iid_train = get_auc_and_optimal_accuracy(Y_iid_train, lin_reg_Y_iid_train)
    lin_reg_auc_iid_val, lin_reg_acc_iid_val = get_auc_and_optimal_accuracy(Y_iid_val, lin_reg_Y_iid_val)
    lin_reg_auc_ood, lin_reg_acc_ood = get_auc_and_optimal_accuracy(Y_ood, lin_reg_Y_ood)
    print('auc_iid_train', lin_reg_auc_iid_train, 'acc_iid_train', lin_reg_acc_iid_train)
    print('auc_iid_val', lin_reg_auc_iid_val, 'acc_iid_val', lin_reg_acc_iid_val)
    print('auc_ood', lin_reg_auc_ood, 'acc_ood', lin_reg_acc_ood)

    # GenCla train on iid
    gen_cla = LinearGenerativeClassifier()
    gen_cla.fit(X_iid_train, Y_iid_train)
    print('LinearGenerativeClassifier, var exp', gen_cla.variance_explained(X_iid_train))
    gen_cla_Y_iid_train = gen_cla.transform(X_iid_train)
    gen_cla_Y_iid_val = gen_cla.transform(X_iid_val)
    gen_cla_Y_ood = gen_cla.transform(X_ood)
    gen_cla_auc_iid_train, gen_cla_acc_iid_train = get_auc_and_optimal_accuracy(Y_iid_train, gen_cla_Y_iid_train)
    gen_cla_auc_iid_val, gen_cla_acc_iid_val = get_auc_and_optimal_accuracy(Y_iid_val, gen_cla_Y_iid_val)
    gen_cla_auc_ood, gen_cla_acc_ood = get_auc_and_optimal_accuracy(Y_ood, gen_cla_Y_ood)
    print('auc_iid_train', gen_cla_auc_iid_train, 'acc_iid_train', gen_cla_acc_iid_train)
    print('auc_iid_val', gen_cla_auc_iid_val, 'acc_iid_val', gen_cla_acc_iid_val)
    print('auc_ood', gen_cla_auc_ood, 'acc_ood', gen_cla_acc_ood)

    # Sparse Coding
    print('Sparse coding')
    log = {
        'log_reg': log_reg, 'gen_cla': gen_cla,
        'LinReg_acc_iid_train': lin_reg_acc_iid_train, 'LinReg_auc_iid_train': lin_reg_auc_iid_train,
        'LinReg_acc_iid_val': lin_reg_acc_iid_val, 'LinReg_auc_iid_val': lin_reg_auc_iid_val,
        'LinReg_acc_ood': lin_reg_acc_ood, 'LinReg_auc_ood': lin_reg_auc_ood,
        'GenCla_acc_iid_train': gen_cla_acc_iid_train, 'GenCla_auc_iid_train': gen_cla_auc_iid_train,
        'GenCla_acc_iid_val': gen_cla_acc_iid_val, 'GenCla_auc_iid_val': gen_cla_auc_iid_val,
        'GenCla_acc_ood': gen_cla_acc_ood, 'GenCla_auc_ood': gen_cla_auc_ood,
        'SpaCod_auc_iid_train': [], 'SpaCod_acc_iid_train': [],
        'SpaCod_auc_iid_val': [], 'SpaCod_acc_iid_val': [],
        'SpaCod_auc_ood': [], 'SpaCod_acc_ood': [],
        'var_exp_iid_train': [], 'var_exp_iid_val': [], 'var_exp_ood': [],
    }

    # combine all data for joint inference
    X = np.concatenate([X_iid_train, X_iid_val, X_ood], axis=0)
    normalize = lambda x: (x - x.mean()) / x.std()
    y_train = normalize(Y_iid_train)
    if args.alpha2_mix == 'bin': # binarized
        val_a = gen_cla_Y_iid_val > np.median(gen_cla_Y_iid_val)
        val_b = lin_reg_Y_iid_val > np.median(lin_reg_Y_iid_val)
        ood_a = gen_cla_Y_ood > np.median(gen_cla_Y_ood)
        ood_b = lin_reg_Y_ood > np.median(lin_reg_Y_ood)
    elif args.alpha2_mix == 'cont': # continuous
        val_a = normalize(gen_cla_Y_iid_val)
        val_b = normalize(lin_reg_Y_iid_val)
        ood_a = normalize(gen_cla_Y_ood)
        ood_b = normalize(lin_reg_Y_ood)
    # join
    y_val = normalize(args.alpha2 * val_a + (1 - args.alpha2) * val_b)
    y_ood = normalize(args.alpha2 * ood_a + (1 - args.alpha2) * ood_b)
    Y = np.concatenate([y_train, y_val, y_ood], axis=0)
        

    # random init, with first column set to GenCla beta
    A = np.random.normal(0, 1, (args.N, M))
    A[0] = gen_cla.beta.copy()
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1
    A /= norms

    # let's go!!!
    for rep in tqdm(range(args.max_steps)):
        ### Sparse Inference: get codes
        Z, reconstructed_signals, residuals = matching_pursuit(
            signals=X.T, 
            dictionary=A.T, 
            verbose=False, 
            K=args.K,
            force_first_atom_step=args.K - 1, # project on first column last 
        )
        
        # auc of codes
        Z_iid_train = Z[:num_iid_train]
        Z_iid_val = Z[num_iid_train:num_iid_train + num_iid_val]
        Z_ood = Z[num_iid_train + num_iid_val:]
        
        spa_cod_auc_iid_train, spa_cod_acc_iid_train = get_auc_and_optimal_accuracy(Y_iid_train, Z_iid_train[:, 0])
        spa_cod_auc_iid_val, spa_cod_acc_iid_val = get_auc_and_optimal_accuracy(Y_iid_val, Z_iid_val[:, 0])
        spa_cod_auc_ood, spa_cod_acc_ood = get_auc_and_optimal_accuracy(Y_ood, Z_ood[:, 0])

        # variance explained
        rec_X_iid_train = reconstructed_signals[:num_iid_train]
        rec_X_iid_val = reconstructed_signals[num_iid_train:num_iid_train + num_iid_val]
        rec_X_ood = reconstructed_signals[num_iid_train + num_iid_val:]
        log['var_exp_iid_train'].append(variance_explained(X_iid_train, rec_X_iid_train))
        log['var_exp_iid_val'].append(variance_explained(X_iid_val, rec_X_iid_val))
        log['var_exp_ood'].append(variance_explained(X_ood, rec_X_ood))

        # logging
        log['SpaCod_auc_iid_train'].append(spa_cod_auc_iid_train)
        log['SpaCod_acc_iid_train'].append(spa_cod_acc_iid_train)
        log['SpaCod_auc_iid_val'].append(spa_cod_auc_iid_val)
        log['SpaCod_acc_iid_val'].append(spa_cod_acc_iid_val)
        log['SpaCod_auc_ood'].append(spa_cod_auc_ood)
        log['SpaCod_acc_ood'].append(spa_cod_acc_ood)
        print(
            'auc_iid_train=%.4f, ' % spa_cod_auc_iid_train +\
            'auc_iid_val=%.4f, ' % spa_cod_auc_iid_val +\
            'auc_ood=%.4f, ' % spa_cod_auc_ood,
            'var_exp_iid_train=%.4f, ' % log['var_exp_iid_train'][-1] +\
            'var_exp_iid_val=%.4f, ' % log['var_exp_iid_val'][-1] +\
            'var_exp_ood=%.4f' % log['var_exp_ood'][-1],
        )

        
        ### Dictionary Learning: update dict
            
        # combine inferred latents with labels (true on iid_train, inferred on val and ood)
        label = Y.copy()
        learned = normalize(Z[:, 0])
        Z[:, 0] = args.alpha1 * label + (1 - args.alpha1) * learned

        # select train only or learn dict on val and ood too
        if args.dict_learn == 'combined':
            target = Z.copy()
            inputs = X.copy()
        elif args.dict_learn == 'sep':
            target = Z.copy()[:num_iid_train]
            inputs = X.copy()[:num_iid_train]
        
        # compute dict with OLS
        # A = np.linalg.pinv(target.T @ target) @ target.T @ inputs # SVD might crash
        try:
            # First, try the standard pseudo-inverse
            A = np.linalg.pinv(target.T @ target) @ target.T @ inputs
        except np.linalg.LinAlgError:
            print(f"Warning: SVD did not converge. Retrying with regularization.")
            try:
                # If it fails, add a small identity matrix to stabilize
                A = np.linalg.pinv(target.T @ target + 1e-6 * np.eye(target.shape[1])) @ target.T @ inputs
            except np.linalg.LinAlgError:
                # This is a catastrophic failure, highly unlikely with regularization.
                # In this case, skipping the update is the only option left.
                print("Error: Regularized update also failed. Skipping dictionary update for this step.")
                continue # This will use the old dictionary A for the next loop

        # normalize / fill dead columns with noise
        norms = np.linalg.norm(A, axis=1, keepdims=False)
        ind_dead = norms < 1e-9
        A[ind_dead] = np.random.normal(0, 1, (np.sum(ind_dead), M))
        norms = np.linalg.norm(A, axis=1, keepdims=True) # recompute to include new
        norms[norms < 1e-9] = 1
        A /= norms

        
    ### After training
    log['Z_iid_train'] = Z_iid_train
    log['Z_iid_val'] = Z_iid_val
    log['Z_ood'] = Z_ood
    log['A'] = A
    
    # Save the single result dictionary to a unique file
    filename = os.path.join(args.save_dir, 'log.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Results saved to {filename}")


### For Random Search ###
def randint(low, high):
	return int(np.random.randint(low, high, 1)[0])

def uniform(low, high):
	return np.random.uniform(low, high, 1)[0]

def loguniform(low, high):
	return np.exp(np.random.uniform(np.log(low), np.log(high), 1))[0]

def spike_slab(spike_val, spike_prob, sample_fn):
    dum = uniform(0, 1)
    if dum < spike_prob:
        return spike_val
    else:
        return sample_fn()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A well-structured sparse coding OOD experiment.'
    )

    # -- Core Command-Line Arguments --
    parser.add_argument('--key_ind', type=int, default=0, 
                        help='Index of the dataset key to use.')
    parser.add_argument('--my_home', type=str, default='/grid/klindt/home/klindt/Vitoria/real_data',
                        help='Directory where the data are stored.')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Directory where the results are stored.')
    
    # -- Training & Execution Parameters --
    parser.add_argument('--K', type=int, default=20, 
                        help='Sparsity, L0 of codes, steps in Matching Pursuit.')
    parser.add_argument('--max_steps', type=int, default=10, 
                        help='Maximum number of training steps (inference + dict update).')
    parser.add_argument('--alpha1', type=float, default=1.0, 
                        help='Code update: 0 always learned, 1 always labels, in [0, 1]')
    parser.add_argument('--alpha2', type=float, default=0.5, 
                        help='LinReg vs GenCla guidance: 0 all LinReg, 1 all GenCla, in [0, 1]')
    parser.add_argument('--alpha2_mix', type=str, default='bin',
                        help='binary or continuous LinReg vs GenCla added in mix, [bin, cont]')
    parser.add_argument('--dict_learn', type=str, default='combined',
                        help='whether to learn dict also on val and ood, [combined, sep]')

    # -- Reproducibility --
    parser.add_argument('--seed', type=int, default=984735, 
                        help='Random seed for initialization.')
    parser.add_argument('--num_run', type=int, default=30, 
                        help='Number of runs (e.g., for sampling).')
    parser.add_argument('--num_rep', type=int, default=5, 
                        help='Number of repetitions per run (e.g., for sampling).')

    # Parse all defined arguments
    args = parser.parse_args()

    # It's good practice to print the configuration for the run
    print("--- Experiment Configuration ---")
    print(json.dumps(vars(args), indent=4))
    print("------------------------------")

    # do each dataset
    args.key = KEYS[args.key_ind]

    # prep data
    labels_iid, act_iid, labels_ood, act_ood = load_data(args.my_home)
    X_iid = act_iid[args.key].cpu().numpy()
    Y_iid = labels_iid[args.key]['target'].to_numpy()
    X_ood = act_ood[args.key].cpu().numpy()
    Y_ood = labels_ood[args.key]['target'].to_numpy()

    if not Y_iid.dtype == np.int64:
        Y_iid = np.array([int('not' in _) for _ in Y_iid])
    if not Y_ood.dtype == np.int64:
        Y_ood = np.array([int('not' in _) for _ in Y_ood])
    print(args.key, X_iid.shape, X_ood.shape)

    # split iid into train and val
    kf = KFold(n_splits=args.num_rep, shuffle=True, random_state=args.seed)
    kf_index_list = list(kf.split(X_iid))

    for ind_seed in range(args.num_run):
        # 1. Calculate a base seed for this hyperparameter set and dataset.
        #    This ensures each set's seed range is completely separate.
        #    e.g., ind_seed=0 -> base=20250913; ind_seed=1 -> base=20250918
        base_seed = args.seed * (1 + int(args.key.split('_')[0])) + ind_seed * args.num_rep

        # manual choice:
        # args.C = 1.0 
        # args.N = 1024
        # args.K = 20 # must be smaller than N
        # args.alpha1 = 1.0
        # args.alpha2 = 0.5
        # args.max_steps = 10
        
        # 2. Use this base_seed to generate a reproducible set of hyperparameters.
        np.random.seed(base_seed)

        # Hyperparameter search
        args.C = loguniform(1e-3, 1e3)
        args.N = int(loguniform(128, 2048))
        args.K = int(loguniform(2, 64)) # must be smaller than N
        args.max_steps = int(loguniform(2, 64))
        spike_val = 1.0
        spike_prob = .75 # 75% of the time, it will be spike_val
        sample_fn = lambda: 1 - uniform(0, 1) ** 2 # 25% of the time it will come from this
        args.alpha1 = spike_slab(spike_val, spike_prob, sample_fn)
        args.alpha2 = uniform(0, 1)
        
        
        for rep in range(args.num_rep):
            # 3. Create the final, unique seed for this specific trial.
            #    e.g., base=20250913 -> run_seeds are 20250913, ..., 20250917
            #    e.g., base=20250918 -> run_seeds are 20250918, ..., 20250922
            run_seed = base_seed + rep
            
            args.save_dir = os.path.join(
                args.my_home, args.out_dir, args.key, f'seed_{base_seed}_rep_{rep}'
            )
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Created directory: {args.save_dir}")

            # --- Add the seeds to the args object here ---
            args.base_seed = base_seed
            args.run_seed = run_seed
            
            # 4. Convert the argparse.Namespace object to a dictionary
            params = vars(args)
            
            # 5. Define the file path and save the dictionary as a JSON file
            # os.path.join creates a platform-independent path (e.g., 'experiment_results/params.json')
            file_path = os.path.join(args.save_dir, 'params.json')
            
            with open(file_path, 'w') as f:
                # json.dump writes the dictionary to the file
                # indent=4 makes the JSON file nicely formatted and easy to read
                json.dump(params, f, indent=4)
            
            print(f"Parameters saved to {file_path}")
            print("--- Contents of params.json ---")
            # Optional: Print the content of the JSON for verification
            with open(file_path, 'r') as f:
                print(f.read())

            # --- Data Selection and Normalization (same for Linear Probe and Sparse Coding) ---
            train_index, val_index = kf_index_list[rep]
            X_iid_train = X_iid[train_index]
            X_iid_val = X_iid[val_index]
            # normalization (https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html)
            normalize = lambda x: (x - x.mean(0)) / (x.std(0) + 1e-9)
            X_iid_train_norm = normalize(X_iid_train)
            X_iid_val_norm = normalize(X_iid_val)
            X_ood_norm = normalize(X_ood)
                
            print('run experiment:')
            main(
                args, 
                run_seed,  # <-- Pass the unique, non-overlapping seed
                X_iid_train_norm,
                X_iid_val_norm,
                Y_iid[train_index], 
                Y_iid[val_index],
                X_ood_norm,
                Y_ood
            )

            
    
"""
Needs to be run in data folder downloaded from Dropbox link in https://github.com/JoshEngels/SAE-Probes

# KEY_IND in range(8)

python sem_sup_sparse_code_mp_clean.py --key_ind=KEY_IND --out_dir=results/exp2_reproduce
"""