from json import encoder
import numpy as np

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from tqdm import tqdm

import torch

from src.mcc import mean_corr_coef as mcc
from src.evaluation import count_nonzero_close, downstream_accuracy, accuracy_iid, accuracy_best_z_iid, accuracy_best_all

import wandb

wandb.login()

device = torch.device('cuda')

def train_seed(log_Z, D, inputs, optim, steps, lambda_p, train_Z_iid, train_label_iid, run, true_A, val_inputs=None, val_Z_iid=None):
    best_loss = np.inf
    best_D, best_Z = None, None

    # compute the l0 norm for each z_i (across columns), 
    # then take mean for all samples
    l0_norm_train_z_iid = np.count_nonzero(train_Z_iid, axis=1).mean()
    print(f"L0 'norm' of training Z (iid): {l0_norm_train_z_iid}")

    if val_Z_iid is not None:
        l0_norm_val_z_iid = np.count_nonzero(val_Z_iid, axis=1).mean()
        print(f"L0 'norm' of validation Z (iid): {l0_norm_val_z_iid}")

    ind_iid = train_Z_iid.std(0) > 0

    for i in tqdm(range(steps)):
        Z = torch.nn.functional.softplus(log_Z)
        rec = Z @ D
        mse = torch.mean((inputs - rec)**2)
        l1 = torch.mean(torch.abs(Z) * torch.linalg.norm(D, dim=1))
        loss = mse + lambda_p * l1
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Compute validation loss if validation data is provided
        val_loss = None
        if val_inputs is not None and val_Z_iid is not None:
            with torch.no_grad():
                Z_val = torch.nn.functional.softplus(log_Z)
                rec_val = Z_val @ D
                val_mse = torch.mean((val_inputs - rec_val)**2)
                val_l1 = torch.mean(torch.abs(Z_val) * torch.linalg.norm(D, dim=1))
                val_loss = val_mse + lambda_p * val_l1

        # log and compute every 100 epochs
        if (i % 100 == 0):
            with torch.no_grad():
                mcc_iid = mcc(Z.detach().cpu().numpy(), train_Z_iid[:, ind_iid])
                mcs_D = mcc(D.detach().cpu().numpy(), true_A.T, method='cos')
                l0_norm_rec_z = count_nonzero_close(Z.detach().cpu().numpy(), axis=1).mean()
                # print(f"L0 'norm' of reconstructed Z: {l0_norm_rec_z}")
            log_dict = {
                "loss": loss.item(),
                "mse": mse.item(),
                "l1": l1.item(),
                "step": i,
                "mcc": mcc_iid.item(),
                "mcs_D": mcs_D.item(),
                "l0_norm_rec_z": l0_norm_rec_z.item()
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss.item()
                # val_mcc = mcc(Z.detach().cpu().numpy(), val_Z_iid)
                # log_dict["val_mcc"] = val_mcc.item()

            ## training accuracy
            with torch.no_grad():
                acc_iid_all, _ = accuracy_iid(Z.cpu().detach().numpy(), train_label_iid)
                log_dict["acc_iid_all"] = acc_iid_all

                acc_iid_best, _, _ = accuracy_best_z_iid(train_label_iid, Z.cpu().detach().numpy())
                log_dict["acc_iid_best"] = acc_iid_best

            if run is not None:
                run.log(log_dict)
            # print(log_dict)

        # early stopping
        if i > 10000:
            if loss.item() > best_loss * 1.1:
                break

        if loss.item() < best_loss:
            best_D = D.detach().cpu().numpy()
            best_Z = Z.detach().cpu().numpy()

    return loss, mcc_iid, l0_norm_rec_z, best_Z, best_D

def train_supervised_coding(seed, num_seed, lambda_p, lr, steps, n, n_points, A, inputs, optim, train_Z_iid, train_label_iid, run, val_inputs=None, val_Z_iid=None):
    Ds = []
    Zs = []
    mccs = []
    l0s = []
    losses = []
    for rep in range(num_seed):
        torch.manual_seed(seed + rep)

        # Initialize with smaller values to prevent gradient explosion
        log_Z = torch.randn(n_points//2, n, dtype=torch.float32, device=device).requires_grad_()
        D = torch.tensor(A.T, dtype=torch.float32, device=device) # supervised
        optim = torch.optim.Adam([log_Z, D], lr=lr)

        loss, mcc_iid, l0_norm_rec_z, Z, D = train_seed(log_Z, D, inputs, optim, steps, lambda_p, train_Z_iid, train_label_iid, run, A, val_inputs, val_Z_iid)

        Ds.append(D)
        Zs.append(Z)
        mccs.append(mcc_iid)
        l0s.append(l0_norm_rec_z)
        losses.append(loss.item())

    
    return Ds, Zs, mccs, l0s, losses

def train_unsupervised_coding(seed, num_seed, lambda_p, lr, steps, n, n_points, A, inputs, optim, train_Z_iid, train_label_iid, run, m, Y_ood, label_ood, true_Z_ood, val_inputs=None, val_Z_iid=None):
    Ds = []
    Zs = []
    mccs = []
    l0s = []
    losses = []
    for rep in range(num_seed):
        torch.manual_seed(seed + rep)

        # Initialize with smaller values to prevent gradient explosion
        log_Z = torch.randn(n_points//2, n, dtype=torch.float32, device=device).requires_grad_()
        D = torch.randn(n, m, dtype=torch.float32, device=device).requires_grad_() # unsupervised
        optim = torch.optim.Adam([log_Z, D], lr=lr)

        loss, mcc_iid, l0_norm_rec_z, Z, D = train_seed(log_Z, D, inputs, optim, steps, lambda_p, train_Z_iid, train_label_iid, run, A, val_inputs, val_Z_iid)

        Ds.append(D)
        Zs.append(Z)
        mccs.append(mcc_iid)
        l0s.append(l0_norm_rec_z)
        losses.append(loss.item())

    # l0_norm_rec_z = np.count_nonzero(Z, axis=1).mean()
    # print(f"L0 'norm' of reconstructed Z: {l0_norm_rec_z}")

    _, z_ood, _, _, _ = train_supervised_coding(seed, 1, lambda_p, lr, steps, n, n_points, D.T, Y_ood, optim, true_Z_ood, label_ood, run)

    acc_iid_all, acc_ood_all = downstream_accuracy(Z, z_ood[0], train_label_iid, label_ood)
    acc_iid_best, acc_ood_best = accuracy_best_all(train_label_iid, label_ood, Z, z_ood[0])

    # print(f"Downstream accuracy (iid, ood) using all data: {acc_iid_all}, {acc_ood_all}")
    # print(f"Downstream accuracy (iid, ood) using best z: {acc_iid_best}, {acc_ood_best}")

    accuracy_dict = {
        'acc_iid_all': acc_iid_all,
        'acc_ood_all': acc_ood_all,
        'acc_iid_best': acc_iid_best,
        'acc_ood_best': acc_ood_best        
    }

    return Ds, Zs, mccs, l0s, losses, accuracy_dict