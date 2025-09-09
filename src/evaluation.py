import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
# from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment


def accuracy_iid(z, label_iid, C=np.inf, max_iter=1000):
    '''
    Compute the accuracy on the IID dataset, splitting into train/val and reporting both.
    '''
    from sklearn.model_selection import train_test_split
    z_train, z_val, y_train, y_val = train_test_split(z, label_iid, test_size=0.5, random_state=42, stratify=label_iid)
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(z_train, y_train)
    acc_train = clf.score(z_train, y_train)
    acc_val = clf.score(z_val, y_val)
    return (acc_train, acc_val), clf

def accuracy_ood(clf, rec_z_ood, label_ood):
    """Compute the accuracy on the OOD dataset.

    Args:
        clf (_type_): classifier
        rec_z_ood (_type_): _description_
        label_ood (_type_): _description_

    Returns:
        _type_: _description_
    """
    acc_o = clf.score(rec_z_ood, label_ood)
    return acc_o

def downstream_accuracy(z_iid, z_ood, label_iid, label_ood):
    '''
    evaluate everything (not used in training loop because we want to hide ood accuracy)
    '''
    acc_iid, clf = accuracy_iid(z_iid, label_iid)
    acc_ood = accuracy_ood(clf, z_ood, label_ood)
    return acc_iid, acc_ood

## functions for selecting best z_i and using it downstream
def normalize(x):
    # Center the data
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    # Normalize by the L2 norm of the centered data
    x_normed = x_centered / np.linalg.norm(x_centered, axis=0, keepdims=True)
    return x_normed

def compute_correlation(a, b):
    # The dot product of centered and normalized vectors gives the correlation.
    return normalize(a).T @ normalize(b)

def accuracy_best_z_iid(label_iid, Z_iid_, C=np.inf, max_iter=100):
    # pick best matching dim on iid
    correlations_iid = compute_correlation(Z_iid_, label_iid[:, None])[:, 0]
    ind_fit_first = np.argmax(abs(correlations_iid))

    # regression from best latent 
    clf = LogisticRegression(C=C, max_iter=max_iter).fit(Z_iid_[:, ind_fit_first][:, None], label_iid)
    acc_iid_best = clf.score(Z_iid_[:, ind_fit_first][:, None], label_iid)
    # acc_ood_best = clf.score(Z_ood_[:, ind_fit_first][:, None], label_ood)
    return acc_iid_best, clf, ind_fit_first

def accuracy_best_z_ood(clf, label_ood, Z_ood_, ind_fit_first):
    acc_ood_best = clf.score(Z_ood_[:, ind_fit_first][:, None], label_ood)
    return acc_ood_best

def accuracy_best_all(label_iid, label_ood, Z_iid_, Z_ood_, C=np.inf, max_iter=100):
    acc_iid_best, clf, ind_fit_first = accuracy_best_z_iid(label_iid, Z_iid_, C=C, max_iter=max_iter)
    acc_ood_best = accuracy_best_z_ood(clf, label_ood, Z_ood_, ind_fit_first)
    return acc_iid_best, acc_ood_best

def eval_codes(encoder, inputs_iid, inputs_ood, label_iid, label_ood):
    '''
    Isa's code for evaluating SAE.
    '''
    encoder.eval()
    with torch.no_grad():
        Zi = encoder(inputs_iid).cpu().numpy()
        Zo = encoder(inputs_ood).cpu().numpy()
    Zall = np.vstack([Zi, Zo])
    z_iid = Zall[:Zi.shape[0]]
    z_ood = Zall[Zi.shape[0]:]
    # C = 5
    clf = LogisticRegression(C=np.inf, max_iter=1000).fit(z_iid, label_iid)
    # print("isabela has made this change")
    acc_i = clf.score(z_iid, label_iid)
    acc_o = clf.score(z_ood, label_ood)
    return Zi, Zo, acc_i, acc_o

def compute_mcc(correlations, return_ind=False):
    """
    David's MCC implementation.

    Args:
        correlations (_type_): _description_
        return_ind (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    assert correlations.shape[0] == correlations.shape[1]
    if type(correlations) == np.ndarray:
        cost = - np.abs(correlations)
    elif type(correlations) == torch.Tensor:
        cost = - np.abs(correlations.detach().cpu().numpy())
    else:
        raise ValueError("Type %s not recognized" % type(correlations))

    ind = [(row_ind, col_ind) for (row_ind, col_ind) in zip(*linear_sum_assignment(cost))]
    mcc = np.mean([abs(correlations[i, j]) for i, j in ind])

    if return_ind:
        return mcc, ind
    else:
        return mcc


def as_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else x


def save_z_arrays(sup_iid, sup_ood, unsup_iid, unsup_ood, prefix='Z'):
    np.save(f"{prefix}_sup_iid.npy",   sup_iid)
    np.save(f"{prefix}_sup_ood.npy",   sup_ood)
    np.save(f"{prefix}_unsup_iid.npy", unsup_iid)
    np.save(f"{prefix}_unsup_ood.npy", unsup_ood)
    print("Saved Z arrays to disk.")


def count_nonzero_close(arr, tol=1e-4, axis=None):
    # Create a boolean mask where True indicates elements close to zero
    close_to_zero_mask = np.isclose(arr, 0, atol=tol)

    # Count the number of elements close to zero
    count_nonzero_close = np.count_nonzero(~close_to_zero_mask, axis=axis)

    return count_nonzero_close