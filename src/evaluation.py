import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment


def eval_codes(encoder, inputs_iid, inputs_ood, label_iid, label_ood):
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