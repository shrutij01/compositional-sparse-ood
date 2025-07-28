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


def mean_corr(z_pred, z_true, method='pearson'):
    corrs = []
    for i in range(z_true.shape[0]):
        if method == 'pearson':
            corrs.append(pearsonr(z_pred[i], z_true[i])[0])
        else:
            corrs.append(spearmanr(z_pred[i], z_true[i])[0])
    return np.mean(corrs)


def compute_mcc(correlations, return_ind=False):
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
