"""
Shared evaluation metrics for sparse OOD experiments.

Provides: MCC, IID/OOD accuracy (logistic regression), AUC.
All functions accept numpy arrays. Randomness is controlled via explicit
np.random.RandomState objects — no global state is touched.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ============================================================================
# MCC (Mean Correlation Coefficient)
# ============================================================================


def _pad_to_match(a, b, rng):
    """Pad the narrower array with noise columns so both have the same width."""
    if a.shape[1] < b.shape[1]:
        noise = rng.normal(0, 1, (a.shape[0], b.shape[1] - a.shape[1]))
        a = np.concatenate([a, noise], axis=1)
    elif b.shape[1] < a.shape[1]:
        noise = rng.normal(0, 1, (b.shape[0], a.shape[1] - b.shape[1]))
        b = np.concatenate([b, noise], axis=1)
    return a, b


def _normalize(x):
    """Center and L2-normalize each column."""
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    norms = np.linalg.norm(x_centered, axis=0, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x_centered / norms


def compute_mcc(Z_true, Z_pred, seed=42):
    """
    Mean Correlation Coefficient between true and predicted latents.

    Computes column-wise absolute Pearson correlations, then finds the
    best one-to-one matching via linear sum assignment.

    Parameters
    ----------
    Z_true : np.ndarray, shape (n_samples, d_true)
        Ground-truth latent variables. Typically filtered to active columns
        (columns with nonzero variance).
    Z_pred : np.ndarray, shape (n_samples, d_pred)
        Predicted/learned latent codes from the model.
    seed : int
        Seed for the RNG used for noise padding when dimensions don't match.

    Returns
    -------
    float
        MCC score in [0, 1]. Higher is better.
    """
    rng = np.random.RandomState(seed)
    a, b = _pad_to_match(Z_true.copy(), Z_pred.copy(), rng)
    corr = _normalize(a).T @ _normalize(b)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    cost = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost)
    return np.mean([abs(corr[i, j]) for i, j in zip(row_ind, col_ind)])


# ============================================================================
# Support recovery
# ============================================================================


def compute_support_metrics(Z_true, codes, row_ind, col_ind, threshold=1e-4):
    """Measure how well learned codes recover the true sparsity pattern.

    After Hungarian-matching columns (via match_columns), compares the binary
    support of matched code columns against the ground-truth support.

    Parameters
    ----------
    Z_true : np.ndarray, shape (n_samples, num_latents)
        Ground-truth latent variables.
    codes : np.ndarray, shape (n_samples, width)
        Learned codes from the SAE.
    row_ind, col_ind : arrays from match_columns()
        row_ind[i] is the ground-truth column matched to codes column col_ind[i].
    threshold : float
        Absolute value threshold for considering a code entry "active".

    Returns
    -------
    dict with per-sample-averaged precision, recall, F1, and average L0 stats.
    """
    n_samples = Z_true.shape[0]
    n_matched = len(row_ind)

    # Build aligned views: (n_samples, n_matched)
    Z_aligned = Z_true[:, row_ind]
    codes_aligned = codes[:, col_ind]

    gt_support = np.abs(Z_aligned) > threshold      # (n_samples, n_matched)
    pred_support = np.abs(codes_aligned) > threshold

    # Per-sample metrics
    tp = (gt_support & pred_support).sum(axis=1).astype(float)
    pred_pos = pred_support.sum(axis=1).astype(float)
    true_pos = gt_support.sum(axis=1).astype(float)

    precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
    recall = np.where(true_pos > 0, tp / true_pos, 0.0)
    f1_denom = precision + recall
    f1 = np.where(f1_denom > 0, 2 * precision * recall / f1_denom, 0.0)

    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "gt_l0": float(true_pos.mean()),
        "pred_l0": float(pred_pos.mean()),
        "pred_total_l0": float((np.abs(codes) > threshold).sum(axis=1).mean()),
    }


def reestimate_magnitudes(X, D, support_mask, nonneg=True):
    """Given a binary support mask, solve least-squares for code magnitudes.

    For each sample, solves  min_z ||x - D[:, S] z||^2  over the active set S.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, input_dim)
        Observations (bias-corrected).
    D : np.ndarray, shape (input_dim, width)
        Dictionary.
    support_mask : np.ndarray, shape (n_samples, width), bool
        Binary mask indicating which code entries are active.
    nonneg : bool
        If True, clamp negative values to zero after solving.

    Returns
    -------
    np.ndarray, shape (n_samples, width)
        Codes with re-estimated magnitudes on the given support.
    """
    n_samples, width = support_mask.shape
    codes = np.zeros((n_samples, width), dtype=np.float64)

    # Group samples by identical support pattern for efficiency
    unique_supports = {}
    for i in range(n_samples):
        key = tuple(np.where(support_mask[i])[0])
        if key not in unique_supports:
            unique_supports[key] = []
        unique_supports[key].append(i)

    for active_cols, sample_indices in unique_supports.items():
        if len(active_cols) == 0:
            continue
        active_cols = list(active_cols)
        D_sub = D[:, active_cols]  # (input_dim, |S|)
        X_sub = X[sample_indices]  # (batch, input_dim)
        # Least squares: X_sub = Z_sub @ D_sub.T  =>  Z_sub = X_sub @ D_sub @ (D_sub.T @ D_sub)^-1
        DtD = D_sub.T @ D_sub
        try:
            z_sub = X_sub @ D_sub @ np.linalg.inv(DtD + 1e-8 * np.eye(len(active_cols)))
        except np.linalg.LinAlgError:
            z_sub = X_sub @ np.linalg.pinv(D_sub.T)
        if nonneg:
            z_sub = np.maximum(z_sub, 0)
        codes[np.ix_(sample_indices, active_cols)] = z_sub

    return codes


# ============================================================================
# Column matching (dictionary analysis)
# ============================================================================


def match_columns(D, A):
    """Hungarian-match decoder columns D to ground-truth columns A by cosine similarity.

    Parameters
    ----------
    D : np.ndarray, shape (input_dim, width)
        Learned dictionary (e.g. SAE decoder columns).
    A : np.ndarray, shape (input_dim, num_latents)
        Ground-truth mixing matrix.

    Returns
    -------
    dict with:
        row_ind, col_ind : matched (A column index, D column index) pairs
        cosines : absolute cosine similarity per matched pair
        angular_errors : arccos of cosines (radians)
        norm_ratios : ||d_matched|| / ||a_matched|| per pair
        mean_cosine, mean_angular_error, mean_norm_ratio : summary stats
        frac_close : fraction of matched pairs with cosine > 0.9
    """
    D_norms = np.linalg.norm(D, axis=0, keepdims=True).clip(min=1e-8)
    A_norms = np.linalg.norm(A, axis=0, keepdims=True).clip(min=1e-8)

    cos_matrix = np.abs((A / A_norms).T @ (D / D_norms))  # (num_latents, width)

    row_ind, col_ind = linear_sum_assignment(-cos_matrix)

    cosines = cos_matrix[row_ind, col_ind]
    angular_errors = np.arccos(np.clip(cosines, 0, 1))
    norm_ratios = np.linalg.norm(D[:, col_ind], axis=0) / np.linalg.norm(A[:, row_ind], axis=0).clip(min=1e-8)

    return {
        "row_ind": row_ind,
        "col_ind": col_ind,
        "cosines": cosines,
        "angular_errors": angular_errors,
        "norm_ratios": norm_ratios,
        "mean_cosine": float(cosines.mean()),
        "mean_angular_error": float(angular_errors.mean()),
        "mean_norm_ratio": float(norm_ratios.mean()),
        "frac_close": float((cosines > 0.9).mean()),
    }


def replace_column_norms(D, A, row_ind, col_ind):
    """Replace norms of matched D columns with ground-truth A norms.

    Keeps D's column directions but substitutes A's magnitudes.
    Unmatched D columns are zeroed out.

    Parameters
    ----------
    D : np.ndarray, shape (input_dim, width)
    A : np.ndarray, shape (input_dim, num_latents)
    row_ind, col_ind : from match_columns()

    Returns
    -------
    np.ndarray, shape (input_dim, width)
    """
    D_new = np.zeros_like(D)
    for a_idx, d_idx in zip(row_ind, col_ind):
        d_col = D[:, d_idx]
        d_norm = np.linalg.norm(d_col)
        if d_norm < 1e-8:
            continue
        a_norm = np.linalg.norm(A[:, a_idx])
        D_new[:, d_idx] = (d_col / d_norm) * a_norm
    return D_new


# ============================================================================
# Downstream accuracy (logistic regression)
# ============================================================================


def evaluate_accuracy(codes_iid, labels_iid, codes_ood, labels_ood,
                      C=np.inf, max_iter=1000):
    """
    Train logistic regression on IID codes, evaluate on IID and OOD.

    Parameters
    ----------
    codes_iid : np.ndarray, shape (n_iid, d)
        Learned latent codes for IID data.
    labels_iid : np.ndarray, shape (n_iid,)
        Binary labels for IID data.
    codes_ood : np.ndarray, shape (n_ood, d)
        Learned latent codes for OOD data.
    labels_ood : np.ndarray, shape (n_ood,)
        Binary labels for OOD data.
    C : float
        Inverse regularization strength for logistic regression.
    max_iter : int
        Maximum iterations for logistic regression solver.

    Returns
    -------
    dict with 'acc_iid' and 'acc_ood'.
    """
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(codes_iid, labels_iid)
    return {
        "acc_iid": clf.score(codes_iid, labels_iid),
        "acc_ood": clf.score(codes_ood, labels_ood),
    }


# ============================================================================
# AUC (per-feature, best single feature)
# ============================================================================


def _safe_auc(labels, scores):
    """ROC AUC with fallback for degenerate cases."""
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5


def _per_feature_auc(codes, labels):
    """ROC-AUC per feature → (d,). Uses max(AUC, 1-AUC) for sign invariance."""
    d = codes.shape[1]
    aucs = np.zeros(d)
    for j in range(d):
        col = codes[:, j]
        if col.std() < 1e-10:
            aucs[j] = 0.5
            continue
        auc = _safe_auc(labels, col)
        aucs[j] = max(auc, 1 - auc)
    return aucs


def evaluate_auc(codes_iid, labels_iid, codes_ood, labels_ood, **_kwargs):
    """
    Per-feature AUC: select the single best feature on IID, report on both splits.

    For each code dimension, computes ROC-AUC using the raw feature activation
    as the score (no trained probe), with max(AUC, 1-AUC) for sign invariance.
    The feature with the highest IID AUC is selected, and that same feature's
    AUC is reported on OOD. This tests whether individual features isolate the
    label, which is what disentangled SAE features should achieve.

    Parameters
    ----------
    codes_iid : np.ndarray, shape (n_iid, d)
    labels_iid : np.ndarray, shape (n_iid,)
    codes_ood : np.ndarray, shape (n_ood, d)
    labels_ood : np.ndarray, shape (n_ood,)

    Returns
    -------
    dict with 'auc_iid' and 'auc_ood'.
    """
    iid_aucs = _per_feature_auc(codes_iid, labels_iid)
    best_idx = iid_aucs.argmax()

    ood_aucs = _per_feature_auc(codes_ood, labels_ood)

    return {
        "auc_iid": float(iid_aucs[best_idx]),
        "auc_ood": float(ood_aucs[best_idx]),
    }


def evaluate_auc_probe(codes_iid, labels_iid, codes_ood, labels_ood,
                       C=np.inf, max_iter=1000):
    """
    AUC via logistic regression probe on all code dimensions (legacy).

    Trains logistic regression on IID codes, then uses predicted probabilities
    to compute AUC on both IID and OOD splits. This measures linear
    accessibility across all features, not single-feature separability.
    """
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(codes_iid, labels_iid)

    proba_iid = clf.predict_proba(codes_iid)[:, 1]
    proba_ood = clf.predict_proba(codes_ood)[:, 1]

    return {
        "auc_probe_iid": float(_safe_auc(labels_iid, proba_iid)),
        "auc_probe_ood": float(_safe_auc(labels_ood, proba_ood)),
    }


# ============================================================================
# Combined evaluation
# ============================================================================


def evaluate_all(codes_iid, labels_iid, codes_ood, labels_ood,
                 Z_true_iid=None, Z_true_ood=None):
    """
    Run all metrics: accuracy, AUC, and optionally MCC.

    Parameters
    ----------
    codes_iid, labels_iid, codes_ood, labels_ood :
        As in evaluate_accuracy.
    Z_true_iid : np.ndarray, optional
        Ground-truth latents for IID (for MCC computation).
    Z_true_ood : np.ndarray, optional
        Ground-truth latents for OOD (for MCC computation).

    Returns
    -------
    dict with all metric values.
    """
    acc = evaluate_accuracy(codes_iid, labels_iid, codes_ood, labels_ood)
    auc = evaluate_auc(codes_iid, labels_iid, codes_ood, labels_ood)

    result = {**acc, **auc}

    if Z_true_iid is not None:
        active_iid = Z_true_iid[:, Z_true_iid.var(0) > 1e-8]
        result["mcc_iid"] = compute_mcc(active_iid, codes_iid)

    if Z_true_ood is not None:
        active_ood = Z_true_ood[:, Z_true_ood.var(0) > 1e-8]
        n_active = active_ood.shape[1]
        top_dims = np.argsort(-codes_ood.var(0))[:n_active]
        result["mcc_ood"] = compute_mcc(active_ood, codes_ood[:, top_dims])

    return result
