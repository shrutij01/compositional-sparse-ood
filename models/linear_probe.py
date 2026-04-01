"""
Linear baselines for comparison with sparse coding methods.

Provides three baselines that use the same evaluate_all pipeline:
  - raw:          use observations Y directly as codes (no representation learning)
  - pca:          project Y to k principal components (unsupervised)
  - linear_probe: supervised least-squares regression from Y to Z (oracle ceiling)

raw and pca establish a floor; linear_probe establishes a ceiling for linear methods.
"""

import numpy as np
from sklearn.decomposition import PCA


def pca_codes(Y_iid, Y_ood, n_components):
    """Fit PCA on IID observations, project both IID and OOD.

    Parameters
    ----------
    Y_iid : np.ndarray, shape (n_iid, m)
        IID observations (used to fit PCA).
    Y_ood : np.ndarray, shape (n_ood, m)
        OOD observations (transformed only).
    n_components : int
        Number of principal components (typically k).

    Returns
    -------
    codes_iid : np.ndarray, shape (n_iid, n_components)
    codes_ood : np.ndarray, shape (n_ood, n_components)
    """
    pca = PCA(n_components=n_components)
    codes_iid = pca.fit_transform(Y_iid)
    codes_ood = pca.transform(Y_ood)
    return codes_iid, codes_ood


def linear_probe_codes(Y_train, Z_train, Y_iid, Y_ood,
                       alphas=(1e-6, 1e-4, 1e-2, 1.0, 10.0)):
    """Supervised linear regression from observations to ground-truth codes.

    Uses RidgeCV with built-in leave-one-out cross-validation to select
    the regularization strength alpha.

    This is an oracle baseline (uses ground-truth Z) that upper-bounds
    what any linear method can achieve.

    Parameters
    ----------
    Y_train : np.ndarray, shape (n_train, m)
        Training observations.
    Z_train : np.ndarray, shape (n_train, d)
        Ground-truth latent codes for training data.
    Y_iid : np.ndarray, shape (n_iid, m)
        IID evaluation observations.
    Y_ood : np.ndarray, shape (n_ood, m)
        OOD evaluation observations.
    alphas : tuple of float
        Candidate regularization strengths for LOO-CV.

    Returns
    -------
    codes_iid : np.ndarray, shape (n_iid, d)
    codes_ood : np.ndarray, shape (n_ood, d)
    """
    from sklearn.linear_model import RidgeCV

    reg = RidgeCV(alphas=list(alphas), fit_intercept=False)
    reg.fit(Y_train, Z_train)
    print(f"    RidgeCV selected alpha={reg.alpha_}")

    codes_iid = reg.predict(Y_iid)
    codes_ood = reg.predict(Y_ood)
    return codes_iid, codes_ood
