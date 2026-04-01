import numpy as np


# ============================================================================
# Sampling functions — all take an explicit rng for reproducibility
# ============================================================================


def sample_comb(ind, n=3, k=2, power=1, rng=None):
    """Given k indices, sample sources.

    Parameters
    ----------
    ind : np.ndarray
        Indices of the latent variables to sample.
    n : int
        Total number of latent variables.
    k : int
        Number of sources to sample.
    power : int
        Power to which the uniform distribution is raised.
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    z : np.ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.RandomState()
    z = np.zeros(n)
    z[ind] = rng.uniform(0, 1, k) ** power
    return z


def sample_setting_a(n=3, k=2, num_ood=None, rng=None):
    """Include first index and sample randomly from IID.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample.
    num_ood : int, optional
        Number of OOD sources. Defaults to n//2.
    rng : np.random.RandomState
        Random state for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()
    if num_ood is None:
        num_ood = n // 2
    ind_distractors = rng.choice(
        np.arange(1, n - num_ood), k - 1, replace=False
    )
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    return sample_comb(ind, n=n, k=k, rng=rng)


def sample_setting_b(n=3, k=2, num_ood=None, rng=None):
    """Sample randomly from all but first.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample.
    num_ood : int, optional
        Number of OOD sources. Defaults to n//2.
    rng : np.random.RandomState
        Random state for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()
    if num_ood is None:
        num_ood = n // 2
    ind = rng.choice(np.arange(1, n), k, replace=False)
    return sample_comb(ind, n=n, k=k, rng=rng)


def sample_setting_c(n=3, k=2, num_ood=None, rng=None):
    """Include first index and sample randomly from OOD.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample.
    num_ood : int, optional
        Number of OOD sources. Defaults to n//2.
    rng : np.random.RandomState
        Random state for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()
    if num_ood is None:
        num_ood = n // 2
    ind_distractors = rng.choice(
        np.arange(n - num_ood, n), k - 1, replace=False
    )
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    return sample_comb(ind, n=n, k=k, rng=rng)


def sample_iid(n=3, k=2, num_ood=None, rng=None):
    """Only sample from IID latent combinations.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample.
    num_ood : int, optional
        Number of OOD sources. Defaults to n//2.
    rng : np.random.RandomState
        Random state for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()
    if num_ood is None:
        num_ood = n // 2
    first_one_in = rng.binomial(n=1, p=0.5)
    if first_one_in:
        return sample_setting_a(n=n, k=k, num_ood=num_ood, rng=rng)
    else:
        return sample_setting_b(n=n, k=k, num_ood=num_ood, rng=rng)


def sample_ood(n=3, k=2, num_ood=None, rng=None):
    """Only sample from OOD latent combinations.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample.
    num_ood : int, optional
        Number of OOD sources. Defaults to n//2.
    rng : np.random.RandomState
        Random state for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()
    if num_ood is None:
        num_ood = n // 2
    return sample_setting_c(n=n, k=k, num_ood=num_ood, rng=rng)


def sample_all(n=3, k=2, rng=None):
    """No IID/OOD split, just sample all.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample.
    rng : np.random.RandomState
        Random state for reproducibility.
    """
    if rng is None:
        rng = np.random.RandomState()
    ind = rng.choice(np.arange(n), k, replace=False)
    return sample_comb(ind, n=n, k=k, rng=rng)


# ============================================================================
# Data generation
# ============================================================================


def generate_matrix(m=3, n=2, rng=None):
    """Generate a random matrix A with restricted isometry property (RIP).

    Parameters
    ----------
    m : int
        Number of rows (observation dimension).
    n : int
        Number of columns (latent dimension).
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    A : np.ndarray of shape (m, n) with normalized columns.
    """
    if rng is None:
        rng = np.random.RandomState()
    A = rng.normal(0, 1, (m, n))
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    A /= norms
    return A


def generate_data(n=3, k=2, m=2, n_samples=100, rng=None):
    """Generate IID and OOD datasets.

    Parameters
    ----------
    n : int
        Number of latent variables.
    k : int
        Number of sources to sample (sparsity level).
    m : int
        Observation dimension.
    n_samples : int
        Number of IID samples (OOD gets n_samples // 2).
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    (Z_iid, Y_iid, label_iid), (Z_ood, Y_ood, label_ood), A
    """
    if rng is None:
        rng = np.random.RandomState()

    A = generate_matrix(m=m, n=n, rng=rng)

    Z_iid = np.array([sample_iid(n=n, k=k, rng=rng) for _ in range(n_samples)])
    Y_iid = Z_iid @ A.T
    label_iid = Z_iid[:, 0] > 0.5

    Z_ood = np.array([sample_ood(n=n, k=k, rng=rng) for _ in range(n_samples // 2)])
    Y_ood = Z_ood @ A.T
    label_ood = Z_ood[:, 0] > 0.5

    return (Z_iid, Y_iid, label_iid), (Z_ood, Y_ood, label_ood), A


def generate_datasets(seed=0, num_latents=3, k=2, n_samples=100, input_dim=None):
    """Generate training, validation, and OOD test datasets.

    Creates a single RandomState from the seed and threads it through all
    sampling, so results are fully reproducible regardless of external state.

    Parameters
    ----------
    seed : int
        Random seed — the single source of randomness.
    num_latents : int
        Number of latent variables.
    k : int
        Number of sources to sample (sparsity level).
    n_samples : int
        Number of IID samples (split into train/val halves).
    input_dim : int, optional
        Observation dimension. Defaults to 2 * k * log(num_latents/k) (CS bound).

    Returns
    -------
    (Z_train, Y_train, labels_train),
    (Z_val, Y_val, labels_val),
    (Z_ood, Y_ood, labels_ood),
    A
    """
    rng = np.random.RandomState(seed)

    if input_dim is None:
        input_dim = int(np.ceil(2 * k * np.log(num_latents / k)))

    (Z_iid, Y_iid, label_iid), (Z_ood, Y_ood, label_ood), A = generate_data(
        n=num_latents, k=k, m=input_dim, n_samples=n_samples, rng=rng
    )

    half = n_samples // 2
    train = (Z_iid[:half], Y_iid[:half], label_iid[:half])
    val = (Z_iid[half:], Y_iid[half:], label_iid[half:])
    ood = (Z_ood, Y_ood, label_ood)

    return train, val, ood, A


# ============================================================================
# Torch data setup
# ============================================================================


def data_setup(
    Z_iid,
    Y_iid,
    val_Z_iid,
    val_Y_iid,
    Z_ood,
    Y_ood,
    batch_size=64,
    device=None,
):
    """Convert numpy arrays to torch tensors and build a DataLoader.

    Returns (device, train_loader, inputs_iid, targets_iid, inputs_ood, targets_ood).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs_iid = torch.tensor(Y_iid, dtype=torch.float32, device=device)
    targets_iid = torch.tensor(Z_iid, dtype=torch.float32, device=device)
    train_loader = DataLoader(
        TensorDataset(inputs_iid, targets_iid),
        batch_size=batch_size,
        shuffle=True,
    )

    inputs_ood = torch.tensor(Y_ood, dtype=torch.float32, device=device)
    targets_ood = torch.tensor(Z_ood, dtype=torch.float32, device=device)

    return (device, train_loader, inputs_iid, targets_iid, inputs_ood, targets_ood)
