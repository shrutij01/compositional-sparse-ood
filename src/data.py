import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# sampling functions

def sample_comb(ind, n=3, k=2, power=1):
    """given k indices, sample sources
    Parameters:
    ind: np.ndarray     
        Indices of the latent variables to sample.
    n: int, optional
        Total number of latent variables.
    k: int, optional
        Number of sources to sample.
    power: int, optional
        Power to which the uniform distribution is raised.
    Returns:
    z: np.ndarray
        Sampled latent variables.       
    """
    z = np.zeros(n)
    z[ind] = np.random.uniform(0, 1, k) ** power
    return z

def sample_setting_a(seed=None, n=3, k=2, num_ood=None):
    """include first index and sample randomly from IID
    Parameters:
    seed: int, optional         
        Random seed for reproducibility.            
    n: int, optional
        Number of latent variables.
    k: int, optional                                    
        Number of sources to sample.
    num_ood: int, optional  
        Number of OOD samples to generate. If None, defaults to n//2.
    Returns:
    z: np.ndarray
        Sampled latent variables.   
    """
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = n//2
    ind_distractors = np.random.choice(
        np.arange(1, n - num_ood), k - 1, replace=False)
    # add first latent
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    z = sample_comb(ind, n=n, k=k)
    return z

def sample_setting_b(seed=None, n=3, k=2, num_ood=None):
    """sample randomly from all but first
    Parameters:
    seed: int, optional
        Random seed for reproducibility.
    n: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.
    num_ood: int, optional
        Number of OOD samples to generate. If None, defaults to n//2.
    Returns:
    z: np.ndarray
        Sampled latent variables.   
    """
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = n//2
    ind = np.random.choice(np.arange(1, n), k, replace=False)
    z = sample_comb(ind, n=n, k=k)
    return z

def sample_setting_c(seed=None, n=3, k=2, num_ood=None):
    """include first index and sample randomly from OOD
    Parameters:
    seed: int, optional
        Random seed for reproducibility.
    n: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.
    num_ood: int, optional
        Number of OOD samples to generate. If None, defaults to n//2.
    Returns:
    z: np.ndarray
        Sampled latent variables.

    """
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = n//2
    ind_distractors = np.random.choice(
        np.arange(n - num_ood, n), k - 1, replace=False)
    # add first latent
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    z = sample_comb(ind, n=n, k=k)
    return z

def sample_iid(seed=None, n=3, k=2, num_ood=None):
    """only sample from IID latent combinations
    Parameters:
    seed: int, optional         
        Random seed for reproducibility.
    n: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.
    num_ood: int, optional
        Number of OOD samples to generate. If None, defaults to n//2.
    Returns:
    z: np.ndarray
        Sampled latent variables.       
    """
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = n//2
    # is variable of interest in sample?
    first_one_in = np.random.binomial(n=1, p=.5)
    if first_one_in: # a) setting
        return sample_setting_a(seed=None, n=n, k=k, num_ood=num_ood)
    else: # b) setting
        return sample_setting_b(seed=None, n=n, k=k, num_ood=num_ood)

def sample_ood(seed=None, n=3, k=2, num_ood=None):
    """only sample from OOD latent combinations
    Parameters: 
    seed: int, optional         
        Random seed for reproducibility.
    n: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.
    num_ood: int, optional
        Number of OOD samples to generate. If None, defaults to n//2.
    Returns:
    z: np.ndarray
        Sampled latent variables.   
    """
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = n//2
    return sample_setting_c(seed=None, n=n, k=k, num_ood=num_ood)

def sample_all(seed=None, n=3, k=2):
    """
    no IID/OOD split, just sample all"
    Parameters:
    seed: int, optional
        Random seed for reproducibility.
    N: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.
    Returns:
    tuple: (Z, Y, label)
        - Z: sampled latent variables.
    """
    if seed is not None:
        np.random.seed(seed)
    ind = np.random.choice(np.arange(n), k, replace=False)
    z = sample_comb(ind, n=n, k=k)
    return z

def generate_matrix(m=3, n=2, seed=None):
    """ Generate a random matrix A with restricted isometry property (RIP).
    Parameters:
    m: int, optional
        Number of rows in the matrix.
    n: int, optional
        Number of columns in the matrix.
    seed: int, optional
        Random seed for reproducibility.
    Returns:
    np.ndarray:
        Random matrix A of shape (M, N) with normalized columns.    
    """
    if seed is not None:
        np.random.seed(seed)
    # lazy example (might not be perfect), just draw random A
    # https://en.wikipedia.org/wiki/Restricted_isometry_property
    A = np.random.normal(0, 1, (m, n)) # random normal has RIP -> CS works =)
    A /= np.linalg.norm(A, axis=0, keepdims=True) # normalize columns
    return A

def generate_data(seed=None, n=3, k=2, m=2, D=100):
    """
    Generate data for high-dimensional setting. 
    Generates IID and OOD data based on the specified parameters.
    Parameters:
    seed: int, optional
        Random seed for reproducibility.
    n: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.    
    D: int, optional
        Total number of samples to generate.
    Returns:
    tuple:      
        - (Z_iid, Y_iid, label_iid): IID dataset.
        - (Z_ood, Y_ood, label_ood): OOD dataset.
    """
    # generate data
    if seed is not None:
        np.random.seed(seed)

    A = generate_matrix(m=m, n=n, seed=seed)  # generate random matrix A

    Z_iid = np.array([sample_iid(n=n, k=k) for _ in range(D)])
    Y_iid = Z_iid @ A.T
    label_iid = Z_iid[:, 0] > .5

    Z_ood = np.array([sample_ood(n=n, k=k) for _ in range(D//2)])
    Y_ood = Z_ood @ A.T
    label_ood = Z_ood[:, 0] > .5

    return (Z_iid, Y_iid, label_iid), (Z_ood, Y_ood, label_ood)

    """
    Generate dasets for IID and OOD settings.
    """

# vitoria's version of generate_datasets

"""
def generate_datasets(seed=None, n=3, k=2, n_samples=100):
    '''
    Generate training and validation datasets for IID and OOD settings. 
    The training set is the first half of the IID data, and the validation set is the second half.
    The OOD data is generated separately.       

    Parameters:
    seed: int, optional
        Random seed for reproducibility.
    n: int, optional
        Number of latent variables.
    k: int, optional
        Number of sources to sample.
    D: int, optional
        Total number of samples to generate.                

    Returns:
    tuple:
        - (train_Z_iid, train_Y_iid, train_label_iid): Training dataset for IID.
        - (val_Z_iid, val_Y_iid, val_label_iid): Validation dataset for IID.
        - (Z_ood, Y_ood, label_ood): OOD dataset. 
    '''
    (Z_iid, Y_iid, label_iid), (Z_ood, Y_ood, label_ood) = generate_data(seed=42, n=3, k=2, D=100, m=2)
    train_Z_iid = Z_iid[0:n_samples//2]
    train_Y_iid = Y_iid[0:n_samples//2]
    train_label_iid = label_iid[0:n_samples//2]

    val_Z_iid = Z_iid[n_samples//2:]
    val_Y_iid = Y_iid[n_samples//2:]
    val_label_iid = label_iid[n_samples//2:]

    return (train_Z_iid, train_Y_iid, train_label_iid), (val_Z_iid, val_Y_iid, val_label_iid), (Z_ood, Y_ood, label_ood)
"""

# my version of generate_datasets to modify paramters
def generate_datasets( seed: int = None, n: int = 3, k: int = 2, d: int = 100, m: int = 2):
    """
    Generate training and validation datasets for IID and OOD settings. 
    Training is the first half of the IID data, validation the second half.
    OOD is generated separately.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    n : int, optional
        Number of latent variables.
    k : int, optional
        Number of sources to sample.
    D : int, optional
        Total number of IID samples to generate.
    m : int, optional
        Observation dimension (compressed measurement size).

    Returns
    -------
    (Z_train_iid, Y_train_iid, labels_train_iid),
    (Z_val_iid,   Y_val_iid,   labels_val_iid),
    (Z_ood,       Y_ood,       labels_ood)
    """
    # call generate_data with your parameters
    (Z_iid, Y_iid, labels_iid), (Z_ood, Y_ood, labels_ood) = generate_data(
        seed=seed,
        n=n,
        k=k,
        D=d,
        m=m,
    )

    # split IID into train / validation
    half = d // 2
    train = (Z_iid[:half], Y_iid[:half], labels_iid[:half])
    val   = (Z_iid[half:], Y_iid[half:], labels_iid[half:])

    return train, val, (Z_ood, Y_ood, labels_ood)


# from original sae.py code
def data_setup(
    Z_iid, Y_iid,
    val_Z_iid,   val_Y_iid,
    Z_ood,       Y_ood,
    batch_size=64,
    device=None
):
    """
    1) Chooses GPU if available.
    2) Converts NumPy arrays to torch tensors on that device.
    3) Builds train DataLoader (Y→Z for supervised; unsup ignores Z).
    4) Returns device, loader, and IID/OOD tensors.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # IID tensors & loader
    inputs_iid  = torch.tensor(Y_iid, dtype=torch.float32, device=device)
    targets_iid = torch.tensor(Z_iid, dtype=torch.float32, device=device)
    train_loader = DataLoader(TensorDataset(inputs_iid, targets_iid),batch_size=batch_size,shuffle=True)

    # OOD tensors
    inputs_ood  = torch.tensor(Y_ood, dtype=torch.float32, device=device)
    targets_ood = torch.tensor(Z_ood, dtype=torch.float32, device=device)

    return device, train_loader, inputs_iid, targets_iid, inputs_ood, targets_ood


