import numpy as np

# sampling functions

def sample_comb(ind, N=3, K=2, power=1):
    """given K indices, sample sources"""
    z = np.zeros(N)
    z[ind] = np.random.uniform(0, 1, K) ** power
    return z

def sample_setting_a(seed=None, N=3, K=2, num_ood=None):
    """include first index and sample randomly from IID"""
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = N//2
    ind_distractors = np.random.choice(
        np.arange(1, N - num_ood), K - 1, replace=False)
    # add first latent
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    z = sample_comb(ind, N=N, K=K)
    return z

def sample_setting_b(seed=None, N=3, K=2, num_ood=None):
    """sample randomly from all but first"""
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = N//2
    ind = np.random.choice(np.arange(1, N), K, replace=False)
    z = sample_comb(ind, N=N, K=K)
    return z

def sample_setting_c(seed=None, N=3, K=2, num_ood=None):
    """include first index and sample randomly from OOD"""
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = N//2
    ind_distractors = np.random.choice(
        np.arange(N - num_ood, N), K - 1, replace=False)
    # add first latent
    ind = np.concatenate([np.zeros(1, dtype=int), ind_distractors])
    z = sample_comb(ind, N=N, K=K)
    return z

def sample_iid(seed=None, N=3, K=2, num_ood=None):
    """only sample from IID latent combinations"""
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = N//2
    # is variable of interest in sample?
    first_one_in = np.random.binomial(n=1, p=.5)
    if first_one_in: # a) setting
        return sample_setting_a(seed=None, N=N, K=K, num_ood=num_ood)
    else: # b) setting
        return sample_setting_b(seed=None, N=N, K=K, num_ood=num_ood)

def sample_ood(seed=None, N=3, K=2, num_ood=None):
    if seed is not None:
        np.random.seed(seed)
    if num_ood is None:
        num_ood = N//2
    return sample_setting_c(seed=None, N=N, K=K, num_ood=num_ood)

def sample_all(seed=None, N=3, K=2):
    """
    no IID/OOD split, just sample all"""
    if seed is not None:
        np.random.seed(seed)
    ind = np.random.choice(np.arange(N), K, replace=False)
    z = sample_comb(ind, N=N, K=K)
    return z