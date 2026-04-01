# Data Generation

## Generative Model

The synthetic data follows a sparse linear model:

$$\mathbf{y} = A\mathbf{z}$$

where $A \in \mathbb{R}^{m \times n}$ is a mixing matrix with unit-norm columns and $\mathbf{z} \in \mathbb{R}^n$ is $k$-sparse (exactly $k$ non-zero entries).

## ID / OOD Split

Compositional generalisation is tested by controlling which *combinations* of active latents appear at train vs. test time. The split is defined by the first latent ($z_0$):

| Setting | Active latents | Used for |
|---------|---------------|----------|
| **A** | $z_0$ active + $(k-1)$ from ID pool | ID training |
| **B** | $k$ from ID pool (no $z_0$) | ID training |
| **C** | $z_0$ active + $(k-1)$ from OOD pool | OOD test |

!!! tip "Key idea"
    The ID and OOD pools are disjoint subsets of $\{z_1, \ldots, z_{n-1}\}$, so OOD samples contain $z_0$ paired with latents **never seen active alongside it** during training.

## Usage

```python
from src.data import generate_datasets

train, val, ood, A = generate_datasets(
    seed=0,
    num_latents=10,   # n: number of latent sources
    k=3,              # sparsity level
    n_samples=2000,   # ID training samples
    input_dim=None,   # m: observation dim (defaults to num_latents // 2)
)

Z_train, Y_train, labels_train = train  # Z: sparse codes, Y: observations
Z_ood, Y_ood, labels_ood = ood
# A: ground-truth mixing matrix (m x n)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_latents` | $n$ — number of latent sources | required |
| `k` | number of simultaneously active sources | required |
| `n_samples` | number of ID training samples | required |
| `input_dim` | $m$ — observation dimension | `num_latents // 2` |
| `seed` | random seed for reproducibility | required |

## Mixing Matrix

`generate_matrix(m, n)` draws entries i.i.d. from $\mathcal{N}(0, 1)$ and normalises each column to unit $\ell_2$ norm, satisfying the restricted isometry property (RIP) in expectation when $m = \mathcal{O}(k \log(n/k))$.
