# API Reference

## Data — `src.data`

### `generate_datasets(seed, num_latents, k, n_samples, input_dim=None)`

Main entry point. Returns `(train, val, ood, A)` where each split is `(Z, Y, labels)` and `A` is the ground-truth mixing matrix.

```python
from src.data import generate_datasets

train, val, ood, A = generate_datasets(
    seed=0, num_latents=10, k=3, n_samples=2000
)
```

### `data_setup(Z_iid, Y_iid, val_Z, val_Y, Z_ood, Y_ood, batch_size, device)`

Convert numpy arrays to torch tensors and build a DataLoader for training.

---

## Models

=== "SAEs"

    ### `SAEConfig`

    Dataclass for SAE experiments.

    ```python
    from models.saes import SAEConfig, run_sae_experiment

    cfg = SAEConfig(
        num_latents=10, k=3, n_samples=2000,
        width=20, sae_type="topk",   # "relu", "topk", "jumprelu", "MP"
        epochs=1000, lr=5e-4, seed=0,
    )
    model, results, data, A, cfg = run_sae_experiment(cfg)
    ```

    ### `SAE(input_dim, width, sae_type, kval_topk=None, mp_kval=None)`

    `torch.nn.Module`. Call `forward(x, return_hidden=False)` for codes + reconstruction, or `decode(codes)` for decoding only.

    ### `train_sae(model, train_loader, inputs_iid, inputs_ood, device, cfg)`

    Train a single SAE. Returns dict with codes, reconstructions, and per-epoch metrics.

    ### `run_sae_experiment(cfg=None, **overrides)`

    End-to-end: generate data, train, evaluate. Returns `(model, results, data, A, cfg)`.

=== "Sparse Coding"

    ### `fista(x, D, lam, n_iter=100, lr=None, z_init=None, nonneg=False)`

    FISTA with Nesterov momentum. Use with the ground-truth dictionary for oracle sparse inference.

    ```python
    from models.sparse_coding import fista
    import torch

    codes = fista(Y, D=torch.from_numpy(A), lam=0.1, n_iter=100)
    ```

    ### `ista(x, D, lam, n_iter=100, lr=None, z_init=None)`

    Plain ISTA (no momentum).

    ### `SparseCodingConfig`

    ```python
    from models.sparse_coding import SparseCodingConfig

    cfg = SparseCodingConfig(
        input_dim=10, num_latents=20,
        method="fista",  # "fista", "ista", "lista", "direct"
        lam=0.1, n_iter=100,
    )
    ```

    ### `train_sparse_coding(X_iid, X_ood, cfg, A=None, D_init=None, device=None)`

    Train sparse coding with method specified in `cfg.method`. Returns dict with codes, learned dictionary, and training history.

    ### `LISTA(n_obs, n_latent, n_unroll=16)`

    Learned ISTA encoder. Call `init_from_dictionary(D)` to initialise, then `forward(x)` for fast inference.

    ### `refine_from_sae(sae_model, X, lam, n_iter=20, nonneg=True)`

    Extract an SAE's decoder and warm-start FISTA from its encoder output. Returns dict with original codes, refined codes, dictionary, and MSE gap.

=== "Linear Probe"

    ### `pca_codes(Y_iid, Y_ood, n_components)`

    PCA baseline. Fit on IID, project both splits.

    ### `linear_probe_codes(Y_train, Z_train, Y_iid, Y_ood, alphas=(...))`

    Supervised RidgeCV regression from observations to ground-truth codes. Oracle upper bound.

---

## Metrics — `utils.metrics`

### `compute_mcc(Z_true, Z_pred, seed=42) -> float`

Mean Correlation Coefficient via Hungarian matching. Returns score in [0, 1].

### `match_columns(D, A) -> dict`

Match learned dictionary `D` to ground-truth `A` by cosine similarity (Hungarian algorithm). Returns matched indices, angular errors, and summary statistics.

### `compute_support_metrics(Z_true, codes, row_ind, col_ind, threshold=1e-4) -> dict`

Sparsity pattern recovery: precision, recall, F1, L0.

### `evaluate_accuracy(codes_iid, labels_iid, codes_ood, labels_ood) -> dict`

Logistic regression on IID, evaluate on both splits. Returns `{acc_iid, acc_ood}`.

### `evaluate_auc(codes_iid, labels_iid, codes_ood, labels_ood) -> dict`

Per-feature ROC-AUC: best feature on IID, report both. Returns `{auc_iid, auc_ood}`.

### `evaluate_all(codes_iid, labels_iid, codes_ood, labels_ood, Z_true_iid=None, Z_true_ood=None) -> dict`

Combined evaluation: accuracy + AUC + optional MCC.

---

## Experiment Helpers — `experiments._common`

### `eval_and_tag(codes_iid, codes_ood, data, method, **extra) -> dict`

Run `evaluate_all` and tag the result with method name and metadata.

### `run_all_saes(data, input_dim, width, k, num_latents, ...) -> list`

Train and evaluate all SAE variants (ReLU, TopK, JumpReLU, MP).

### `run_sparse_coding_methods(data, A, input_dim, num_latents, ...) -> list`

Train and evaluate FISTA oracle, DL-FISTA, and Softplus-Adam.

### `run_linear_baselines(data, k, tag) -> list`

Evaluate supervised linear probe baseline.

### `save_incremental(all_results, out_path)`

Save results list to JSON with automatic directory creation.
