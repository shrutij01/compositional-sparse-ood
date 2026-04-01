"""
Sparse coding variants for synthetic sparse OOD experiments.

Implements multiple inference methods under a unified interface:
  - Direct optimization (Adam on pre-activation codes, existing approach)
  - ISTA  (Iterative Shrinkage-Thresholding Algorithm)
  - FISTA (Fast ISTA with Nesterov momentum)
  - LISTA (Learned ISTA — unrolled, learned weights)

All methods share the same loss:  MSE(X, Z @ D^T) + lam * weighted_L1(Z, D)
and the same IID/OOD asymmetry: dictionary is trained on IID only.

Usage
-----
1) Quick end-to-end experiment (generates data, trains, evaluates):

    from models.sparse_coding import SparseCodingConfig, run_sparse_coding_experiment

    # FISTA with known dictionary (supervised)
    out = run_sparse_coding_experiment(method="fista", num_latents=10, k=3, supervised=True, seed=42)
    print(out["metrics"])  # MCC, accuracy, AUC on IID and OOD

    # Direct optimization, unsupervised (learns dictionary)
    out = run_sparse_coding_experiment(method="direct", num_latents=10, k=3, supervised=False)

    # LISTA (trains LISTA encoder after learning D with FISTA)
    out = run_sparse_coding_experiment(method="lista", num_latents=10, k=3, supervised=True)


2) Standalone ISTA/FISTA on your own data (fixed dictionary, infer codes):

    from models.sparse_coding import fista, ista

    # D: (n_obs, n_latent) dictionary tensor, X: (batch, n_obs) observations
    z = fista(X, D, lam=0.1, n_iter=100)                     # standard FISTA
    z = fista(X, D, lam=0.1, n_iter=100, nonneg=True)        # non-negative codes
    z = fista(X, D, lam=0.1, n_iter=20, z_init=z_warmstart)  # warm-started
    z = ista(X, D, lam=0.1, n_iter=200)                      # plain ISTA (slower)


3) LISTA — train a learned encoder, then use it for fast inference:

    from models.sparse_coding import LISTA, train_lista
    from torch.utils.data import DataLoader, TensorDataset

    # Build and initialize from a known dictionary D
    lista = LISTA(n_obs=m, n_latent=n, n_unroll=16)
    lista.init_from_dictionary(D)  # warm-start from ISTA matrices

    # Train on IID data (D stays fixed, only encoder weights are learned)
    loader = DataLoader(TensorDataset(X_iid), batch_size=64, shuffle=True)
    history = train_lista(lista, D, loader, lam=0.1, epochs=200, lr=1e-3)

    # Fast inference (single forward pass, no iteration)
    lista.eval()
    with torch.no_grad():
        codes_iid = lista(X_iid)
        codes_ood = lista(X_ood)


4) Measure amortization gap (SAE vs FISTA refinement):

    from models.sparse_coding import refine_from_sae
    from models.saes import SAE

    # sae_model: a trained SAE with .Ad (decoder) attribute
    gap = refine_from_sae(sae_model, X, lam=0.1, n_iter=20)
    print(gap["mse_sae"])       # reconstruction error from SAE encoder
    print(gap["mse_refined"])   # reconstruction error after FISTA refinement
    print(gap["gap_mse"])       # the amortization gap (mse_sae - mse_refined >= 0)
    # gap["z_sae"] and gap["z_refined"] are the code arrays for further analysis


5) Fair cross-method comparison (same dictionary, same data, same metrics):

    from models.sparse_coding import compare_methods

    # D: shared dictionary (e.g. ground truth A, or SAE decoder, or FISTA-learned)
    # All methods infer codes with THIS D — only the inference procedure differs.
    results = compare_methods(
        X_iid, X_ood, D,
        labels_iid, labels_ood,
        Z_true_iid=Z_true_iid,  # optional, for MCC (synthetic data)
        Z_true_ood=Z_true_ood,
        lam=0.1,
        fista_iters=100,
        sae_model=trained_sae,   # optional, include SAE in comparison
        lista_model=trained_lista,  # optional, include LISTA in comparison
    )
    # results is {"fista": {metrics}, "ista": {metrics}, "sae": {metrics},
    #             "sae_refined": {metrics}, "lista": {metrics}}
    # Each metrics dict has: acc_iid, acc_ood, auc_iid, auc_ood,
    #                         mcc_iid, mcc_ood, mse_iid, mse_ood, l0_iid, l0_ood


6) train_sparse_coding() — the main training entry point with full control:

    from models.sparse_coding import SparseCodingConfig, train_sparse_coding

    cfg = SparseCodingConfig(
        input_dim=30, num_latents=100, method="fista",
        lam=0.1, max_steps=10_000, n_iter=100,
        supervised=False, seed=42,
    )
    # X_iid, X_ood: torch tensors of observations
    # A: numpy array of ground-truth mixing matrix (required if supervised=True)
    result = train_sparse_coding(X_iid, X_ood, cfg, A=A)
    # result["codes_iid"], result["codes_ood"]: numpy arrays of inferred codes
    # result["dictionary"]: numpy array of learned/fixed dictionary
    # result["history"]: list of per-step metric dicts
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import generate_datasets
from src.opt import AdaptiveLR
from utils.metrics import compute_mcc, evaluate_all


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class SparseCodingConfig:
    # Data dimensions
    input_dim: int = 10        # observation dim
    num_latents: int = 20      # code dim (typically num_latents > input_dim)

    # Inference method: "direct", "ista", "fista", "lista"
    method: str = "fista"

    # Sparsity
    lam: float = 0.1
    nonlinearity: str = "softplus"  # for direct method: "softplus" or "relu"

    # Optimization
    lr: float = 1e-3
    max_steps: int = 50_000

    # ISTA/FISTA
    n_iter: int = 100

    # LISTA
    n_unroll: int = 16
    lista_epochs: int = 200
    lista_batch_size: int = 64

    # Dictionary learning
    supervised: bool = False
    dict_update_every: int = 50  # for alternating methods

    # Data generation (for convenience runner)
    k: int = 3
    n_samples: int = 2000

    # Misc
    seed: int = 0
    print_every: int = 5000

    @property
    def run_name(self) -> str:
        sup = "sup" if self.supervised else "unsup"
        return f"sc_{self.method}_{sup}_n{self.num_latents}_lam{self.lam:g}_seed{self.seed}"

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SparseCodingConfig":
        return cls(**json.loads(path.read_text()))


# ============================================================================
# PRIMITIVES
# ============================================================================


def soft_threshold(x: Tensor, threshold: float | Tensor) -> Tensor:
    """Proximal operator for L1: sign(x) * max(|x| - threshold, 0)."""
    return torch.sign(x) * F.relu(x.abs() - threshold)


def _lipschitz_constant(D: Tensor) -> float:
    """Largest eigenvalue of D^T D = ||D||_2^2."""
    return max((torch.linalg.norm(D, ord=2) ** 2).item(), 1e-8)


# ============================================================================
# ISTA / FISTA
# ============================================================================


def ista(
    x: Tensor, D: Tensor, lam: float,
    n_iter: int = 100, lr: float | None = None,
    z_init: Tensor | None = None,
) -> Tensor:
    """
    ISTA: Iterative Shrinkage-Thresholding Algorithm.

    Solves  min_z  0.5 * ||x - z @ D^T||^2  +  lam * ||z||_1

    Parameters
    ----------
    x : (batch, input_dim)
    D : (input_dim, num_latents) — dictionary with atoms as columns
    lam : sparsity weight
    n_iter : number of iterations
    lr : step size (default: 1/L where L is Lipschitz constant of grad)
    z_init : optional warm-start

    Returns
    -------
    z : (batch, num_latents)
    """
    L = _lipschitz_constant(D)
    if lr is None:
        lr = 1.0 / L

    n_latent = D.shape[1]
    # Use precomputed DtD for small problems; on-the-fly gradient for large ones
    use_dtd = (n_latent <= 10_000)

    if use_dtd:
        DtD = D.T @ D                # (num_latents, num_latents)
    Dtx = x @ D                      # (batch, num_latents)

    z = z_init.clone() if z_init is not None else torch.zeros(
        x.shape[0], D.shape[1], device=x.device
    )

    with torch.no_grad():
        for _ in range(n_iter):
            if use_dtd:
                grad = z @ DtD - Dtx
            else:
                grad = (z @ D.T) @ D - Dtx
            z = soft_threshold(z - lr * grad, lr * lam)

    return z


def fista(
    x: Tensor, D: Tensor, lam: float,
    n_iter: int = 100, lr: float | None = None,
    z_init: Tensor | None = None, nonneg: bool = False,
) -> Tensor:
    """
    FISTA: Fast ISTA with Nesterov momentum.  O(1/k^2) convergence.

    Parameters
    ----------
    x : (batch, input_dim)
    D : (input_dim, num_latents)
    lam, n_iter, lr, z_init : same as ista()
    nonneg : if True, project to non-negative orthant after each step

    Returns
    -------
    z : (batch, num_latents)
    """
    L = _lipschitz_constant(D)
    if lr is None:
        lr = 1.0 / L

    n_latent = D.shape[1]
    # Use precomputed DtD for small problems; on-the-fly gradient for large ones
    use_dtd = (n_latent <= 10_000)

    if use_dtd:
        DtD = D.T @ D
    Dtx = x @ D

    z = z_init.clone() if z_init is not None else torch.zeros(
        x.shape[0], D.shape[1], device=x.device
    )
    y = z.clone()
    t = 1.0

    with torch.no_grad():
        for _ in range(n_iter):
            z_old = z
            if use_dtd:
                grad = y @ DtD - Dtx
            else:
                grad = (y @ D.T) @ D - Dtx
            z_tilde = y - lr * grad
            z = soft_threshold(z_tilde, lr * lam)
            if nonneg:
                z = z.clamp(min=0)
            t_old = t
            t = (1 + (1 + 4 * t ** 2) ** 0.5) / 2
            y = z + ((t_old - 1) / t) * (z - z_old)

    return z


# ============================================================================
# LISTA  (Learned ISTA)
# ============================================================================


class LISTA(nn.Module):
    """
    Learned ISTA (Gregor & LeCun 2010).

    Unrolls ISTA for `n_unroll` steps with per-layer learnable weights
    W_k, S_k, and thresholds.  After training, inference is a single
    forward pass.
    """

    def __init__(self, n_obs: int, n_latent: int, n_unroll: int = 16):
        super().__init__()
        self.n_obs = n_obs
        self.n_latent = n_latent
        self.n_unroll = n_unroll

        # Per-layer parameters: W_k (lateral), S_k (input), threshold_k
        self.W = nn.ParameterList([
            nn.Parameter(torch.zeros(n_latent, n_latent)) for _ in range(n_unroll)
        ])
        self.S = nn.ParameterList([
            nn.Parameter(torch.zeros(n_latent, n_obs)) for _ in range(n_unroll)
        ])
        self.thresholds = nn.ParameterList([
            nn.Parameter(0.01 * torch.ones(n_latent)) for _ in range(n_unroll)
        ])

    def init_from_dictionary(self, D: Tensor):
        """
        Initialize weights from analytical ISTA matrices.

        D : (n_obs, n_latent)
        Sets W_k = I - (1/L) D^T D,  S_k = (1/L) D^T  for all k.
        """
        with torch.no_grad():
            L = _lipschitz_constant(D)
            W_init = torch.eye(self.n_latent, device=D.device) - (1 / L) * (D.T @ D)
            S_init = (1 / L) * D.T
            for k in range(self.n_unroll):
                self.W[k].data.copy_(W_init)
                self.S[k].data.copy_(S_init)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (batch, n_obs)
        Returns codes : (batch, n_latent)
        """
        z = torch.zeros(x.shape[0], self.n_latent, device=x.device)
        for k in range(self.n_unroll):
            z = soft_threshold(
                z @ self.W[k].T + x @ self.S[k].T,
                self.thresholds[k].abs(),
            )
        return z


def train_lista(
    lista: LISTA,
    D: Tensor,
    train_loader: DataLoader,
    lam: float,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[dict]:
    """
    Train LISTA with a fixed dictionary D.

    Loss = MSE(x, codes @ D^T) + lam * L1(codes)
    Only the LISTA encoder parameters (W_k, S_k, thresholds) are updated.
    """
    if device is None:
        device = D.device
    lista = lista.to(device)
    D = D.to(device)
    optimizer = torch.optim.Adam(lista.parameters(), lr=lr)
    history = []

    for ep in range(epochs):
        lista.train()
        ep_loss = 0.0
        n_batches = 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            codes = lista(X_batch)
            recon = codes @ D.T
            loss_mse = F.mse_loss(recon, X_batch)
            loss_l1 = lam * codes.abs().mean()
            loss = loss_mse + loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        history.append({"epoch": ep, "loss": ep_loss / max(n_batches, 1)})
    return history


# ============================================================================
# DICTIONARY UPDATE
# ============================================================================


def update_dictionary(X: Tensor, Z: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Closed-form dictionary update:  D = X^T Z (Z^T Z + eps I)^{-1}

    Then normalize columns to unit norm.

    X : (n_samples, n_obs)
    Z : (n_samples, n_latent)
    Returns D : (n_obs, n_latent)
    """
    ZtZ = Z.T @ Z + eps * torch.eye(Z.shape[1], device=Z.device)
    D = torch.linalg.solve(ZtZ.T, (X.T @ Z).T).T  # equivalent to X^T Z (Z^T Z)^{-1}
    # normalize columns
    norms = torch.linalg.norm(D, dim=0, keepdim=True).clamp(min=1e-8)
    D = D / norms
    return D


# ============================================================================
# TRAINING: DIRECT OPTIMIZATION
# ============================================================================


def _train_direct(
    X_iid: Tensor, X_ood: Tensor, D_init: Tensor,
    cfg: SparseCodingConfig, device: torch.device,
    supervised: bool = False,
) -> dict:
    """
    Direct optimization approach (existing method from src/trainers.py).

    Optimizes pre-activation codes with Adam; applies softplus to get Z >= 0.
    Dictionary is co-optimized (unsupervised) or fixed (supervised).
    """
    n_iid, n_ood = X_iid.shape[0], X_ood.shape[0]
    N = cfg.num_latents

    # Pre-activation codes
    pre_Z_iid = torch.randn(n_iid, N, device=device).requires_grad_()
    pre_Z_ood = torch.randn(n_ood, N, device=device).requires_grad_()
    pre_Z_iid.data -= 10  # softplus(-10) ≈ 0  →  sparse init
    pre_Z_ood.data -= 10

    params = [pre_Z_iid, pre_Z_ood]

    D = D_init.clone().to(device)
    if supervised:
        D.requires_grad_(False)
    else:
        D.requires_grad_(True)
        params.append(D)

    nonlin = F.softplus if cfg.nonlinearity == "softplus" else F.relu
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = AdaptiveLR(optimizer, verbose=False)
    history = []
    best_loss = float("inf")
    patience_counter = 0
    patience = 5000  # early stop if no improvement for this many steps

    for step in tqdm(range(cfg.max_steps), desc="direct"):
        Z_iid = nonlin(pre_Z_iid)
        Z_ood = nonlin(pre_Z_ood)

        D_norms = torch.linalg.norm(D, dim=0)

        # Reconstruction  (D stored as n_obs x n_latent → rec = Z @ D^T)
        rec_iid = Z_iid @ D.T
        rec_ood = Z_ood @ D.T.detach()  # dictionary trained on IID only

        mse_iid = F.mse_loss(rec_iid, X_iid)
        mse_ood = F.mse_loss(rec_ood, X_ood)

        l1_iid = (Z_iid.abs() * D_norms).mean()
        l1_ood = (Z_ood.abs() * D_norms.detach()).mean()

        loss = mse_iid + mse_ood + cfg.lam * (l1_iid + l1_ood)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        # Early stopping
        loss_val = loss.item()
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience and step > 10000:
            print(f"  early stop at step {step}: loss plateaued at {best_loss:.6f}")
            break

        if step > 0 and step % cfg.print_every == 0:
            history.append({
                "step": step,
                "loss": loss_val,
                "mse_iid": mse_iid.item(),
                "mse_ood": mse_ood.item(),
                "l1_iid": l1_iid.item(),
                "l1_ood": l1_ood.item(),
                "lr": optimizer.param_groups[0]["lr"],
            })
            print(
                f"  step {step:6d}  loss {loss_val:.4f}  "
                f"mse_iid {mse_iid.item():.4f}  mse_ood {mse_ood.item():.4f}"
            )

    Z_iid_np = nonlin(pre_Z_iid).detach().cpu().numpy()
    Z_ood_np = nonlin(pre_Z_ood).detach().cpu().numpy()
    D_np = D.detach().cpu().numpy()
    return {"codes_iid": Z_iid_np, "codes_ood": Z_ood_np, "dictionary": D_np, "history": history}


# ============================================================================
# TRAINING: ISTA / FISTA  (alternating optimisation)
# ============================================================================


def _train_iterative(
    X_iid: Tensor, X_ood: Tensor, D_init: Tensor,
    cfg: SparseCodingConfig, device: torch.device,
    supervised: bool = False,
    use_momentum: bool = True,
) -> dict:
    """
    Alternating sparse coding with ISTA or FISTA inference.

    Outer loop: alternate between inferring codes (ISTA/FISTA) and
    updating dictionary (closed-form least squares).

    Optimizations:
    - Supervised mode: single FISTA pass (D is fixed, no alternation needed)
    - Warm-starting: codes from previous outer iteration seed the next
    - torch.no_grad: FISTA inference doesn't need autograd
    - Early stopping: halt if dictionary change < tol
    """
    infer = fista if use_momentum else ista
    nonneg = use_momentum  # FISTA uses nonneg, ISTA does not
    D = D_init.clone().to(device)
    history = []

    # --- Supervised: just run inference once (D never changes) ---
    if supervised:
        with torch.no_grad():
            Z_iid = infer(X_iid, D, cfg.lam, n_iter=cfg.n_iter, nonneg=nonneg) if use_momentum else infer(X_iid, D, cfg.lam, n_iter=cfg.n_iter)
            Z_ood = infer(X_ood, D, cfg.lam, n_iter=cfg.n_iter, nonneg=nonneg) if use_momentum else infer(X_ood, D, cfg.lam, n_iter=cfg.n_iter)
        with torch.no_grad():
            mse_iid = F.mse_loss(Z_iid @ D.T, X_iid).item()
            mse_ood = F.mse_loss(Z_ood @ D.T, X_ood).item()
            l0_iid = (Z_iid.abs() > 0.01).float().sum(dim=-1).mean().item()
            l0_ood = (Z_ood.abs() > 0.01).float().sum(dim=-1).mean().item()
        history.append({"step": 0, "mse_iid": mse_iid, "mse_ood": mse_ood, "l0_iid": l0_iid, "l0_ood": l0_ood})
        print(f"  supervised: mse_iid {mse_iid:.4f}  mse_ood {mse_ood:.4f}  l0_iid {l0_iid:.1f}  l0_ood {l0_ood:.1f}")
        return {"codes_iid": Z_iid.cpu().numpy(), "codes_ood": Z_ood.cpu().numpy(), "dictionary": D.cpu().numpy(), "history": history}

    # --- Unsupervised: alternating optimisation with warm-start + early stopping ---
    n_outer = cfg.max_steps // cfg.dict_update_every
    Z_iid = None  # will be initialized on first iter
    Z_ood = None
    tol = 1e-5  # early stopping tolerance on dictionary change

    for outer in tqdm(range(n_outer), desc=cfg.method):
        # --- Infer codes with current D, warm-started from previous ---
        with torch.no_grad():
            if use_momentum:
                Z_iid = infer(X_iid, D, cfg.lam, n_iter=cfg.n_iter, z_init=Z_iid, nonneg=True)
                Z_ood = infer(X_ood, D, cfg.lam, n_iter=cfg.n_iter, z_init=Z_ood, nonneg=True)
            else:
                Z_iid = infer(X_iid, D, cfg.lam, n_iter=cfg.n_iter, z_init=Z_iid)
                Z_ood = infer(X_ood, D, cfg.lam, n_iter=cfg.n_iter, z_init=Z_ood)

        # --- Update dictionary from IID only ---
        D_old = D
        D = update_dictionary(X_iid, Z_iid)

        # --- Early stopping on dictionary convergence ---
        dict_change = (D - D_old).norm().item() / max(D_old.norm().item(), 1e-8)
        if dict_change < tol and outer > 10:
            print(f"  early stop at outer {outer}: dict_change={dict_change:.2e}")
            break

        # --- Logging ---
        step = (outer + 1) * cfg.dict_update_every
        with torch.no_grad():
            mse_iid = F.mse_loss(Z_iid @ D.T, X_iid).item()
            mse_ood = F.mse_loss(Z_ood @ D.T, X_ood).item()
            l0_iid = (Z_iid.abs() > 0.01).float().sum(dim=-1).mean().item()
            l0_ood = (Z_ood.abs() > 0.01).float().sum(dim=-1).mean().item()

        history.append({"step": step, "mse_iid": mse_iid, "mse_ood": mse_ood, "l0_iid": l0_iid, "l0_ood": l0_ood})

        if outer % max(1, n_outer // 10) == 0:
            print(
                f"  outer {outer:4d}/{n_outer}  "
                f"mse_iid {mse_iid:.4f}  mse_ood {mse_ood:.4f}  "
                f"l0_iid {l0_iid:.1f}  l0_ood {l0_ood:.1f}"
            )

    Z_iid_np = Z_iid.detach().cpu().numpy()
    Z_ood_np = Z_ood.detach().cpu().numpy()
    D_np = D.detach().cpu().numpy()
    return {"codes_iid": Z_iid_np, "codes_ood": Z_ood_np, "dictionary": D_np, "history": history}


# ============================================================================
# TRAINING: LISTA
# ============================================================================


def _train_lista_method(
    X_iid: Tensor, X_ood: Tensor, D_init: Tensor,
    cfg: SparseCodingConfig, device: torch.device,
    supervised: bool = False,
) -> dict:
    """
    Train LISTA encoder with a fixed dictionary, then infer codes.
    """
    D = D_init.clone().to(device)

    # If unsupervised, first learn D with FISTA, then train LISTA on that D
    if not supervised:
        print("LISTA unsupervised: learning dictionary with FISTA first...")
        fista_result = _train_iterative(
            X_iid, X_ood, D, cfg, device,
            supervised=False, use_momentum=True,
        )
        D = torch.tensor(fista_result["dictionary"], dtype=torch.float32, device=device)

    # Build LISTA and initialize from D
    lista = LISTA(cfg.input_dim, cfg.num_latents, cfg.n_unroll).to(device)
    lista.init_from_dictionary(D)

    # Train LISTA on IID data with fixed D
    train_loader = DataLoader(
        TensorDataset(X_iid), batch_size=cfg.lista_batch_size, shuffle=True,
    )
    history = train_lista(
        lista, D, train_loader,
        lam=cfg.lam, epochs=cfg.lista_epochs, lr=cfg.lr, device=device,
    )

    # Infer codes
    lista.eval()
    with torch.no_grad():
        codes_iid = lista(X_iid).cpu().numpy()
        codes_ood = lista(X_ood).cpu().numpy()

    return {
        "codes_iid": codes_iid,
        "codes_ood": codes_ood,
        "dictionary": D.cpu().numpy(),
        "history": history,
        "lista_model": lista,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def train_sparse_coding(
    X_iid: Tensor, X_ood: Tensor,
    cfg: SparseCodingConfig,
    A: np.ndarray | None = None,
    D_init: np.ndarray | Tensor | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Train sparse coding with the specified method.

    Parameters
    ----------
    X_iid : (n_iid, n_obs)  IID observations
    X_ood : (n_ood, n_obs)  OOD observations
    cfg   : SparseCodingConfig
    A     : ground-truth mixing matrix (n_obs, n_latent). Required if supervised=True.
    D_init : optional initial dictionary (n_obs, n_latent). When provided in
             unsupervised mode, warm-starts from this dictionary instead of random.
    device: torch device

    Returns
    -------
    dict with 'codes_iid', 'codes_ood', 'dictionary' (np arrays), 'history'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed torch for reproducible weight init (no global numpy seeding)
    torch.manual_seed(cfg.seed)

    X_iid = X_iid.to(device)
    X_ood = X_ood.to(device)

    # Dictionary initialization
    if cfg.supervised:
        assert A is not None, "Must provide A for supervised mode"
        D_init = torch.tensor(A, dtype=torch.float32, device=device)
    elif D_init is not None:
        if isinstance(D_init, np.ndarray):
            D_init = torch.tensor(D_init, dtype=torch.float32, device=device)
        else:
            D_init = D_init.clone().to(device)
    else:
        D_init = torch.randn(cfg.input_dim, cfg.num_latents, device=device)
        D_init = F.normalize(D_init, dim=0)  # unit-norm columns

    dispatch = {
        "direct": lambda: _train_direct(X_iid, X_ood, D_init, cfg, device, cfg.supervised),
        "ista": lambda: _train_iterative(X_iid, X_ood, D_init, cfg, device, cfg.supervised, use_momentum=False),
        "fista": lambda: _train_iterative(X_iid, X_ood, D_init, cfg, device, cfg.supervised, use_momentum=True),
        "lista": lambda: _train_lista_method(X_iid, X_ood, D_init, cfg, device, cfg.supervised),
    }

    if cfg.method not in dispatch:
        raise ValueError(f"Unknown method {cfg.method!r}. Choose from {list(dispatch)}")

    print(f"Training sparse coding: method={cfg.method}, supervised={cfg.supervised}")
    return dispatch[cfg.method]()


# ============================================================================
# REFINE FROM SAE  (measure amortization gap)
# ============================================================================


def refine_from_sae(
    sae_model: nn.Module,
    X: Tensor,
    lam: float,
    n_iter: int = 20,
    nonneg: bool = True,
) -> dict:
    """
    Measure the amortization gap: take a trained SAE, extract its decoder
    as dictionary D, get amortized codes via forward pass, then refine with
    FISTA warm-started from those codes.

    Parameters
    ----------
    sae_model : trained SAE (from models/saes.py) with .Ad attribute
    X : (batch, n_obs) observations
    lam : sparsity penalty for FISTA
    n_iter : FISTA iterations for refinement
    nonneg : enforce non-negative codes

    Returns
    -------
    dict with 'z_sae', 'z_refined', 'dictionary',
              'mse_sae', 'mse_refined', 'gap_mse'.
    """
    device = next(sae_model.parameters()).device
    X = X.to(device)

    sae_model.eval()
    with torch.no_grad():
        _, z_sae = sae_model(X, return_hidden=True)

    # Extract decoder as dictionary: Ad is (n_obs, width)
    D = sae_model.Ad.data.detach()

    # Refine with FISTA, warm-started from SAE codes
    z_refined = fista(X - sae_model.bd.data, D, lam, n_iter=n_iter, z_init=z_sae, nonneg=nonneg)

    # Reconstruction errors
    with torch.no_grad():
        rec_sae = z_sae @ D.T + sae_model.bd.data
        rec_refined = z_refined @ D.T + sae_model.bd.data
        mse_sae = F.mse_loss(rec_sae, X).item()
        mse_refined = F.mse_loss(rec_refined, X).item()

    return {
        "z_sae": z_sae.cpu().numpy(),
        "z_refined": z_refined.detach().cpu().numpy(),
        "dictionary": D.cpu().numpy(),
        "mse_sae": mse_sae,
        "mse_refined": mse_refined,
        "gap_mse": mse_sae - mse_refined,
    }


# ============================================================================
# COMPARE METHODS  (fair cross-method evaluation)
# ============================================================================


def compare_methods(
    X_iid: Tensor, X_ood: Tensor,
    D: Tensor,
    labels_iid: np.ndarray, labels_ood: np.ndarray,
    Z_true_iid: np.ndarray | None = None,
    Z_true_ood: np.ndarray | None = None,
    lam: float = 0.1,
    fista_iters: int = 100,
    lista_model: LISTA | None = None,
    sae_model: nn.Module | None = None,
) -> dict[str, dict]:
    """
    Compare inference methods on the same (X, D) pair.

    All methods use the SAME dictionary D and the SAME data — the only
    variable is the inference procedure.

    Returns dict mapping method name → evaluation metrics from
    utils/metrics.evaluate_all().
    """
    device = D.device
    X_iid = X_iid.to(device)
    X_ood = X_ood.to(device)
    results = {}

    def _eval(codes_iid_np, codes_ood_np, name):
        metrics = evaluate_all(
            codes_iid_np, labels_iid, codes_ood_np, labels_ood,
            Z_true_iid=Z_true_iid, Z_true_ood=Z_true_ood,
        )
        # Add reconstruction MSE
        with torch.no_grad():
            z_iid_t = torch.tensor(codes_iid_np, dtype=torch.float32, device=device)
            z_ood_t = torch.tensor(codes_ood_np, dtype=torch.float32, device=device)
            metrics["mse_iid"] = F.mse_loss(z_iid_t @ D.T, X_iid).item()
            metrics["mse_ood"] = F.mse_loss(z_ood_t @ D.T, X_ood).item()
            metrics["l0_iid"] = (z_iid_t.abs() > 0.01).float().sum(-1).mean().item()
            metrics["l0_ood"] = (z_ood_t.abs() > 0.01).float().sum(-1).mean().item()
        results[name] = metrics
        print(f"  {name:15s}  mse_iid={metrics['mse_iid']:.4f}  mse_ood={metrics['mse_ood']:.4f}  "
              f"l0_iid={metrics['l0_iid']:.1f}  l0_ood={metrics['l0_ood']:.1f}")

    print("Comparing methods (same dictionary D):")

    # 1. FISTA  (always run)
    z_fista_iid = fista(X_iid, D, lam, n_iter=fista_iters, nonneg=True)
    z_fista_ood = fista(X_ood, D, lam, n_iter=fista_iters, nonneg=True)
    _eval(z_fista_iid.cpu().numpy(), z_fista_ood.cpu().numpy(), "fista")

    # 2. ISTA
    z_ista_iid = ista(X_iid, D, lam, n_iter=fista_iters)
    z_ista_ood = ista(X_ood, D, lam, n_iter=fista_iters)
    _eval(z_ista_iid.cpu().numpy(), z_ista_ood.cpu().numpy(), "ista")

    # 3. SAE  (if provided)
    if sae_model is not None:
        sae_model.eval()
        with torch.no_grad():
            _, z_sae_iid = sae_model(X_iid, return_hidden=True)
            _, z_sae_ood = sae_model(X_ood, return_hidden=True)
        _eval(z_sae_iid.cpu().numpy(), z_sae_ood.cpu().numpy(), "sae")

        # SAE + FISTA refinement
        ref_iid = refine_from_sae(sae_model, X_iid, lam, n_iter=fista_iters)
        ref_ood = refine_from_sae(sae_model, X_ood, lam, n_iter=fista_iters)
        _eval(ref_iid["z_refined"], ref_ood["z_refined"], "sae_refined")

    # 4. LISTA  (if provided)
    if lista_model is not None:
        lista_model.eval()
        with torch.no_grad():
            z_lista_iid = lista_model(X_iid)
            z_lista_ood = lista_model(X_ood)
        _eval(z_lista_iid.cpu().numpy(), z_lista_ood.cpu().numpy(), "lista")

    return results


# ============================================================================
# CONVENIENCE: generate data + train + evaluate
# ============================================================================


def run_sparse_coding_experiment(cfg: SparseCodingConfig = None, **overrides):
    """
    End-to-end: generate synthetic data, train sparse coding, evaluate.

    Mirrors run_sae_experiment() from models/saes.py.
    """
    if cfg is None:
        cfg = SparseCodingConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(cfg, k, v)

    # generate_datasets handles its own seeding via RandomState(seed)
    train, val, ood, A = generate_datasets(
        seed=cfg.seed, num_latents=cfg.num_latents, k=cfg.k, n_samples=cfg.n_samples,
    )
    Z_train, Y_train, labels_train = train
    Z_val, Y_val, labels_val = val
    Z_ood, Y_ood, labels_ood = ood

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_iid = torch.tensor(Y_val, dtype=torch.float32, device=device)
    X_ood = torch.tensor(Y_ood, dtype=torch.float32, device=device)

    # Update config dims from data
    cfg.input_dim = Y_train.shape[1]

    result = train_sparse_coding(X_iid, X_ood, cfg, A=A, device=device)

    # Evaluate
    metrics = evaluate_all(
        result["codes_iid"], labels_val,
        result["codes_ood"], labels_ood,
        Z_true_iid=Z_val, Z_true_ood=Z_ood,
    )
    result["metrics"] = metrics
    result["data"] = {
        "Z_train": Z_train, "Y_train": Y_train, "labels_train": labels_train,
        "Z_val": Z_val, "Y_val": Y_val, "labels_val": labels_val,
        "Z_ood": Z_ood, "Y_ood": Y_ood, "labels_ood": labels_ood,
    }
    result["A"] = A
    result["cfg"] = cfg

    print(f"\nFinal metrics: {metrics}")
    return result


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    cfg = SparseCodingConfig(
        num_latents=10, k=3, n_samples=2000,
        method="fista", lam=0.1,
        max_steps=5000, n_iter=100, dict_update_every=50,
        supervised=True, seed=42,
    )
    out = run_sparse_coding_experiment(cfg)
    print(f"Codes IID shape: {out['codes_iid'].shape}")
    print(f"Codes OOD shape: {out['codes_ood'].shape}")
