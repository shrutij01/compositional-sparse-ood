"""
Sparse Autoencoder (SAE) for synthetic sparse OOD data.

Supports sparsity mechanisms: ReLU, TopK, JumpReLU, MP (Matching Pursuit).
Trains on IID observations from src/data.py, evaluates on both IID and OOD.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import generate_datasets


# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class SAEConfig:
    # Data parameters
    num_latents: int = 10
    k: int = 3
    n_samples: int = 2000

    # Model parameters
    width: int = 20
    sae_type: str = "relu"  # 'relu', 'topk', 'jumprelu', 'MP'
    kval_topk: int = 3
    mp_kval: int = 3

    # Training parameters
    epochs: int = 1000
    lr: float = 5e-4
    gamma_reg: float = 1e-4
    batch_size: int = 64
    grad_clip: float = 0.0
    renorm_every: int = 50

    # Misc
    seed: int = 0
    print_every: int = 100

    @property
    def run_name(self) -> str:
        """Descriptive name for this experiment run."""
        name = f"sae_{self.sae_type}_n{self.num_latents}_k{self.k}_w{self.width}_g{self.gamma_reg:g}"
        if self.sae_type == "topk":
            name += f"_topk{self.kval_topk}"
        elif self.sae_type == "MP":
            name += f"_mp{self.mp_kval}"
        name += f"_seed{self.seed}"
        return name

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "SAEConfig":
        return cls(**json.loads(path.read_text()))


# ============================================================================
# UTILS
# ============================================================================


def rectangle(x):
    return ((x >= -0.5) & (x <= 0.5)).float()


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, threshold, bandwidth)
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return x_grad, threshold_grad, None


def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(
                threshold, dtype=input.dtype, device=input.device
            )
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(
                bandwidth, dtype=input.dtype, device=input.device
            )
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0 * grad_output
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None


def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)


def renorm_decoder_cols_(Ad):
    """Normalize decoder columns to unit norm in-place."""
    norms = torch.linalg.norm(Ad, dim=0, keepdim=True).clamp(min=1e-8)
    Ad.div_(norms)


def project_decoder_grads_(Ad):
    """Project decoder gradients orthogonal to columns (maintain unit norm)."""
    dots = (Ad.grad * Ad).sum(dim=0, keepdim=True)
    Ad.grad.sub_(dots * Ad)


# ============================================================================
# SAE MODEL
# ============================================================================


class SAE(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        width=5,
        sae_type="relu",
        kval_topk=None,
        mp_kval=None,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Input dimension (observation dim from data.py).
        width : int
            Width of the hidden/code layer.
        sae_type : str
            One of 'relu', 'topk', 'jumprelu', 'MP'.
        kval_topk : int
            k for topk sae_type.
        mp_kval : int
            Number of pursuit iterations for MP sae_type.
        """
        super(SAE, self).__init__()
        self.sae_type = sae_type
        self.width = width
        self.input_dim = input_dim

        # Encoder parameters — Kaiming uniform init
        self.Ae = nn.Parameter(torch.empty((width, input_dim)))
        init.kaiming_uniform_(self.Ae, a=math.sqrt(5))
        self.be = nn.Parameter(torch.zeros((1, width)))

        # Decoder parameters — init from encoder, then normalize columns
        self.bd = nn.Parameter(torch.zeros((1, input_dim)))
        self.Ad = nn.Parameter(self.Ae.data.T.clone())
        with torch.no_grad():
            renorm_decoder_cols_(self.Ad)

        # JumpReLU parameters
        if sae_type == "jumprelu":
            self.logthreshold = nn.Parameter(
                torch.log(1e-3 * torch.ones((1, width)))
            )
            self.bandwidth = 1e-3

        # Topk parameter
        if sae_type == "topk":
            self.kval_topk = kval_topk

        # MP parameter
        if sae_type == "MP":
            self.mp_kval = mp_kval

    def forward(self, x, return_hidden=False, inf_k=None):
        if self.sae_type == "relu":
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            codes = F.relu(x)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        elif self.sae_type == "topk":
            kval = self.kval_topk if inf_k is None else inf_k
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            topk_values, topk_indices = torch.topk(x, kval, dim=-1)
            codes = torch.zeros_like(x)
            codes.scatter_(-1, topk_indices, F.relu(topk_values))
            x = torch.matmul(codes, self.Ad.T) + self.bd

        elif self.sae_type == "MP":
            kval = self.mp_kval if inf_k is None else inf_k
            residual = x - self.bd
            # Use decoder columns as dictionary (MP procedure replaces encoder)
            Ad_normed = self.Ad / self.Ad.norm(dim=0, keepdim=True).clamp(min=1e-8)
            codes = torch.zeros(x.shape[0], self.width, device=x.device)
            # Track selected atoms to prevent re-selection
            selected_mask = torch.zeros(x.shape[0], self.width, device=x.device)
            for _ in range(kval):
                # Correlations with normalized decoder atoms
                z = residual @ Ad_normed                            # (batch, width)
                # Mask out already-selected atoms
                z = z.masked_fill(selected_mask.bool(), 0.0)
                # Select atom with highest absolute correlation
                idx = z.abs().argmax(dim=1)                         # (batch,)
                # Coefficient = projection onto the selected decoder atom
                val = z.gather(1, idx.unsqueeze(1))                 # (batch, 1)
                mask = F.one_hot(idx, num_classes=self.width).float()
                codes = codes + mask * val
                selected_mask = selected_mask + mask
                # Residual update using normalized decoder atoms (consistent)
                residual = residual - (mask * val) @ Ad_normed.T
            # Reconstruct with normalized atoms (consistent with how codes were computed)
            x = codes @ Ad_normed.T + self.bd

        elif self.sae_type == "jumprelu":
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            x = F.relu(x)
            threshold = torch.exp(self.logthreshold.clamp(max=20.0))
            codes = jumprelu(x, threshold, self.bandwidth)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        else:
            raise ValueError(f"Invalid sae_type: {self.sae_type!r}")

        if return_hidden:
            return x, codes
        return x

    def decode(self, codes):
        return torch.matmul(codes, self.Ad.T) + self.bd


# ============================================================================
# TYPE-SPECIFIC REGULARISATION
# ============================================================================


def compute_reg_loss(model: SAE, codes: Tensor, dev: torch.device) -> Tensor:
    """Return the regularisation loss appropriate for each SAE type."""
    if model.sae_type == "relu":
        return torch.norm(codes, p=1, dim=-1).mean()

    if model.sae_type == "jumprelu":
        bandwidth = 1e-3
        return torch.mean(
            torch.sum(
                step_fn(codes, torch.exp(model.logthreshold.clamp(max=20.0)), bandwidth),
                dim=-1,
            )
        )

    # TopK and MP: sparsity is structural, no regularisation needed
    if model.sae_type in ("topk", "MP"):
        return torch.tensor(0.0, device=dev)

    raise ValueError(f"Unknown sae_type: {model.sae_type!r}")


# ============================================================================
# TRAINING
# ============================================================================


def train_sae(
    model: SAE,
    train_loader: DataLoader,
    inputs_iid: Tensor,
    inputs_ood: Tensor,
    device: torch.device,
    cfg: SAEConfig,
) -> dict:
    """
    Train SAE on IID observations Y, evaluate on IID and OOD Y.

    Returns
    -------
    dict with keys:
        'codes_iid', 'codes_ood': np.ndarray of latent codes
        'recon_iid', 'recon_ood': np.ndarray of reconstructions
        'history': list of per-epoch metric dicts
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []

    for ep in range(cfg.epochs):
        # --- Train on IID data ---
        model.train()
        epoch_recon = 0.0
        epoch_reg = 0.0
        n_batches = 0

        for (Y_batch,) in train_loader:
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            x_hat, codes = model(Y_batch, return_hidden=True)
            loss_recon = F.mse_loss(x_hat, Y_batch)
            loss_reg = compute_reg_loss(model, codes, device)
            loss = loss_recon + cfg.gamma_reg * loss_reg

            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if model.Ad.grad is not None:
                with torch.no_grad():
                    project_decoder_grads_(model.Ad)
            optimizer.step()

            epoch_recon += loss_recon.item()
            epoch_reg += loss_reg.item()
            n_batches += 1

        # Periodic decoder renormalization
        if cfg.renorm_every > 0 and ep % cfg.renorm_every == 0:
            with torch.no_grad():
                renorm_decoder_cols_(model.Ad)

        # --- Evaluate on IID and OOD ---
        model.eval()
        with torch.no_grad():
            rec_iid, codes_iid = model(inputs_iid, return_hidden=True)
            mse_iid = F.mse_loss(rec_iid, inputs_iid).item()
            l0_iid = (codes_iid.abs() > 0.01).float().sum(dim=-1).mean().item()

            rec_ood, codes_ood = model(inputs_ood, return_hidden=True)
            mse_ood = F.mse_loss(rec_ood, inputs_ood).item()
            l0_ood = (codes_ood.abs() > 0.01).float().sum(dim=-1).mean().item()

        metrics = {
            "epoch": ep,
            "train_recon": epoch_recon / max(n_batches, 1),
            "train_reg": epoch_reg / max(n_batches, 1),
            "mse_iid": mse_iid,
            "mse_ood": mse_ood,
            "l0_iid": l0_iid,
            "l0_ood": l0_ood,
        }
        history.append(metrics)

        if ep % cfg.print_every == 0:
            print(
                f"ep {ep:04d}  train_recon {metrics['train_recon']:.4f}  "
                f"reg {metrics['train_reg']:.4f}  "
                f"mse_iid {mse_iid:.4f}  mse_ood {mse_ood:.4f}  "
                f"l0_iid {l0_iid:.1f}  l0_ood {l0_ood:.1f}"
            )

    # Final codes and reconstructions
    model.eval()
    with torch.no_grad():
        rec_iid, codes_iid = model(inputs_iid, return_hidden=True)
        rec_ood, codes_ood = model(inputs_ood, return_hidden=True)

    return {
        "codes_iid": codes_iid.cpu().numpy(),
        "codes_ood": codes_ood.cpu().numpy(),
        "recon_iid": rec_iid.cpu().numpy(),
        "recon_ood": rec_ood.cpu().numpy(),
        "history": history,
    }


# ============================================================================
# SAVING / LOADING
# ============================================================================


def save_run(model: SAE, cfg: SAEConfig, results: dict, root: Path = None):
    """Save model weights, config, and final metrics under root/run_name/."""
    if root is None:
        root = Path(__file__).resolve().parent.parent / "runs"
    run_dir = root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "weights.pth")
    cfg.save(run_dir / "config.json")

    final = results["history"][-1] if results["history"] else {}
    (run_dir / "metrics.json").write_text(json.dumps(final, indent=2))

    print(f"Saved run to {run_dir}")
    return run_dir


def load_run(run_dir: Path, device: torch.device = None):
    """Load a saved SAE model and its config from a run directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SAEConfig.load(run_dir / "config.json")

    # Recover observation dim from weights
    state = torch.load(run_dir / "weights.pth", map_location=device, weights_only=True)
    input_dim = state["bd"].shape[1]

    model = SAE(
        input_dim=input_dim,
        width=cfg.width,
        sae_type=cfg.sae_type,
        kval_topk=cfg.kval_topk if cfg.sae_type == "topk" else None,
        mp_kval=cfg.mp_kval if cfg.sae_type == "MP" else None,
    )
    model.load_state_dict(state)
    model.to(device)
    return model, cfg


# ============================================================================
# CONVENIENCE: generate data + train
# ============================================================================


def run_sae_experiment(cfg: SAEConfig = None, **overrides):
    """
    End-to-end: generate synthetic data, train SAE, save and return results.

    All randomness is derived from ``cfg.seed``:
      - Data generation uses ``np.random.RandomState(seed)`` internally.
      - Model init and DataLoader shuffling use a ``torch.Generator``
        seeded from ``cfg.seed``.

    No global RNG state is modified.

    Parameters
    ----------
    cfg : SAEConfig, optional
        Experiment config. If None, a default is created.
    **overrides
        Override any SAEConfig field (e.g. sae_type="topk", seed=42).

    Returns
    -------
    dict with 'model', 'results', 'data', 'A', 'cfg' keys.
    """
    if cfg is None:
        cfg = SAEConfig(**overrides)
    else:
        for k, v in overrides.items():
            setattr(cfg, k, v)

    # Data generation — uses its own np.random.RandomState(seed) internally
    train, val, ood, A = generate_datasets(
        seed=cfg.seed, num_latents=cfg.num_latents, k=cfg.k, n_samples=cfg.n_samples
    )
    Z_train, Y_train, labels_train = train
    Z_val, Y_val, labels_val = val
    Z_ood, Y_ood, labels_ood = ood

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Torch generator for reproducible DataLoader shuffling
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(cfg.seed)

    # Build DataLoader for IID training Y data (unsupervised)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    train_loader = DataLoader(
        TensorDataset(Y_train_t),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch_gen,
    )

    # Eval tensors
    inputs_iid = torch.tensor(Y_val, dtype=torch.float32, device=device)
    inputs_ood = torch.tensor(Y_ood, dtype=torch.float32, device=device)

    # Observation dimension
    input_dim = Y_train.shape[1]

    # Seed torch before model init so weight initialization is reproducible
    torch.manual_seed(cfg.seed)
    model = SAE(
        input_dim=input_dim,
        width=cfg.width,
        sae_type=cfg.sae_type,
        kval_topk=cfg.kval_topk if cfg.sae_type == "topk" else None,
        mp_kval=cfg.mp_kval if cfg.sae_type == "MP" else None,
    )

    results = train_sae(
        model=model,
        train_loader=train_loader,
        inputs_iid=inputs_iid,
        inputs_ood=inputs_ood,
        device=device,
        cfg=cfg,
    )

    save_run(model, cfg, results)

    return {
        "model": model,
        "results": results,
        "cfg": cfg,
        "data": {
            "Z_train": Z_train, "Y_train": Y_train, "labels_train": labels_train,
            "Z_val": Z_val, "Y_val": Y_val, "labels_val": labels_val,
            "Z_ood": Z_ood, "Y_ood": Y_ood, "labels_ood": labels_ood,
        },
        "A": A,
    }


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    cfg = SAEConfig(
        num_latents=10, k=3, n_samples=2000, width=20,
        sae_type="relu", epochs=500, lr=5e-4, gamma_reg=1e-4,
        batch_size=64, seed=42,
    )
    out = run_sae_experiment(cfg)

    h = out["results"]["history"][-1]
    print(f"\nFinal IID MSE: {h['mse_iid']:.4f}")
    print(f"Final OOD MSE: {h['mse_ood']:.4f}")
    print(f"Final IID L0:  {h['l0_iid']:.1f}")
    print(f"Final OOD L0:  {h['l0_ood']:.1f}")
    print(f"Codes IID shape: {out['results']['codes_iid'].shape}")
    print(f"Codes OOD shape: {out['results']['codes_ood'].shape}")
