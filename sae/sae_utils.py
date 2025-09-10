from typing import Dict
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import wandb
import cooper

# import debug_tools as dbg  # Commented out to fix sys.excepthook error


def rectangle(x):
    """Rectangle function for smooth gradient approximation."""
    return ((x >= -0.5) & (x <= 0.5)).float()


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
        grad_input = 0.0 * grad_output  # no ste to input
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return (
            grad_input,
            grad_threshold,
            None,
        )  # None for bandwidth since const


def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)


class JumpReLU(nn.Module):
    """
    JumpReLU activation function with per-neuron learnable thresholds
    and smooth gradient approximation.
    """

    def __init__(self, width: int):
        super().__init__()
        # Per-neuron log thresholds (ensures positive thresholds)
        self.logthreshold = nn.Parameter(
            torch.log(1e-3 * torch.ones((1, width)))
        )
        self.bandwidth = 1e-3  # Width of rectangle for gradient approximation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = torch.exp(self.logthreshold)
        return self._JumpReLUFunction.apply(x, threshold, self.bandwidth)

    class _JumpReLUFunction(torch.autograd.Function):
        """Custom autograd function for JumpReLU with smooth gradients."""

        @staticmethod
        def forward(ctx, x, threshold, bandwidth):
            if not isinstance(bandwidth, torch.Tensor):
                bandwidth = torch.tensor(
                    bandwidth, dtype=x.dtype, device=x.device
                )
            ctx.save_for_backward(x, threshold, bandwidth)
            return x * (x > threshold)

        @staticmethod
        def backward(ctx, grad_output):
            x, threshold, bandwidth = ctx.saved_tensors

            # Compute gradients
            x_grad = (x > threshold).float() * grad_output
            threshold_grad = (
                -(threshold / bandwidth)
                * JumpReLU._rectangle((x - threshold) / bandwidth)
                * grad_output
            ).sum(
                dim=0, keepdim=True
            )  # Aggregating across batch dimension

            return (
                x_grad,
                threshold_grad,
                None,
            )  # None for bandwidth since const


# ----------------------------------------------------------------------
# 1. Hard projection: W ← W / ||W||₂ column-wise
# ----------------------------------------------------------------------
@torch.jit.script
def renorm_decoder_cols(W: Tensor, eps: float = 1e-8) -> None:
    """
    In-place column l_2 normalisation.
    Safe for zero columns (leaves them unchanged).
    """
    col_norms = W.norm(p=2, dim=0, keepdim=True).clamp_min_(eps)
    W.data.div_(col_norms)


class DictAE(nn.Module):
    """
    Implements:
        coefficients = encoder(delta_z) + encoder_bias
        recons = decoder(coefficients) + decoder_bias
        delta_z --> rep_dim x 1, coefficients --> num_concepts x 1
        encoder --> num_concepts x rep_dim, decoder --> rep_dim x num_concepts
        encoder_bias --> num_concepts x 1, decoder_bias --> rep_dim x 1
    """

    def __init__(
        self,
        rep_dim: int,
        hid: int,
        norm_type: str,
        activation_type: str = "relu",
        use_lambda: bool = False,
    ) -> None:
        super(DictAE, self).__init__()
        self.encoder = nn.Linear(rep_dim, hid, bias=True)
        self.input_norm = nn.LayerNorm(rep_dim)
        self.enc_norm = dict(
            ln=nn.LayerNorm(hid),
            gn=nn.GroupNorm(1, hid),
            bn=nn.BatchNorm1d(hid),
        ).get(norm_type) or self._bad_norm(norm_type)

        # Lambda scaling parameter (only for ReLU-based activations)
        if use_lambda and activation_type.lower() not in ["relu", "jumprelu"]:
            raise ValueError(
                f"Lambda scaling is only supported with ReLU-based activations (relu, jumprelu), "
                f"but got activation_type='{activation_type}'. Lambda scaling doesn't make sense with "
                f"'{activation_type}' activation as it cannot control sparsity."
            )

        self.use_lambda = use_lambda and activation_type.lower() in [
            "relu",
            "jumprelu",
        ]
        if self.use_lambda:
            self.lambda_pre = nn.Parameter(
                torch.tensor(0.0)
            )  # Will be passed through softplus

        # Set activation function
        self.activation = dict(
            relu=nn.ReLU(),
            jumprelu=JumpReLU(hid),
            identity=nn.Identity(),
        ).get(activation_type.lower()) or self._bad_activation(activation_type)

        self.decoder = nn.Linear(hid, rep_dim, bias=True)
        # copy, do NOT tie; we only want identical init
        self.decoder.weight.data.copy_(self.encoder.weight.T)
        renorm_decoder_cols(self.decoder.weight)

    @staticmethod
    def _bad_norm(nt):
        raise ValueError(f"norm_type {nt!r} not in ln/gn/bn")

    @staticmethod
    def _bad_activation(at):
        raise ValueError(
            f"activation_type {at!r} not in relu/jumprelu/identity"
        )

    @property
    def lambda_val(self):
        """Lambda value forced to be positive via softplus."""
        if hasattr(self, "lambda_pre"):
            return F.softplus(self.lambda_pre)
        else:
            return 1.0  # Default scaling when lambda is not used

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.input_norm(x)
        # trainable layernorm at the input
        x_biased = x_norm - self.decoder.bias
        h_encoded = self.encoder(x_biased)
        h_normalized = self.enc_norm(h_encoded)

        # Apply lambda scaling after normalization (only for ReLU-based activations)
        if self.use_lambda:
            h_scaled = self.lambda_val * h_normalized
            h = self.activation(h_scaled)
        else:
            h = self.activation(h_normalized)

        # Decode and denormalize efficiently
        x_hat = self.decoder(h)
        return x_hat, h

    # remove denormalisation at the output


class SSAE(cooper.ConstrainedMinimizationProblem):
    """MSE + l_1 sparsity constraint on a linear schedule."""

    _LOSS: dict[str, callable] = {
        "relative": lambda z, z_hat: (
            torch.sum((z_hat - z).pow(2), dim=1) / torch.sum(z.pow(2), dim=1)
        ).mean(),  # Vectorized relative loss computation
        "absolute": lambda z, z_hat: nn.functional.mse_loss(
            z_hat, z, reduction="mean"
        ),
    }

    def __init__(
        self,
        dev: torch.device,  # Device for the model
        *,
        batch: int,
        hid: int,
        n_concepts: int,
        warmup: int = 2_000,
        schedule: int = 5_000,
        target: float = 0.10,
    ) -> None:
        super().__init__()
        self.batch, self.hid = batch, hid

        # sparsity schedule
        self._lvl0 = n_concepts
        self._lvl1 = target
        self._warm = warmup
        self._T = max(schedule, 1)
        self.level = n_concepts

        self.epoch = 0

        # Cache last forward pass results to avoid recomputation
        self._cached_h: torch.Tensor | None = None
        self.last_cmp_state: cooper.CMPState | None = None
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1).to(
            dev
        )

        self.sparsity_constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.Lagrangian,
            multiplier=multiplier,
        )

    def _tick_schedule(self) -> None:
        if self.epoch < self._warm:
            return
        t = (self.epoch - self._warm) / self._T
        self.level = self._lvl0 + min(1.0, t) * (self._lvl1 - self._lvl0)

    def compute_cmp_state(
        self,
        model: nn.Module,
        delta_z: Tensor,
        loss_type: str = "relative",
    ) -> cooper.CMPState:
        """Return CMPState used by Cooper’s optimiser."""
        delta_z_hat, h = model(delta_z)
        loss = self._LOSS[loss_type](delta_z, delta_z_hat)

        # Cache hidden states for sparsity computation
        self._cached_h = h
        actual_batch_size = h.shape[0]

        self._tick_schedule()
        ineq = h.abs().sum() / (actual_batch_size * self.hid) - self.level
        constraint_state = cooper.ConstraintState(violation=ineq)
        observed_constraints = {self.sparsity_constraint: constraint_state}
        cmp_state = cooper.CMPState(
            loss=loss, observed_constraints=observed_constraints
        )
        self.last_cmp_state = cmp_state  # Cache for metrics
        return cmp_state

    # ──────────────────────────────────────────────────────────────
    def metrics(self) -> tuple[float, float]:
        """(loss, ineq) as floats – call *after* the last closure."""
        if self.last_cmp_state is None:
            raise RuntimeError(
                "metrics() called before compute_cmp_state(). No cached state available."
            )
        state = self.last_cmp_state
        return float(state.loss), float(
            state.observed_constraints[self.sparsity_constraint].violation
        )

    def get_cached_hidden(self) -> Tensor:
        """Get hidden states from last forward pass to avoid recomputation."""
        if self._cached_h is None:
            raise RuntimeError(
                "get_cached_hidden() called before compute_cmp_state()"
            )
        return self._cached_h


class Logger:
    "A thin, side-effect-free wrapper around wandb with memory monitoring."

    def __init__(self, **kws):
        self._run = wandb.init(
            settings=wandb.Settings(code_dir="."),
            dir="./wandb",
            **kws,
        )
        self._buf: Dict[str, float] = {}

    def logkv(self, k, v):
        self._buf[k] = float(v) if torch.is_tensor(v) else v
        return v

    def log_memory_stats(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            self._buf.update(
                {
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
                    / 1024**2,
                    "gpu_memory_reserved_mb": torch.cuda.memory_reserved()
                    / 1024**2,
                    "gpu_memory_cached_mb": torch.cuda.memory_cached()
                    / 1024**2,
                }
            )

    def dumpkvs(self):
        if self._buf:
            self._run.log(self._buf)
            self._buf = {}


class SimpleCPUData(Dataset):
    def __init__(
        self,
        h5_path_or_data=None,
        key: str = "cfc_train",
        ground_truth=None,
        labels=None,
    ):
        """
        Initialize dataset from either h5 file or numpy array.

        Args:
            h5_path_or_data: Either path to h5 file or numpy array (observations)
            key: Key for h5 file data
            ground_truth: Ground truth sparse vectors (gtz) - optional for synthetic data
            labels: Binary/categorical labels - optional for synthetic data
        """
        if isinstance(h5_path_or_data, (str, Path)):
            with h5py.File(h5_path_or_data, "r") as f:
                self.data = f[key][:]  # Load entire dataset into RAM
        else:
            # Assume it's already a numpy array (z_iid - compressed observations)
            self.data = h5_path_or_data

        self.ground_truth = ground_truth  # gtz_iid
        self.labels = labels  # labels_iid

        # Validate lengths match for synthetic data
        if ground_truth is not None and len(self.data) != len(ground_truth):
            raise ValueError("Data and ground_truth must have same length")
        if labels is not None and len(self.data) != len(labels):
            raise ValueError("Data and labels must have same length")

    def __len__(self):
        if hasattr(self, "ground_truth") and self.ground_truth is not None:
            # Synthetic data case - return actual length
            return len(self.data)
        else:
            # H5 embedding case - return length-1 for delta computation
            return (
                len(self.data) - 1
                if len(self.data.shape) == 2
                else len(self.data)
            )

    def __getitem__(self, idx):
        if hasattr(self, "ground_truth") and self.ground_truth is not None:
            # Synthetic data case - return observations directly
            return torch.from_numpy(self.data[idx]).float()
        else:
            # H5 embedding case - compute deltas
            if len(self.data.shape) == 3:  # (N, 2, rep_dim)
                return torch.from_numpy(self.data[idx, 1] - self.data[idx, 0])
            else:  # (N, rep_dim)
                return torch.from_numpy(self.data[idx + 1] - self.data[idx])
