"""
SAEBench adapters for the sparse coding methods dl_firsta and softplus_adam.

Both classes wrap a pre-trained dictionary inside BaseSAE so that SAEBench
evaluations (sparse probing, etc.) can call encode() / decode() / forward()
exactly as they would on any other SAE.

Usage
-----
    from sae_bench_adapter.sparse_coding_sae import DLFistaSAE, SoftplusAdamSAE

    # dictionary: numpy array (d_in, d_sae) from train_sparse_coding()["dictionary"]
    sae = DLFistaSAE.from_trained(dictionary, model_name="pythia-70m-deduped",
                                   hook_layer=4, device=device, dtype=dtype)

    # or for the softplus-adam variant:
    sae = SoftplusAdamSAE.from_trained(dictionary, model_name="pythia-70m-deduped",
                                        hook_layer=4, device=device, dtype=dtype)

    selected_saes = [("dl_fista_layer4", sae)]
    results = run_eval(config, selected_saes, device, output_path)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Resolve project root and SAEBench root from this file's location
_ADAPTER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ADAPTER_DIR.parent
_SAEBENCH_ROOT = _PROJECT_ROOT / "SAEBench"

for _p in [str(_PROJECT_ROOT), str(_SAEBENCH_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.sparse_coding import fista  # noqa: E402
from sae_bench.custom_saes.base_sae import BaseSAE  # noqa: E402


# ---------------------------------------------------------------------------
# DL-FISTA SAE
# ---------------------------------------------------------------------------


class DLFistaSAE(BaseSAE):
    """
    SAEBench adapter for the DL-FISTA method.

    The dictionary is learned offline via alternating FISTA + closed-form
    dictionary update (train_sparse_coding(method="fista")).
    At inference time, encode() runs FISTA with the fixed learned dictionary.

    W_dec shape (SAEBench convention): (d_sae, d_in)
    Dictionary shape (sparse_coding convention): (d_in, d_sae) = W_dec.T
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
        lam: float = 0.1,
        n_iter: int = 200,
        nonneg: bool = True,
    ):
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.lam = lam
        self.n_iter = n_iter
        self.nonneg = nonneg
        self.cfg.architecture = "dl_fista"
        self.cfg.activation_fn_str = "fista"

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run FISTA on x using the learned dictionary (W_dec.T).

        Handles both 2-D (batch, d_in) and 3-D (batch, seq_len, d_in) inputs.
        """
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).float()
        D = self.W_dec.detach().T.float()  # (d_in, d_sae)
        z = fista(x_2d, D, self.lam, n_iter=self.n_iter, nonneg=self.nonneg)
        return z.to(dtype=self.dtype).reshape(*original_shape[:-1], -1)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @classmethod
    def from_trained(
        cls,
        dictionary: np.ndarray,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
        lam: float = 0.1,
        n_iter: int = 200,
        nonneg: bool = True,
    ) -> "DLFistaSAE":
        """Load from a pre-trained dictionary.

        Parameters
        ----------
        dictionary : np.ndarray, shape (d_in, d_sae)
            Output of train_sparse_coding()["dictionary"].
        """
        d_in, d_sae = dictionary.shape
        sae = cls(d_in, d_sae, model_name, hook_layer, device, dtype,
                  hook_name, lam, n_iter, nonneg)
        D = torch.tensor(dictionary, dtype=dtype, device=device)
        # SAEBench W_dec: (d_sae, d_in) = D.T
        sae.W_dec.data = D.T.contiguous()
        # Ensure unit-norm rows (SAEBench convention)
        norms = torch.norm(sae.W_dec.data, dim=1, keepdim=True).clamp(min=1e-8)
        sae.W_dec.data = sae.W_dec.data / norms
        return sae


# ---------------------------------------------------------------------------
# Softplus-Adam SAE
# ---------------------------------------------------------------------------


class SoftplusAdamSAE(BaseSAE):
    """
    SAEBench adapter for the Softplus-Adam method.

    The dictionary is learned offline via joint Adam optimization of codes
    and dictionary (train_sparse_coding(method="direct")).
    At inference time, encode() runs Adam to optimize pre-activation codes
    with the fixed learned dictionary — faithful to the training procedure.

    If speed is a concern, set n_encode_steps to a smaller value (e.g. 100)
    at the cost of slightly worse code quality.

    W_dec shape (SAEBench convention): (d_sae, d_in)
    Dictionary shape (sparse_coding convention): (d_in, d_sae) = W_dec.T
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
        lam: float = 0.1,
        n_encode_steps: int = 300,
        encode_lr: float = 1e-2,
    ):
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.lam = lam
        self.n_encode_steps = n_encode_steps
        self.encode_lr = encode_lr
        self.cfg.architecture = "softplus_adam"
        self.cfg.activation_fn_str = "softplus"

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Optimize pre-activation codes with Adam, then apply softplus.

        Handles both 2-D (batch, d_in) and 3-D (batch, seq_len, d_in) inputs.
        """
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).detach().float()
        n_samples = x_2d.shape[0]

        D = self.W_dec.detach().T.float()  # (d_in, d_sae)
        D_norms = torch.linalg.norm(D, dim=0)

        pre_z = torch.full(
            (n_samples, self.cfg.d_sae), -10.0,
            device=self.device, dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([pre_z], lr=self.encode_lr)

        with torch.enable_grad():
            for _ in range(self.n_encode_steps):
                z = F.softplus(pre_z)
                rec = z @ D.T
                loss = F.mse_loss(rec, x_2d) + self.lam * (z * D_norms).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            z = F.softplus(pre_z).to(dtype=self.dtype)

        return z.reshape(*original_shape[:-1], -1)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @classmethod
    def from_trained(
        cls,
        dictionary: np.ndarray,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
        lam: float = 0.1,
        n_encode_steps: int = 300,
        encode_lr: float = 1e-2,
    ) -> "SoftplusAdamSAE":
        """Load from a pre-trained dictionary.

        Parameters
        ----------
        dictionary : np.ndarray, shape (d_in, d_sae)
            Output of train_sparse_coding()["dictionary"].
        """
        d_in, d_sae = dictionary.shape
        sae = cls(d_in, d_sae, model_name, hook_layer, device, dtype,
                  hook_name, lam, n_encode_steps, encode_lr)
        D = torch.tensor(dictionary, dtype=dtype, device=device)
        sae.W_dec.data = D.T.contiguous()
        norms = torch.norm(sae.W_dec.data, dim=1, keepdim=True).clamp(min=1e-8)
        sae.W_dec.data = sae.W_dec.data / norms
        return sae
