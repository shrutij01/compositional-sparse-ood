import argparse
import random
from typing import Any, Dict, Tuple
from types import SimpleNamespace
import numpy as np
import hashlib, json, yaml
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields


import torch
from torch import Tensor
from torch.utils.data import DataLoader
import cooper
from box import Box

from sae.sae_utils import SimpleCPUData, Logger, renorm_decoder_cols
from sae.sae_utils import DictAE, SSAE, step_fn
from data import sample_iid, generate_matrix


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """
    Initialise every RNG we care about.

    Args
    ----
    seed : int
        Non-negative seed.
    deterministic : bool, default=True
        Toggle PyTorch deterministic algorithms and turn off CUDNN bench-
        mark mode (bench = speed, determinism = repro).

    Notes
    -----
    • `torch.cuda.manual_seed_all` is a no-op on CPU-only boxes.
    • We do *not* touch env-vars here (caller decides).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # reduces speed so only use when need bit-perfect reproducibility
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# 2. Project decoder gradients to the tangent space of unit-norm columns
#    g ← g − (wᵀg) w     (vectorised, no Python loop)
# ----------------------------------------------------------------------
@torch.jit.script
def project_decoder_grads_(W: Tensor) -> None:
    """
    In-place orthogonal-projection of ∇W so future SGD preserves
    unit-norm columns.  Assumes `W.grad` exists and `W` is already
    normalised.
    """
    g = W.grad
    if g is None:
        return  # nothing to do

    # (wᵀg) for each column  → shape (1, C)
    col_dot = (W * g).sum(dim=0, keepdim=True)
    # g ← g − w (wᵀg)
    g.sub_(W * col_dot)


KEYS = ("fanin", "hid", "n_concepts", "seed")  # choose what matters


def _hash_cfg(cfg) -> str:
    cfg_dict = asdict(cfg)
    # Convert Path objects and SimpleNamespace to JSON serializable types
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)  # Convert SimpleNamespace to dict
    blob = json.dumps(cfg_dict, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:6]  # short & stable


def dump_run(root: Path, model: torch.nn.Module, cfg, dataset=None) -> Path:
    from datetime import datetime
    import h5py

    tag = "_".join(f"{k}{getattr(cfg, k)}" for k in KEYS)
    now = datetime.now()
    time_id = (
        f"{now.strftime('%m%d')}{now.strftime('%H%M')}"  # MMDDHHMM format
    )
    run = root / f"{tag}_{_hash_cfg(cfg)}"
    run.mkdir(
        parents=True, exist_ok=True
    )  # Allow multiple runs with same config

    torch.save(model.state_dict(), run / f"weights_{time_id}.pth")

    # Convert config to YAML-serializable format
    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)

    (run / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))

    # Save synthetic dataset if provided
    if (
        dataset is not None
        and hasattr(dataset, "ground_truth")
        and dataset.ground_truth is not None
    ):
        dataset_file = run / f"synthetic_data_{time_id}.h5"
        with h5py.File(dataset_file, "w") as f:
            f.create_dataset(
                "observations", data=dataset.data, compression="gzip"
            )
            f.create_dataset(
                "ground_truth", data=dataset.ground_truth, compression="gzip"
            )
            f.create_dataset("labels", data=dataset.labels, compression="gzip")
            # Add metadata
            f.attrs["data_shape"] = dataset.data.shape
            f.attrs["ground_truth_shape"] = dataset.ground_truth.shape
            f.attrs["dataset_length"] = len(dataset)

    return run


def train_epoch(
    dataloader,
    dict,
    ssae,
    optim,
    cfg,
    dev,
) -> Tuple[float, float, float]:
    dict.train()

    # Accumulators
    recon_sum = 0.0
    sparsity_sum = 0.0
    concept_counts = 0

    # Pre-allocated tensors to eliminate dynamic allocation
    gpu_tensor = None
    threshold_mask = None
    batch_count = 0
    renorm_counter = 0

    # Get batch and hidden dimensions for pre-allocation
    sample_batch = next(iter(dataloader))
    batch_size, input_dim = sample_batch.shape
    hid_dim = cfg.hid

    # Pre-allocate all working tensors on GPU
    if torch.cuda.is_available():
        gpu_tensor = torch.empty(
            (batch_size, input_dim), device=dev, dtype=torch.float32
        )
        threshold_mask = torch.empty(
            (batch_size, hid_dim), device=dev, dtype=torch.bool
        )

    # Mixed precision scaler for 2x speedup
    scaler = (
        torch.cuda.amp.GradScaler()
        if cfg.use_amp and torch.cuda.is_available()
        else None
    )

    for batch_idx, delta_z_cpu in enumerate(dataloader):
        with torch.cuda.device(dev):
            # Efficient GPU transfer with pre-allocated tensor
            gpu_tensor.copy_(delta_z_cpu, non_blocking=True)

            # Zero gradients before forward pass
            optim_kwargs = {
                "model": dict,
                "delta_z": gpu_tensor,
                "loss_type": cfg.loss,
            }
            # Mixed precision forward pass
            if ssae:  # SSAE model
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        optim.roll(compute_cmp_state_kwargs=optim_kwargs)
                    # Note: Cooper handles its own scaling internally
                else:
                    optim.roll(compute_cmp_state_kwargs=optim_kwargs)
            else:  # Regular DictAE model with PyTorch optimizer (AdamW)
                optim.zero_grad()

                # Helper function for loss computation
                def compute_loss(computed, target):
                    if cfg.loss == "relative":
                        return (
                            torch.sum((computed - target).pow(2), dim=1)
                            / torch.sum(target.pow(2), dim=1)
                        ).mean()
                    else:  # absolute
                        return torch.nn.functional.mse_loss(
                            computed, target, reduction="mean"
                        )

                # Helper function for sparsity regularization
                def compute_sparsity_loss(h):
                    if cfg.activation == "relu":
                        return torch.norm(h, p=1, dim=-1).mean()
                    elif cfg.activation == "jumprelu":
                        bandwidth = 1e-3
                        return torch.mean(
                            torch.sum(
                                step_fn(
                                    h,
                                    torch.exp(dict.activation.logthreshold),
                                    bandwidth,
                                ),
                                dim=-1,
                            )
                        )
                    else:  # identity, etc.
                        return torch.norm(h, p=1, dim=-1).mean()

                # Forward and backward pass with optional mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        delta_z_hat, h = dict(gpu_tensor)
                        recon_loss = compute_loss(delta_z_hat, gpu_tensor)
                        sparsity_loss = compute_sparsity_loss(h)
                        total_loss = recon_loss + cfg.beta * sparsity_loss
                    scaler.scale(total_loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    delta_z_hat, h = dict(gpu_tensor)
                    recon_loss = compute_loss(delta_z_hat, gpu_tensor)
                    sparsity_loss = compute_sparsity_loss(h)
                    total_loss = recon_loss + cfg.beta * sparsity_loss
                    total_loss.backward()
                    optim.step()

            # Post-step weight operations
            with torch.no_grad():
                if dict.decoder.weight.grad is not None:
                    project_decoder_grads_(dict.decoder.weight)

                # Renormalize decoder columns at configured frequency
                renorm_counter += 1
                if renorm_counter % cfg.renorm_epochs == 0:
                    renorm_decoder_cols(dict.decoder.weight)

            # Extract metrics
            if ssae is not None:  # SSAE model
                recon_loss_val, sparsity_penalty = ssae.metrics()
                recon_sum += recon_loss_val
                sparsity_sum += sparsity_penalty

                # Efficient sparsity counting using cached hidden states
                with torch.no_grad():
                    h_cached = (
                        ssae.get_cached_hidden()
                    )  # Reuse from forward pass
                    torch.ge(
                        torch.abs(h_cached), cfg.ind_th, out=threshold_mask
                    )
                    concept_counts += threshold_mask.sum().item()
            else:  # DictAE model
                # Use the computed losses from forward pass
                recon_sum += recon_loss.item()
                sparsity_penalty = sparsity_loss.item() - cfg.target
                sparsity_sum += sparsity_penalty

                # Use hidden activations from forward pass for sparsity counting
                with torch.no_grad():
                    torch.ge(torch.abs(h), cfg.ind_th, out=threshold_mask)
                    concept_counts += threshold_mask.sum().item()

            batch_count += 1

            # Memory cleanup every 100 batches (less frequent, more efficient)
            if batch_count % 100 == 0:
                torch.cuda.empty_cache()

    # Final cleanup
    del gpu_tensor, threshold_mask
    torch.cuda.empty_cache()

    return recon_sum, sparsity_sum, concept_counts


def main():
    cfg = parse_cfg()
    set_seeds(cfg.seed)

    logger = Logger(
        entity="causalrepl", project="sparseood", config=asdict(cfg)
    )
    dataloader = make_dataloader(cfg)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if cfg.model not in ["sae", "ssae"]:
        raise ValueError(f"Unknown model {cfg.model}")
    dict = make_dict(cfg).to(dev)
    ssae = make_ssae(cfg, dev) if cfg.model == "ssae" else None
    optim = make_optim(dict=dict, ssae=ssae, cfg=cfg)

    for ep in range(cfg.epochs):
        if ssae is not None:
            ssae.epoch = ep  # Update epoch for sparsity scheduling
        total_recon_loss, total_sparsity_defect, total_active_concepts = (
            train_epoch(
                dataloader,
                dict=dict,
                ssae=ssae,
                optim=optim,
                cfg=cfg,
                dev=dev,
            )
        )

        # Calculate correct averages
        num_batches = len(dataloader)
        dataset_size: int = len(dataloader.dataset)

        # Log epoch metrics with correct normalization
        epoch_metrics = {
            "epoch": ep,
            "recon_loss": total_recon_loss
            / num_batches,  # Average loss per batch
            "l0_sparsity": total_active_concepts
            / dataset_size,  # Number of active neurons
            "constraint_violation": total_sparsity_defect
            / num_batches,  # Same as sparsity_defect but clearer name
        }

        for k, v in epoch_metrics.items():
            logger.logkv(k, v)

        # Log memory stats every 10 epochs
        if ep % 10 == 0:
            logger.log_memory_stats()

        logger.dumpkvs()
        print(
            f"ep {ep:04d}  recon_loss {epoch_metrics['recon_loss']:.4f}  "
            f"sparsity_defect {epoch_metrics['constraint_violation']:.4f}  "
            f"l0 {epoch_metrics['l0_sparsity']:.4f}"
        )

    dump_run(Path(".") / "run_out", dict, cfg, dataloader.dataset)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return Box(yaml.safe_load(fh) or {})


# ===== run config ============================================================
@dataclass(frozen=True)
class Cfg:
    datatype: str  # "emb" or "synth"
    emb: Path
    fanin: int  # input dimension
    batch: int = 32
    model: str = "ssae"
    activation: str = "identity"
    epochs: int = 100
    lr: float = 5e-4
    hid: int = 4096  # don't exceed embedding dim for injectivity
    n_concepts: int = 1
    n_iid: int = 10000  # only for synthetic data
    label_iid: float = 0.5  # only for synthetic data
    warmup: int = 2_000
    schedule: int = 5_000
    target: float = 0.1
    beta: float = 0.001  # only for sae
    norm: str = "ln"
    loss: str = "relative"
    ind_th: float = 0.1
    seed: int = 0
    renorm_epochs: int = 10
    use_amp: bool = True
    activation: str = "relu"
    use_lambda: bool = False

    # spill-over lives here (read-only)
    extra: Dict[str, Any] = field(default_factory=dict, init=False)


def parse_cfg() -> Cfg:
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("datatype", choices=["emb", "synth"])
    add("--emb", type=Path)
    add("--emb-cfg", type=Path)
    add("--model", choices=["ssae", "sae"], default="ssae")
    add(
        "--activation",
        choices=["relu", "jumprelu", "identity"],
        default="identity",
    )
    add(
        "--use-lambda",
        action="store_true",
        help="For sparsity control with threshold activation functions",
    )
    add(
        "--beta",
        type=float,
        default=0.001,
        help="Sparsity penalty weight for SAEs",
    )
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=100)
    add("--lr", type=float, default=5e-4)
    add("--hid", type=int, default=4096)
    add("--fanin", type=int, help="Input dimension (required for synth data)")
    add("--n-concepts", "-C", type=int, default=1)
    add("--warmup", type=int, default=2_000)
    add("--schedule", type=int, default=10)
    add("--target", type=float, default=0.1)
    add("--norm", choices=["ln", "gn", "bn"], default="ln")
    add("--loss", default="relative", choices=["relative", "absolute"])
    add("--ind-th", type=float, default=0.1)
    add("--seed", type=int, default=0)
    add("--renorm-epochs", type=int, default=50)
    add("--use-amp", action="store_true", default=True)

    cli: Dict[str, Any] = vars(p.parse_args())

    # Handle YAML config loading based on datatype
    if cli["datatype"] == "emb":
        if not cli.get("emb_cfg"):
            raise ValueError("--emb-cfg is required for emb datatype")
        yaml_path = cli.pop("emb_cfg")
        yaml_cfg = _load_yaml(yaml_path)
        # Split YAML into (known, extra)
        field_names = {f.name for f in fields(Cfg)}
        extra = {k: v for k, v in yaml_cfg.items() if k not in field_names}
        # For emb data, fanin comes from YAML (rep_dim)
        cli["fanin"] = yaml_cfg.get("rep_dim")
    else:  # synth datatype
        if not cli.get("fanin"):
            raise ValueError("--fanin is required for synth datatype")
        extra = {}
        # Remove emb_cfg if provided (not needed for synth)
        cli.pop("emb_cfg", None)

    cfg = Cfg(**cli)
    object.__setattr__(cfg, "extra", SimpleNamespace(**extra))

    return cfg


def make_dataloader(cfg) -> DataLoader:
    # Optimize for memory efficiency over throughput
    if cfg.datatype == "emb":
        dataset = SimpleCPUData(h5_path_or_data=cfg.emb, key="cfc_train")
    elif cfg.datatype == "synth":
        # m : fanin, n : hid, k : n_concepts
        gtz_iid = np.array(
            [
                sample_iid(seed=cfg.seed, n=cfg.hid, k=cfg.n_concepts)
                for _ in range(cfg.n_iid)
            ]
        )
        A = generate_matrix(m=cfg.fanin, n=cfg.hid, seed=cfg.seed)
        z_iid = gtz_iid @ A.T
        label_iid = z_iid[:, 0] > cfg.label_iid
        dataset = SimpleCPUData(
            h5_path_or_data=z_iid, ground_truth=gtz_iid, labels=label_iid
        )
    else:
        raise ValueError(f"Unknown data type {cfg.datatype}")
    train_loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch),
        shuffle=True,
        num_workers=1,  # Reduced to minimize memory overhead
        prefetch_factor=1,  # Minimal prefetch to reduce memory
        persistent_workers=False,  # Let workers die to free memory
        pin_memory=torch.cuda.is_available(),  # Only if GPU available
        drop_last=True,  # Consistent batch sizes for memory pool
    )

    # Store dataset reference for cleanup
    train_loader._dataset_ref = dataset
    return train_loader


def make_dict(cfg: Cfg) -> torch.nn.Module:
    return DictAE(cfg.fanin, cfg.hid, cfg.norm)


def make_ssae(cfg: Cfg, dev: torch.device):
    return SSAE(
        dev=dev,
        batch=cfg.batch,
        hid=cfg.hid,
        n_concepts=cfg.n_concepts,
        warmup=cfg.warmup,
        schedule=cfg.schedule,
        target=cfg.target,
    )


def make_optim(dict: torch.nn.Module, ssae: None, cfg: Cfg):
    if ssae:
        primal_optimizer = cooper.optim.ExtraAdam(dict.parameters(), lr=cfg.lr)
        dual_lr = cfg.lr / 2  # dual_lr should be less than primal_lr
        # 0.5 is common from extra gradient method literature
        # todo: how to select the dual optimizer?
        dual_optimizer = cooper.optim.ExtraAdam(
            ssae.dual_parameters(), lr=dual_lr, maximize=True
        )

        # Setup the constrained optimizer using Cooper's Lagrangian formulation
        optimizer = cooper.optim.ExtrapolationConstrainedOptimizer(
            cmp=ssae,
            primal_optimizers=primal_optimizer,
            dual_optimizers=dual_optimizer,
        )
    else:
        optimizer = torch.optim.AdamW(dict.parameters(), lr=cfg.lr)
        # use Adam with proper weight decay
    return optimizer


if __name__ == "__main__":
    # with dbg.debug_on_exception():
    main()
