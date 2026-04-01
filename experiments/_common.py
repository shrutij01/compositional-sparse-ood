"""Shared helpers for experiment scripts."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.linear_probe import linear_probe_codes
from utils.metrics import evaluate_all


def _lazy_torch_imports():
    """Import torch and model modules lazily (not needed for linear baselines)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.saes import SAE, SAEConfig, train_sae
    from models.sparse_coding import SparseCodingConfig, train_sparse_coding, fista
    return torch, DataLoader, TensorDataset, SAE, SAEConfig, train_sae, SparseCodingConfig, train_sparse_coding, fista

def save_incremental(all_results, out_path):
    """Write all_results to out_path (creates parent dirs). Called after each config."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"  [checkpoint] {len(all_results)} results saved to {out_path}")


# SAE variants to compare.  topk and MP use structural sparsity (kval = k).
SAE_TYPES = ("relu", "topk", "jumprelu", "MP")


def eval_and_tag(codes_iid, codes_ood, data, method, **extra):
    """Run evaluate_all and tag the resulting metrics dict."""
    metrics = evaluate_all(
        codes_iid=codes_iid,
        labels_iid=data["labels_val"],
        codes_ood=codes_ood,
        labels_ood=data["labels_ood"],
        Z_true_iid=data["Z_val"],
        Z_true_ood=data["Z_ood"],
    )
    metrics["method"] = method
    metrics.update(extra)
    print(
        f"  [{method:20s}] acc_iid={metrics['acc_iid']:.3f}  "
        f"acc_ood={metrics['acc_ood']:.3f}  "
        f"mcc_iid={metrics.get('mcc_iid', 0):.3f}  "
        f"mcc_ood={metrics.get('mcc_ood', 0):.3f}  "
        f"auc_iid={metrics['auc_iid']:.3f}  "
        f"auc_ood={metrics['auc_ood']:.3f}"
    )
    return metrics


def run_linear_baselines(data, k, tag):
    """Evaluate supervised linear probe baseline (oracle ceiling).

    Returns list with one tagged metric dict.
    """
    results = []

    print("  Evaluating linear probe (oracle) baseline...")
    codes_iid, codes_ood = linear_probe_codes(
        data["Y_train"], data["Z_train"],
        data["Y_val"], data["Y_ood"],
    )
    results.append(eval_and_tag(
        codes_iid, codes_ood, data, "linear_probe", **tag,
    ))

    return results


def run_all_saes(
    data, input_dim, width, k, num_latents, n_samples, epochs, gamma_reg,
    seed, device, tag,
    sae_types=SAE_TYPES,
):
    """Train and evaluate all SAE variants on the same data.

    Returns list of tagged metric dicts (one per SAE type).
    """
    torch, DataLoader, TensorDataset, SAE, SAEConfig, train_sae, _, _, _ = _lazy_torch_imports()

    Y_train_t = torch.tensor(data["Y_train"], dtype=torch.float32, device=device)
    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    # Reproducible DataLoader shuffling
    loader_gen = torch.Generator(device="cpu")
    loader_gen.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(Y_train_t), batch_size=64, shuffle=True, generator=loader_gen,
    )

    results = []
    for sae_type in sae_types:
        print(f"  Training SAE ({sae_type})...")
        sae_cfg = SAEConfig(
            num_latents=num_latents, k=k, n_samples=n_samples, width=width,
            sae_type=sae_type,
            kval_topk=k if sae_type == "topk" else 3,
            mp_kval=k if sae_type == "MP" else 3,
            epochs=epochs, gamma_reg=gamma_reg,
            seed=seed, print_every=epochs,
        )
        # Seed torch for reproducible weight init
        torch.manual_seed(seed)
        sae = SAE(
            input_dim=input_dim, width=width, sae_type=sae_type,
            kval_topk=k if sae_type == "topk" else None,
            mp_kval=k if sae_type == "MP" else None,
        )
        sae_out = train_sae(sae, train_loader, X_iid, X_ood, device, sae_cfg)
        results.append(eval_and_tag(
            sae_out["codes_iid"], sae_out["codes_ood"],
            data, f"sae_{sae_type}", **tag,
        ))
    return results


def run_sparse_coding_methods(data, A, input_dim, num_latents, sc_lam,
                               sc_max_steps, seed, device, tag):
    """Train and evaluate sparse coding variants on the same data.

    Runs FISTA oracle, DL-FISTA, and Softplus-Adam.
    LISTA is available in models/sparse_coding.py but not run in sweep
    experiments (too slow at scale; run separately for illustrative purposes).

    Returns list of tagged metric dicts.
    """
    torch, _, _, _, _, _, SparseCodingConfig, train_sparse_coding, _ = _lazy_torch_imports()

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    results = []

    # FISTA oracle (known ground-truth dictionary A)
    print("  Training FISTA (oracle)...")
    sc_cfg = SparseCodingConfig(
        input_dim=input_dim, num_latents=num_latents, method="fista", lam=sc_lam,
        max_steps=sc_max_steps, n_iter=100, dict_update_every=50,
        supervised=True, seed=seed, print_every=sc_max_steps,
    )
    sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, A=A, device=device)
    results.append(eval_and_tag(
        sc_out["codes_iid"], sc_out["codes_ood"], data, "fista_oracle", **tag,
    ))

    # Dictionary Learning + FISTA (alternating: FISTA inference + LS dict update)
    print("  Training DL-FISTA...")
    sc_cfg.supervised = False
    sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, device=device)
    results.append(eval_and_tag(
        sc_out["codes_iid"], sc_out["codes_ood"], data, "dl_fista", **tag,
    ))

    # Softplus-Adam (joint code + dict optimization with softplus parameterization)
    print("  Training Softplus-Adam...")
    sc_cfg.method = "direct"
    sc_cfg.supervised = False
    sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, device=device)
    results.append(eval_and_tag(
        sc_out["codes_iid"], sc_out["codes_ood"], data, "softplus_adam", **tag,
    ))

    return results


# ============================================================================
# Frozen decoder + FISTA helpers
# ============================================================================


def train_all_saes(
    data, input_dim, width, k, num_latents, n_samples, epochs, gamma_reg,
    seed, device,
    sae_types=SAE_TYPES,
):
    """Train all SAE variants and return models + codes.

    Unlike run_all_saes(), this keeps the trained models alive for
    downstream use (e.g., extracting the decoder for FISTA).

    Returns
    -------
    dict mapping sae_type -> {
        "model": SAE,
        "codes_iid": np.ndarray,
        "codes_ood": np.ndarray,
    }
    """
    torch, DataLoader, TensorDataset, SAE, SAEConfig, train_sae, _, _, _ = _lazy_torch_imports()

    Y_train_t = torch.tensor(data["Y_train"], dtype=torch.float32, device=device)
    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    loader_gen = torch.Generator(device="cpu")
    loader_gen.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(Y_train_t), batch_size=64, shuffle=True, generator=loader_gen,
    )

    trained = {}
    for sae_type in sae_types:
        print(f"  Training SAE ({sae_type})...")
        sae_cfg = SAEConfig(
            num_latents=num_latents, k=k, n_samples=n_samples, width=width,
            sae_type=sae_type,
            kval_topk=k if sae_type == "topk" else 3,
            mp_kval=k if sae_type == "MP" else 3,
            epochs=epochs, gamma_reg=gamma_reg,
            seed=seed, print_every=epochs,
        )
        torch.manual_seed(seed)
        sae = SAE(
            input_dim=input_dim, width=width, sae_type=sae_type,
            kval_topk=k if sae_type == "topk" else None,
            mp_kval=k if sae_type == "MP" else None,
        )
        sae_out = train_sae(sae, train_loader, X_iid, X_ood, device, sae_cfg)
        trained[sae_type] = {
            "model": sae,
            "codes_iid": sae_out["codes_iid"],
            "codes_ood": sae_out["codes_ood"],
        }
    return trained


def run_frozen_decoder_fista(trained_saes, data, fista_lam, fista_iters, device, tag):
    """For each trained SAE, extract decoder and run FISTA (cold + warm-start).

    Produces three methods per SAE type:
      - sae_{type}:     amortized encoder codes (from training)
      - fista+{type}:   FISTA cold-start with frozen SAE decoder
      - refined_{type}: FISTA warm-started from SAE codes

    Parameters
    ----------
    trained_saes : dict from train_all_saes()
    data : standard data dict with Y_val, Y_ood, labels, Z_true
    fista_lam : L1 penalty for FISTA
    fista_iters : number of FISTA iterations
    device : torch device
    tag : dict of sweep-variable tags

    Returns
    -------
    list of tagged metric dicts
    """
    torch, _, _, _, _, _, _, _, fista = _lazy_torch_imports()

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    results = []
    for sae_type, info in trained_saes.items():
        sae = info["model"]
        sae_codes_iid = info["codes_iid"]
        sae_codes_ood = info["codes_ood"]

        # 1) SAE encoder codes
        results.append(eval_and_tag(
            sae_codes_iid, sae_codes_ood,
            data, f"sae_{sae_type}", **tag,
        ))

        # Extract frozen decoder as dictionary
        D = sae.Ad.data.detach()
        bd = sae.bd.data.detach()
        X_iid_c = X_iid - bd  # bias-corrected input
        X_ood_c = X_ood - bd

        # 2) FISTA cold-start with frozen decoder
        print(f"  FISTA cold-start with {sae_type} decoder...")
        with torch.no_grad():
            z_fista_iid = fista(X_iid_c, D, fista_lam, n_iter=fista_iters, nonneg=True)
            z_fista_ood = fista(X_ood_c, D, fista_lam, n_iter=fista_iters, nonneg=True)
        results.append(eval_and_tag(
            z_fista_iid.cpu().numpy(), z_fista_ood.cpu().numpy(),
            data, f"fista+{sae_type}", **tag,
        ))

        # 3) FISTA warm-started from SAE codes (refinement)
        print(f"  FISTA refinement from {sae_type} codes...")
        z_init_iid = torch.tensor(sae_codes_iid, dtype=torch.float32, device=device)
        z_init_ood = torch.tensor(sae_codes_ood, dtype=torch.float32, device=device)
        with torch.no_grad():
            z_ref_iid = fista(X_iid_c, D, fista_lam, n_iter=fista_iters,
                              z_init=z_init_iid, nonneg=True)
            z_ref_ood = fista(X_ood_c, D, fista_lam, n_iter=fista_iters,
                              z_init=z_init_ood, nonneg=True)
        results.append(eval_and_tag(
            z_ref_iid.cpu().numpy(), z_ref_ood.cpu().numpy(),
            data, f"refined_{sae_type}", **tag,
        ))

    return results


def run_warmstart_dl_fista(trained_saes, data, fista_lam, sc_max_steps, device, tag):
    """For each trained SAE, use its decoder as a warm-start for DL-FISTA.

    Unlike run_frozen_decoder_fista (which freezes the decoder), this refines
    both dictionary and codes via alternating optimisation.

    Produces one method per SAE type:
      - dl_fista+{type}: DL-FISTA warm-started from SAE decoder

    Parameters
    ----------
    trained_saes : dict from train_all_saes()
    data : standard data dict with Y_val, Y_ood, labels, Z_true
    fista_lam : L1 penalty for FISTA
    sc_max_steps : max optimisation steps for DL-FISTA
    device : torch device
    tag : dict of sweep-variable tags

    Returns
    -------
    list of tagged metric dicts
    """
    torch, _, _, _, _, _, SparseCodingConfig, train_sparse_coding, _ = _lazy_torch_imports()

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    results = []
    for sae_type, info in trained_saes.items():
        sae = info["model"]

        # Extract decoder and bias-correct observations
        D_sae = sae.Ad.data.detach().cpu().numpy()
        bd = sae.bd.data.detach()
        X_iid_c = X_iid - bd
        X_ood_c = X_ood - bd

        input_dim = D_sae.shape[0]
        num_latents = D_sae.shape[1]

        print(f"  DL-FISTA warm-started from {sae_type} decoder...")
        sc_cfg = SparseCodingConfig(
            input_dim=input_dim, num_latents=num_latents,
            method="fista", lam=fista_lam,
            max_steps=sc_max_steps, n_iter=100, dict_update_every=50,
            supervised=False, seed=tag.get("seed", 0),
            print_every=sc_max_steps,
        )
        sc_out = train_sparse_coding(
            X_iid_c, X_ood_c, sc_cfg, D_init=D_sae, device=device,
        )
        results.append(eval_and_tag(
            sc_out["codes_iid"], sc_out["codes_ood"],
            data, f"dl_fista+{sae_type}", **tag,
        ))

    return results


# ============================================================================
# Large latents helper
# ============================================================================


def run_large_latents_methods(data, A, input_dim, num_latents, sc_lam,
                               sc_max_steps, fista_lam, fista_iters,
                               epochs, gamma_reg, n_samples,
                               seed, device, tag):
    """Run methods for the large-latents experiment, skipping infeasible ones at scale.

    Tiers by num_latents:
      - Always: FISTA oracle, linear baselines (raw, PCA)
      - num_latents <= 50K: also DL-FISTA, Softplus-Adam, SAE + frozen decoder FISTA
      - 50K < num_latents <= 100K: also Softplus-Adam, SAE + frozen decoder FISTA
      - > 100K: FISTA oracle only (+ linear baselines)

    Returns list of tagged metric dicts.
    """
    torch, _, _, _, _, _, SparseCodingConfig, train_sparse_coding, _ = _lazy_torch_imports()

    k = tag["k"]
    results = []

    # --- Linear baselines (always) ---
    results.extend(run_linear_baselines(data, k, tag))

    # --- FISTA oracle (always) ---
    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    print("  Training FISTA (oracle)...")
    sc_cfg = SparseCodingConfig(
        input_dim=input_dim, num_latents=num_latents, method="fista", lam=sc_lam,
        max_steps=sc_max_steps, n_iter=fista_iters, dict_update_every=50,
        supervised=True, seed=seed, print_every=sc_max_steps,
    )
    sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, A=A, device=device)
    results.append(eval_and_tag(
        sc_out["codes_iid"], sc_out["codes_ood"], data, "fista_oracle", **tag,
    ))

    # --- DL-FISTA (num_latents <= 50K) ---
    # update_dictionary() computes ZtZ = Z.T @ Z which is (num_latents x num_latents)
    if num_latents <= 50_000:
        print("  Training DL-FISTA...")
        sc_cfg.supervised = False
        sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, device=device)
        results.append(eval_and_tag(
            sc_out["codes_iid"], sc_out["codes_ood"], data, "dl_fista", **tag,
        ))

    # --- Softplus-Adam (num_latents <= 100K) ---
    if num_latents <= 100_000:
        print("  Training Softplus-Adam...")
        sc_cfg.method = "direct"
        sc_cfg.supervised = False
        sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, device=device)
        results.append(eval_and_tag(
            sc_out["codes_iid"], sc_out["codes_ood"], data, "softplus_adam", **tag,
        ))

    # --- SAE + frozen decoder FISTA (num_latents <= 100K) ---
    if num_latents <= 100_000:
        width = num_latents
        trained_saes = train_all_saes(
            data, input_dim, width, k, num_latents, n_samples,
            epochs, gamma_reg, seed, device,
        )
        results.extend(run_frozen_decoder_fista(
            trained_saes, data, fista_lam, fista_iters, device, tag,
        ))

    return results


def print_summary(all_results, sweep_var, sweep_values):
    """Print a summary table averaged over seeds, broken out by method."""
    methods = sorted(set(r["method"] for r in all_results))
    header = (
        f"{sweep_var:>10} {'method':>20} "
        f"{'acc_iid':>8} {'acc_ood':>8} {'mcc_iid':>8} {'mcc_ood':>8} "
        f"{'auc_iid':>8} {'auc_ood':>8}"
    )
    print("\n--- Summary (mean over seeds) ---")
    print(header)
    for val in sweep_values:
        for method in methods:
            rows = [r for r in all_results if r[sweep_var] == val and r["method"] == method]
            if not rows:
                continue
            print(
                f"{val:>10} "
                f"{method:>20} "
                f"{np.mean([r['acc_iid'] for r in rows]):>8.3f} "
                f"{np.mean([r['acc_ood'] for r in rows]):>8.3f} "
                f"{np.mean([r.get('mcc_iid', 0) for r in rows]):>8.3f} "
                f"{np.mean([r.get('mcc_ood', 0) for r in rows]):>8.3f} "
                f"{np.mean([r['auc_iid'] for r in rows]):>8.3f} "
                f"{np.mean([r['auc_ood'] for r in rows]):>8.3f}"
            )
