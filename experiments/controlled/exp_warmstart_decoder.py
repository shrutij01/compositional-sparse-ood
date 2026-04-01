"""
Experiment: Warm-Start DL-FISTA from SAE Decoder.

Question: Does the SAE learn a dictionary that's at least a useful starting
point, even if it's not good enough on its own?

The frozen-decoder experiment (exp_frozen_decoder.py) showed that running
FISTA with a *frozen* SAE decoder yields poor MCC -- the SAE decoder is not
a good dictionary by itself.  But it might still be a useful *initialization*
for dictionary learning.

This experiment trains each SAE type, extracts its decoder, and uses it as
the initial dictionary for unsupervised DL-FISTA (alternating FISTA inference
+ least-squares dictionary updates).  We sweep the number of DL-FISTA outer
rounds and compare convergence from the SAE decoder vs from random init.

Unlike the encoder warm-start experiment (which is convex and must converge
to the same optimum), DL-FISTA is non-convex -- the two initializations may
converge to different local minima.  Whether the SAE decoder leads to a
better or worse basin is an empirical question.

For each (num_latents, sae_type, seed) and each outer-round budget:
  cold_dl_{type}@N  -- DL-FISTA from random init, N dict update rounds
  warm_dl_{type}@N  -- DL-FISTA from SAE decoder, N dict update rounds
  fista_oracle      -- FISTA with ground-truth A (ceiling)

Sweeps num_latents while holding k fixed.  Observation dimension
input_dim is set from the compressed-sensing bound via param_check.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from experiments._common import (
    eval_and_tag,
    train_all_saes,
    save_incremental,
    SAE_TYPES,
)
from experiments.param_check import get_frozen_decoder_configs

# Number of dictionary update rounds to evaluate at.
# n_outer=0 means frozen decoder (no dict updates), matching exp_frozen_decoder.
OUTER_ROUND_BUDGETS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)

# Each outer round runs this many FISTA iterations internally.
DICT_UPDATE_EVERY = 50
FISTA_ITERS_PER_ROUND = 100


def run_decoder_convergence_sweep(
    trained_saes, data, A, fista_lam, device, tag,
):
    """Sweep DL-FISTA outer rounds for random-init and SAE-decoder-init.

    For each SAE type and each outer-round budget, runs:
      - cold_dl_{type}: DL-FISTA from random dictionary
      - warm_dl_{type}: DL-FISTA from SAE decoder

    At n_outer=0, cold is FISTA with random dict (uninformative), warm is
    FISTA with frozen SAE decoder (same as fista+{type} in exp_frozen_decoder).

    Returns list of tagged metric dicts, each with an 'n_outer' field.
    """
    from models.sparse_coding import SparseCodingConfig, train_sparse_coding, fista

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    input_dim = X_iid.shape[1]
    seed = tag.get("seed", 0)

    results = []
    for sae_type, info in trained_saes.items():
        sae = info["model"]

        # SAE decoder as warm-start dictionary
        D_sae = sae.Ad.data.detach().cpu().numpy()
        bd = sae.bd.data.detach()
        X_iid_c = X_iid - bd
        X_ood_c = X_ood - bd

        num_latents = D_sae.shape[1]

        for n_outer in OUTER_ROUND_BUDGETS:
            round_tag = {**tag, "n_outer": n_outer}

            if n_outer == 0:
                # Frozen decoder: single FISTA pass, no dict updates
                print(f"  {sae_type} n_outer=0 (frozen)...", end=" ", flush=True)

                with torch.no_grad():
                    D_sae_t = torch.tensor(D_sae, dtype=torch.float32, device=device)
                    z_warm_iid = fista(X_iid_c, D_sae_t, fista_lam,
                                       n_iter=FISTA_ITERS_PER_ROUND, nonneg=True)
                    z_warm_ood = fista(X_ood_c, D_sae_t, fista_lam,
                                       n_iter=FISTA_ITERS_PER_ROUND, nonneg=True)
                warm_metrics = eval_and_tag(
                    z_warm_iid.cpu().numpy(), z_warm_ood.cpu().numpy(),
                    data, f"warm_dl_{sae_type}", **round_tag,
                )
                results.append(warm_metrics)

                # Cold at n_outer=0 is FISTA with random dict -- skip, uninformative
                print(f"warm_mcc={warm_metrics.get('mcc_ood', 0):.3f}")
                continue

            max_steps = n_outer * DICT_UPDATE_EVERY
            print(f"  {sae_type} n_outer={n_outer}...", end=" ", flush=True)

            # --- Warm: DL-FISTA from SAE decoder ---
            sc_cfg = SparseCodingConfig(
                input_dim=input_dim, num_latents=num_latents,
                method="fista", lam=fista_lam,
                max_steps=max_steps, n_iter=FISTA_ITERS_PER_ROUND,
                dict_update_every=DICT_UPDATE_EVERY,
                supervised=False, seed=seed, print_every=max_steps + 1,
            )
            sc_out = train_sparse_coding(
                X_iid_c, X_ood_c, sc_cfg, D_init=D_sae, device=device,
            )
            warm_metrics = eval_and_tag(
                sc_out["codes_iid"], sc_out["codes_ood"],
                data, f"warm_dl_{sae_type}", **round_tag,
            )
            results.append(warm_metrics)

            # --- Cold: DL-FISTA from random init ---
            sc_out = train_sparse_coding(
                X_iid_c, X_ood_c, sc_cfg, device=device,
            )
            cold_metrics = eval_and_tag(
                sc_out["codes_iid"], sc_out["codes_ood"],
                data, f"cold_dl_{sae_type}", **round_tag,
            )
            results.append(cold_metrics)

            print(f"cold_mcc={cold_metrics.get('mcc_ood', 0):.3f}  "
                  f"warm_mcc={warm_metrics.get('mcc_ood', 0):.3f}")

    return results


def run(
    epochs=500,
    gamma_reg=1e-4,
    fista_lam=0.1,
    seeds=(0, 1, 2),
    min_n=0,
    max_n=float("inf"),
    out_suffix="",
):
    configs = get_frozen_decoder_configs()
    configs = [c for c in configs if min_n <= c["num_latents"] <= max_n]

    if not configs:
        print(f"No configs match min_n={min_n}, max_n={max_n}")
        return []

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_warmstart_decoder{out_suffix}.json"

    from models.sparse_coding import SparseCodingConfig, train_sparse_coding

    for cfg in configs:
        num_latents, k, input_dim = cfg["num_latents"], cfg["k"], cfg["input_dim"]
        n_samples = cfg["n_samples"]
        width = cfg["width"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  num_latents={num_latents}, k={k}, input_dim={input_dim}, seed={seed}")
            print(f"{'='*60}")

            train, val, ood, A = generate_datasets(
                seed=seed, num_latents=num_latents, k=k, n_samples=n_samples,
                input_dim=input_dim,
            )
            input_dim_actual = train[1].shape[1]

            data = {
                "Z_train": train[0], "Y_train": train[1], "labels_train": train[2],
                "Z_val": val[0], "Y_val": val[1], "labels_val": val[2],
                "Z_ood": ood[0], "Y_ood": ood[1], "labels_ood": ood[2],
            }
            tag = dict(num_latents=num_latents, k=k, input_dim=input_dim_actual,
                       width=width, n_samples=n_samples, seed=seed)

            # --- Train SAEs ---
            trained_saes = train_all_saes(
                data, input_dim_actual, width, k, num_latents, n_samples,
                epochs, gamma_reg, seed, device,
            )

            # --- Convergence sweep: cold vs warm dictionary init ---
            all_results.extend(run_decoder_convergence_sweep(
                trained_saes, data, A, fista_lam, device, tag,
            ))

            # --- FISTA oracle (ceiling) ---
            X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
            X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

            print("  FISTA oracle...")
            sc_cfg = SparseCodingConfig(
                input_dim=input_dim_actual, num_latents=num_latents,
                method="fista", lam=fista_lam,
                max_steps=1, n_iter=500, dict_update_every=1,
                supervised=True, seed=seed, print_every=1,
            )
            sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, A=A, device=device)
            all_results.append(eval_and_tag(
                sc_out["codes_iid"], sc_out["codes_ood"],
                data, "fista_oracle", **{**tag, "n_outer": max(OUTER_ROUND_BUDGETS)},
            ))

            save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename, e.g. '_large'")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
