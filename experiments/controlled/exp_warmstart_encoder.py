"""
Experiment: SAE Encoder as Warm-Start for FISTA Inference.

Question: Does the SAE learn an encoder that's at least a useful starting
point, even if it's not good enough on its own?

The frozen-decoder experiment showed that refined_{type} (FISTA warm-started
from SAE codes) matches fista+{type} (FISTA cold-start) when both run for
100 iterations -- because the convex objective fully converges regardless of
initialization.

This experiment sweeps the FISTA iteration budget to reveal the convergence
curves.  At low iteration counts, warm-start should outperform cold-start
(the SAE encoder gives a head start).  The crossover point tells us how many
FISTA iterations the SAE encoder is "worth".

For each (num_latents, sae_type, seed) and each iteration budget:
  cold_{type}@N   -- FISTA from zeros, N iterations, frozen SAE decoder
  warm_{type}@N   -- FISTA from SAE encoder codes, N iterations, frozen SAE decoder
  fista_oracle    -- FISTA with ground-truth A (ceiling, at max iterations)

Sweeps num_latents while holding k fixed.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from experiments._common import (
    eval_and_tag,
    train_all_saes,
    save_incremental,
    print_summary,
    SAE_TYPES,
)
from experiments.param_check import get_frozen_decoder_configs

ITER_BUDGETS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)


def run_convergence_sweep(trained_saes, data, fista_lam, device, tag):
    """Sweep FISTA iteration budget for cold-start and warm-start.

    For each SAE type and each iteration budget, runs:
      - cold_{type}: FISTA from zeros
      - warm_{type}: FISTA from SAE encoder codes

    Returns list of tagged metric dicts, each with an 'n_iter' field.
    """
    from models.sparse_coding import fista

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    results = []
    for sae_type, info in trained_saes.items():
        sae = info["model"]
        sae_codes_iid = info["codes_iid"]
        sae_codes_ood = info["codes_ood"]

        # Extract frozen decoder
        D = sae.Ad.data.detach()
        bd = sae.bd.data.detach()
        X_iid_c = X_iid - bd
        X_ood_c = X_ood - bd

        z_init_iid = torch.tensor(sae_codes_iid, dtype=torch.float32, device=device)
        z_init_ood = torch.tensor(sae_codes_ood, dtype=torch.float32, device=device)

        for n_iter in ITER_BUDGETS:
            iter_tag = {**tag, "n_iter": n_iter}

            if n_iter == 0:
                # Warm-start at 0 iterations = raw SAE encoder output
                results.append(eval_and_tag(
                    sae_codes_iid, sae_codes_ood,
                    data, f"warm_{sae_type}", **iter_tag,
                ))
                # Cold-start at 0 iterations = zeros (skip, uninformative)
                continue

            print(f"  {sae_type} n_iter={n_iter}...", end=" ", flush=True)

            # Cold-start: FISTA from zeros
            with torch.no_grad():
                z_cold_iid = fista(X_iid_c, D, fista_lam, n_iter=n_iter, nonneg=True)
                z_cold_ood = fista(X_ood_c, D, fista_lam, n_iter=n_iter, nonneg=True)
            cold_metrics = eval_and_tag(
                z_cold_iid.cpu().numpy(), z_cold_ood.cpu().numpy(),
                data, f"cold_{sae_type}", **iter_tag,
            )
            results.append(cold_metrics)

            # Warm-start: FISTA from SAE encoder codes
            with torch.no_grad():
                z_warm_iid = fista(X_iid_c, D, fista_lam, n_iter=n_iter,
                                   z_init=z_init_iid.clone(), nonneg=True)
                z_warm_ood = fista(X_ood_c, D, fista_lam, n_iter=n_iter,
                                   z_init=z_init_ood.clone(), nonneg=True)
            warm_metrics = eval_and_tag(
                z_warm_iid.cpu().numpy(), z_warm_ood.cpu().numpy(),
                data, f"warm_{sae_type}", **iter_tag,
            )
            results.append(warm_metrics)

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
    out_path = out_dir / f"exp_warmstart_encoder{out_suffix}.json"

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

            # --- Convergence sweep: cold vs warm at each iteration budget ---
            all_results.extend(run_convergence_sweep(
                trained_saes, data, fista_lam, device, tag,
            ))

            # --- FISTA oracle (ceiling, at max iterations) ---
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
                data, "fista_oracle", **{**tag, "n_iter": 500},
            ))

            save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename, e.g. '_small'")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
