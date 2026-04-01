"""
Experiment: Lambda Sensitivity.

Question: Are the frozen decoder results an artifact of the lambda mismatch
between SAE training (gamma_reg=1e-4) and FISTA inference (fista_lam=0.1)?

SAEs are trained with gamma_reg=1e-4 controlling sparsity, but FISTA uses
fista_lam=0.1 --- a 1000x difference. If the frozen decoder results depend
critically on this choice, the finding that "swapping inference doesn't help"
could be an artifact.

This experiment sweeps fista_lam across a wide range for the frozen decoder
condition and reports MCC. If the result is robust to lambda, the dictionary
quality conclusion holds. If there exists a lambda where frozen FISTA
substantially beats the SAE encoder, the lambda mismatch was a confound.

For each (num_latents, sae_type, seed) and each lambda:
  frozen_{type}@lam   -- FISTA with frozen SAE decoder at this lambda
  sae_{type}          -- raw SAE encoder (lambda-independent baseline)
  fista_oracle@lam    -- FISTA with oracle dictionary at this lambda

Sweeps num_latents while holding k fixed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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

LAMBDA_VALUES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def run_lambda_sweep(trained_saes, data, A, device, tag, fista_iters=100):
    """Sweep fista_lam for frozen decoder and oracle conditions."""
    from models.sparse_coding import fista

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    results = []

    for sae_type, info in trained_saes.items():
        sae = info["model"]
        codes_iid = info["codes_iid"]
        codes_ood = info["codes_ood"]

        D = sae.Ad.data.detach()
        bd = sae.bd.data.detach()
        X_iid_c = X_iid - bd
        X_ood_c = X_ood - bd

        # SAE encoder baseline (lambda-independent)
        results.append(eval_and_tag(
            codes_iid, codes_ood, data, f"sae_{sae_type}",
            **{**tag, "lam": -1},  # -1 signals "not applicable"
        ))

        for lam in LAMBDA_VALUES:
            lam_tag = {**tag, "lam": lam}

            # Frozen decoder at this lambda
            with torch.no_grad():
                z_iid = fista(X_iid_c, D, lam, n_iter=fista_iters, nonneg=True)
                z_ood = fista(X_ood_c, D, lam, n_iter=fista_iters, nonneg=True)
            results.append(eval_and_tag(
                z_iid.cpu().numpy(), z_ood.cpu().numpy(),
                data, f"frozen_{sae_type}", **lam_tag,
            ))

        print(f"  {sae_type}: swept {len(LAMBDA_VALUES)} lambdas")

    # Oracle at each lambda
    for lam in LAMBDA_VALUES:
        with torch.no_grad():
            z_iid = fista(X_iid, A_t, lam, n_iter=fista_iters, nonneg=True)
            z_ood = fista(X_ood, A_t, lam, n_iter=fista_iters, nonneg=True)
        results.append(eval_and_tag(
            z_iid.cpu().numpy(), z_ood.cpu().numpy(),
            data, "fista_oracle", **{**tag, "lam": lam},
        ))
    print(f"  oracle: swept {len(LAMBDA_VALUES)} lambdas")

    return results


def run(
    epochs=500,
    gamma_reg=1e-4,
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
    out_path = out_dir / f"exp_lambda_sensitivity{out_suffix}.json"

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

            trained_saes = train_all_saes(
                data, input_dim_actual, width, k, num_latents, n_samples,
                epochs, gamma_reg, seed, device,
            )

            all_results.extend(run_lambda_sweep(
                trained_saes, data, A, device, tag,
            ))

            save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--out-suffix", type=str, default="")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
