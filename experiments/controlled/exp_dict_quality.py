"""
Experiment: Dictionary Quality Decomposition.

Question: Is the SAE decoder bad because of wrong column directions, or wrong
column norms?

The frozen-decoder experiment showed the SAE decoder is a poor dictionary
(low MCC when used with FISTA).  But "poor" could mean two different things:

  1. The decoder columns point in the wrong directions (angular error).
  2. The decoder columns have the right directions but wrong magnitudes.

This experiment decomposes the error by comparing:

  frozen_{type}          -- FISTA with raw SAE decoder (baseline)
  renormed_{type}        -- FISTA with SAE decoder columns re-normalized
                            to unit norm
  oracle_norms_{type}    -- FISTA with SAE decoder directions but ground-truth
                            magnitudes per column (strongest intervention)
  fista_oracle           -- FISTA with ground-truth A (ceiling)

If oracle_norms closes most of the gap to fista_oracle, the SAE learns
correct directions but wrong scales.  If it doesn't help, the directions
themselves are off.

Additionally, reports per-column diagnostics:
  - Angular error: arccos(|cos(d_i, a_matched)|) per matched pair
  - Norm ratio: ||d_i|| / ||a_matched|| per pair
  - Fraction of matched columns with cosine > 0.9

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
from utils.metrics import match_columns, replace_column_norms
from experiments._common import (
    eval_and_tag,
    train_all_saes,
    save_incremental,
)
from experiments.param_check import get_frozen_decoder_configs


def run_dict_quality_sweep(trained_saes, data, A, fista_lam, fista_iters, device, tag):
    """For each SAE type, decompose dictionary error and test interventions."""
    from models.sparse_coding import fista

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    results = []
    diagnostics = []

    for sae_type, info in trained_saes.items():
        sae = info["model"]

        D = sae.Ad.data.detach().cpu().numpy()
        bd = sae.bd.data.detach()
        X_iid_c = X_iid - bd
        X_ood_c = X_ood - bd

        # --- Column matching diagnostics ---
        match = match_columns(D, A)
        diag = {
            "sae_type": sae_type,
            "mean_cosine": match["mean_cosine"],
            "mean_angular_error": match["mean_angular_error"],
            "mean_norm_ratio": match["mean_norm_ratio"],
            "frac_close": match["frac_close"],
            **tag,
        }
        diagnostics.append(diag)
        print(f"  {sae_type}: cos={match['mean_cosine']:.3f}  "
              f"ang_err={match['mean_angular_error']:.3f}rad  "
              f"norm_ratio={match['mean_norm_ratio']:.3f}  "
              f"frac_close={match['frac_close']:.3f}")

        # --- 1) Frozen SAE decoder (baseline) ---
        D_t = torch.tensor(D, dtype=torch.float32, device=device)
        with torch.no_grad():
            z_iid = fista(X_iid_c, D_t, fista_lam, n_iter=fista_iters, nonneg=True)
            z_ood = fista(X_ood_c, D_t, fista_lam, n_iter=fista_iters, nonneg=True)
        results.append(eval_and_tag(
            z_iid.cpu().numpy(), z_ood.cpu().numpy(),
            data, f"frozen_{sae_type}", **tag,
        ))

        # --- 2) Re-normalized decoder (unit-norm columns) ---
        D_renorm = D / np.linalg.norm(D, axis=0, keepdims=True).clip(min=1e-8)
        D_renorm_t = torch.tensor(D_renorm, dtype=torch.float32, device=device)
        with torch.no_grad():
            z_iid = fista(X_iid_c, D_renorm_t, fista_lam, n_iter=fista_iters, nonneg=True)
            z_ood = fista(X_ood_c, D_renorm_t, fista_lam, n_iter=fista_iters, nonneg=True)
        results.append(eval_and_tag(
            z_iid.cpu().numpy(), z_ood.cpu().numpy(),
            data, f"renormed_{sae_type}", **tag,
        ))

        # --- 3) Oracle norms: SAE directions + ground-truth magnitudes ---
        D_oracle = replace_column_norms(D, A, match["row_ind"], match["col_ind"])
        D_oracle_t = torch.tensor(D_oracle, dtype=torch.float32, device=device)
        with torch.no_grad():
            z_iid = fista(X_iid_c, D_oracle_t, fista_lam, n_iter=fista_iters, nonneg=True)
            z_ood = fista(X_ood_c, D_oracle_t, fista_lam, n_iter=fista_iters, nonneg=True)
        results.append(eval_and_tag(
            z_iid.cpu().numpy(), z_ood.cpu().numpy(),
            data, f"oracle_norms_{sae_type}", **tag,
        ))

    return results, diagnostics


def run(
    epochs=500,
    gamma_reg=1e-4,
    fista_lam=0.1,
    fista_iters=100,
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
    all_diagnostics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_dict_quality{out_suffix}.json"
    diag_path = out_dir / f"exp_dict_quality_diagnostics{out_suffix}.json"

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

            # --- Dictionary quality decomposition ---
            results, diagnostics = run_dict_quality_sweep(
                trained_saes, data, A, fista_lam, fista_iters, device, tag,
            )
            all_results.extend(results)
            all_diagnostics.extend(diagnostics)

            # --- FISTA oracle (ceiling) ---
            X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
            X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

            print("  FISTA oracle...")
            sc_cfg = SparseCodingConfig(
                input_dim=input_dim_actual, num_latents=num_latents,
                method="fista", lam=fista_lam,
                max_steps=1, n_iter=fista_iters, dict_update_every=1,
                supervised=True, seed=seed, print_every=1,
            )
            sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, A=A, device=device)
            all_results.append(eval_and_tag(
                sc_out["codes_iid"], sc_out["codes_ood"],
                data, "fista_oracle", **tag,
            ))

            save_incremental(all_results, out_path)
            save_incremental(all_diagnostics, diag_path)

    print(f"\nResults saved to {out_path}")
    print(f"Diagnostics saved to {diag_path}")

    # --- Summary ---
    print("\n--- Dictionary Quality Summary (mean over seeds) ---")
    sae_types = sorted(set(d["sae_type"] for d in all_diagnostics))
    nvals = sorted(set(d["num_latents"] for d in all_diagnostics))
    print("%6s %10s %8s %8s %8s %10s" % ("n", "sae_type", "cosine", "ang_err", "norm_r", "frac_close"))
    for n in nvals:
        for st in sae_types:
            rows = [d for d in all_diagnostics if d["num_latents"] == n and d["sae_type"] == st]
            if rows:
                print("%6d %10s %8.3f %8.3f %8.3f %10.3f" % (
                    n, st,
                    np.mean([r["mean_cosine"] for r in rows]),
                    np.mean([r["mean_angular_error"] for r in rows]),
                    np.mean([r["mean_norm_ratio"] for r in rows]),
                    np.mean([r["frac_close"] for r in rows]),
                ))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename, e.g. '_small'")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
