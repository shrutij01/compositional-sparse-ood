"""
Experiment: Frozen SAE Decoder + FISTA.

Trains each SAE type, then uses its learned decoder (Ad) as a fixed
dictionary for FISTA inference.  This disentangles dictionary quality
from encoder quality.

For each (num_latents, seed) combination, compares:
  sae_{type}       — amortized SAE encoder
  fista+{type}     — FISTA cold-start with frozen SAE decoder
  refined_{type}   — FISTA warm-started from SAE codes
  fista_oracle     — FISTA with ground-truth A
  dl_fista         — FISTA with dictionary learned from scratch
  softplus_adam    — joint code + dict optimization

Sweeps num_latents while holding k fixed.  Observation dimension
input_dim is set from the compressed-sensing bound via param_check.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from experiments._common import (
    train_all_saes,
    run_frozen_decoder_fista,
    run_sparse_coding_methods,
    run_linear_baselines,
    print_summary,
    save_incremental,
)
from experiments.param_check import get_frozen_decoder_configs


def run(
    epochs=500,
    gamma_reg=1e-4,
    fista_lam=0.1,
    fista_iters=100,
    sc_max_steps=50_000,
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
    out_path = out_dir / f"exp_frozen_decoder{out_suffix}.json"

    for cfg in configs:
        num_latents, k, input_dim = cfg["num_latents"], cfg["k"], cfg["input_dim"]
        n_samples = cfg["n_samples"]
        width = cfg["width"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  num_latents={num_latents}, k={k}, input_dim={input_dim}, width={width}, seed={seed}")
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

            all_results.extend(run_linear_baselines(data, k, tag))

            trained_saes = train_all_saes(
                data, input_dim_actual, width, k, num_latents, n_samples,
                epochs, gamma_reg, seed, device,
            )

            all_results.extend(run_frozen_decoder_fista(
                trained_saes, data, fista_lam, fista_iters, device, tag,
            ))

            all_results.extend(run_sparse_coding_methods(
                data, A, input_dim_actual, num_latents, fista_lam,
                sc_max_steps, seed, device, tag,
            ))

        save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")

    n_values = [c["num_latents"] for c in configs]
    print_summary(all_results, "num_latents", n_values)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename, e.g. '_large'")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
