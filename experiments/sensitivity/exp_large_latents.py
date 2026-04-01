"""
Experiment: Large latent dimensionality with FISTA.

Scales num_latents far beyond existing experiments (up to 1,000,000).
Methods are tiered by feasibility at each scale:

  num_latents <= 50K:  FISTA oracle, DL-FISTA, Softplus-Adam,
                       SAE + frozen decoder FISTA, raw, PCA
  50K < num_latents <= 100K:  FISTA oracle, Softplus-Adam,
                              SAE + frozen decoder FISTA, raw, PCA
  num_latents > 100K:  FISTA oracle, raw, PCA

LISTA is not run (available in models/sparse_coding.py for standalone use).

The FISTA/ISTA implementations use on-the-fly gradient computation for
num_latents > 10K, avoiding the O(num_latents^2) memory of DtD = D.T @ D.

Usage
-----
    # Full run (default: 1K to 1M, 3 seeds)
    python experiments/exp_large_latents.py

    # Quick test with small scale
    python -c "
    from experiments.sensitivity.exp_large_latents import run
    run(num_latents_values=(1000, 10000), seeds=(0,), fista_iters=10)
    "

    # Run specific range
    python experiments/exp_large_latents.py --min-n 100000 --max-n 1000000
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from experiments._common import run_large_latents_methods, print_summary, save_incremental
from experiments.param_check import get_large_latents_configs


def run(
    num_latents_values=None,
    epochs=500,
    gamma_reg=1e-4,
    sc_max_steps=50_000,
    sc_lam=0.1,
    fista_lam=0.1,
    fista_iters=100,
    seeds=(0, 1, 2),
    min_n=0,
    max_n=float("inf"),
    out_suffix="",
):
    if num_latents_values is not None:
        configs = get_large_latents_configs(num_latents_values=num_latents_values)
    else:
        configs = get_large_latents_configs()

    configs = [c for c in configs if min_n <= c["num_latents"] <= max_n]

    if not configs:
        print(f"No configs match min_n={min_n}, max_n={max_n}")
        return []

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_large_latents{out_suffix}.json"

    for cfg in configs:
        num_latents, k, input_dim = cfg["num_latents"], cfg["k"], cfg["input_dim"]
        n_samples = cfg["n_samples"]
        width = cfg["width"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  num_latents={num_latents}, k={k}, input_dim={input_dim}, "
                  f"width={width}, seed={seed}")
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

            all_results.extend(run_large_latents_methods(
                data, A, input_dim_actual, num_latents,
                sc_lam, sc_max_steps, fista_lam, fista_iters,
                epochs, gamma_reg, n_samples,
                seed, device, tag,
            ))

        save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")

    n_values = [c["num_latents"] for c in configs]
    print_summary(all_results, "num_latents", n_values)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Large latent dimensionality experiment (FISTA-only at scale).",
    )
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999999)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fista-iters", type=int, default=100)
    parser.add_argument("--sc-max-steps", type=int, default=50_000)
    parser.add_argument("--sc-lam", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    run(
        min_n=args.min_n,
        max_n=args.max_n,
        out_suffix=args.out_suffix,
        seeds=tuple(args.seeds),
        fista_iters=args.fista_iters,
        sc_max_steps=args.sc_max_steps,
        sc_lam=args.sc_lam,
        epochs=args.epochs,
    )
