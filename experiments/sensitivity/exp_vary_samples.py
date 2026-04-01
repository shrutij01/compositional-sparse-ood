"""
Experiment: Vary number of training samples (n_samples).

Sweeps n_samples while holding num_latents, k fixed.  Observation
dimension input_dim is set from the compressed-sensing bound via
param_check.

Compares: SAE (ReLU, TopK, JumpReLU, MP) vs FISTA oracle, DL-FISTA,
Softplus-Adam.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from experiments._common import run_all_saes, run_sparse_coding_methods, run_linear_baselines, print_summary, save_incremental
from experiments.param_check import get_vary_samples_configs


def run(
    num_latents=100,
    epochs=500,
    gamma_reg=1e-4,
    sc_max_steps=50_000,
    sc_lam=0.1,
    seeds=(0, 1, 2),
    out_suffix="",
):
    configs = get_vary_samples_configs(num_latents=num_latents)
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_vary_samples{out_suffix}.json"

    # All configs share the same (num_latents, k, input_dim) — only n_samples varies
    nl = configs[0]["num_latents"]
    k = configs[0]["k"]
    input_dim = configs[0]["input_dim"]
    width = configs[0]["width"]

    for cfg in configs:
        n_samples = cfg["n_samples"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  num_latents={nl}, k={k}, input_dim={input_dim}, n_samples={n_samples}, seed={seed}")
            print(f"{'='*60}")

            train, val, ood, A = generate_datasets(
                seed=seed, num_latents=nl, k=k, n_samples=n_samples,
                input_dim=input_dim,
            )
            input_dim_actual = train[1].shape[1]

            data = {
                "Z_train": train[0], "Y_train": train[1], "labels_train": train[2],
                "Z_val": val[0], "Y_val": val[1], "labels_val": val[2],
                "Z_ood": ood[0], "Y_ood": ood[1], "labels_ood": ood[2],
            }
            tag = dict(num_latents=nl, k=k, input_dim=input_dim_actual,
                       width=width, n_samples=n_samples, seed=seed)

            all_results.extend(run_linear_baselines(data, k, tag))
            all_results.extend(run_all_saes(
                data, input_dim_actual, width, k, nl, n_samples,
                epochs, gamma_reg, seed, device, tag,
            ))
            all_results.extend(run_sparse_coding_methods(
                data, A, input_dim_actual, nl, sc_lam, sc_max_steps,
                seed, device, tag,
            ))

        save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")

    sample_sizes = [c["n_samples"] for c in configs]
    print_summary(all_results, "n_samples", sample_sizes)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-latents", type=int, default=100,
                        help="Latent dimensionality (default: 100)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename, e.g. '_10k'")
    args = parser.parse_args()
    run(num_latents=args.num_latents, out_suffix=args.out_suffix)
