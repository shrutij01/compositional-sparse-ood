"""
Experiment: Vary latent dimensionality (num_latents).

Sweeps num_latents while holding k and n_samples fixed.  Observation
dimension input_dim and width are set from the compressed-sensing bound
via param_check.

Compares: SAE (ReLU, TopK, JumpReLU, MP) vs FISTA oracle, DL-FISTA,
Softplus-Adam.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from experiments._common import (
    run_all_saes,
    run_sparse_coding_methods,
    run_linear_baselines,
    print_summary,
    save_incremental,
)
from experiments.param_check import get_vary_latents_configs


def run(
    epochs=500,
    gamma_reg=1e-4,
    sc_max_steps=50_000,
    sc_lam=0.1,
    seeds=(0, 1, 2),
    min_n=0,
    max_n=float("inf"),
    out_suffix="",
):
    configs = get_vary_latents_configs()
    configs = [c for c in configs if min_n <= c["num_latents"] <= max_n]

    if not configs:
        print(f"No configs match min_n={min_n}, max_n={max_n}")
        return []

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_vary_latents{out_suffix}.json"

    for cfg in configs:
        num_latents, k, input_dim = (
            cfg["num_latents"],
            cfg["k"],
            cfg["input_dim"],
        )
        n_samples = cfg["n_samples"]
        width = cfg["width"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(
                f"  num_latents={num_latents}, k={k}, input_dim={input_dim}, width={width}, seed={seed}"
            )
            print(f"{'='*60}")

            train, val, ood, A = generate_datasets(
                seed=seed,
                num_latents=num_latents,
                k=k,
                n_samples=n_samples,
                input_dim=input_dim,
            )
            input_dim_actual = train[1].shape[1]

            data = {
                "Z_train": train[0],
                "Y_train": train[1],
                "labels_train": train[2],
                "Z_val": val[0],
                "Y_val": val[1],
                "labels_val": val[2],
                "Z_ood": ood[0],
                "Y_ood": ood[1],
                "labels_ood": ood[2],
            }
            tag = dict(
                num_latents=num_latents,
                k=k,
                input_dim=input_dim_actual,
                width=width,
                n_samples=n_samples,
                seed=seed,
            )

            all_results.extend(run_linear_baselines(data, k, tag))
            all_results.extend(
                run_all_saes(
                    data,
                    input_dim_actual,
                    width,
                    k,
                    num_latents,
                    n_samples,
                    epochs,
                    gamma_reg,
                    seed,
                    device,
                    tag,
                )
            )
            all_results.extend(
                run_sparse_coding_methods(
                    data,
                    A,
                    input_dim_actual,
                    num_latents,
                    sc_lam,
                    sc_max_steps,
                    seed,
                    device,
                    tag,
                )
            )

        save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")

    n_values = [c["num_latents"] for c in configs]
    print_summary(all_results, "num_latents", n_values)

    plot_runtime(load_combined_results(out_dir), out_dir)

    return all_results


def load_combined_results(out_dir, pattern="exp_vary_latents*.json"):
    """Load and concatenate all matching result JSONs in out_dir."""
    combined = []
    for p in sorted(Path(out_dir).glob(pattern)):
        with open(p) as f:
            combined.extend(json.load(f))
    return combined


# Methods to compare in the runtime plot
_RUNTIME_METHODS = {
    "dl_fista":     {"label": "DL-FISTA",        "color": "#8c564b", "ls": "-",  "marker": "s"},
    "sae_relu":     {"label": "SAE (ReLU)",       "color": "#1f77b4", "ls": "--", "marker": "o"},
    "sae_topk":     {"label": "SAE (TopK)",       "color": "#ff7f0e", "ls": "--", "marker": "s"},
    "sae_jumprelu": {"label": "SAE (JumpReLU)",   "color": "#2ca02c", "ls": "--", "marker": "^"},
    "sae_MP":       {"label": "SAE (MP)",         "color": "#d62728", "ls": "--", "marker": "D"},
}


def plot_runtime(all_results, out_dir):
    """Plot mean runtime vs. num_latents for DL-FISTA and SAE variants."""
    # Only keep rows that have timing info and belong to target methods
    buckets = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if "runtime_s" not in r or r["method"] not in _RUNTIME_METHODS:
            continue
        buckets[r["method"]][r["num_latents"]].append(r["runtime_s"])

    if not buckets:
        print("  [runtime plot] No timing data found, skipping.")
        return

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.labelweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, style in _RUNTIME_METHODS.items():
        if method not in buckets:
            continue
        xs = sorted(buckets[method].keys())
        means = [np.mean(buckets[method][x]) for x in xs]
        stds = [np.std(buckets[method][x]) for x in xs]
        ax.plot(xs, means, label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"],
                markersize=6, linewidth=1.8)
        ax.fill_between(xs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=style["color"], alpha=0.12)

    ax.set_xscale("log")
    ax.set_xlabel(r"Number of latents ($d$)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Running time vs. number of latents")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    save_path = out_dir / "exp_vary_latents_runtime.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"  [runtime plot] Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Suffix for output filename, e.g. '_large'",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiments; just rebuild the runtime PDF from existing JSONs.",
    )
    args = parser.parse_args()
    if args.plot_only:
        out_dir = ROOT / "results"
        plot_runtime(load_combined_results(out_dir), out_dir)
    else:
        run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
