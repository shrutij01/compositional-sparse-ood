"""
Parameter validation and configuration for sparse OOD experiments.

Ensures all (num_latents, k, input_dim, n_samples) configurations respect the
compressed sensing recovery bound:

    input_dim  >=  c * k * ln(num_latents / k)

where c >= 2 is a safety factor.  Configurations that violate this are
flagged or adjusted automatically.

Usage
-----

    # Print validated sweep values for all three experiments
    python experiments/param_check.py

    # Check a specific configuration
    python experiments/param_check.py --n 1000 --k 50

    # Import in experiment scripts
    from experiments.param_check import (
        compute_input_dim, validate_config, get_vary_latents_configs,
        get_vary_sparsity_configs, get_vary_samples_configs,
    )
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================================
# Core bound computation
# ============================================================================


def compute_input_dim(num_latents: int, k: int, c: float = 2.0) -> int:
    """Compute observation dimension from the CS bound:
    input_dim = ceil(c * k * ln(num_latents/k)).

    Parameters
    ----------
    num_latents : int  — number of latent variables
    k : int  — sparsity level (number of active latents per sample)
    c : float — safety factor (c=2 is the standard heuristic)

    Returns
    -------
    input_dim : int — minimum observation dimension satisfying the bound

    Raises
    ------
    ValueError
        If the CS bound requires input_dim > num_latents.
    """
    if k <= 0 or num_latents <= 0 or k >= num_latents:
        raise ValueError(
            f"Need 0 < k < num_latents, got k={k}, num_latents={num_latents}"
        )
    log_ratio = math.log(num_latents / k)
    input_dim = max(k + 1, math.ceil(c * k * log_ratio))
    if input_dim > num_latents:
        raise ValueError(
            f"CS bound requires input_dim={input_dim} > num_latents={num_latents} "
            f"(k={k}, c={c}, num_latents/k={num_latents/k:.1f}). "
            f"Increase num_latents or decrease k."
        )
    return input_dim


def validate_config(
    num_latents: int,
    k: int,
    input_dim: int = None,
    n_samples: int = None,
    c: float = 2.0,
):
    """Validate a single (num_latents, k, input_dim, n_samples) configuration.

    Returns dict with validated values and diagnostic info.
    """
    input_dim_min = compute_input_dim(num_latents, k, c=c)

    if input_dim is None:
        input_dim = input_dim_min

    warnings = []

    if input_dim < input_dim_min:
        warnings.append(
            f"input_dim={input_dim} < input_dim_min={input_dim_min} (c={c}): recovery unlikely"
        )
    if k >= num_latents:
        warnings.append(f"k={k} >= num_latents={num_latents}: not sparse")
    if n_samples is not None and n_samples < 2 * input_dim:
        warnings.append(
            f"n_samples={n_samples} < 2*input_dim={2*input_dim}: severely underdetermined"
        )

    eps = (
        input_dim / (k * math.log(num_latents / k))
        if k < num_latents
        else float("inf")
    )

    return {
        "num_latents": num_latents,
        "k": k,
        "input_dim": input_dim,
        "input_dim_min": input_dim_min,
        "n_samples": n_samples,
        "eps": round(eps, 3),
        "valid": len(warnings) == 0,
        "warnings": warnings,
    }


# ============================================================================
# Experiment sweep configurations
# ============================================================================


def get_vary_latents_configs(
    num_latents_values=(10, 50, 100, 500, 1000, 5000, 10000),
    k=10,
    n_samples=5000,
    c=2.0,
):
    """Validated configs for the vary-latents experiment.

    Filters out (num_latents, k) pairs where k >= num_latents.
    Width scales as 2*num_latents (overcomplete).
    """
    configs = []
    for num_latents in num_latents_values:
        if k >= num_latents:
            continue
        try:
            input_dim = compute_input_dim(num_latents, k, c=c)
        except ValueError:
            continue  # CS bound infeasible
        v = validate_config(num_latents, k, input_dim, n_samples, c=c)
        v["width"] = num_latents
        configs.append(v)
    return configs


def get_vary_sparsity_configs(
    num_latents=1000,
    k_values=(2, 5, 10, 20, 50, 100),
    n_samples=5000,
    c=2.0,
):
    """Validated configs for the vary-sparsity experiment.

    Filters out k values that violate k < num_latents and
    k < num_latents - num_latents//2 + 1 (needed for IID/OOD split).
    """
    configs = []
    num_ood = num_latents // 2
    for k in k_values:
        if k >= num_latents or k >= num_latents - num_ood + 1:
            continue
        try:
            input_dim = compute_input_dim(num_latents, k, c=c)
        except ValueError:
            continue  # CS bound infeasible
        v = validate_config(num_latents, k, input_dim, n_samples, c=c)
        v["width"] = num_latents
        configs.append(v)
    return configs


def get_vary_samples_configs(
    num_latents=100,
    k=10,
    n_samples_values=(100, 500, 1000, 5000, 10000, 50000, 100000),
    c=2.0,
):
    """Validated configs for the vary-samples experiment.

    All configs use the same (num_latents, k, input_dim) — only n_samples changes.
    """
    input_dim = compute_input_dim(num_latents, k, c=c)
    configs = []
    for ns in n_samples_values:
        v = validate_config(num_latents, k, input_dim, ns, c=c)
        v["width"] = num_latents
        configs.append(v)
    return configs


def get_frozen_decoder_configs(
    num_latents_values=(10, 50, 100, 500, 1000, 5000, 10000),
    k=10,
    n_samples=5000,
    c=2.0,
):
    """Validated configs for the frozen-decoder experiment.

    Sweeps num_latents with fixed k.  Same range, k, and n_samples as
    get_vary_latents_configs() for direct comparison.
    """
    configs = []
    for num_latents in num_latents_values:
        if k >= num_latents:
            continue
        try:
            input_dim = compute_input_dim(num_latents, k, c=c)
        except ValueError:
            continue  # CS bound infeasible
        v = validate_config(num_latents, k, input_dim, n_samples, c=c)
        v["width"] = num_latents
        configs.append(v)
    return configs


def get_large_latents_configs(
    num_latents_values=(1_000, 10_000, 100_000, 500_000, 1_000_000),
    k=10,
    n_samples=1000,
    c=2.0,
):
    """Validated configs for the large-latents FISTA experiment.

    Scales num_latents far beyond existing experiments.  n_samples is
    reduced (from 5000 to 1000) to manage memory at 1M latents.
    Width scales as 2*num_latents (overcomplete).
    """
    configs = []
    for num_latents in num_latents_values:
        if k >= num_latents:
            continue
        try:
            input_dim = compute_input_dim(num_latents, k, c=c)
        except ValueError:
            continue
        v = validate_config(num_latents, k, input_dim, n_samples, c=c)
        v["width"] = num_latents
        configs.append(v)
    return configs


# ============================================================================
# Pretty printing
# ============================================================================


def print_configs(configs, title):
    """Print a table of configurations with validation status."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(
        f"{'num_latents':>12} {'k':>6} {'input_dim':>10} {'id_min':>6} {'eps':>8} {'n_samples':>10} {'width':>8} {'status':>8}"
    )
    print("-" * 78)
    for c in configs:
        status = "OK" if c["valid"] else "WARN"
        ns = c["n_samples"] if c["n_samples"] is not None else "-"
        w = c.get("width", "-")
        print(
            f"{c['num_latents']:>12} {c['k']:>6} {c['input_dim']:>10} {c['input_dim_min']:>6} {c['eps']:>8.3f} {ns:>10} {w:>8} {status:>8}"
        )
        for warn in c["warnings"]:
            print(f"             ^ {warn}")


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate experiment parameters."
    )
    parser.add_argument(
        "--n", type=int, default=None, help="Check specific num_latents."
    )
    parser.add_argument(
        "--k", type=int, default=None, help="Check specific k."
    )
    parser.add_argument(
        "--m", type=int, default=None, help="Check specific input_dim."
    )
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--c", type=float, default=2.0, help="Safety factor (default: 2)."
    )
    args = parser.parse_args()

    if args.n is not None and args.k is not None:
        v = validate_config(args.n, args.k, args.m, args.n_samples, args.c)
        print(
            f"\nConfig: num_latents={v['num_latents']}, k={v['k']}, "
            f"input_dim={v['input_dim']} (min={v['input_dim_min']}), eps={v['eps']}"
        )
        if v["valid"]:
            print("Status: OK")
        else:
            for w in v["warnings"]:
                print(f"WARNING: {w}")
    else:
        # Print all experiment configurations
        print_configs(
            get_vary_latents_configs(),
            "Vary Latents: num_latents = (10..10000), k=10 fixed",
        )
        print_configs(
            get_vary_sparsity_configs(),
            "Vary Sparsity: k = (2..100), num_latents=1000 fixed",
        )
        print_configs(
            get_vary_samples_configs(),
            "Vary Samples: n_samples = (100..100000), num_latents=100, k=10 fixed",
        )
        print_configs(
            get_frozen_decoder_configs(),
            "Frozen Decoder: num_latents = (10..10000), k=10 fixed",
        )
        print_configs(
            get_large_latents_configs(),
            "Large Latents: num_latents = (1K..1M), k=10 fixed",
        )
