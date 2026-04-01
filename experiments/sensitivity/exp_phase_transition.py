"""
Experiment: Phase transition in sparse recovery.

Tests the compressed sensing recovery condition:

    input_dim / k  >=  c * ln(num_latents / k)

where input_dim = observation dim, k = sparsity, num_latents = latent dim,
and c is an unknown constant.  The ratio
ε = input_dim / (k * ln(num_latents/k)) controls whether recovery succeeds
(ε >> 1) or fails (ε << 1).  The phase transition is the critical value of
ε (equivalently, of c) at which recovery breaks down.

The experiment constructs a grid of (num_latents, k, input_dim) triples that
tile the ratio ε from well below to well above the expected transition.  For
each triple it runs every method (SAE variants + sparse coding variants) and
records MCC, accuracy, and AUC.  Plotting MCC vs ε should reveal a sharp
transition — and potentially different transition points for different methods
(the core claim: proper sparse coding transitions earlier / more sharply than
SAEs).

All code uses natural log consistently: input_dim = ceil(c * k * ln(num_latents/k)).

Compares: SAE (relu, topk, jumprelu, MP) vs FISTA (sup/unsup) vs Direct (unsup).

Usage
-----

1) CLI commands:

    # Run the full experiment (default grid, 3 seeds)
    python experiments/exp_phase_transition.py

    # Run with custom grid / training params
    python experiments/exp_phase_transition.py --n 50 100 --k 3 5 --c 1.0 2.0 4.0 --seeds 0 1
    python experiments/exp_phase_transition.py --n-samples 5000 --sae-epochs 1000 --sc-lam 0.05

    # Preview the grid without running anything
    python experiments/exp_phase_transition.py --show-grid
    python experiments/exp_phase_transition.py --show-grid --n 50 100 --k 3 5 --c 0.5 1.0 2.0 4.0

    # Plot from saved results (combined plot, all methods on one axis)
    python experiments/exp_phase_transition.py --plot
    python experiments/exp_phase_transition.py --plot --metric mcc_ood
    python experiments/exp_phase_transition.py --plot --metric acc_iid --save figures/phase_acc.pdf
    python experiments/exp_phase_transition.py --plot --results-path results/exp_phase_transition.json

    # Plot faceted (one subplot per (num_latents, k) combination)
    python experiments/exp_phase_transition.py --plot-faceted
    python experiments/exp_phase_transition.py --plot-faceted --metric mcc_ood --save figures/faceted.pdf

2) From Python / notebook with custom parameters:

    from experiments.sensitivity.exp_phase_transition import run

    # Full run
    results = run()

    # Quick test: fewer grid points and one seed
    results = run(
        n_values=(50, 100),
        k_values=(3, 5),
        c_values=(1.0, 2.0, 4.0),
        seeds=(0,),
    )

    # Fine-grained sweep around the expected transition (c ~ 2)
    results = run(
        n_values=(100,),
        k_values=(5,),
        c_values=(0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0),
        seeds=(0, 1, 2, 3, 4),
    )

    # Adjust training budget
    results = run(
        sae_epochs=1000,       # more SAE training
        sc_max_steps=100_000,  # more sparse coding steps
        sc_lam=0.05,           # different sparsity penalty
    )

3) Inspect the grid without running (see which triples will be tested):

    from experiments.sensitivity.exp_phase_transition import _build_grid
    grid = _build_grid(n_values=(50, 100), k_values=(3, 5))
    for g in grid:
        print(f"num_latents={g['num_latents']}  k={g['k']}  input_dim={g['input_dim']}  eps={g['eps']:.3f}")

4) Plot results — combined (all methods on one axis, mean +/- std):

    from experiments.sensitivity.exp_phase_transition import plot_phase_transition

    # Default: MCC_iid vs eps, loaded from results/exp_phase_transition.json
    plot_phase_transition()

    # Specify path, metric, and save to file
    plot_phase_transition(
        results_path="results/exp_phase_transition.json",
        metric="mcc_ood",
        save_path="figures/phase_ood.pdf",
    )

    # Plot from in-memory results (e.g. right after run())
    results = run(...)
    plot_phase_transition(results=results, metric="acc_iid")

5) Plot results — faceted (one subplot per (num_latents, k) combination):

    from experiments.sensitivity.exp_phase_transition import plot_phase_transition_faceted

    # Default: MCC_iid, one panel per (num_latents, k)
    plot_phase_transition_faceted()

    # OOD metric, save to file
    plot_phase_transition_faceted(
        results_path="results/exp_phase_transition.json",
        metric="mcc_ood",
        save_path="figures/phase_faceted_ood.pdf",
    )

    # Available metrics for both plot functions:
    #   "mcc_iid", "mcc_ood"   — latent recovery (main result)
    #   "acc_iid", "acc_ood"   — downstream classification accuracy
    #   "auc_iid", "auc_ood"   — AUC from best-matching latent dim
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _build_grid(
    n_values=(50, 100, 200),
    k_values=(3, 5, 10),
    c_values=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0),
):
    """Build (num_latents, k, input_dim, c, r) grid from target c values.

    For each (num_latents, k, c) compute input_dim = ceil(c * k * ln(num_latents/k))
    and the realised ratio r = input_dim / (k * ln(num_latents/k)).

    Returns list of dicts with keys: num_latents, k, input_dim, c_target, eps.
    Deduplicates on (num_latents, k, input_dim) — different target c's can
    map to the same integer input_dim.
    """
    grid = []
    seen = set()
    for n in n_values:
        for k in k_values:
            if k >= n:
                continue
            log_ratio = math.log(n / k)
            if log_ratio <= 0:
                continue
            for c in c_values:
                m = max(1, math.ceil(c * k * log_ratio))
                m = min(m, n)
                key = (n, k, m)
                if key in seen:
                    continue
                seen.add(key)
                eps = m / (k * log_ratio)
                rho = k / n            # sparsity (k/d)
                delta = m / n          # undersampling ratio (m/d)
                grid.append(dict(num_latents=n, k=k, input_dim=m,
                                 c_target=round(eps, 3), eps=round(eps, 3),
                                 rho=round(rho, 4), delta=round(delta, 4)))
    grid.sort(key=lambda g: (g["num_latents"], g["k"], g["eps"]))
    return grid


def run(
    n_values=(50, 100, 200),
    k_values=(3, 5, 10),
    c_values=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0),
    n_samples=2000,
    sae_epochs=500,
    gamma_reg=1e-4,
    sc_max_steps=50_000,
    sc_lam=0.1,
    seeds=(0, 1, 2),
):
    import torch
    from src.data import generate_datasets
    from experiments._common import run_all_saes, run_sparse_coding_methods, run_linear_baselines

    grid = _build_grid(n_values, k_values, c_values)
    print(f"Phase transition grid: {len(grid)} unique (num_latents, k, input_dim) triples")
    for g in grid:
        print(f"  num_latents={g['num_latents']:>4d}  k={g['k']:>3d}  input_dim={g['input_dim']:>4d}  eps={g['eps']:.3f}")

    from experiments._common import save_incremental

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp_phase_transition.json"

    for gi, g in enumerate(grid):
        num_latents, k, input_dim = g["num_latents"], g["k"], g["input_dim"]
        eps = g["eps"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  [{gi+1}/{len(grid)}] num_latents={num_latents}, k={k}, input_dim={input_dim}, eps={eps:.3f}, seed={seed}")
            print(f"{'='*60}")

            train, val, ood, A = generate_datasets(
                seed=seed, num_latents=num_latents, k=k, n_samples=n_samples,
                input_dim=input_dim,
            )
            obs_dim = train[1].shape[1]

            data = {
                "Z_train": train[0], "Y_train": train[1], "labels_train": train[2],
                "Z_val": val[0], "Y_val": val[1], "labels_val": val[2],
                "Z_ood": ood[0], "Y_ood": ood[1], "labels_ood": ood[2],
            }

            width = num_latents
            rho = k / num_latents              # sparsity (k/d)
            delta = input_dim / num_latents   # undersampling ratio (m/d)
            tag = dict(num_latents=num_latents, k=k, input_dim=input_dim,
                       eps=eps, rho=round(rho, 4), delta=round(delta, 4), seed=seed)

            all_results.extend(run_linear_baselines(data, k, tag))

            all_results.extend(run_all_saes(
                data, obs_dim, width, k, num_latents, n_samples,
                sae_epochs, gamma_reg, seed, device, tag,
            ))

            all_results.extend(run_sparse_coding_methods(
                data, A, obs_dim, num_latents, sc_lam, sc_max_steps,
                seed, device, tag,
            ))

        save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")

    # Summary: for each method, show MCC vs eps (averaged over seeds)
    import numpy as np
    methods = sorted(set(r_["method"] for r_ in all_results))
    eps_values = sorted(set(r_["eps"] for r_ in all_results))

    print("\n--- Phase transition summary: MCC_iid vs eps (mean over seeds) ---")
    header = f"{'eps':>8} " + " ".join(f"{m:>15s}" for m in methods)
    print(header)
    for eps_val in eps_values:
        row = f"{eps_val:>8.3f} "
        for method in methods:
            rows = [x for x in all_results if x["eps"] == eps_val and x["method"] == method]
            if rows:
                mcc = np.mean([x.get("mcc_iid", 0) for x in rows])
                row += f"{mcc:>15.3f} "
            else:
                row += f"{'---':>15s} "
        print(row)

    print("\n--- Phase transition summary: MCC_ood vs eps (mean over seeds) ---")
    print(header)
    for eps_val in eps_values:
        row = f"{eps_val:>8.3f} "
        for method in methods:
            rows = [x for x in all_results if x["eps"] == eps_val and x["method"] == method]
            if rows:
                mcc = np.mean([x.get("mcc_ood", 0) for x in rows])
                row += f"{mcc:>15.3f} "
            else:
                row += f"{'---':>15s} "
        print(row)

    return all_results


# ============================================================================
# PLOTTING
# ============================================================================


_METRIC_LABELS = {
    "mcc_iid": "MCC (IID)",
    "mcc_ood": "MCC (OOD)",
    "acc_iid": "Accuracy (IID)",
    "acc_ood": "Accuracy (OOD)",
    "auc_iid": "AUC (IID)",
    "auc_ood": "AUC (OOD)",
}


def _metric_label(metric):
    return _METRIC_LABELS.get(metric, metric.replace("_", " "))


_METHOD_LABELS = {
    "sae_relu":     "SAE (ReLU)",
    "sae_topk":     "SAE (TopK)",
    "sae_jumprelu": "SAE (JumpReLU)",
    "sae_MP":       "SAE (MP)",
    "fista_oracle": "FISTA (oracle)",
    "dl_fista":     "DL-FISTA",
    "softplus_adam": "Softplus-Adam",
    "lista_oracle": "LISTA (oracle)",
    "dl_lista":     "DL-LISTA",
    "linear_probe": "Linear probe (oracle)",
}


def _method_label(method):
    return _METHOD_LABELS.get(method, method.replace("_", " "))


def plot_phase_transition(
    results_path: str | Path = None,
    results: list[dict] | None = None,
    metric: str = "mcc_iid",
    save_path: str | Path | None = None,
):
    """
    Plot phase transition curves: recovery metric vs eps for each method.

    Parameters
    ----------
    results_path : path to exp_phase_transition.json (used if results is None)
    results : list of result dicts (if already loaded)
    metric : which metric to plot on y-axis.
        "mcc_iid", "mcc_ood", "acc_iid", "acc_ood", "auc_iid", "auc_ood"
    save_path : if provided, save figure here instead of showing
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    if results is None:
        if results_path is None:
            results_path = ROOT / "results" / "exp_phase_transition.json"
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(
                f"No results at {results_path}. "
                "Run the experiment first, or use --plot from the CLI (which runs automatically)."
            )
        with open(results_path) as f:
            results = json.load(f)

    methods = sorted(set(r["method"] for r in results))

    # Style: group methods by family for visual clarity
    sc_methods = [m for m in methods if not m.startswith("sae_")]
    sae_methods = [m for m in methods if m.startswith("sae_")]
    ordered = sc_methods + sae_methods

    # Compute delta for each result if not already present
    for row in results:
        if "delta" not in row:
            row["delta"] = row["input_dim"] / row["num_latents"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ordered:
        rows = [r for r in results if r["method"] == method]
        delta_vals = sorted(set(r["delta"] for r in rows))
        means = []
        stds = []
        for dv in delta_vals:
            vals = [r.get(metric, 0) for r in rows if abs(r["delta"] - dv) < 1e-6]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)

        linestyle = "--" if method.startswith("sae_") else "-"
        ax.plot(delta_vals, means, marker="o", markersize=4, linestyle=linestyle, label=_method_label(method))
        ax.fill_between(delta_vals, means - stds, means + stds, alpha=0.1)

    ax.set_xlabel(r"$\delta = m/d$ (undersampling ratio)", fontweight="bold")
    ax.set_ylabel(_metric_label(metric), fontweight="bold")
    ax.set_title("Phase transition in sparse recovery", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    return fig, ax


def plot_phase_transition_faceted(
    results_path: str | Path = None,
    results: list[dict] | None = None,
    metric: str = "mcc_iid",
    save_path: str | Path | None = None,
):
    """
    Faceted phase transition plot: one subplot per (num_latents, k) combination.

    Each panel shares a common sparsity ratio rho = k/d. The x-axis is the
    undersampling ratio delta = m/d (Donoho-Tanner convention).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    if results is None:
        if results_path is None:
            results_path = ROOT / "results" / "exp_phase_transition.json"
        results_path = Path(results_path)
        if not results_path.exists():
            print(f"Error: results file not found: {results_path}")
            print("Run the experiment first:  python experiments/exp_phase_transition.py")
            return None, None
        with open(results_path) as f:
            results = json.load(f)

    # Compute delta and rho for each result if not already present
    for row in results:
        if "delta" not in row:
            row["delta"] = row["input_dim"] / row["num_latents"]
        if "rho" not in row:
            row["rho"] = row["k"] / row["num_latents"]

    methods = sorted(set(r["method"] for r in results))
    sc_methods = [m for m in methods if not m.startswith("sae_")]
    sae_methods = [m for m in methods if m.startswith("sae_")]
    ordered = sc_methods + sae_methods

    # Facet by (num_latents, k) — label shows ρ = k/d (sparsity)
    nk_pairs = sorted(set((r["num_latents"], r["k"]) for r in results))
    n_panels = len(nk_pairs)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (n, k) in enumerate(nk_pairs):
        ax = axes[idx // ncols][idx % ncols]
        panel_rows = [r for r in results if r["num_latents"] == n and r["k"] == k]

        for method in ordered:
            mrows = [r for r in panel_rows if r["method"] == method]
            if not mrows:
                continue
            delta_vals = sorted(set(r["delta"] for r in mrows))
            means = [np.mean([r.get(metric, 0) for r in mrows if abs(r["delta"] - dv) < 1e-6])
                     for dv in delta_vals]

            linestyle = "--" if method.startswith("sae_") else "-"
            ax.plot(delta_vals, means, marker="o", markersize=3, linestyle=linestyle, label=_method_label(method))

        rho_val = k / n
        ax.set_title(rf"$\rho = {rho_val:.3f}$ ($d={n}$, $k={k}$)")
        ax.set_xlabel(r"$\delta = m/d$", fontweight="bold")
        ax.set_ylabel(_metric_label(metric), fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        r"Phase transition: " + _metric_label(metric)
        + r" vs $\delta = m/d$, faceted by $\rho = k/d$",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    return fig, axes


# ============================================================================
# CLI
# ============================================================================


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase transition experiment for sparse recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mode flags (mutually exclusive) ---
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plot", action="store_true",
        help="Plot combined phase transition curve from saved results.",
    )
    mode.add_argument(
        "--plot-faceted", action="store_true",
        help="Plot faceted phase transition (one subplot per (num_latents, k)).",
    )
    mode.add_argument(
        "--show-grid", action="store_true",
        help="Print the (num_latents, k, input_dim, eps) grid and exit (no training).",
    )

    # --- Grid parameters ---
    parser.add_argument("--n", type=int, nargs="+", default=[50, 100, 200],
                        help="Latent dimensions to sweep (default: 50 100 200).")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10],
                        help="Sparsity levels to sweep (default: 3 5 10).")
    parser.add_argument("--c", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
                        help="Target c values for input_dim = c*k*ln(num_latents/k).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Random seeds (default: 0 1 2).")

    # --- Training parameters ---
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples per dataset (default: 2000).")
    parser.add_argument("--sae-epochs", type=int, default=500,
                        help="SAE training epochs (default: 500).")
    parser.add_argument("--gamma-reg", type=float, default=1e-4,
                        help="SAE regularisation weight (default: 1e-4).")
    parser.add_argument("--sc-max-steps", type=int, default=50_000,
                        help="Sparse coding max training steps (default: 50000).")
    parser.add_argument("--sc-lam", type=float, default=0.1,
                        help="Sparse coding L1 penalty (default: 0.1).")

    # --- Plot parameters ---
    parser.add_argument("--metric", type=str, default="mcc_iid",
                        choices=["mcc_iid", "mcc_ood", "acc_iid", "acc_ood",
                                 "auc_iid", "auc_ood"],
                        help="Metric for y-axis when plotting (default: mcc_iid).")
    parser.add_argument("--results-path", type=str, default=None,
                        help="Path to results JSON for plotting.")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to this path instead of showing.")

    # --- SLURM array support ---
    parser.add_argument("--grid-index", type=int, default=None,
                        help="Run only this grid point index (0-based). "
                             "For SLURM array jobs: --grid-index $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--merge", type=str, default=None,
                        help="Merge per-grid-point JSON files from this directory "
                             "into a single results file and exit.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    n_values = tuple(args.n)
    k_values = tuple(args.k)
    c_values = tuple(args.c)
    seeds = tuple(args.seeds)

    def _ensure_results():
        """Run experiment if results don't exist yet, return results list."""
        rpath = Path(args.results_path) if args.results_path else ROOT / "results" / "exp_phase_transition.json"
        if rpath.exists():
            with open(rpath) as f:
                return json.load(f)
        print(f"No results at {rpath} — running experiment first.\n")
        return run(
            n_values=n_values,
            k_values=k_values,
            c_values=c_values,
            n_samples=args.n_samples,
            sae_epochs=args.sae_epochs,
            gamma_reg=args.gamma_reg,
            sc_max_steps=args.sc_max_steps,
            sc_lam=args.sc_lam,
            seeds=seeds,
        )

    # --- Merge mode: combine per-grid-point JSON files ---
    if args.merge is not None:
        merge_dir = Path(args.merge)
        parts = sorted(merge_dir.glob("phase_transition_*.json"))
        if not parts:
            print(f"No phase_transition_*.json files found in {merge_dir}")
            sys.exit(1)
        merged = []
        for p in parts:
            merged.extend(json.loads(p.read_text()))
        out_path = ROOT / "results" / "exp_phase_transition.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(merged, indent=2))
        print(f"Merged {len(parts)} files ({len(merged)} results) -> {out_path}")
        sys.exit(0)

    if args.show_grid:
        grid = _build_grid(n_values, k_values, c_values)
        print(f"Grid: {len(grid)} unique (num_latents, k, input_dim) triples\n")
        print(f"{'num_latents':>12} {'k':>4} {'input_dim':>10} {'eps':>8}")
        print("-" * 38)
        for g in grid:
            print(f"{g['num_latents']:>12} {g['k']:>4} {g['input_dim']:>10} {g['eps']:>8.3f}")

    elif args.plot:
        results = _ensure_results()
        plot_phase_transition(
            results=results,
            metric=args.metric,
            save_path=args.save,
        )

    elif args.plot_faceted:
        results = _ensure_results()
        plot_phase_transition_faceted(
            results=results,
            metric=args.metric,
            save_path=args.save,
        )

    elif args.grid_index is not None:
        # --- Single grid point mode (for SLURM array jobs) ---
        import torch
        from src.data import generate_datasets
        from experiments._common import run_all_saes, run_sparse_coding_methods, run_linear_baselines

        grid = _build_grid(n_values, k_values, c_values)
        gi = args.grid_index
        if gi < 0 or gi >= len(grid):
            print(f"grid-index {gi} out of range [0, {len(grid)})")
            sys.exit(1)

        g = grid[gi]
        num_latents, k, input_dim, eps = g["num_latents"], g["k"], g["input_dim"], g["eps"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  grid[{gi}] num_latents={num_latents}, k={k}, input_dim={input_dim}, eps={eps:.3f}, seed={seed}")
            print(f"{'='*60}")

            train, val, ood, A = generate_datasets(
                seed=seed, num_latents=num_latents, k=k,
                n_samples=args.n_samples, input_dim=input_dim,
            )
            obs_dim = train[1].shape[1]
            data = {
                "Z_train": train[0], "Y_train": train[1], "labels_train": train[2],
                "Z_val": val[0], "Y_val": val[1], "labels_val": val[2],
                "Z_ood": ood[0], "Y_ood": ood[1], "labels_ood": ood[2],
            }
            width = num_latents
            rho = k / num_latents              # sparsity (k/d)
            delta = input_dim / num_latents   # undersampling ratio (m/d)
            tag = dict(num_latents=num_latents, k=k, input_dim=input_dim,
                       eps=eps, rho=round(rho, 4), delta=round(delta, 4), seed=seed)

            results.extend(run_linear_baselines(data, k, tag))
            results.extend(run_all_saes(
                data, obs_dim, width, k, num_latents, args.n_samples,
                args.sae_epochs, args.gamma_reg, seed, device, tag,
            ))
            results.extend(run_sparse_coding_methods(
                data, A, obs_dim, num_latents, args.sc_lam,
                args.sc_max_steps, seed, device, tag,
            ))

        # Save per-grid-point results
        out_dir = ROOT / "results" / "phase_transition_parts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"phase_transition_{gi:04d}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nGrid point {gi} saved to {out_path}")

    else:
        run(
            n_values=n_values,
            k_values=k_values,
            c_values=c_values,
            n_samples=args.n_samples,
            sae_epochs=args.sae_epochs,
            gamma_reg=args.gamma_reg,
            sc_max_steps=args.sc_max_steps,
            sc_lam=args.sc_lam,
            seeds=seeds,
        )
