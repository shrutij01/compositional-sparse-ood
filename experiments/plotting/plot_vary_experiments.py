"""
Plot results from the three vary_ experiments.

Reads JSON results saved by exp_vary_samples.py, exp_vary_latents.py,
and exp_vary_sparsity.py.  For each experiment, produces one figure per
metric (6 metrics × 3 experiments = 18 panels, laid out as 3 figures
with 6 subplots each).

Usage
-----

    # Plot all three experiments (reads from results/ directory)
    python experiments/plot_vary_experiments.py

    # Save to directory instead of showing
    python experiments/plot_vary_experiments.py --save figures/

    # Plot only one experiment
    python experiments/plot_vary_experiments.py --only samples
    python experiments/plot_vary_experiments.py --only latents
    python experiments/plot_vary_experiments.py --only sparsity

    # Custom results directory
    python experiments/plot_vary_experiments.py --results-dir results/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================================
# Global matplotlib style — LaTeX rendering
# ============================================================================

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})


# ============================================================================
# Style
# ============================================================================

# Method display names and grouping
METHOD_META = {
    # SAE variants — dashed lines
    "sae_relu":      {"label": "SAE (ReLU)",      "color": "#1f77b4", "ls": "--", "marker": "o", "group": "SAE"},
    "sae_topk":      {"label": "SAE (TopK)",      "color": "#ff7f0e", "ls": "--", "marker": "s", "group": "SAE"},
    "sae_jumprelu":  {"label": "SAE (JumpReLU)",  "color": "#2ca02c", "ls": "--", "marker": "^", "group": "SAE"},
    "sae_MP":        {"label": "SAE (MP)",        "color": "#d62728", "ls": "--", "marker": "D", "group": "SAE"},
    # Sparse coding variants — solid lines
    "fista_oracle":  {"label": "FISTA (oracle)",   "color": "#9467bd", "ls": "-",  "marker": "o", "group": "SC"},
    "dl_fista":      {"label": "DL-FISTA",         "color": "#8c564b", "ls": "-",  "marker": "s", "group": "SC"},
    "softplus_adam":  {"label": "Softplus-Adam",    "color": "#e377c2", "ls": "-",  "marker": "^", "group": "SC"},
    "lista_oracle":  {"label": "LISTA (oracle)",   "color": "#17becf", "ls": "-",  "marker": "D", "group": "SC"},
    "dl_lista":      {"label": "DL-LISTA",         "color": "#bcbd22", "ls": "-",  "marker": "v", "group": "SC"},
    # Frozen decoder FISTA — dash-dot lines
    "fista+relu":      {"label": "FISTA+ReLU dec",      "color": "#1f77b4", "ls": "-.",  "marker": "p", "group": "Frozen"},
    "fista+topk":      {"label": "FISTA+TopK dec",      "color": "#ff7f0e", "ls": "-.",  "marker": "p", "group": "Frozen"},
    "fista+jumprelu":  {"label": "FISTA+JumpReLU dec",  "color": "#2ca02c", "ls": "-.",  "marker": "p", "group": "Frozen"},
    "fista+MP":        {"label": "FISTA+MP dec",        "color": "#d62728", "ls": "-.",  "marker": "p", "group": "Frozen"},
    # Refined (warm-started FISTA from SAE codes) — dash-dot lines
    "refined_relu":      {"label": "Refined ReLU",      "color": "#1f77b4", "ls": "-.",  "marker": "h", "group": "Refined"},
    "refined_topk":      {"label": "Refined TopK",      "color": "#ff7f0e", "ls": "-.",  "marker": "h", "group": "Refined"},
    "refined_jumprelu":  {"label": "Refined JumpReLU",  "color": "#2ca02c", "ls": "-.",  "marker": "h", "group": "Refined"},
    "refined_MP":        {"label": "Refined MP",        "color": "#d62728", "ls": "-.",  "marker": "h", "group": "Refined"},
    # Linear baseline — dotted line
    "linear_probe":  {"label": "Linear probe (oracle)", "color": "#555555", "ls": ":",  "marker": "*", "group": "Baseline"},
    # Unsupervised baselines — dotted line
    "raw":           {"label": "Raw (Y)",               "color": "#888888", "ls": ":",  "marker": "x", "group": "Baseline"},
    "pca":           {"label": "PCA",                   "color": "#aaaaaa", "ls": ":",  "marker": "+", "group": "Baseline"},
}

METRICS_MAIN = [
    ("mcc_iid",  "MCC (IID)"),
    ("auc_ood",  "AUC (OOD)"),
]

METRICS_FULL = [
    ("mcc_iid",  "MCC (IID)"),
    ("mcc_ood",  "MCC (OOD)"),
    ("acc_iid",  "Accuracy (IID)"),
    ("acc_ood",  "Accuracy (OOD)"),
    ("auc_iid",  "AUC (IID)"),
    ("auc_ood",  "AUC (OOD)"),
]

# Per-experiment captions explaining what each metric tells us
CAPTIONS = {
    "samples": {
        "mcc_iid": (
            "Latent recovery on IID data as training set size grows. "
            "FISTA (oracle) with the true dictionary should be near-perfect regardless of sample count. "
            "SAE methods need enough samples to learn a good encoder; "
            "a flat curve means the method has already saturated."
        ),
        "mcc_ood": (
            "Latent recovery under distribution shift. "
            "A gap between IID and OOD MCC reveals how much each method overfits "
            "to the training distribution's support patterns. "
            "Per-sample optimization (DL-FISTA) should degrade less than amortized inference (SAEs)."
        ),
        "acc_iid": (
            "Downstream classification accuracy on IID data from a logistic probe on the learned codes. "
            "Higher accuracy means the first latent (the label-relevant variable) is well-recovered. "
            "All methods should improve with more training data."
        ),
        "acc_ood": (
            "Downstream accuracy under distribution shift. "
            "This is the core metric: can the model still classify correctly "
            "when the combination of active latents changes at test time? "
            "A large IID-OOD gap indicates the representation entangles the label variable "
            "with spurious co-occurrence patterns."
        ),
        "auc_iid": (
            "AUC of the single best-matching latent dimension on IID data. "
            "Unlike accuracy (which uses all dims), this isolates whether "
            "any individual code dimension captures the label signal."
        ),
        "auc_ood": (
            "AUC of the best IID-selected dimension evaluated on OOD data. "
            "If the best IID dimension truly isolates the causal variable, "
            "it should transfer to OOD. A drop indicates the dimension "
            "captures a spurious IID-specific pattern instead."
        ),
    },
    "latents": {
        "mcc_iid": (
            "Latent recovery vs. num_latents (d) with k=10 fixed. "
            "Larger d means sparser codes relative to dictionary size (rho = k/d decreases). "
            "FISTA (oracle) with known A is immune; "
            "unsupervised methods should degrade as d grows. "
            "For d > 10K, only feasible methods run (SAEs drop out, FISTA continues to 1M)."
        ),
        "mcc_ood": (
            "OOD latent recovery vs. num_latents. "
            "Larger d means more possible support patterns, so distribution shift "
            "can be more severe. Per-sample methods should be more robust. "
            "At extreme scale (d > 100K), only FISTA oracle is feasible."
        ),
        "acc_iid": (
            "IID accuracy vs. num_latents. "
            "As d grows, the label variable is a smaller fraction of the code, "
            "making it harder for a linear probe to isolate the right signal."
        ),
        "acc_ood": (
            "OOD accuracy vs. num_latents. "
            "Larger d compounds combinatorial difficulty with distribution shift. "
            "Methods that recover true generative factors should maintain accuracy."
        ),
        "auc_iid": (
            "Best-dimension AUC vs. num_latents. "
            "Even if MCC drops, a method might still isolate the label variable "
            "in one code dimension."
        ),
        "auc_ood": (
            "OOD transfer of the best IID dimension vs. num_latents. "
            "With larger d, the risk of selecting a "
            "spuriously correlated dimension increases."
        ),
    },
    "sparsity": {
        "mcc_iid": (
            "Latent recovery vs. rho = k/d (sweeping k with d fixed, so rho increases). "
            "Higher rho means denser codes. "
            "As rho grows, recovery gets harder and all methods degrade."
        ),
        "mcc_ood": (
            "OOD latent recovery vs. rho. "
            "Higher rho means more active latents per sample, which can increase "
            "IID/OOD support overlap (helping OOD) but also makes recovery harder."
        ),
        "acc_iid": (
            "IID accuracy vs. rho. "
            "With higher rho, the label signal may be diluted among distractors. "
            "But richer codes can also provide more information to the classifier."
        ),
        "acc_ood": (
            "OOD accuracy vs. rho. "
            "Does increasing rho help OOD generalization "
            "(more support overlap) or hurt it (harder recovery)?"
        ),
        "auc_iid": (
            "Best-dimension AUC vs. rho. "
            "Tests whether any single code dimension captures the label, "
            "even as more latents are simultaneously active."
        ),
        "auc_ood": (
            "OOD transfer of the best dimension vs. rho. "
            "Higher rho makes the combinatorial structure denser, "
            "which can either help (more overlap) or hurt (more confusion) OOD transfer."
        ),
    },
    "frozen": {
        "mcc_iid": (
            "Latent recovery with frozen SAE decoders vs. rho. "
            "Compares amortized SAE inference against FISTA using the same learned dictionary. "
            "The gap between sae_{type} and fista+{type} isolates the amortization gap."
        ),
        "mcc_ood": (
            "OOD latent recovery with frozen decoders. "
            "FISTA with the SAE decoder (fista+{type}) should degrade less than "
            "the amortized encoder (sae_{type}) under distribution shift."
        ),
        "acc_iid": (
            "IID accuracy: amortized vs. per-sample inference with the same dictionary. "
            "If the SAE learned a good dictionary, FISTA with that dictionary should match or exceed SAE accuracy."
        ),
        "acc_ood": (
            "OOD accuracy with frozen decoders. "
            "The core test: does per-sample optimization (FISTA) with an SAE-learned dictionary "
            "generalize better than the SAE's own encoder?"
        ),
        "auc_iid": (
            "Best-dimension AUC with frozen decoders. "
            "Tests whether FISTA refinement recovers a cleaner single-dimension signal."
        ),
        "auc_ood": (
            "OOD transfer of the best dimension with frozen decoders. "
            "Refined codes (warm-started FISTA from SAE) should transfer better if "
            "FISTA corrects encoder errors that are specific to the IID distribution."
        ),
    },
    "samples_10k": {
        "mcc_iid": (
            "Latent recovery on IID data as training set size grows (d=10K). "
            "With 10x more latents than the d=100 experiment, all methods need more data "
            "to learn the dictionary/encoder. FISTA (oracle) should still be near-perfect."
        ),
        "mcc_ood": (
            "OOD latent recovery at d=10K. "
            "A gap between IID and OOD MCC reveals how much each method overfits "
            "at higher dimensionality. Sample efficiency may differ from the d=100 case."
        ),
        "acc_iid": (
            "IID accuracy at d=10K. "
            "The label variable is a much smaller fraction of the code (1/10K vs 1/100), "
            "so more samples may be needed for accurate classification."
        ),
        "acc_ood": (
            "OOD accuracy at d=10K. "
            "Tests whether methods can still isolate the label under distribution shift "
            "when the latent space is 100x larger than in the d=100 experiment."
        ),
        "auc_iid": (
            "Best-dimension AUC at d=10K. "
            "With more dimensions, spurious correlations are more likely. "
            "Tests whether any individual code dimension captures the label signal."
        ),
        "auc_ood": (
            "OOD transfer of the best IID dimension at d=10K. "
            "A drop from d=100 would indicate that higher dimensionality makes "
            "spurious dimension selection more likely."
        ),
    },
}


# ============================================================================
# Data loading
# ============================================================================


def _load_results(path):
    """Load JSON results, return list of dicts."""
    p = Path(path)
    if not p.exists():
        print(f"Results not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def _aggregate(results, sweep_key):
    """Group results by (method, sweep_value) and compute mean/std over seeds.

    Returns: {method: {"x": [...], "mean": [...], "std": [...]}}
    """
    # Collect values per (method, sweep_val)
    buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        method = r["method"]
        sweep_val = r[sweep_key]
        buckets[method][sweep_val].append(r)

    agg = {}
    for method, val_dict in buckets.items():
        x_vals = sorted(val_dict.keys())
        agg[method] = {"x": x_vals}
        for metric_key, _ in METRICS_FULL:
            means = []
            stds = []
            for xv in x_vals:
                vals = [r.get(metric_key, 0) for r in val_dict[xv]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            agg[method][f"{metric_key}_mean"] = means
            agg[method][f"{metric_key}_std"] = stds
    return agg


# ============================================================================
# Plotting
# ============================================================================


# Per-experiment subtitle: what's fixed, what varies, and the key ratio held constant
_EXPERIMENT_SUBTITLES = {
    "samples": (
        r"$d=100$, $k=10$, $m=47$, $\varepsilon \approx 2$ fixed; varying $n$"
    ),
    "latents": (
        r"$k=10$, $\varepsilon \approx 2$ fixed; $n=5000$ for $d \leq 10K$, $n=1000$ for $d > 10K$"
    ),
    "sparsity": (
        r"$d=1000$, $n=5000$, $\varepsilon \approx 2$ fixed; $m = \lceil 2k \ln(d/k) \rceil$ scales with $k$"
    ),
    "frozen": (
        r"$k=10$, $n=5000$, $\varepsilon \approx 2$ fixed; $m = \lceil 2k \ln(d/k) \rceil$ scales with $d$"
    ),
    "samples_10k": (
        r"$d=10000$, $k=10$, $m=139$, $\varepsilon \approx 2$ fixed; varying $n$"
    ),
}


CORE_GROUPS = {"SAE", "SC", "Baseline"}


def _plot_single(agg, sweep_key, sweep_label, metrics, nrows, ncols,
                  figsize, exp_name, suffix, save_dir=None,
                  groups=None):
    """Generic plotter for any metric subset and layout.

    Parameters
    ----------
    groups : set of str or None
        If given, only plot methods whose group is in this set.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]

        for method, style in METHOD_META.items():
            if method not in agg:
                continue
            if groups is not None and style.get("group") not in groups:
                continue
            data = agg[method]
            x = data["x"]
            y_mean = data[f"{metric_key}_mean"]
            y_std = data[f"{metric_key}_std"]

            ax.plot(
                x, y_mean,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.8,
            )
            ax.fill_between(
                x,
                np.array(y_mean) - np.array(y_std),
                np.array(y_mean) + np.array(y_std),
                color=style["color"],
                alpha=0.08,
            )

        ax.set_xlabel(sweep_label)
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        if "mcc" in metric_key or "auc" in metric_key:
            ax.set_ylim(-0.05, 1.05)
        elif "acc" in metric_key:
            ax.set_ylim(0.4, 1.05)

        if sweep_key in ("n_samples", "num_latents"):
            ax.set_xscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1.0])

    if save_dir is not None:
        save_path = Path(save_dir) / f"vary_{exp_name}{suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _plot_experiment(agg, sweep_key, sweep_label, exp_name, save_dir=None,
                     subtitle="", groups=None):
    """Create main (1x2) and appendix (2x3) figures."""
    # Main: MCC (IID) + AUC (OOD)
    _plot_single(agg, sweep_key, sweep_label, METRICS_MAIN,
                 nrows=1, ncols=2, figsize=(12, 4.5),
                 exp_name=exp_name, suffix="", save_dir=save_dir,
                 groups=groups)
    # Appendix: all 6 metrics
    _plot_single(agg, sweep_key, sweep_label, METRICS_FULL,
                 nrows=2, ncols=3, figsize=(16, 9),
                 exp_name=exp_name, suffix="_appendix", save_dir=save_dir,
                 groups=groups)


def _plot_combined_grid(agg_a, agg_b, sweep_key, sweep_label,
                        label_a, label_b, metrics, nrows_per, ncols,
                        figsize, exp_name, suffix, save_dir=None,
                        groups=None):
    """Create a combined figure: two experiment rows x metric columns."""
    total_rows = 2 * nrows_per
    fig, axes = plt.subplots(total_rows, ncols, figsize=figsize, squeeze=False)

    for row_block, (agg, row_label) in enumerate([(agg_a, label_a), (agg_b, label_b)]):
        for idx, (metric_key, metric_label) in enumerate(metrics):
            r = row_block * nrows_per + idx // ncols
            c = idx % ncols
            ax = axes[r, c]

            for method, style in METHOD_META.items():
                if method not in agg:
                    continue
                if groups is not None and style.get("group") not in groups:
                    continue
                data = agg[method]
                x = data["x"]
                y_mean = data[f"{metric_key}_mean"]
                y_std = data[f"{metric_key}_std"]

                ax.plot(
                    x, y_mean,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["ls"],
                    marker=style["marker"],
                    markersize=5,
                    linewidth=1.8,
                )
                ax.fill_between(
                    x,
                    np.array(y_mean) - np.array(y_std),
                    np.array(y_mean) + np.array(y_std),
                    color=style["color"],
                    alpha=0.08,
                )

            ax.set_xlabel(sweep_label)
            if c == 0:
                ax.set_ylabel(f"{row_label}\n{metric_label}")
            else:
                ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.25, linewidth=0.5)

            if "mcc" in metric_key or "auc" in metric_key:
                ax.set_ylim(-0.05, 1.05)
            elif "acc" in metric_key:
                ax.set_ylim(0.4, 1.05)

            if sweep_key in ("n_samples", "num_latents"):
                ax.set_xscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])

    if save_dir is not None:
        save_path = Path(save_dir) / f"vary_{exp_name}{suffix}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _plot_combined(agg_a, agg_b, sweep_key, sweep_label,
                   label_a, label_b, exp_name, save_dir=None,
                   groups=None):
    """Create main (2x2) and appendix (4x3) combined figures."""
    # Main: 2 rows x 2 cols (MCC_IID, AUC_OOD per experiment)
    _plot_combined_grid(
        agg_a, agg_b, sweep_key, sweep_label, label_a, label_b,
        METRICS_MAIN, nrows_per=1, ncols=2,
        figsize=(12, 8), exp_name=exp_name, suffix="",
        save_dir=save_dir, groups=groups,
    )
    # Appendix: 2 rows x 3 cols x 2 experiments = 4 rows x 3 cols
    _plot_combined_grid(
        agg_a, agg_b, sweep_key, sweep_label, label_a, label_b,
        METRICS_FULL, nrows_per=2, ncols=3,
        figsize=(16, 16), exp_name=exp_name, suffix="_appendix",
        save_dir=save_dir, groups=groups,
    )


def _plot_captions(exp_name, save_dir=None):
    """Create a text figure with captions for each metric."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    captions = CAPTIONS.get(exp_name)
    if captions is None:
        return
    title_map = {
        "samples": "Varying Training Samples (d=100)",
        "samples_10k": "Varying Training Samples (d=10000)",
        "latents": "Varying num_latents (d=10 to 1M), k=10 fixed",
        "sparsity": "Varying rho = k/d (sweeping k, d fixed)",
        "frozen": "Frozen Decoder + FISTA (sweeping d, k fixed)",
    }

    title = title_map.get(exp_name, exp_name)
    text = f"{title}\n{'=' * len(title)}\n\n"
    for metric_key, metric_label in METRICS_FULL:
        text += f"{metric_label}:\n{captions[metric_key]}\n\n"

    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        wrap=True,
    )

    plt.tight_layout()

    if save_dir is not None:
        save_path = Path(save_dir) / f"vary_{exp_name}_captions.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _plot_iid_vs_ood(agg, sweep_key, sweep_label, exp_name, save_dir=None, subtitle=""):
    """Create a 1x3 figure showing IID-OOD gap for MCC, accuracy, AUC."""
    metric_pairs = [
        ("mcc_iid", "mcc_ood", "MCC"),
        ("acc_iid", "acc_ood", "Accuracy"),
        ("auc_iid", "auc_ood", "AUC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    title = "IID vs OOD Gap — " + sweep_label
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (iid_key, ood_key, name) in zip(axes, metric_pairs):
        for method, style in METHOD_META.items():
            if method not in agg:
                continue
            data = agg[method]
            x = data["x"]
            y_iid = np.array(data[f"{iid_key}_mean"])
            y_ood = np.array(data[f"{ood_key}_mean"])
            gap = y_iid - y_ood

            ax.plot(
                x, gap,
                label=style["label"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.8,
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel(sweep_label, fontweight="bold")
        ax.set_ylabel(name + " gap (IID − OOD)", fontweight="bold")
        ax.set_title(name + ": IID − OOD gap")
        ax.grid(True, alpha=0.3)

        if sweep_key in ("n_samples", "num_latents"):
            ax.set_xscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    if save_dir is not None:
        save_path = Path(save_dir) / f"vary_{exp_name}_iid_ood_gap.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _frozen_bar_panels(agg, sweep_key, sweep_label):
    """Return shared config for frozen bar plots."""
    sae_types = ["relu", "topk", "jumprelu", "MP"]

    sae_type_colors = {
        "relu":     "#1f77b4",
        "topk":     "#ff7f0e",
        "jumprelu": "#2ca02c",
        "MP":       "#d62728",
    }
    sc_colors = {
        "fista_oracle": "#9467bd",
        "dl_fista":     "#8c564b",
        "softplus_adam": "#e377c2",
        "linear_probe": "#555555",
    }
    hatch_sae = ""
    hatch_frozen = "///"
    hatch_refined = "xxx"

    def get_bar_style(method):
        if method in sc_colors:
            return sc_colors[method], "", "black"
        for stype, col in sae_type_colors.items():
            if method == f"sae_{stype}":
                return col, hatch_sae, "black"
            if method == f"fista+{stype}":
                return col, hatch_frozen, "black"
            if method == f"refined_{stype}":
                return col, hatch_refined, "black"
        return "#888888", "", "black"

    sc_methods = ["dl_fista", "softplus_adam"]
    sae_methods = [f"sae_{s}" for s in sae_types]

    # Supervised methods shown as line plots on all panels
    line_methods = ["fista_oracle", "linear_probe"]
    line_styles = {
        "fista_oracle": {"color": "#9467bd", "ls": "-",  "marker": "o", "label": "FISTA (oracle)"},
        "linear_probe": {"color": "#555555", "ls": ":",  "marker": "*", "label": "Linear probe (oracle)"},
    }

    frozen_interleaved = []
    refined_interleaved = []
    for stype in sae_types:
        frozen_interleaved.extend([f"sae_{stype}", f"fista+{stype}"])
        refined_interleaved.extend([f"sae_{stype}", f"refined_{stype}"])

    panels = [
        sc_methods + sae_methods,
        frozen_interleaved,
        refined_interleaved,
    ]

    all_x = sorted(set(xv for m_data in agg.values() for xv in m_data["x"]))
    x_labels = [f"{int(v)}" if v < 1000 else f"{int(v/1000)}K" for v in all_x]

    return panels, all_x, x_labels, get_bar_style, line_methods, line_styles


def _draw_frozen_row(axes_row, agg, metric_key, metric_label,
                     panels, all_x, x_labels, get_bar_style,
                     line_methods, line_styles,
                     show_x_labels=True):
    """Draw one row of 3 bar panels + supervised line overlays.

    Returns col_handles dict and a list of line handles (for shared legend).
    """
    col_handles = {c: [] for c in range(3)}
    col_labels_seen = {c: set() for c in range(3)}
    line_handles = []
    line_labels_seen = set()

    for col, method_list in enumerate(panels):
        ax = axes_row[col]
        methods = [m for m in method_list if m in agg and m in METHOD_META]
        n_methods = len(methods)
        if n_methods == 0:
            continue

        n_groups = len(all_x)
        bar_width = 0.8 / n_methods
        group_positions = np.arange(n_groups)
        panel_vals = []

        for i, method in enumerate(methods):
            style = METHOD_META[method]
            data = agg[method]
            offset = (i - n_methods / 2 + 0.5) * bar_width
            color, hatch, ec = get_bar_style(method)

            means, stds = [], []
            for xv in all_x:
                if xv in data["x"]:
                    idx = data["x"].index(xv)
                    means.append(data[f"{metric_key}_mean"][idx])
                    stds.append(data[f"{metric_key}_std"][idx])
                else:
                    means.append(0)
                    stds.append(0)

            panel_vals.extend(m for m in means if m > 0)

            bar = ax.bar(
                group_positions + offset, means, bar_width,
                yerr=stds, label=style["label"],
                color=color, alpha=0.55, hatch=hatch,
                capsize=2, edgecolor=ec, linewidth=0.6,
            )
            if style["label"] not in col_labels_seen[col]:
                col_handles[col].append(bar[0])
                col_labels_seen[col].add(style["label"])

        # Overlay supervised methods as line plots
        for lm in line_methods:
            if lm not in agg:
                continue
            ls = line_styles[lm]
            data = agg[lm]
            means = []
            for xv in all_x:
                if xv in data["x"]:
                    idx = data["x"].index(xv)
                    means.append(data[f"{metric_key}_mean"][idx])
                else:
                    means.append(np.nan)
            panel_vals.extend(m for m in means if not np.isnan(m))
            line, = ax.plot(
                group_positions, means,
                color=ls["color"], linestyle=ls["ls"], marker=ls["marker"],
                markersize=7, linewidth=2, label=ls["label"], zorder=5,
            )
            if ls["label"] not in line_labels_seen:
                line_handles.append(line)
                line_labels_seen.add(ls["label"])

        ax.set_xticks(group_positions)
        if show_x_labels:
            ax.set_xticklabels(x_labels, fontsize=9)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

        # Left panel: full range; middle/right: zoom to data range
        if col == 0:
            ax.set_ylim(0, 1.05)
        elif panel_vals:
            ymin = max(0, min(panel_vals) - 0.08)
            ymax = min(1.05, max(panel_vals) + 0.08)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0, 1.05)

    return col_handles, line_handles


def _style_red_ylim_ticks(fig, axes_list):
    """Bold red the top and bottom y-tick labels on all given axes.

    Uses explicit set_yticklabels so styling persists through savefig.
    """
    fig.canvas.draw()
    for ax in axes_list:
        yticks = ax.get_yticks()
        ylim = ax.get_ylim()
        visible = [t for t in yticks if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9]
        if len(visible) < 2:
            continue
        # Freeze the ticks and labels so matplotlib doesn't regenerate them
        ax.set_yticks(visible)
        labels = []
        for i, t in enumerate(visible):
            txt = f"{t:g}"
            if i == 0 or i == len(visible) - 1:
                labels.append(txt)  # placeholder, styled below
            else:
                labels.append(txt)
        tick_labels = ax.set_yticklabels(labels)
        for i, tl in enumerate(tick_labels):
            if i == 0 or i == len(tick_labels) - 1:
                tl.set_color("red")
                tl.set_fontweight("bold")
                tl.set_fontsize(12)
                tl.set_fontsize(11)


def _add_frozen_legends(fig, col_handles, line_handles, col_centers,
                        legend_y=0.10, line_y=0.02):
    """Add one legend per column + shared line legend below the figure."""
    for col in range(3):
        handles = [h for h in col_handles[col]
                   if not h.get_label().startswith("_")]
        if handles:
            labels = [h.get_label() for h in handles]
            ncol_leg = 2 if len(labels) > 4 else 1
            fig.legend(
                handles, labels,
                fontsize=8, loc="upper center",
                bbox_to_anchor=(col_centers[col], legend_y),
                framealpha=0.9, ncol=ncol_leg,
                edgecolor="black",
            )
    if line_handles:
        line_labels = [h.get_label() for h in line_handles]
        fig.legend(
            line_handles, line_labels,
            fontsize=9, loc="upper center",
            bbox_to_anchor=(0.5, line_y),
            framealpha=0.9, ncol=len(line_handles),
            edgecolor="black", title="Supervised (all panels)",
            title_fontsize=9,
        )


def _save_or_show(fig, save_dir, filename):
    if save_dir is not None:
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _plot_frozen_experiment(agg, sweep_key, sweep_label, exp_name, save_dir=None):
    """Frozen decoder plots: 2x3 combined + 1x3 per metric."""
    panels, all_x, x_labels, get_bar_style, line_methods, line_styles = \
        _frozen_bar_panels(agg, sweep_key, sweep_label)

    metrics = [("mcc_iid", "MCC (IID)"), ("auc_ood", "AUC (OOD)")]
    col_centers = [0.185, 0.5, 0.815]

    # ---- Combined 2x3 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 14), squeeze=False)
    all_col_handles = {c: [] for c in range(3)}
    all_line_handles = []

    for row, (metric_key, metric_label) in enumerate(metrics):
        show_x = (row == len(metrics) - 1)
        ch, lh = _draw_frozen_row(
            axes[row], agg, metric_key, metric_label,
            panels, all_x, x_labels, get_bar_style,
            line_methods, line_styles,
            show_x_labels=show_x)
        if row == 0:
            all_col_handles = ch
            all_line_handles = lh

    plt.tight_layout(rect=[0, 0.25, 1, 1.0])
    _style_red_ylim_ticks(fig, [axes[r, c] for r in range(2) for c in range(3)])

    fig.text(0.5, 0.24, sweep_label, ha="center", fontsize=13, fontweight="bold")
    _add_frozen_legends(fig, all_col_handles, all_line_handles, col_centers,
                        legend_y=0.22, line_y=0.05)
    _save_or_show(fig, save_dir, f"vary_{exp_name}.pdf")

    # ---- Individual 1x3 per metric ----
    for metric_key, metric_label in metrics:
        fig, axes_row = plt.subplots(1, 3, figsize=(18, 8), squeeze=False)
        ch, lh = _draw_frozen_row(
            axes_row[0], agg, metric_key, metric_label,
            panels, all_x, x_labels, get_bar_style,
            line_methods, line_styles,
            show_x_labels=True)

        plt.tight_layout(rect=[0, 0.30, 1, 1.0])
        _style_red_ylim_ticks(fig, list(axes_row[0]))

        fig.text(0.5, 0.29, sweep_label, ha="center", fontsize=13,
                 fontweight="bold")
        _add_frozen_legends(fig, ch, lh, col_centers,
                            legend_y=0.26, line_y=0.06)
        _save_or_show(fig, save_dir, f"vary_{exp_name}_{metric_key}.pdf")


def _plot_method_comparison(agg, sweep_key, sweep_label, exp_name, save_dir=None, subtitle=""):
    """Bar chart at the last sweep value: SAE vs SC side-by-side for OOD metrics."""
    ood_metrics = [
        ("mcc_ood", "MCC (OOD)"),
        ("acc_ood", "Accuracy (OOD)"),
        ("auc_ood", "AUC (OOD)"),
    ]

    # Get the last (largest) sweep value across all methods
    all_x = set()
    for m_data in agg.values():
        all_x.update(m_data["x"])
    last_x = max(all_x)

    # Only include methods that have results at last_x
    def _has_x(m):
        return m in agg and last_x in agg[m]["x"]

    sae_methods = [m for m in METHOD_META if METHOD_META[m]["group"] == "SAE" and _has_x(m)]
    sc_methods = [m for m in METHOD_META if METHOD_META[m]["group"] == "SC" and _has_x(m)]
    baseline_methods = [m for m in METHOD_META if METHOD_META[m]["group"] == "Baseline" and _has_x(m)]
    frozen_methods = [m for m in METHOD_META if METHOD_META[m]["group"] in ("Frozen", "Refined") and _has_x(m)]
    all_methods = baseline_methods + sae_methods + sc_methods + frozen_methods

    if not all_methods:
        print(f"  [bar chart] No methods have data at {last_x}, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    last_x_str = f"{last_x:g}" if isinstance(last_x, (int, float)) else str(last_x)
    title = f"Method comparison at {sweep_label} = {last_x_str} (OOD metrics)"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (metric_key, metric_label) in zip(axes, ood_metrics):
        x_pos = np.arange(len(all_methods))
        means = []
        stds = []
        colors = []
        for m in all_methods:
            idx = agg[m]["x"].index(last_x)
            means.append(agg[m][f"{metric_key}_mean"][idx])
            stds.append(agg[m][f"{metric_key}_std"][idx])
            colors.append(METHOD_META[m]["color"])

        ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.8,
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [METHOD_META[m]["label"] for m in all_methods],
            rotation=35, ha="right",
        )
        ax.set_ylabel(metric_label, fontweight="bold")
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis="y")

        # Add separators between groups
        group_sizes = [len(baseline_methods), len(sae_methods), len(sc_methods), len(frozen_methods)]
        pos = 0
        for gs in group_sizes[:-1]:
            pos += gs
            if gs > 0 and pos < len(all_methods):
                ax.axvline(pos - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_dir is not None:
        save_path = Path(save_dir) / f"vary_{exp_name}_bar_comparison.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

EXPERIMENTS = {
    "samples": {
        "file": "exp_vary_samples.json",
        "sweep_key": "n_samples",
        "sweep_label": r"Number of training samples ($n$)",
    },
    "samples_10k": {
        "file": "exp_vary_samples_10k.json",
        "sweep_key": "n_samples",
        "sweep_label": r"Number of training samples ($n$)",
    },
    "latents": {
        "file": "exp_vary_latents.json",
        "extend_file": "exp_large_latents.json",
        "sweep_key": "num_latents",
        "sweep_label": r"Number of latents ($d$)",
    },
    "sparsity": {
        "file": "exp_vary_sparsity.json",
        "sweep_key": "rho",
        "sweep_label": r"$\rho = k/d$ (sparsity)",
        "compute_key": lambda r: r["k"] / r["num_latents"],
    },
    "frozen": {
        "file": "exp_frozen_decoder.json",
        "extend_file": "exp_frozen_decoder_10k.json",
        "sweep_key": "num_latents",
        "sweep_label": r"Number of latents ($d$)",
    },
}


def _load_and_prepare(results_dir, exp_info):
    """Load results, merge with extension file if needed, compute derived keys."""
    results_path = results_dir / exp_info["file"]
    results = _load_results(results_path)
    if results is None:
        return None

    extend_file = exp_info.get("extend_file")
    if extend_file:
        extend_path = results_dir / extend_file
        extend_results = _load_results(extend_path)
        if extend_results:
            sweep_key = exp_info["sweep_key"]
            # Build set of (method, sweep_val) already in primary results
            existing = set((r["method"], r[sweep_key]) for r in results)
            # Add entries from extension only for (method, sweep_val) gaps
            extra = [r for r in extend_results
                     if (r["method"], r.get(sweep_key, 0)) not in existing]
            results.extend(extra)
            print(f"  Merged {len(extra)} new (method, d) entries from {extend_file}")

    compute_fn = exp_info.get("compute_key")
    if compute_fn is not None:
        for r in results:
            r[exp_info["sweep_key"]] = compute_fn(r)

    return results


def plot_all(results_dir=None, save_dir=None, only=None):
    """Plot all experiments.

    Parameters
    ----------
    results_dir : str or Path, optional
        Directory containing result JSON files. Default: ROOT/results/
    save_dir : str or Path, optional
        If given, save plots here instead of showing.
    only : str, optional
        If given, plot only this experiment ("samples", "latents", "sparsity").
    """
    if results_dir is None:
        results_dir = ROOT / "results"
    results_dir = Path(results_dir)

    # ---- Combined: samples (d=100) + samples_10k (d=10K) ----
    if only in (None, "samples"):
        results_100 = _load_and_prepare(results_dir, EXPERIMENTS["samples"])
        results_10k = _load_and_prepare(results_dir, EXPERIMENTS["samples_10k"])

        if results_100 and results_10k:
            print(f"\n{'='*60}")
            print(f"  Plotting: samples_combined (d=100 + d=10K)")
            print(f"{'='*60}")
            agg_100 = _aggregate(results_100, "n_samples")
            agg_10k = _aggregate(results_10k, "n_samples")
            _plot_combined(
                agg_100, agg_10k, "n_samples",
                r"Number of training samples ($n$)",
                r"$d=100$", r"$d=10\,000$",
                "samples_combined", save_dir,
                groups=CORE_GROUPS,
            )
        elif results_100:
            agg = _aggregate(results_100, "n_samples")
            _plot_experiment(agg, "n_samples", r"Number of training samples ($n$)",
                             "samples", save_dir, groups=CORE_GROUPS)

    # ---- Standalone experiments ----
    standalone = ["latents", "sparsity", "frozen"]
    for exp_name in standalone:
        if only is not None and only != exp_name:
            continue
        exp_info = EXPERIMENTS[exp_name]
        results = _load_and_prepare(results_dir, exp_info)
        if results is None:
            continue

        print(f"\n{'='*60}")
        print(f"  Plotting: {exp_name} ({len(results)} result entries)")
        print(f"{'='*60}")

        agg = _aggregate(results, exp_info["sweep_key"])

        if exp_name == "frozen":
            # Custom 2x3 layout for frozen decoder experiment
            _plot_frozen_experiment(
                agg, exp_info["sweep_key"], exp_info["sweep_label"],
                exp_name, save_dir,
            )
            # Also generate the appendix (all 6 metrics, all methods)
            _plot_single(agg, exp_info["sweep_key"], exp_info["sweep_label"],
                         METRICS_FULL, nrows=2, ncols=3, figsize=(16, 9),
                         exp_name=exp_name, suffix="_appendix", save_dir=save_dir,
                         groups=None)
        else:
            _plot_experiment(
                agg, exp_info["sweep_key"], exp_info["sweep_label"],
                exp_name, save_dir, groups=CORE_GROUPS,
            )

        _plot_method_comparison(
            agg, exp_info["sweep_key"], exp_info["sweep_label"],
            exp_name, save_dir,
        )

    if save_dir:
        print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results from vary_ experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing result JSON files (default: results/).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save plots to this directory instead of showing.",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["samples", "samples_10k", "latents", "sparsity", "frozen"],
        help="Plot only this experiment.",
    )

    args = parser.parse_args()
    plot_all(results_dir=args.results_dir, save_dir=args.save, only=args.only)
