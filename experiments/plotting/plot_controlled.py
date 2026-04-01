"""
Plot results for controlled experiments (warmstart encoder/decoder,
dictionary quality, support recovery).

Design principles (learned from paper_figures legibility issues):
  - Facet by SAE type: one panel per type, max 3 lines per panel.
  - Large fonts, thick lines, minimal clutter.
  - Shared legend at figure level, not per-panel.
  - Log x-axis for iteration/round sweeps.
  - Grouped bars for intervention comparisons.

Usage
-----
    python experiments/plotting/plot_controlled.py
    python experiments/plotting/plot_controlled.py --only encoder
    python experiments/plotting/plot_controlled.py --only decoder
    python experiments/plotting/plot_controlled.py --only dict_quality
    python experiments/plotting/plot_controlled.py --only support
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
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "paper_figures" / "controlled"

# ============================================================================
# Style
# ============================================================================

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

SAE_TYPES = ("relu", "topk", "jumprelu", "MP")
SAE_LABELS = {"relu": "ReLU", "topk": "TopK", "jumprelu": "JumpReLU", "MP": "MP"}

# Two-colour palette: warm-start vs cold-start
C_WARM = "#d94801"   # orange
C_COLD = "#2171b5"   # blue
C_ORACLE = "#333333" # dark gray
C_LSTSQ = "#238b45"  # green
C_RENORM = "#7b4ea3"  # purple
C_ORNORM = "#c0392b"  # red

METRIC_LABELS = {
    "mcc_iid": "MCC (ID)",
    "mcc_ood": "MCC (OOD)",
    "auc_iid": "AUC (ID)",
    "auc_ood": "AUC (OOD)",
    "acc_iid": "Accuracy (ID)",
    "acc_ood": "Accuracy (OOD)",
}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def _load(filename):
    """Load results, trying both single file and split files."""
    p = RESULTS_DIR / filename
    if p.exists():
        return json.loads(p.read_text())

    # Try merging split files: exp_foo_n10.json, exp_foo_n50.json, ...
    stem = p.stem
    parts = sorted(RESULTS_DIR.glob(f"{stem}_n*.json"))
    if parts:
        merged = []
        for part in parts:
            merged.extend(json.loads(part.read_text()))
        print(f"  Merged {len(parts)} part files for {stem} ({len(merged)} entries)")
        return merged

    print(f"  [skip] Not found: {p}")
    return None


# ============================================================================
# Warmstart encoder: convergence curves faceted by SAE type
# ============================================================================


def plot_warmstart_encoder(metric="mcc_ood"):
    """Faceted convergence curves: cold vs warm FISTA, one panel per SAE type."""
    data = _load("exp_warmstart_encoder.json")
    if data is None:
        return

    nvals = sorted(set(r.get("num_latents", 0) for r in data))

    for n in nvals:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        fig.suptitle(f"Encoder Warm-Start Convergence (d = {n})", fontsize=18, y=1.02)

        for ax_idx, sae_type in enumerate(SAE_TYPES):
            ax = axes[ax_idx]

            # Gather cold and warm data
            for prefix, color, label, marker in [
                (f"cold_{sae_type}", C_COLD, "Cold-start (zeros)", "s"),
                (f"warm_{sae_type}", C_WARM, "Warm-start (SAE)", "o"),
            ]:
                rows = [r for r in data
                        if r["method"] == prefix
                        and r.get("num_latents") == n
                        and "n_iter" in r]
                if not rows:
                    continue

                # Aggregate over seeds
                by_iter = defaultdict(list)
                for r in rows:
                    by_iter[r["n_iter"]].append(r.get(metric, 0))

                iters = sorted(by_iter.keys())
                means = [np.mean(by_iter[i]) for i in iters]
                stds = [np.std(by_iter[i]) for i in iters]

                ax.plot(iters, means, color=color, marker=marker,
                        markersize=7, linewidth=2.5, label=label)
                ax.fill_between(iters,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                color=color, alpha=0.12)

            # Oracle ceiling
            oracle_rows = [r for r in data
                           if r["method"] == "fista_oracle"
                           and r.get("num_latents") == n]
            if oracle_rows:
                oracle_val = np.mean([r.get(metric, 0) for r in oracle_rows])
                ax.axhline(oracle_val, color=C_ORACLE, ls="--", lw=1.5,
                           label="Oracle", zorder=1)

            ax.set_title(SAE_LABELS[sae_type])
            ax.set_xlabel("FISTA iterations")
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_xscale("symlog", linthresh=1)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2, linewidth=0.5)

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, edgecolor="black", framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.08), fontsize=13)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])
        _save(fig, OUT_DIR / f"warmstart_encoder_n{n}_{metric}.pdf")


# ============================================================================
# Warmstart decoder: convergence curves faceted by SAE type
# ============================================================================


def plot_warmstart_decoder(metric="mcc_ood"):
    """Faceted convergence curves: cold vs warm DL-FISTA, one panel per SAE type."""
    data = _load("exp_warmstart_decoder.json")
    if data is None:
        return

    nvals = sorted(set(r.get("num_latents", 0) for r in data))

    for n in nvals:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        fig.suptitle(f"Decoder Warm-Start Convergence (d = {n})", fontsize=18, y=1.02)

        for ax_idx, sae_type in enumerate(SAE_TYPES):
            ax = axes[ax_idx]

            for prefix, color, label, marker in [
                (f"cold_dl_{sae_type}", C_COLD, "DL-FISTA (random)", "s"),
                (f"warm_dl_{sae_type}", C_WARM, "DL-FISTA (SAE decoder)", "o"),
            ]:
                rows = [r for r in data
                        if r["method"] == prefix
                        and r.get("num_latents") == n
                        and "n_outer" in r]
                if not rows:
                    continue

                by_round = defaultdict(list)
                for r in rows:
                    by_round[r["n_outer"]].append(r.get(metric, 0))

                rounds = sorted(by_round.keys())
                means = [np.mean(by_round[rd]) for rd in rounds]
                stds = [np.std(by_round[rd]) for rd in rounds]

                ax.plot(rounds, means, color=color, marker=marker,
                        markersize=7, linewidth=2.5, label=label)
                ax.fill_between(rounds,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                color=color, alpha=0.12)

            # Oracle ceiling
            oracle_rows = [r for r in data
                           if r["method"] == "fista_oracle"
                           and r.get("num_latents") == n]
            if oracle_rows:
                oracle_val = np.mean([r.get(metric, 0) for r in oracle_rows])
                ax.axhline(oracle_val, color=C_ORACLE, ls="--", lw=1.5,
                           label="Oracle", zorder=1)

            ax.set_title(SAE_LABELS[sae_type])
            ax.set_xlabel("Dict update rounds")
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_xscale("symlog", linthresh=1)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2, linewidth=0.5)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, edgecolor="black", framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.08), fontsize=13)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])
        _save(fig, OUT_DIR / f"warmstart_decoder_n{n}_{metric}.pdf")


# ============================================================================
# Dictionary quality: grouped bar chart faceted by SAE type
# ============================================================================


def plot_dict_quality(metric="mcc_ood"):
    """Grouped bars: frozen vs renormed vs oracle_norms vs fista_oracle."""
    data = _load("exp_dict_quality.json")
    if data is None:
        return

    nvals = sorted(set(r.get("num_latents", 0) for r in data))

    # One figure per num_latents: 4 panels (SAE types), 4 bars each
    for n in nvals:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        fig.suptitle(f"Dictionary Quality Decomposition (d = {n})", fontsize=18, y=1.02)

        bar_defs = [
            ("frozen_{t}",       C_COLD,   "Frozen decoder"),
            ("renormed_{t}",     C_RENORM, "Re-normalized"),
            ("oracle_norms_{t}", C_ORNORM, "Oracle norms"),
        ]
        bar_width = 0.22

        for ax_idx, sae_type in enumerate(SAE_TYPES):
            ax = axes[ax_idx]
            x_pos = 0

            for bi, (method_tmpl, color, label) in enumerate(bar_defs):
                method = method_tmpl.replace("{t}", sae_type)
                rows = [r for r in data
                        if r["method"] == method and r.get("num_latents") == n]
                if not rows:
                    continue
                val = np.mean([r.get(metric, 0) for r in rows])
                std = np.std([r.get(metric, 0) for r in rows])
                ax.bar(bi * bar_width, val, bar_width * 0.85, yerr=std,
                       color=color, edgecolor="white", linewidth=0.5,
                       label=label if ax_idx == 0 else None,
                       capsize=3, error_kw={"linewidth": 1.5})

            # Oracle ceiling as horizontal line
            oracle_rows = [r for r in data
                           if r["method"] == "fista_oracle"
                           and r.get("num_latents") == n]
            if oracle_rows:
                oracle_val = np.mean([r.get(metric, 0) for r in oracle_rows])
                ax.axhline(oracle_val, color=C_ORACLE, ls="--", lw=1.5,
                           label="Oracle" if ax_idx == 0 else None, zorder=1)

            ax.set_title(SAE_LABELS[sae_type])
            ax.set_xticks([])
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, edgecolor="black", framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.08), fontsize=13)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])
        _save(fig, OUT_DIR / f"dict_quality_n{n}_{metric}.pdf")

    # Diagnostics plot: cosine + angular error + norm ratio across num_latents
    diag_data = _load("exp_dict_quality_diagnostics.json")
    if diag_data is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle("Dictionary Column Diagnostics vs num_latents", fontsize=18, y=1.02)

    diag_metrics = [
        ("mean_cosine", "Cosine similarity"),
        ("mean_angular_error", "Angular error (rad)"),
        ("mean_norm_ratio", "Norm ratio (D/A)"),
    ]

    sae_colors = {
        "relu": "#c0392b", "topk": "#2980b9",
        "jumprelu": "#27ae60", "MP": "#8e44ad",
    }

    for ax_idx, (dkey, dlabel) in enumerate(diag_metrics):
        ax = axes[ax_idx]
        for sae_type in SAE_TYPES:
            rows_by_n = defaultdict(list)
            for d in diag_data:
                if d["sae_type"] == sae_type:
                    rows_by_n[d["num_latents"]].append(d[dkey])
            if not rows_by_n:
                continue
            ns = sorted(rows_by_n.keys())
            means = [np.mean(rows_by_n[nv]) for nv in ns]
            stds = [np.std(rows_by_n[nv]) for nv in ns]
            ax.plot(ns, means, color=sae_colors[sae_type], marker="o",
                    markersize=7, linewidth=2.5, label=SAE_LABELS[sae_type])
            ax.fill_between(ns,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=sae_colors[sae_type], alpha=0.12)

        ax.set_xlabel("num_latents")
        ax.set_ylabel(dlabel)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        if dkey == "mean_cosine":
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color=C_ORACLE, ls="--", lw=1, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=True, edgecolor="black", framealpha=0.95,
               bbox_to_anchor=(0.5, -0.08), fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    _save(fig, OUT_DIR / "dict_quality_diagnostics.pdf")


# ============================================================================
# Support recovery: grouped bar chart faceted by SAE type
# ============================================================================


def plot_support_recovery(metric="mcc_ood"):
    """Grouped bars: sae vs support_lstsq vs fista, plus diagnostics."""
    data = _load("exp_support_recovery.json")
    if data is None:
        return

    nvals = sorted(set(r.get("num_latents", 0) for r in data))

    for n in nvals:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        fig.suptitle(f"Support Recovery (d = {n})", fontsize=18, y=1.02)

        bar_defs = [
            ("sae_{t}",            C_COLD,  "SAE (raw)"),
            ("support_lstsq_{t}",  C_LSTSQ, "SAE support + LSTSQ"),
            ("fista+{t}",          C_WARM,  "FISTA (frozen dec)"),
        ]
        bar_width = 0.22

        for ax_idx, sae_type in enumerate(SAE_TYPES):
            ax = axes[ax_idx]

            for bi, (method_tmpl, color, label) in enumerate(bar_defs):
                method = method_tmpl.replace("{t}", sae_type)
                rows = [r for r in data
                        if r["method"] == method and r.get("num_latents") == n]
                if not rows:
                    continue
                val = np.mean([r.get(metric, 0) for r in rows])
                std = np.std([r.get(metric, 0) for r in rows])
                ax.bar(bi * bar_width, val, bar_width * 0.85, yerr=std,
                       color=color, edgecolor="white", linewidth=0.5,
                       label=label if ax_idx == 0 else None,
                       capsize=3, error_kw={"linewidth": 1.5})

            oracle_rows = [r for r in data
                           if r["method"] == "fista_oracle"
                           and r.get("num_latents") == n]
            if oracle_rows:
                oracle_val = np.mean([r.get(metric, 0) for r in oracle_rows])
                ax.axhline(oracle_val, color=C_ORACLE, ls="--", lw=1.5,
                           label="Oracle" if ax_idx == 0 else None, zorder=1)

            ax.set_title(SAE_LABELS[sae_type])
            ax.set_xticks([])
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, edgecolor="black", framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.08), fontsize=13)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])
        _save(fig, OUT_DIR / f"support_recovery_n{n}_{metric}.pdf")

    # Diagnostics: precision/recall/F1 across num_latents
    diag_data = _load("exp_support_recovery_diagnostics.json")
    if diag_data is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.suptitle("Support Recovery Diagnostics (OOD)", fontsize=18, y=1.02)

    diag_metrics = [
        ("precision_ood", "Precision"),
        ("recall_ood", "Recall"),
        ("f1_ood", "F1"),
    ]

    sae_colors = {
        "relu": "#c0392b", "topk": "#2980b9",
        "jumprelu": "#27ae60", "MP": "#8e44ad",
    }

    for ax_idx, (dkey, dlabel) in enumerate(diag_metrics):
        ax = axes[ax_idx]
        for sae_type in SAE_TYPES:
            rows_by_n = defaultdict(list)
            for d in diag_data:
                if d["sae_type"] == sae_type:
                    rows_by_n[d["num_latents"]].append(d[dkey])
            if not rows_by_n:
                continue
            ns = sorted(rows_by_n.keys())
            means = [np.mean(rows_by_n[nv]) for nv in ns]
            stds = [np.std(rows_by_n[nv]) for nv in ns]
            ax.plot(ns, means, color=sae_colors[sae_type], marker="o",
                    markersize=7, linewidth=2.5, label=SAE_LABELS[sae_type])
            ax.fill_between(ns,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=sae_colors[sae_type], alpha=0.12)

        ax.set_xlabel("num_latents")
        ax.set_ylabel(dlabel)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=True, edgecolor="black", framealpha=0.95,
               bbox_to_anchor=(0.5, -0.08), fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    _save(fig, OUT_DIR / "support_recovery_diagnostics.pdf")


# ============================================================================
# Main
# ============================================================================


# ============================================================================
# Learning dynamics: cosine similarity during training
# ============================================================================


def plot_learning_dynamics(metric=None):
    """SAE vs DL-FISTA dictionary cosine during training, faceted by SAE type."""
    data = _load("exp_learning_dynamics.json")
    if data is None:
        return

    nvals = sorted(set(r.get("num_latents", 0) for r in data))

    sae_colors = {
        "relu": "#c0392b", "topk": "#2980b9",
        "jumprelu": "#27ae60", "MP": "#8e44ad",
    }

    for n in nvals:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        fig.suptitle(f"Dictionary Quality During Training (d = {n})", fontsize=18, y=1.02)

        for ax_idx, sae_type in enumerate(SAE_TYPES):
            ax = axes[ax_idx]

            # SAE learning curve (epoch vs cosine)
            sae_rows = [r for r in data
                        if r["method"] == f"sae_{sae_type}"
                        and r.get("num_latents") == n
                        and "epoch" in r]
            if sae_rows:
                by_ep = defaultdict(list)
                for r in sae_rows:
                    by_ep[r["epoch"]].append(r["mean_cosine"])
                eps = sorted(by_ep.keys())
                means = [np.mean(by_ep[e]) for e in eps]
                stds = [np.std(by_ep[e]) for e in eps]
                ax.plot(eps, means, color=sae_colors[sae_type], marker="o",
                        markersize=4, linewidth=2.5, label=f"SAE ({SAE_LABELS[sae_type]})")
                ax.fill_between(eps,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                color=sae_colors[sae_type], alpha=0.12)

            # DL-FISTA learning curve (round vs cosine) — same for all panels
            if ax_idx == 0:
                dl_rows = [r for r in data
                           if r["method"] == "dl_fista"
                           and r.get("num_latents") == n
                           and "round" in r]
                if dl_rows:
                    by_rd = defaultdict(list)
                    for r in dl_rows:
                        by_rd[r["round"]].append(r["mean_cosine"])
                    rds = sorted(by_rd.keys())
                    dl_means = [np.mean(by_rd[rd]) for rd in rds]
                    dl_label = "DL-FISTA"

            # Plot DL-FISTA on all panels for comparison
            if dl_rows:
                ax.plot(rds, dl_means, color="#333333", ls="--", lw=2,
                        label="DL-FISTA" if ax_idx == 0 else None, zorder=1)

            ax.set_title(SAE_LABELS[sae_type])
            ax.set_xlabel("Epoch / Round")
            if ax_idx == 0:
                ax.set_ylabel("Cosine similarity to ground truth")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, linewidth=0.5)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, edgecolor="black", framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.08), fontsize=13)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])
        _save(fig, OUT_DIR / f"learning_dynamics_n{n}.pdf")


# ============================================================================
# Lambda sensitivity: frozen decoder MCC vs lambda
# ============================================================================


def plot_lambda_sensitivity(metric="mcc_ood"):
    """Frozen decoder MCC vs fista_lam, faceted by SAE type."""
    data = _load("exp_lambda_sensitivity.json")
    if data is None:
        return

    nvals = sorted(set(r.get("num_latents", 0) for r in data))

    for n in nvals:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        fig.suptitle(f"Lambda Sensitivity (d = {n})", fontsize=18, y=1.02)

        for ax_idx, sae_type in enumerate(SAE_TYPES):
            ax = axes[ax_idx]

            # Frozen decoder at each lambda
            frozen_rows = [r for r in data
                           if r["method"] == f"frozen_{sae_type}"
                           and r.get("num_latents") == n
                           and "lam" in r]
            if frozen_rows:
                by_lam = defaultdict(list)
                for r in frozen_rows:
                    by_lam[r["lam"]].append(r.get(metric, 0))
                lams = sorted(by_lam.keys())
                means = [np.mean(by_lam[l]) for l in lams]
                stds = [np.std(by_lam[l]) for l in lams]
                ax.plot(lams, means, color="#2171b5", marker="s",
                        markersize=7, linewidth=2.5, label="FISTA (frozen dec)")
                ax.fill_between(lams,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                color="#2171b5", alpha=0.12)

            # SAE baseline (horizontal line)
            sae_rows = [r for r in data
                        if r["method"] == f"sae_{sae_type}"
                        and r.get("num_latents") == n]
            if sae_rows:
                sae_val = np.mean([r.get(metric, 0) for r in sae_rows])
                ax.axhline(sae_val, color="#d94801", ls="--", lw=2,
                           label="SAE encoder" if ax_idx == 0 else None)

            # Oracle at each lambda
            oracle_rows = [r for r in data
                           if r["method"] == "fista_oracle"
                           and r.get("num_latents") == n
                           and "lam" in r]
            if oracle_rows:
                by_lam = defaultdict(list)
                for r in oracle_rows:
                    by_lam[r["lam"]].append(r.get(metric, 0))
                lams = sorted(by_lam.keys())
                means = [np.mean(by_lam[l]) for l in lams]
                ax.plot(lams, means, color="#333333", ls="--", lw=1.5,
                        marker=".", markersize=5,
                        label="Oracle" if ax_idx == 0 else None, zorder=1)

            ax.set_title(SAE_LABELS[sae_type])
            ax.set_xlabel(r"$\lambda$ (FISTA)")
            ax.set_xscale("log")
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2, linewidth=0.5)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, edgecolor="black", framealpha=0.95,
                   bbox_to_anchor=(0.5, -0.08), fontsize=13)
        plt.tight_layout(rect=[0, 0.05, 1, 1.0])
        _save(fig, OUT_DIR / f"lambda_sensitivity_n{n}_{metric}.pdf")


PLOT_FNS = {
    "encoder": plot_warmstart_encoder,
    "decoder": plot_warmstart_decoder,
    "dict_quality": plot_dict_quality,
    "support": plot_support_recovery,
    "dynamics": plot_learning_dynamics,
    "lambda": plot_lambda_sensitivity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, choices=list(PLOT_FNS.keys()),
                        help="Plot only one experiment type")
    parser.add_argument("--metric", type=str, default="mcc_ood",
                        choices=list(METRIC_LABELS.keys()),
                        help="Primary metric to plot (default: mcc_ood)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        fns = {args.only: PLOT_FNS[args.only]}
    else:
        fns = PLOT_FNS

    for name, fn in fns.items():
        print(f"\n--- {name} ---")
        fn(metric=args.metric)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
