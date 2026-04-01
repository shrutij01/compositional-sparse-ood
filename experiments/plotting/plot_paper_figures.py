"""
Generate paper-ready figures (main text + appendix).

Produces 4 figures, each with a main-text and appendix version:

  1. vary_sparsity   — line plots over ρ = k/d
  2. vary_frozen     — dumbbell plots with frozen decoder comparisons
  3. vary_samples    — line plots over n (d=100 only)
  4. phase_faceted   — faceted phase transition plots

Main text:  MCC (ID),  AUC (OOD)
Appendix:   MCC (ID),  MCC (OOD),  Accuracy (ID),  Accuracy (OOD),
            AUC (ID),  AUC (OOD)

Output goes to paper_figures/.

Usage
-----
    python experiments/plot_paper_figures.py
    python experiments/plot_paper_figures.py --only sparsity
    python experiments/plot_paper_figures.py --only frozen
    python experiments/plot_paper_figures.py --only samples
    python experiments/plot_paper_figures.py --only phase
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "paper_figures"

# ============================================================================
# Global matplotlib style
# ============================================================================

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.labelweight": "bold",
})

# ============================================================================
# Metric definitions  (IID → ID for paper)
# ============================================================================

METRICS_MAIN = [
    ("mcc_iid", "MCC (ID)"),
    ("auc_ood", "AUC (OOD)"),
]

METRICS_APX = [
    ("mcc_iid", "MCC (ID)"),
    ("mcc_ood", "MCC (OOD)"),
    ("acc_iid", "Accuracy (ID)"),
    ("acc_ood", "Accuracy (OOD)"),
    ("auc_iid", "AUC (ID)"),
    ("auc_ood", "AUC (OOD)"),
]

METRIC_LABEL = {k: v for k, v in METRICS_APX}

# ============================================================================
# Category colour scheme
# ============================================================================

CAT_COLORS = {
    "SC":       "#2171b5",   # blue   — sparse coding
    "SAE":      "#d94801",   # orange — SAEs
    "Frozen":   "#7b4ea3",   # purple — FISTA on frozen SAE decoder
    "Refined":  "#238b45",   # green  — hybrid (warm-started FISTA)
    "Baseline": "#555555",   # gray
}

# ============================================================================
# Method metadata — colours follow the category scheme
# ============================================================================

METHOD_META = {
    # Sparse coding — solid lines, blue (dark→medium, all clearly visible)
    "fista_oracle":   {"label": "FISTA (oracle)",   "color": "#084594", "ls": "-",  "marker": "o",  "group": "SC"},
    "lista_oracle":   {"label": "LISTA (oracle)",   "color": "#1a6fb5", "ls": "-",  "marker": "D",  "group": "SC"},
    "dl_fista":       {"label": "DL-FISTA",         "color": "#3787c0", "ls": "-",  "marker": "s",  "group": "SC"},
    "dl_lista":       {"label": "DL-LISTA",         "color": "#5198ca", "ls": "-",  "marker": "v",  "group": "SC"},
    "softplus_adam":  {"label": "Softplus-Adam",    "color": "#6aaed6", "ls": "-",  "marker": "^",  "group": "SC"},
    # SAE — dashed lines, orange (all medium-to-dark)
    "sae_relu":       {"label": "SAE (ReLU)",       "color": "#8c2d04", "ls": "--", "marker": "o",  "group": "SAE"},
    "sae_topk":       {"label": "SAE (TopK)",       "color": "#c44103", "ls": "--", "marker": "s",  "group": "SAE"},
    "sae_jumprelu":   {"label": "SAE (JumpReLU)",   "color": "#e66a1e", "ls": "--", "marker": "^",  "group": "SAE"},
    "sae_MP":         {"label": "SAE (MP)",         "color": "#f08a40", "ls": "--", "marker": "D",  "group": "SAE"},
    # Frozen decoder — dash-dot, purple (tighter range, all visible)
    "fista+relu":     {"label": "FISTA+SAE (RL)",   "color": "#4a1486", "ls": "-.", "marker": "o",  "group": "Frozen"},
    "fista+topk":     {"label": "FISTA+SAE (TK)",   "color": "#6a3d9a", "ls": "-.", "marker": "s",  "group": "Frozen"},
    "fista+jumprelu": {"label": "FISTA+SAE (JR)",   "color": "#8b6baf", "ls": "-.", "marker": "^",  "group": "Frozen"},
    "fista+MP":       {"label": "FISTA+SAE (MP)",   "color": "#a68cc2", "ls": "-.", "marker": "D",  "group": "Frozen"},
    # Refined (hybrid) — dash-dot, green (tighter range)
    "refined_relu":     {"label": "Refined (RL)",   "color": "#005a32", "ls": "-.", "marker": "o",  "group": "Refined"},
    "refined_topk":     {"label": "Refined (TK)",   "color": "#1e8c49", "ls": "-.", "marker": "s",  "group": "Refined"},
    "refined_jumprelu": {"label": "Refined (JR)",   "color": "#3da660", "ls": "-.", "marker": "^",  "group": "Refined"},
    "refined_MP":       {"label": "Refined (MP)",   "color": "#5cb778", "ls": "-.", "marker": "D",  "group": "Refined"},
    # Baselines — dotted, gray
    "linear_probe":   {"label": "Linear probe",    "color": "#1a1a1a", "ls": ":",  "marker": "*",  "group": "Baseline"},
    "raw":            {"label": "Raw (Y)",          "color": "#555555", "ls": ":",  "marker": "x",  "group": "Baseline"},
    "pca":            {"label": "PCA",              "color": "#888888", "ls": ":",  "marker": "+",  "group": "Baseline"},
    # Aliases used in exp_phase_transition.json
    "fista_sup":      {"label": "FISTA (oracle)",   "color": "#084594", "ls": "-",  "marker": "o",  "group": "SC"},
    "fista_unsup":    {"label": "DL-FISTA",         "color": "#3787c0", "ls": "-",  "marker": "s",  "group": "SC"},
    "direct_unsup":   {"label": "Softplus-Adam",    "color": "#6aaed6", "ls": "-",  "marker": "^",  "group": "SC"},
}

CORE_GROUPS = {"SAE", "SC", "Baseline"}

# ============================================================================
# Data loading & aggregation
# ============================================================================


def _load_results(path):
    p = Path(path)
    if not p.exists():
        print(f"  [skip] Results not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def _aggregate(results, sweep_key):
    """Group by (method, sweep_value), compute mean/std over seeds."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        buckets[r["method"]][r[sweep_key]].append(r)

    agg = {}
    for method, val_dict in buckets.items():
        x_vals = sorted(val_dict.keys())
        agg[method] = {"x": x_vals}
        for metric_key, _ in METRICS_APX:
            means, stds = [], []
            for xv in x_vals:
                vals = [r.get(metric_key, 0) for r in val_dict[xv]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            agg[method][f"{metric_key}_mean"] = means
            agg[method][f"{metric_key}_std"] = stds
    return agg


def _load_and_merge(primary_file, extend_file, sweep_key, compute_key=None):
    """Load primary results, optionally merge extension file, apply key transform."""
    results = _load_results(RESULTS_DIR / primary_file)
    if results is None:
        return None

    if extend_file:
        ext = _load_results(RESULTS_DIR / extend_file)
        if ext:
            existing = set((r["method"], r[sweep_key]) for r in results)
            extra = [r for r in ext if (r["method"], r.get(sweep_key, 0)) not in existing]
            results.extend(extra)
            print(f"  Merged {len(extra)} entries from {extend_file}")

    if compute_key is not None:
        for r in results:
            r[sweep_key] = compute_key(r)

    return results


# ============================================================================
# Generic line plot
# ============================================================================


def _plot_lines(agg, sweep_key, sweep_label, metrics, nrows, ncols,
                figsize, save_path, groups=None, show_legend=True):
    """Line plot with mean ± std shading."""
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
                label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"],
                markersize=8, linewidth=2.4,
            )
            ax.fill_between(
                x,
                np.array(y_mean) - np.array(y_std),
                np.array(y_mean) + np.array(y_std),
                color=style["color"], alpha=0.08,
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

    # Hide unused panels
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    if show_legend:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center",
            ncol=len(handles), frameon=True,
            bbox_to_anchor=(0.5, -0.03), fontsize=10,
            edgecolor="black", framealpha=0.9,
        )
    plt.tight_layout(rect=[0, 0.07 if show_legend else 0.0, 1, 1.0])
    _save(fig, save_path)


# ============================================================================
# Frozen decoder dumbbell chart
# ============================================================================


def _frozen_dumbbell_config(agg):
    """Build configuration for frozen dumbbell plots."""
    sae_types = ["relu", "topk", "jumprelu", "MP"]

    # Activation-type colours for dumbbell pairs
    act_colors = {
        "relu":     "#c0392b",
        "topk":     "#2980b9",
        "jumprelu": "#27ae60",
        "MP":       "#8e44ad",
    }
    act_labels = {
        "relu": "ReLU", "topk": "TopK", "jumprelu": "JumpReLU", "MP": "MP",
    }

    # Panel 1: SC + SAE overview (dot plot with METHOD_META colours)
    panel1_methods = ["dl_fista", "softplus_adam"] + [f"sae_{s}" for s in sae_types]

    # Panels 2 & 3: dumbbell pairs (SAE vs Frozen, SAE vs Refined)
    frozen_pairs = [(f"sae_{s}", f"fista+{s}", s) for s in sae_types]
    refined_pairs = [(f"sae_{s}", f"refined_{s}", s) for s in sae_types]

    # Line overlay methods
    line_methods = ["fista_oracle", "linear_probe"]
    line_styles = {
        "fista_oracle": {
            "color": METHOD_META["fista_oracle"]["color"],
            "ls": "-", "marker": "o", "label": "FISTA (oracle)",
        },
        "linear_probe": {
            "color": METHOD_META["linear_probe"]["color"],
            "ls": ":", "marker": "*", "label": "Linear probe",
        },
    }

    all_x = sorted(set(xv for m_data in agg.values() for xv in m_data["x"]))
    x_labels = [f"{int(v)}" if v < 1000 else f"{int(v/1000)}K" for v in all_x]

    return (panel1_methods, frozen_pairs, refined_pairs,
            act_colors, act_labels, all_x, x_labels,
            line_methods, line_styles)


def _draw_frozen_dumbbell_row(axes_row, agg, metric_key, metric_label,
                              panel1_methods, frozen_pairs, refined_pairs,
                              act_colors, act_labels, all_x, x_labels,
                              line_methods, line_styles, show_x_labels=True):
    """Draw one row: panel 1 line plot + panels 2-3 dumbbells (no oracle overlays)."""
    n_groups = len(all_x)
    group_positions = np.arange(n_groups)
    line_handles, line_labels_seen = [], set()
    act_handles = {}  # for dumbbell legend
    is_mcc = "mcc" in metric_key

    def _get_value(method, xv):
        if method not in agg:
            return np.nan, 0.0
        data = agg[method]
        if xv in data["x"]:
            idx = data["x"].index(xv)
            return data[f"{metric_key}_mean"][idx], data[f"{metric_key}_std"][idx]
        return np.nan, 0.0

    def _draw_line_overlays(ax, panel_vals):
        for lm in line_methods:
            if lm not in agg:
                continue
            ls = line_styles[lm]
            means = [_get_value(lm, xv)[0] for xv in all_x]
            panel_vals.extend(m for m in means if not np.isnan(m))
            line, = ax.plot(
                group_positions, means,
                color=ls["color"], linestyle=ls["ls"], marker=ls["marker"],
                markersize=9, linewidth=2.4, label=ls["label"], zorder=5,
            )
            if ls["label"] not in line_labels_seen:
                line_handles.append(line)
                line_labels_seen.add(ls["label"])

    def _setup_ax(ax, col, panel_vals):
        ax.set_xticks(group_positions)
        if show_x_labels:
            ax.set_xticklabels(x_labels, fontsize=9)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")
        if col == 0:
            ax.set_ylim(0, 1.05)
        elif is_mcc:
            ax.set_ylim(0, 0.6)
        elif panel_vals:
            ymin = max(0, min(panel_vals) - 0.08)
            ymax = min(1.05, max(panel_vals) + 0.08)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0, 1.05)

    # --- Panel 1: line plot for SC + SAE methods ---
    ax0 = axes_row[0]
    methods_p1 = [m for m in panel1_methods if m in agg and m in METHOD_META]
    panel_vals_0 = []
    p1_handles, p1_labels_seen = [], set()

    for method in methods_p1:
        style = METHOD_META[method]
        means = [_get_value(method, xv)[0] for xv in all_x]
        stds = [_get_value(method, xv)[1] for xv in all_x]
        panel_vals_0.extend(m for m in means if not np.isnan(m))
        h, = ax0.plot(
            group_positions, means,
            color=style["color"], linestyle=style["ls"], marker=style["marker"],
            markersize=8, linewidth=2.4, label=style["label"], zorder=3,
        )
        ax0.fill_between(
            group_positions,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=style["color"], alpha=0.08,
        )
        if style["label"] not in p1_labels_seen:
            p1_handles.append(h)
            p1_labels_seen.add(style["label"])

    _draw_line_overlays(ax0, panel_vals_0)
    _setup_ax(ax0, 0, panel_vals_0)

    # --- Panels 2 & 3: dumbbells (no oracle overlays) ---
    for col, pairs in [(1, frozen_pairs), (2, refined_pairs)]:
        ax = axes_row[col]
        n_pairs = len(pairs)
        jw = 0.6
        panel_vals = []

        for pi, (sae_method, alt_method, act_type) in enumerate(pairs):
            color = act_colors[act_type]
            offset = (pi - n_pairs / 2 + 0.5) * (jw / n_pairs)

            for gi, xv in enumerate(all_x):
                y_sae, std_sae = _get_value(sae_method, xv)
                y_alt, std_alt = _get_value(alt_method, xv)
                if np.isnan(y_sae) and np.isnan(y_alt):
                    continue
                xpos = group_positions[gi] + offset

                # Connecting line
                if not np.isnan(y_sae) and not np.isnan(y_alt):
                    ax.plot([xpos, xpos], [y_sae, y_alt],
                            color=color, linewidth=2.0, zorder=2)

                # SAE endpoint: filled circle
                if not np.isnan(y_sae):
                    panel_vals.append(y_sae)
                    ax.plot(xpos, y_sae, marker="o", color=color,
                            markersize=10, zorder=4)

                # Frozen/Refined endpoint: filled triangle
                if not np.isnan(y_alt):
                    panel_vals.append(y_alt)
                    ax.plot(xpos, y_alt, marker="^", color=color,
                            markersize=10, zorder=4)

            # Collect one handle per activation type (for legend)
            if act_type not in act_handles:
                h_line, = ax.plot([], [], color=color, linewidth=1.2)
                h_dot, = ax.plot([], [], marker="o", color=color,
                                 markersize=10, linestyle="None")
                h_tri, = ax.plot([], [], marker="^", color=color,
                                 markersize=10, linestyle="None")
                act_handles[act_type] = (h_line, h_dot, h_tri)

        _setup_ax(ax, col, panel_vals)

    return p1_handles, act_handles, line_handles


def _style_red_ylim_ticks(fig, axes_list):
    """Bold red the extremal y-tick labels."""
    fig.canvas.draw()
    for ax in axes_list:
        yticks = ax.get_yticks()
        ylim = ax.get_ylim()
        visible = [t for t in yticks if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9]
        if len(visible) < 2:
            continue
        ax.set_yticks(visible)
        labels = [f"{t:g}" for t in visible]
        tick_labels = ax.set_yticklabels(labels)
        for i, tl in enumerate(tick_labels):
            if i == 0 or i == len(tick_labels) - 1:
                tl.set_color("red")
                tl.set_fontweight("bold")
                tl.set_fontsize(11)


def _add_frozen_dumbbell_legends(fig, p1_handles, act_handles, act_labels,
                                 line_handles, col_centers,
                                 legend_y=0.10, line_y=0.02):
    """Panel A legend + compact dumbbell legend (colour = activation, shape = type)."""
    from matplotlib.lines import Line2D

    # --- Panel A: method legend ---
    handles_p1 = [h for h in p1_handles if not h.get_label().startswith("_")]
    if handles_p1 or line_handles:
        p1_all = list(handles_p1) + list(line_handles)
        p1_labels = [h.get_label() for h in p1_all]
        fig.legend(
            p1_all, p1_labels, fontsize=10, loc="upper center",
            bbox_to_anchor=(col_centers[0], legend_y),
            framealpha=0.9, ncol=4, edgecolor="black",
        )

    # --- Panels B & C: colour = activation, shape = inference type ---
    db_handles, db_labels = [], []

    for act_type in ["relu", "topk", "jumprelu", "MP"]:
        if act_type in act_handles:
            color = act_handles[act_type][0].get_color()
            h = Line2D([0], [0], color=color, marker="s", markersize=8,
                       linewidth=2, linestyle="-")
            db_handles.append(h)
            db_labels.append(act_labels[act_type])

    h_sae = Line2D([0], [0], marker="o", color="black", markersize=9,
                   linestyle="None")
    h_alt = Line2D([0], [0], marker="^", color="black", markersize=9,
                   linestyle="None")
    db_handles.extend([h_sae, h_alt])
    db_labels.extend(["SAE", "Frozen / Refined"])

    mid_x = (col_centers[1] + col_centers[2]) / 2
    fig.legend(
        db_handles, db_labels, fontsize=10, loc="upper center",
        bbox_to_anchor=(mid_x, legend_y), framealpha=0.9,
        ncol=len(db_handles), edgecolor="black", markerscale=1.2,
    )


def _plot_frozen(agg, metrics, sweep_label, save_path, show_legend=True):
    """Frozen decoder dumbbell chart: N_metrics × 3 panels."""
    (panel1_methods, frozen_pairs, refined_pairs,
     act_colors, act_labels, all_x, x_labels,
     line_methods, line_styles) = _frozen_dumbbell_config(agg)

    n_metrics = len(metrics)
    col_centers = [0.185, 0.5, 0.815]

    fig, axes = plt.subplots(n_metrics, 3, figsize=(18, 4.5 * n_metrics),
                             squeeze=False)
    all_p1_handles = []
    all_act_handles = {}
    all_line_handles = []

    for row, (metric_key, metric_label) in enumerate(metrics):
        show_x = (row == n_metrics - 1)
        p1h, ach, lh = _draw_frozen_dumbbell_row(
            axes[row], agg, metric_key, metric_label,
            panel1_methods, frozen_pairs, refined_pairs,
            act_colors, act_labels, all_x, x_labels,
            line_methods, line_styles, show_x_labels=show_x,
        )
        if row == 0:
            all_p1_handles = p1h
            all_act_handles = ach
            all_line_handles = lh

    if show_legend:
        plt.tight_layout(rect=[0, 0.25, 1, 1.0])
        _style_red_ylim_ticks(fig, [axes[r, c] for r in range(n_metrics) for c in range(3)])
        fig.text(0.5, 0.24, sweep_label, ha="center", fontsize=13, fontweight="bold")
        _add_frozen_dumbbell_legends(fig, all_p1_handles, all_act_handles, act_labels,
                                     all_line_handles, col_centers,
                                     legend_y=0.22, line_y=0.05)
    else:
        plt.tight_layout()
        _style_red_ylim_ticks(fig, [axes[r, c] for r in range(n_metrics) for c in range(3)])
    _save(fig, save_path)


# ============================================================================
# Phase transition faceted
# ============================================================================


def _plot_phase_faceted(results, metric_key, metric_label, save_path, show_legend=True):
    """One subplot per (num_latents, k) combination."""
    # Normalise key names: the phase transition JSON may use
    # n/m/r instead of num_latents/input_dim/eps.
    for row in results:
        if "num_latents" not in row and "n" in row:
            row["num_latents"] = row["n"]
        if "input_dim" not in row and "m" in row:
            row["input_dim"] = row["m"]
        if "delta" not in row:
            row["delta"] = row["input_dim"] / row["num_latents"]
        if "rho" not in row:
            row["rho"] = row["k"] / row["num_latents"]

    methods = sorted(set(r["method"] for r in results))
    skip = {"linear_probe"}
    sc_methods = [m for m in methods if not m.startswith("sae_") and m not in skip]
    sae_methods = [m for m in methods if m.startswith("sae_")]
    ordered = sc_methods + sae_methods

    all_nk = sorted(set((r["num_latents"], r["k"]) for r in results))
    # Keep only the four corners: (min_n, min_k), (min_n, max_k),
    #                              (max_n, min_k), (max_n, max_k)
    all_n = sorted(set(n for n, _ in all_nk))
    all_k = sorted(set(k for _, k in all_nk))
    corner_n = {all_n[0], all_n[-1]}
    corner_k = {all_k[0], all_k[-1]}
    nk_pairs = [(n, k) for n, k in all_nk if n in corner_n and k in corner_k]

    n_panels = len(nk_pairs)
    ncols = 2
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (n, k) in enumerate(nk_pairs):
        ax = axes[idx // ncols][idx % ncols]
        panel_rows = [r for r in results if r["num_latents"] == n and r["k"] == k]

        for method in ordered:
            mrows = [r for r in panel_rows if r["method"] == method]
            if not mrows:
                continue
            delta_vals = sorted(set(r["delta"] for r in mrows))
            means = [
                np.mean([r.get(metric_key, 0) for r in mrows
                         if abs(r["delta"] - dv) < 1e-6])
                for dv in delta_vals
            ]
            style = METHOD_META.get(method, {})
            linestyle = style.get("ls", "--" if method.startswith("sae_") else "-")
            color = style.get("color", None)
            marker = style.get("marker", "o")
            label = style.get("label", method.replace("_", " "))
            ax.plot(delta_vals, means, marker=marker, markersize=8,
                    color=color, linestyle=linestyle, linewidth=2.4, label=label)

        ax.set_xlabel(r"$\delta = m/d$", fontweight="bold")
        ax.set_ylabel(metric_label, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if show_legend:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(handles), fontsize=10,
                   bbox_to_anchor=(0.5, -0.02),
                   edgecolor="black", framealpha=0.9)
    fig.tight_layout(rect=[0, 0.04 if show_legend else 0.0, 1, 1.0])
    _save(fig, save_path)


# ============================================================================
# Save helper
# ============================================================================


def _save(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ============================================================================
# Master legend figure
# ============================================================================


def plot_legend():
    """Standalone method legend for the paper.

    Uses LaTeX rendering for \\textsc names with U/S subscripts.
    Falls back to plain text if LaTeX is unavailable.
    """
    print("\n=== method_legend ===")

    # ------------------------------------------------------------------
    # Legend data: (method_key, tex_name, short_description)
    # ------------------------------------------------------------------
    categories = [
        (
            "Sparse Coding",
            "SC",
            r"classical sparse recovery ($\ell_1$ or learned)",
            [
                ("fista_oracle",  r"\textsc{Fista}$_{\mathrm{S}}$",    "oracle $A$"),
                ("lista_oracle",  r"\textsc{Lista}$_{\mathrm{S}}$",    "unrolled, oracle $A$"),
                ("dl_fista",      r"\textsc{Dl-Fista}$_{\mathrm{U}}$", "learned dict."),
                ("dl_lista",      r"\textsc{Dl-Lista}$_{\mathrm{U}}$", "learned, unrolled"),
                ("softplus_adam", r"\textsc{Softplus}$_{\mathrm{U}}$",  "diff.\ relaxation"),
            ],
        ),
        (
            "SAE",
            "SAE",
            r"sparse autoencoders (amortised encoder)",
            [
                ("sae_relu",     r"\textsc{Sae}(ReLU)$_{\mathrm{U}}$",     ""),
                ("sae_topk",     r"\textsc{Sae}(TopK)$_{\mathrm{U}}$",     ""),
                ("sae_jumprelu", r"\textsc{Sae}(JumpReLU)$_{\mathrm{U}}$", ""),
                ("sae_MP",       r"\textsc{Sae}(MP)$_{\mathrm{U}}$",       ""),
            ],
        ),
        (
            "Frozen Decoder",
            "Frozen",
            r"FISTA on frozen SAE decoder $\hat A$",
            [
                ("fista+relu",     r"\textsc{Fista\,+\,Sae}(ReLU)$_{\mathrm{U}}$",     ""),
                ("fista+topk",     r"\textsc{Fista\,+\,Sae}(TopK)$_{\mathrm{U}}$",     ""),
                ("fista+jumprelu", r"\textsc{Fista\,+\,Sae}(JumpReLU)$_{\mathrm{U}}$", ""),
                ("fista+MP",       r"\textsc{Fista\,+\,Sae}(MP)$_{\mathrm{U}}$",       ""),
            ],
        ),
        (
            "Refined (Hybrid)",
            "Refined",
            r"FISTA warm-started from SAE codes",
            [
                ("refined_relu",     r"\textsc{Refined}(ReLU)$_{\mathrm{U}}$",     ""),
                ("refined_topk",     r"\textsc{Refined}(TopK)$_{\mathrm{U}}$",     ""),
                ("refined_jumprelu", r"\textsc{Refined}(JumpReLU)$_{\mathrm{U}}$", ""),
                ("refined_MP",       r"\textsc{Refined}(MP)$_{\mathrm{U}}$",       ""),
            ],
        ),
    ]

    baselines = [
        ("linear_probe", r"\textsc{Linear Probe}$_{\mathrm{S}}$", "supervised oracle"),
        ("raw",          r"\textsc{Raw}$_{\mathrm{U}}$",          "raw observations"),
        ("pca",          r"\textsc{Pca}$_{\mathrm{U}}$",          "principal components"),
    ]

    # ------------------------------------------------------------------
    # Try LaTeX rendering; fall back to plain text
    # ------------------------------------------------------------------
    use_tex = True
    try:
        with matplotlib.rc_context({"text.usetex": True}):
            fig_test = plt.figure()
            fig_test.text(0.5, 0.5, r"\textsc{test}")
            fig_test.savefig(OUT_DIR / ".latex_test.pdf")
            plt.close(fig_test)
            (OUT_DIR / ".latex_test.pdf").unlink(missing_ok=True)
    except Exception:
        use_tex = False
        print("  [info] LaTeX not available, using plain-text fallback")

    # Plain-text fallback names
    _plain = {
        "fista_oracle": "FISTA_S", "lista_oracle": "LISTA_S",
        "dl_fista": "DL-FISTA_U", "dl_lista": "DL-LISTA_U",
        "softplus_adam": "Softplus_U",
        "sae_relu": "SAE(ReLU)_U", "sae_topk": "SAE(TopK)_U",
        "sae_jumprelu": "SAE(JumpReLU)_U", "sae_MP": "SAE(MP)_U",
        "fista+relu": "FISTA+SAE(RL)_U", "fista+topk": "FISTA+SAE(TK)_U",
        "fista+jumprelu": "FISTA+SAE(JR)_U", "fista+MP": "FISTA+SAE(MP)_U",
        "refined_relu": "Refined(RL)_U", "refined_topk": "Refined(TK)_U",
        "refined_jumprelu": "Refined(JR)_U", "refined_MP": "Refined(MP)_U",
        "linear_probe": "Lin.Probe_S", "raw": "Raw_U", "pca": "PCA_U",
    }

    # ------------------------------------------------------------------
    # Layout constants  (figure-fraction coordinates via fig.text / fig axes)
    # ------------------------------------------------------------------
    # We use a single Axes whose data coords map 1:1 to our layout grid.
    # x in [0, 100],  y goes top-to-bottom (we set ylim high→low).

    line_w = 3.5            # width of line+marker sample
    name_x = 5.5            # method name x-offset from column origin
    desc_x = 27             # description x-offset from column origin
    row_h = 1.0             # vertical spacing unit
    cat_gap = 0.6           # extra gap between categories
    header_fs = 10          # header font size
    subtitle_fs = 7.5
    name_fs = 8.5
    desc_fs = 7.5

    # Column layout: [SC + SAE] | [Frozen + Refined]
    col_cats = [categories[:2], categories[2:]]
    col_x0 = [0, 48]       # left edge of each column

    # Pre-compute height of each column
    def _col_height(cats):
        h = 0
        for ci, (_, _, _, methods) in enumerate(cats):
            if ci > 0:
                h += cat_gap
            h += 1.6             # header + subtitle
            h += len(methods)    # one row per method
        return h

    col_heights = [_col_height(c) for c in col_cats]
    body_h = max(col_heights)
    # baselines: header + one row + footer = ~2.2
    total_h = body_h + cat_gap + 2.2
    fig_h = total_h * 0.30  # scale factor: data units → inches
    fig_h = max(fig_h, 2.8)

    with matplotlib.rc_context({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.size": 9,
    }):
        fig, ax = plt.subplots(figsize=(7.2, fig_h))
        ax.set_xlim(-2, 102)
        ax.set_ylim(total_h + 0.5, -0.5)   # y increases downward
        ax.axis("off")

        y_bottoms = []

        for col_idx, cats in enumerate(col_cats):
            cx = col_x0[col_idx]
            y = 0.0

            for ci, (cat_name, cat_key, cat_desc, methods) in enumerate(cats):
                if ci > 0:
                    y += cat_gap

                cat_col = CAT_COLORS[cat_key]

                # ---- Category header ----
                ax.text(cx + name_x, y, cat_name,
                        fontsize=header_fs, fontweight="bold",
                        color=cat_col, va="center")
                y += 0.7

                # ---- Subtitle ----
                if cat_desc:
                    desc_txt = cat_desc
                    if not use_tex:
                        desc_txt = (desc_txt.replace(r"$\ell_1$", "L1")
                                    .replace(r"$\hat A$", "A-hat")
                                    .replace("$A$", "A").replace("$Y$", "Y"))
                    ax.text(cx + name_x, y, desc_txt,
                            fontsize=subtitle_fs, color="#777777",
                            va="center", style="italic")
                    y += 0.9

                # ---- Method entries ----
                for method_key, tex_name, desc in methods:
                    style = METHOD_META.get(method_key, {})
                    color = style.get("color", cat_col)
                    ls = style.get("ls", "-")
                    marker = style.get("marker", "o")

                    # Line + marker sample
                    ax.plot([cx, cx + line_w], [y, y],
                            color=color, linestyle=ls, marker=marker,
                            markersize=5, linewidth=1.8, clip_on=False)

                    # Method name (coloured by category)
                    name = tex_name if use_tex else _plain.get(method_key, method_key)
                    ax.text(cx + name_x, y, name,
                            fontsize=name_fs, color=cat_col, va="center")

                    # Short description (gray)
                    if desc:
                        ax.text(cx + desc_x, y, desc,
                                fontsize=desc_fs, color="#999999", va="center")

                    y += row_h

            y_bottoms.append(y)

        # ------------------------------------------------------------------
        # Baselines — centred row below both columns
        # ------------------------------------------------------------------
        base_y = max(y_bottoms) + cat_gap
        base_col = CAT_COLORS["Baseline"]

        ax.text(50, base_y, "Baselines",
                fontsize=header_fs, fontweight="bold",
                color=base_col, va="center", ha="center")
        base_y += 0.9

        bx_starts = [8, 38, 68]
        for bi, (method_key, tex_name, desc) in enumerate(baselines):
            bx = bx_starts[bi]
            style = METHOD_META.get(method_key, {})
            color = style.get("color", base_col)
            ls = style.get("ls", ":")
            marker = style.get("marker", "*")

            ax.plot([bx, bx + line_w], [base_y, base_y],
                    color=color, linestyle=ls, marker=marker,
                    markersize=5, linewidth=1.8, clip_on=False)

            name = tex_name if use_tex else _plain.get(method_key, method_key)
            ax.text(bx + name_x, base_y, name,
                    fontsize=name_fs, color=base_col, va="center")

        base_y += 0.7
        foot = (r"$_{\mathrm{S}}$ = supervised (oracle);\quad "
                r"$_{\mathrm{U}}$ = unsupervised") if use_tex else \
               "S = supervised (oracle);  U = unsupervised"
        ax.text(50, base_y, foot,
                fontsize=desc_fs, color="#aaaaaa", va="center", ha="center")

    _save(fig, OUT_DIR / "method_legend.pdf")


# ============================================================================
# Top-level figure generators
# ============================================================================


def plot_sparsity():
    """Figure 1: Vary sparsity (ρ = k/d)."""
    print("\n=== vary_sparsity ===")
    results = _load_and_merge("exp_vary_sparsity.json", None, "k")
    if results is None:
        return
    results = [r for r in results if r["method"] != "linear_probe"]
    agg = _aggregate(results, "k")
    sweep_label = r"Sparsity $k$ ($d=1000$ fixed)"

    _plot_lines(agg, "k", sweep_label, METRICS_MAIN,
                nrows=1, ncols=2, figsize=(12, 4.5),
                save_path=OUT_DIR / "vary_sparsity_main.pdf",
                groups=CORE_GROUPS, show_legend=False)

    _plot_lines(agg, "k", sweep_label, METRICS_APX,
                nrows=2, ncols=3, figsize=(16, 9),
                save_path=OUT_DIR / "vary_sparsity_appendix.pdf",
                groups=CORE_GROUPS)


def plot_frozen():
    """Figure 2: Frozen decoder comparison (dumbbell plots)."""
    print("\n=== vary_frozen ===")
    results = _load_and_merge(
        "exp_frozen_decoder.json", "exp_frozen_decoder_10k.json", "num_latents",
    )
    if results is None:
        return
    agg = _aggregate(results, "num_latents")
    sweep_label = r"Number of latents ($d$)"

    # Main: 2×3 dumbbell chart (MCC ID + AUC OOD)
    _plot_frozen(agg, METRICS_MAIN, sweep_label,
                 OUT_DIR / "vary_frozen_main.pdf", show_legend=False)

    # Appendix: 2×3 line plot with all 6 metrics, all method groups
    _plot_lines(agg, "num_latents", sweep_label, METRICS_APX,
                nrows=2, ncols=3, figsize=(16, 9),
                save_path=OUT_DIR / "vary_frozen_appendix.pdf",
                groups=None)


def plot_samples():
    """Figure 3: Vary samples (d=100 only)."""
    print("\n=== vary_samples (d=100) ===")
    results = _load_and_merge("exp_vary_samples.json", None, "n_samples")
    if results is None:
        return
    agg = _aggregate(results, "n_samples")
    sweep_label = r"Number of training samples ($n$)"

    _plot_lines(agg, "n_samples", sweep_label, METRICS_MAIN,
                nrows=1, ncols=2, figsize=(12, 4.5),
                save_path=OUT_DIR / "vary_samples_main.pdf",
                groups=CORE_GROUPS, show_legend=False)

    _plot_lines(agg, "n_samples", sweep_label, METRICS_APX,
                nrows=2, ncols=3, figsize=(16, 9),
                save_path=OUT_DIR / "vary_samples_appendix.pdf",
                groups=CORE_GROUPS)


def plot_phase():
    """Figure 4: Phase transition (faceted)."""
    print("\n=== phase_faceted ===")
    results = _load_results(RESULTS_DIR / "exp_phase_transition.json")
    if results is None:
        return

    # Main: one PDF per main metric
    for metric_key, metric_label in METRICS_MAIN:
        tag = metric_key.replace("_iid", "_id")
        _plot_phase_faceted(results, metric_key, metric_label,
                            OUT_DIR / f"phase_faceted_{tag}.pdf",
                            show_legend=False)

    # Appendix: remaining metrics (skip duplicates already in main)
    main_keys = {k for k, _ in METRICS_MAIN}
    for metric_key, metric_label in METRICS_APX:
        if metric_key in main_keys:
            continue
        tag = metric_key.replace("_iid", "_id")
        _plot_phase_faceted(results, metric_key, metric_label,
                            OUT_DIR / f"phase_faceted_{tag}.pdf")


# ============================================================================
# V2 figures — improved legibility
# ============================================================================

# Core methods for reduced-clutter line plots.
# Includes aliases used in exp_phase_transition.json (fista_sup, fista_unsup).
CORE_METHODS = [
    "fista_oracle", "fista_sup",       # same method, different naming conventions
    "dl_fista", "fista_unsup",
    "sae_topk", "sae_relu",
    "linear_probe",
]

SAE_TYPES_V2 = ["relu", "topk", "jumprelu", "MP"]
SAE_LABELS_V2 = {"relu": "ReLU", "topk": "TopK", "jumprelu": "JumpReLU", "MP": "MP"}

# Per-SAE-type colours for faceted plots
SAE_TYPE_COLORS = {
    "relu": "#c0392b", "topk": "#2980b9",
    "jumprelu": "#27ae60", "MP": "#8e44ad",
}


def _plot_lines_v2(agg, sweep_key, sweep_label, metric_key, metric_label,
                   methods, figsize, save_path, caption=""):
    """Clean single-panel line plot with in-figure legend. Max ~5 methods."""
    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        if method not in agg:
            continue
        style = METHOD_META.get(method, {})
        data = agg[method]
        x = data["x"]
        y_mean = data[f"{metric_key}_mean"]
        y_std = data[f"{metric_key}_std"]

        ax.plot(x, y_mean,
                label=style.get("label", method),
                color=style.get("color"),
                linestyle=style.get("ls", "-"),
                marker=style.get("marker", "o"),
                markersize=9, linewidth=2.8)
        ax.fill_between(x,
                        np.array(y_mean) - np.array(y_std),
                        np.array(y_mean) + np.array(y_std),
                        color=style.get("color"), alpha=0.10)

    ax.set_xlabel(sweep_label)
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if "mcc" in metric_key or "auc" in metric_key:
        ax.set_ylim(-0.05, 1.05)
    if sweep_key in ("n_samples", "num_latents"):
        ax.set_xscale("log")

    ax.legend(fontsize=12, frameon=True, edgecolor="black", framealpha=0.95,
              loc="best")

    if caption:
        fig.text(0.5, -0.02, caption, ha="center", fontsize=11,
                 style="italic", color="#444444", wrap=True)

    plt.tight_layout()
    _save(fig, save_path)


def _plot_frozen_faceted_v2(agg, metric_key, metric_label, sweep_key,
                            sweep_label, save_path, caption=""):
    """Faceted frozen decoder plot: one panel per SAE type, 3 lines each."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharey=True)

    for ax_idx, sae_type in enumerate(SAE_TYPES_V2):
        ax = axes[ax_idx]

        # Distinct colours per method — panel title identifies SAE type
        line_defs = [
            (f"sae_{sae_type}",     "#d94801", "--", "o", "SAE encoder"),       # orange
            (f"fista+{sae_type}",   "#2171b5", "-",  "s", "FISTA (frozen dec)"),# blue
            (f"refined_{sae_type}", "#238b45", "-",  "^", "Refined (warm)"),    # green
            ("dl_fista",            "#7b4ea3", "-",  "D", "DL-FISTA"),          # purple
        ]

        for method, color, ls, marker, label in line_defs:
            if method not in agg:
                continue
            data = agg[method]
            x = data["x"]
            y_mean = data[f"{metric_key}_mean"]
            y_std = data[f"{metric_key}_std"]

            # Darken/lighten slightly for visual separation
            ax.plot(x, y_mean, color=color, linestyle=ls, marker=marker,
                    markersize=8, linewidth=2.5,
                    label=label if ax_idx == 0 else None)
            ax.fill_between(x,
                            np.array(y_mean) - np.array(y_std),
                            np.array(y_mean) + np.array(y_std),
                            color=color, alpha=0.08)

        # Oracle ceiling
        if "fista_oracle" in agg:
            data = agg["fista_oracle"]
            x = data["x"]
            y_mean = data[f"{metric_key}_mean"]
            ax.plot(x, y_mean, color="#333333", ls="--", lw=1.5,
                    label="FISTA (oracle)" if ax_idx == 0 else None, zorder=1)

        ax.set_title(SAE_LABELS_V2[sae_type], fontsize=15, fontweight="bold")
        ax.set_xlabel(sweep_label)
        if ax_idx == 0:
            ax.set_ylabel(metric_label)
        if sweep_key in ("n_samples", "num_latents"):
            ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(handles), frameon=True, edgecolor="black",
               framealpha=0.95, fontsize=12,
               bbox_to_anchor=(0.5, -0.08))

    if caption:
        fig.text(0.5, -0.14, caption, ha="center", fontsize=11,
                 style="italic", color="#444444", wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    _save(fig, save_path)


def _plot_phase_v2(results, metric_key, metric_label, save_path,
                   methods=None, caption=""):
    """Phase transition with reduced methods and in-figure legend."""
    if methods is None:
        methods = CORE_METHODS

    for row in results:
        if "num_latents" not in row and "n" in row:
            row["num_latents"] = row["n"]
        if "input_dim" not in row and "m" in row:
            row["input_dim"] = row["m"]
        if "delta" not in row:
            row["delta"] = row["input_dim"] / row["num_latents"]

    all_nk = sorted(set((r["num_latents"], r["k"]) for r in results))
    all_n = sorted(set(n for n, _ in all_nk))
    all_k = sorted(set(k for _, k in all_nk))
    corner_n = {all_n[0], all_n[-1]}
    corner_k = {all_k[0], all_k[-1]}
    nk_pairs = [(n, k) for n, k in all_nk if n in corner_n and k in corner_k]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)

    for idx, (n, k) in enumerate(nk_pairs):
        ax = axes[idx // 2][idx % 2]
        panel_rows = [r for r in results if r["num_latents"] == n and r["k"] == k]

        for method in methods:
            mrows = [r for r in panel_rows if r["method"] == method]
            if not mrows:
                continue
            delta_vals = sorted(set(r["delta"] for r in mrows))
            means = [
                np.mean([r.get(metric_key, 0) for r in mrows
                         if abs(r["delta"] - dv) < 1e-6])
                for dv in delta_vals
            ]
            style = METHOD_META.get(method, {})
            ax.plot(delta_vals, means,
                    marker=style.get("marker", "o"), markersize=9,
                    color=style.get("color"), linestyle=style.get("ls", "-"),
                    linewidth=2.8, label=style.get("label", method))

        ax.set_xlabel(r"$\delta = d_y / d_z$")
        ax.set_ylabel(metric_label)
        ax.set_title(f"$d_z = {n},\\; k = {k}$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 5), fontsize=12,
               bbox_to_anchor=(0.5, -0.04),
               edgecolor="black", framealpha=0.95)

    if caption:
        fig.text(0.5, -0.08, caption, ha="center", fontsize=11,
                 style="italic", color="#444444", wrap=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save(fig, save_path)


def plot_v2():
    """Generate all v2 (improved legibility) figures."""
    print("\n=== V2 figures ===")
    v2_dir = OUT_DIR / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 5: Phase transition (reduced methods) ---
    print("\n--- phase (v2) ---")
    results = _load_results(RESULTS_DIR / "exp_phase_transition.json")
    if results:
        _plot_phase_v2(
            results, "mcc_iid", "MCC (ID)",
            v2_dir / "phase_mcc_id.pdf",
            caption="Per-sample methods exhibit a sharp phase transition; "
                    "SAEs plateau at 0.2\u20130.5 MCC regardless of \u03b4.",
        )
        _plot_phase_v2(
            results, "auc_ood", "AUC (OOD)",
            v2_dir / "phase_auc_ood.pdf",
            caption="SAE OOD AUC degrades toward chance while "
                    "per-sample methods maintain near-perfect performance.",
        )

    # --- Figure 6a: Vary latents (core methods only) ---
    print("\n--- vary_latents (v2) ---")
    results = _load_and_merge(
        "exp_frozen_decoder.json", "exp_frozen_decoder_10k.json", "num_latents",
    )
    if results:
        agg = _aggregate(results, "num_latents")

        _plot_lines_v2(
            agg, "num_latents",
            r"Number of latents ($d_z$)",
            "mcc_iid", "MCC (ID)",
            CORE_METHODS, figsize=(7, 4.5),
            save_path=v2_dir / "vary_latents_mcc_id.pdf",
            caption="Scaling latent dimension does not close the compositional gap.",
        )
        _plot_lines_v2(
            agg, "num_latents",
            r"Number of latents ($d_z$)",
            "auc_ood", "AUC (OOD)",
            CORE_METHODS, figsize=(7, 4.5),
            save_path=v2_dir / "vary_latents_auc_ood.pdf",
            caption="SAEs exhibit a persistent ID\u2013OOD gap; "
                    "FISTA maintains consistent performance.",
        )

    # --- Vary latents: acc_ood (probe on codes, fair comparison) ---
    results_latents = _load_and_merge(
        "exp_vary_latents.json", None, "num_latents",
    )
    if results_latents:
        agg_latents = _aggregate(results_latents, "num_latents")
        ACC_METHODS = [
            "fista_oracle", "dl_fista",
            "sae_topk", "sae_relu", "sae_jumprelu", "sae_MP",
            "linear_probe",
        ]
        _plot_lines_v2(
            agg_latents, "num_latents",
            r"Number of latents ($d_z$)",
            "acc_ood", "Accuracy (OOD)",
            ACC_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_latents_acc_ood.pdf",
            caption="Same downstream probe on each method\u2019s codes: "
                    "FISTA (oracle) dominates; SAEs match or trail linear probes.",
        )

    if results:
        agg = _aggregate(results, "num_latents")

        # --- Figure 6b: Frozen decoder faceted by SAE type ---
        print("\n--- frozen_faceted (v2) ---")
        _plot_frozen_faceted_v2(
            agg, "mcc_iid", "MCC (ID)", "num_latents",
            r"Number of latents ($d_z$)",
            v2_dir / "frozen_faceted_mcc_id.pdf",
            caption="Swapping the SAE encoder for FISTA on the same frozen dictionary "
                    "does not close the gap to DL-FISTA: the SAE-learned dictionary "
                    "is the bottleneck on ID MCC.",
        )
        _plot_frozen_faceted_v2(
            agg, "auc_ood", "AUC (OOD)", "num_latents",
            r"Number of latents ($d_z$)",
            v2_dir / "frozen_faceted_auc_ood.pdf",
            caption="OOD AUC gains from swapping inference are largest "
                    "for JumpReLU, confirming its decoder learns a "
                    "comparatively better dictionary.",
        )
        _plot_frozen_faceted_v2(
            agg, "acc_ood", "Accuracy (OOD)", "num_latents",
            r"Number of latents ($d_z$)",
            v2_dir / "frozen_faceted_acc_ood.pdf",
            caption="Swapping the encoder for FISTA on the same frozen dictionary "
                    "does not improve OOD accuracy: the dictionary is the bottleneck.",
        )

    # --- Figure 7: Vary samples (all 4 SAEs — text discusses each individually) ---
    SAMPLES_METHODS = [
        "fista_oracle", "dl_fista",
        "sae_topk", "sae_relu", "sae_jumprelu", "sae_MP",
        "linear_probe",
    ]
    print("\n--- vary_samples (v2) ---")
    results = _load_and_merge("exp_vary_samples.json", None, "n_samples")
    if results:
        agg = _aggregate(results, "n_samples")

        _plot_lines_v2(
            agg, "n_samples",
            r"Number of training samples ($p$)",
            "mcc_iid", "MCC (ID)",
            SAMPLES_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_samples_mcc_id.pdf",
            caption="More data closes the dictionary learning gap "
                    "(DL-FISTA improves) but not the amortisation gap "
                    "(SAEs plateau or degrade).",
        )
        _plot_lines_v2(
            agg, "n_samples",
            r"Number of training samples ($p$)",
            "auc_ood", "AUC (OOD)",
            SAMPLES_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_samples_auc_ood.pdf",
            caption="Additional training data benefits dictionary quality "
                    "but does not close the amortisation gap.",
        )
        _plot_lines_v2(
            agg, "n_samples",
            r"Number of training samples ($p$)",
            "acc_ood", "Accuracy (OOD)",
            SAMPLES_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_samples_acc_ood.pdf",
            caption="Same downstream probe on each method\u2019s codes: "
                    "DL-FISTA matches the oracle once data suffices; "
                    "SAEs trail linear probes.",
        )

    # --- Sparsity (all methods) ---
    SPARSITY_METHODS = [
        "fista_oracle", "dl_fista",
        "sae_topk", "sae_relu", "sae_jumprelu", "sae_MP",
        "softplus_adam", "linear_probe",
    ]
    print("\n--- vary_sparsity (v2) ---")
    results = _load_and_merge("exp_vary_sparsity.json", None, "k")
    if results:
        agg = _aggregate(results, "k")

        _plot_lines_v2(
            agg, "k",
            r"Sparsity $k$ ($d_z=1000$ fixed)",
            "mcc_iid", "MCC (ID)",
            SPARSITY_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_sparsity_mcc_id.pdf",
            caption="Per-sample methods degrade gracefully with sparsity; "
                    "SAEs plateau regardless of $k$.",
        )
        _plot_lines_v2(
            agg, "k",
            r"Sparsity $k$ ($d_z=1000$ fixed)",
            "auc_ood", "AUC (OOD)",
            SPARSITY_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_sparsity_auc_ood.pdf",
            caption="OOD generalisation degrades with sparsity for SAEs; "
                    "per-sample methods remain robust.",
        )
        _plot_lines_v2(
            agg, "k",
            r"Sparsity $k$ ($d_z=1000$ fixed)",
            "acc_ood", "Accuracy (OOD)",
            SPARSITY_METHODS, figsize=(8, 5),
            save_path=v2_dir / "vary_sparsity_acc_ood.pdf",
            caption="Same downstream probe on each method\u2019s codes: "
                    "FISTA (oracle) dominates; SAE codes offer no advantage "
                    "over raw activations.",
        )

    print(f"\nV2 figures saved to {v2_dir}/")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["sparsity", "frozen", "samples", "phase", "legend", "v2"],
        help="Generate only this figure set.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = {
        "legend": plot_legend,
        "sparsity": plot_sparsity,
        "frozen": plot_frozen,
        "samples": plot_samples,
        "phase": plot_phase,
        "v2": plot_v2,
    }

    if args.only:
        generators[args.only]()
    else:
        for gen in generators.values():
            gen()

    print(f"\nAll paper figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
