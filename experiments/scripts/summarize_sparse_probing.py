#!/usr/bin/env python3
"""
Summarise sparse-probing results across seeds (42, 1, 2) for:
  - dl_fista   (custom SAE trained with our DL-FISTA method)
  - std_sae    (pre-trained standard SAEBench SAE, trainer_0 res-4k, layer 4)
  - linear     (linear probe on full LLM activations, same for all SAE types)

Usage (run from project root after all SLURM jobs finish):
    python experiments/scripts/summarize_sparse_probing.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEEDS = [42, 1, 2]

# --------------------------------------------------------------------------
# File locations
# --------------------------------------------------------------------------
# dl_fista: seed 42 lives in the original dir; seeds 1/2 in seed-specific dirs
def dl_fista_path(seed: int) -> Path:
    if seed == 42:
        d = ROOT / "results/saebench_sparse_probing"
    else:
        d = ROOT / f"results/saebench_sparse_probing_seed{seed}"
    return d / "sparse_probing/dl_fista_layer4_lam0.1_custom_sae_eval_results.json"


# standard SAE: sae_bench_pythia70m_sweep_standard_ctx128_0712 trainer_0
STD_FILENAME = (
    "sae_bench_pythia70m_sweep_standard_ctx128_0712"
    "_blocks.4.hook_resid_post__trainer_0_eval_results.json"
)

def std_sae_path(seed: int) -> Path:
    return ROOT / f"results/saebench_sparse_probing_std/seed{seed}" / STD_FILENAME


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    m = data["eval_result_metrics"]
    return {
        "sae_test_accuracy":  m["sae"]["sae_test_accuracy"],
        "llm_test_accuracy":  m["llm"]["llm_test_accuracy"],
    }


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    if n == 1:
        return mu, float("nan")
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mu, se


def fmt(mu: float, se: float) -> str:
    if math.isnan(mu):
        return "  N/A  "
    if math.isnan(se):
        return f"{mu:.4f} (SE N/A)"
    return f"{mu:.4f} ± {se:.4f}"


# --------------------------------------------------------------------------
# Collect per-seed results
# --------------------------------------------------------------------------
rows = {}   # method -> seed -> {sae, llm}

for seed in SEEDS:
    # dl_fista
    p = dl_fista_path(seed)
    m = load_metrics(p)
    rows.setdefault("dl_fista", {})[seed] = m
    if not m:
        print(f"[WARN] dl_fista seed={seed} not found: {p}")

    # standard SAE
    p = std_sae_path(seed)
    m = load_metrics(p)
    rows.setdefault("std_sae", {})[seed] = m
    if not m:
        print(f"[WARN] std_sae seed={seed} not found: {p}")

# --------------------------------------------------------------------------
# Print per-seed table
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SPARSE PROBING — per-seed sae_test_accuracy")
print("=" * 72)
header = f"{'Method':<12}  " + "  ".join(f"seed={s}" for s in SEEDS)
print(header)
print("-" * 72)

for method in ["dl_fista", "std_sae"]:
    vals = [rows[method].get(s, {}).get("sae_test_accuracy", float("nan")) for s in SEEDS]
    row = f"{method:<12}  " + "  ".join(f"{v:.4f}" if not math.isnan(v) else " N/A " for v in vals)
    print(row)

# linear probe (llm_test_accuracy from dl_fista; same LLM across all methods)
print("-" * 72)
for method in ["dl_fista", "std_sae"]:
    vals = [rows[method].get(s, {}).get("llm_test_accuracy", float("nan")) for s in SEEDS]
    label = f"linear ({method})"
    row = f"{label:<20}  " + "  ".join(f"{v:.4f}" if not math.isnan(v) else " N/A " for v in vals)
    print(row)

# --------------------------------------------------------------------------
# Mean ± SE table
# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SPARSE PROBING — mean ± SE across seeds (42, 1, 2)")
print("=" * 72)
print(f"{'Metric':<30}  {'dl_fista':<22}  {'std_sae':<22}  {'linear_probe':<22}")
print("-" * 72)

for metric_key, label in [("sae_test_accuracy", "SAE test accuracy"),
                           ("llm_test_accuracy", "Linear probe accuracy")]:
    row_parts = [f"{label:<30}"]
    for method in ["dl_fista", "std_sae"]:
        vals = [rows[method].get(s, {}).get(metric_key, float("nan")) for s in SEEDS]
        vals = [v for v in vals if not math.isnan(v)]
        mu, se = mean_se(vals)
        row_parts.append(f"{fmt(mu, se):<22}")
    # linear probe is the same across both SAE types; use dl_fista source
    if metric_key == "llm_test_accuracy":
        vals_dl = [rows["dl_fista"].get(s, {}).get("llm_test_accuracy", float("nan")) for s in SEEDS]
        vals_std = [rows["std_sae"].get(s, {}).get("llm_test_accuracy", float("nan")) for s in SEEDS]
        # merge non-nan values (they should be close; report dl_fista average)
        vals = [v for v in vals_dl if not math.isnan(v)]
        mu, se = mean_se(vals)
        row_parts.append(f"{fmt(mu, se):<22}")
    else:
        row_parts.append(f"{'(same as above)':<22}")
    print("  ".join(row_parts))

print("=" * 72)

# --------------------------------------------------------------------------
# Save to file
# --------------------------------------------------------------------------
output_lines = []
output_lines.append("SPARSE PROBING COMPARISON — pythia-70m-deduped, layer 4, seeds 42/1/2")
output_lines.append("")
output_lines.append("Methods:")
output_lines.append("  dl_fista : DL-FISTA sparse coding (d_sae=2048, lam=0.1)")
output_lines.append("  std_sae  : SAEBench standard sweep trainer_0 (res-4k ~4096 features)")
output_lines.append("  linear   : linear probe on full LLM residual stream")
output_lines.append("")

output_lines.append("Per-seed sae_test_accuracy:")
header = f"  {'Method':<12}  " + "  ".join(f"seed={s}" for s in SEEDS)
output_lines.append(header)
for method in ["dl_fista", "std_sae"]:
    vals = [rows[method].get(s, {}).get("sae_test_accuracy", float("nan")) for s in SEEDS]
    row = f"  {method:<12}  " + "  ".join(f"{v:.4f}" if not math.isnan(v) else " N/A " for v in vals)
    output_lines.append(row)
output_lines.append("")

output_lines.append("Per-seed llm_test_accuracy (linear probe):")
for method in ["dl_fista", "std_sae"]:
    vals = [rows[method].get(s, {}).get("llm_test_accuracy", float("nan")) for s in SEEDS]
    label = f"  linear ({method})"
    row = f"{label:<22}  " + "  ".join(f"{v:.4f}" if not math.isnan(v) else " N/A " for v in vals)
    output_lines.append(row)
output_lines.append("")

output_lines.append("Mean ± SE across seeds:")
output_lines.append(f"  {'Metric':<30}  {'dl_fista':<26}  {'std_sae':<26}  {'linear_probe':<26}")
for metric_key, label in [("sae_test_accuracy", "SAE test accuracy"),
                           ("llm_test_accuracy", "Linear probe accuracy")]:
    row_parts = [f"  {label:<30}"]
    for method in ["dl_fista", "std_sae"]:
        vals = [rows[method].get(s, {}).get(metric_key, float("nan")) for s in SEEDS]
        vals = [v for v in vals if not math.isnan(v)]
        mu, se = mean_se(vals)
        row_parts.append(f"{fmt(mu, se):<26}")
    if metric_key == "llm_test_accuracy":
        vals = [rows["dl_fista"].get(s, {}).get("llm_test_accuracy", float("nan")) for s in SEEDS]
        vals = [v for v in vals if not math.isnan(v)]
        mu, se = mean_se(vals)
        row_parts.append(f"{fmt(mu, se):<26}")
    else:
        row_parts.append(f"{'(same as above)':<26}")
    output_lines.append("  ".join(row_parts))

out_path = ROOT / "results/sparse_probing_comparison.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(output_lines) + "\n")
print(f"\nSaved to {out_path}")

out_json = ROOT / "results/sparse_probing_comparison.json"
summary = {}
for method in ["dl_fista", "std_sae"]:
    summary[method] = {}
    for seed in SEEDS:
        summary[method][str(seed)] = rows[method].get(seed, {})
    for metric_key in ["sae_test_accuracy", "llm_test_accuracy"]:
        vals = [rows[method].get(s, {}).get(metric_key, float("nan")) for s in SEEDS]
        vals_clean = [v for v in vals if not math.isnan(v)]
        mu, se = mean_se(vals_clean)
        summary[method][f"{metric_key}_mean"] = mu
        summary[method][f"{metric_key}_se"] = se
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved to {out_json}")
