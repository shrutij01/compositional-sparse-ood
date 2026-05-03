"""
Aggregate per-dataset eval results into a single SAEBench-format JSON per seed.

Reads:  OUTPUT_BASE/seed{N}/per_dataset/*.json  (one file per dataset)
Writes: OUTPUT_BASE/seed{N}/sparse_probing/sae_bench_<...>_eval_results.json
        OUTPUT_BASE/summary.json (cross-seed comparison)

Usage:
    python sae_bench_adapter/merge_per_dataset_results.py \
        --output_base /network/scratch/.../saebench_sparse_probing \
        --seeds 0 1 2
"""
import argparse
import json
import statistics
from pathlib import Path

EXPECTED_DATASETS = [
    "LabHC/bias_in_bios_class_set1",
    "LabHC/bias_in_bios_class_set2",
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "canrager/amazon_reviews_mcauley_1and5_sentiment",
    "codeparrot/github-code",
    "fancyzhx/ag_news",
    "Helsinki-NLP/europarl",
]


def safe(name: str) -> str:
    return name.replace("/", "_")


def load_per_dataset(seed_dir: Path) -> dict:
    """Returns {dataset_name: {dataset_results: ..., per_class_dict: ...}}."""
    pd_dir = seed_dir / "per_dataset"
    out = {}
    if not pd_dir.exists():
        return out
    for ds in EXPECTED_DATASETS:
        f = pd_dir / f"{safe(ds)}.json"
        if f.exists():
            out[ds] = json.load(open(f))
    return out


def aggregate_seed(per_ds: dict) -> dict:
    """Compute mean accuracy across datasets for each metric (sae_test_accuracy etc)."""
    if not per_ds:
        return {}
    keys = list(next(iter(per_ds.values()))["dataset_results"].keys())
    agg = {}
    for k in keys:
        vals = [d["dataset_results"].get(k) for d in per_ds.values() if d["dataset_results"].get(k) is not None]
        agg[k] = statistics.mean(vals) if vals else float("nan")
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_base", type=str, required=True,
                   help="Parent directory containing seed{N}/ subdirs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    base = Path(args.output_base)
    summary = {"per_seed": {}, "per_dataset_per_seed": {}}

    for s in args.seeds:
        seed_dir = base / f"seed{s}"
        per_ds = load_per_dataset(seed_dir)
        n_done = len(per_ds)
        n_expected = len(EXPECTED_DATASETS)
        missing = [d for d in EXPECTED_DATASETS if d not in per_ds]
        agg = aggregate_seed(per_ds)
        summary["per_seed"][s] = {
            "n_done": n_done,
            "n_expected": n_expected,
            "complete": n_done == n_expected,
            "missing": missing,
            "aggregate_metrics": agg,
        }
        for ds_name, d in per_ds.items():
            summary["per_dataset_per_seed"].setdefault(ds_name, {})[s] = d["dataset_results"]
        print(f"\n=== seed {s}: {n_done}/{n_expected} datasets ===")
        if missing:
            print(f"  missing: {missing}")
        for k in ("sae_test_accuracy", "llm_test_accuracy"):
            if k in agg:
                print(f"  mean {k}: {agg[k]:.4f}")

    # Cross-seed mean for sae_test_accuracy and llm_test_accuracy
    print("\n=== Cross-seed comparison (mean ± std across seeds) ===")
    for k in ("sae_test_accuracy", "llm_test_accuracy"):
        per_seed_vals = [summary["per_seed"][s]["aggregate_metrics"].get(k)
                         for s in args.seeds
                         if k in summary["per_seed"][s]["aggregate_metrics"]]
        per_seed_vals = [v for v in per_seed_vals if v is not None]
        if len(per_seed_vals) >= 2:
            print(f"  {k}: {statistics.mean(per_seed_vals):.4f} ± {statistics.stdev(per_seed_vals):.4f}")
        elif per_seed_vals:
            print(f"  {k}: {per_seed_vals[0]:.4f} (only 1 seed)")

    # Per-dataset cross-seed table
    print("\n=== Per-dataset (mean ± std across seeds for sae_test_accuracy) ===")
    for ds in EXPECTED_DATASETS:
        vals = []
        for s in args.seeds:
            r = summary["per_dataset_per_seed"].get(ds, {}).get(s, {})
            if r and "sae_test_accuracy" in r:
                vals.append(r["sae_test_accuracy"])
        if len(vals) >= 2:
            print(f"  {ds:55s} {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}  (n={len(vals)})")
        elif vals:
            print(f"  {ds:55s} {vals[0]:.4f}  (n=1)")
        else:
            print(f"  {ds:55s} (no data)")

    out_path = base / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, default=str, indent=2)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
