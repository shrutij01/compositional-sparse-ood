"""
Experiment: Support Recovery.

Question: Does the SAE encoder identify the correct active features, even if
the magnitudes are wrong?

The SAE encoder produces codes in a single forward pass (amortized inference).
These codes might be inaccurate in two distinct ways:

  1. Wrong support: the encoder activates the wrong set of features.
  2. Wrong magnitudes: the encoder activates the right features but with
     incorrect values.

This experiment separates the two by:

  - Measuring support precision/recall/F1 against ground-truth Z (after
    Hungarian-matching SAE features to ground-truth columns).
  - Testing an intervention: keep the SAE's binary support, but re-estimate
    magnitudes via least-squares on the active dictionary columns.

If support is mostly correct but magnitudes are off, support_lstsq should
substantially improve over raw SAE codes.  If support itself is wrong,
re-estimating magnitudes won't help.

For each (num_latents, sae_type, seed), compares:

  sae_{type}             -- raw SAE encoder codes
  support_lstsq_{type}   -- SAE support + least-squares magnitudes
  fista+{type}           -- FISTA cold-start with frozen SAE decoder
  fista_oracle           -- FISTA with ground-truth A (ceiling)

Also reports per-SAE-type diagnostics:
  - Support precision, recall, F1
  - Ground-truth L0 vs predicted L0

Sweeps num_latents while holding k fixed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from utils.metrics import match_columns, compute_support_metrics, reestimate_magnitudes
from experiments._common import (
    eval_and_tag,
    train_all_saes,
    save_incremental,
)
from experiments.param_check import get_frozen_decoder_configs


def run_support_sweep(trained_saes, data, A, fista_lam, fista_iters, device, tag):
    """For each SAE type, measure support recovery and test magnitude re-estimation."""
    from models.sparse_coding import fista

    X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
    X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

    results = []
    diagnostics = []

    for sae_type, info in trained_saes.items():
        sae = info["model"]
        codes_iid = info["codes_iid"]
        codes_ood = info["codes_ood"]

        D = sae.Ad.data.detach().cpu().numpy()
        bd = sae.bd.data.detach()
        X_iid_c = (X_iid - bd).cpu().numpy()
        X_ood_c = (X_ood - bd).cpu().numpy()

        # --- Match SAE columns to ground-truth ---
        match = match_columns(D, A)
        row_ind, col_ind = match["row_ind"], match["col_ind"]

        # --- Support diagnostics ---
        Z_true_iid = data["Z_val"]
        Z_true_ood = data["Z_ood"]

        supp_iid = compute_support_metrics(Z_true_iid, codes_iid, row_ind, col_ind)
        supp_ood = compute_support_metrics(Z_true_ood, codes_ood, row_ind, col_ind)

        diag = {
            "sae_type": sae_type,
            "precision_iid": supp_iid["precision"],
            "recall_iid": supp_iid["recall"],
            "f1_iid": supp_iid["f1"],
            "gt_l0_iid": supp_iid["gt_l0"],
            "pred_l0_iid": supp_iid["pred_l0"],
            "pred_total_l0_iid": supp_iid["pred_total_l0"],
            "precision_ood": supp_ood["precision"],
            "recall_ood": supp_ood["recall"],
            "f1_ood": supp_ood["f1"],
            "gt_l0_ood": supp_ood["gt_l0"],
            "pred_l0_ood": supp_ood["pred_l0"],
            "pred_total_l0_ood": supp_ood["pred_total_l0"],
            **tag,
        }
        diagnostics.append(diag)
        print(f"  {sae_type} IID: prec={supp_iid['precision']:.3f}  "
              f"rec={supp_iid['recall']:.3f}  f1={supp_iid['f1']:.3f}  "
              f"l0={supp_iid['pred_total_l0']:.1f}/{supp_iid['gt_l0']:.1f}")
        print(f"  {sae_type} OOD: prec={supp_ood['precision']:.3f}  "
              f"rec={supp_ood['recall']:.3f}  f1={supp_ood['f1']:.3f}  "
              f"l0={supp_ood['pred_total_l0']:.1f}/{supp_ood['gt_l0']:.1f}")

        # --- 1) Raw SAE codes (baseline) ---
        results.append(eval_and_tag(
            codes_iid, codes_ood, data, f"sae_{sae_type}", **tag,
        ))

        # --- 2) SAE support + least-squares magnitude re-estimation ---
        support_iid = np.abs(codes_iid) > 1e-4
        support_ood = np.abs(codes_ood) > 1e-4

        codes_lstsq_iid = reestimate_magnitudes(X_iid_c, D, support_iid, nonneg=True)
        codes_lstsq_ood = reestimate_magnitudes(X_ood_c, D, support_ood, nonneg=True)

        results.append(eval_and_tag(
            codes_lstsq_iid, codes_lstsq_ood,
            data, f"support_lstsq_{sae_type}", **tag,
        ))

        # --- 3) FISTA cold-start with frozen decoder ---
        D_t = torch.tensor(D, dtype=torch.float32, device=device)
        X_iid_ct = X_iid - bd
        X_ood_ct = X_ood - bd
        with torch.no_grad():
            z_fista_iid = fista(X_iid_ct, D_t, fista_lam, n_iter=fista_iters, nonneg=True)
            z_fista_ood = fista(X_ood_ct, D_t, fista_lam, n_iter=fista_iters, nonneg=True)
        results.append(eval_and_tag(
            z_fista_iid.cpu().numpy(), z_fista_ood.cpu().numpy(),
            data, f"fista+{sae_type}", **tag,
        ))

    return results, diagnostics


def run(
    epochs=500,
    gamma_reg=1e-4,
    fista_lam=0.1,
    fista_iters=100,
    seeds=(0, 1, 2),
    min_n=0,
    max_n=float("inf"),
    out_suffix="",
):
    configs = get_frozen_decoder_configs()
    configs = [c for c in configs if min_n <= c["num_latents"] <= max_n]

    if not configs:
        print(f"No configs match min_n={min_n}, max_n={max_n}")
        return []

    all_results = []
    all_diagnostics = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_support_recovery{out_suffix}.json"
    diag_path = out_dir / f"exp_support_recovery_diagnostics{out_suffix}.json"

    from models.sparse_coding import SparseCodingConfig, train_sparse_coding

    for cfg in configs:
        num_latents, k, input_dim = cfg["num_latents"], cfg["k"], cfg["input_dim"]
        n_samples = cfg["n_samples"]
        width = cfg["width"]

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  num_latents={num_latents}, k={k}, input_dim={input_dim}, seed={seed}")
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

            # --- Train SAEs ---
            trained_saes = train_all_saes(
                data, input_dim_actual, width, k, num_latents, n_samples,
                epochs, gamma_reg, seed, device,
            )

            # --- Support recovery analysis ---
            results, diagnostics = run_support_sweep(
                trained_saes, data, A, fista_lam, fista_iters, device, tag,
            )
            all_results.extend(results)
            all_diagnostics.extend(diagnostics)

            # --- FISTA oracle (ceiling) ---
            X_iid = torch.tensor(data["Y_val"], dtype=torch.float32, device=device)
            X_ood = torch.tensor(data["Y_ood"], dtype=torch.float32, device=device)

            print("  FISTA oracle...")
            sc_cfg = SparseCodingConfig(
                input_dim=input_dim_actual, num_latents=num_latents,
                method="fista", lam=fista_lam,
                max_steps=1, n_iter=fista_iters, dict_update_every=1,
                supervised=True, seed=seed, print_every=1,
            )
            sc_out = train_sparse_coding(X_iid, X_ood, sc_cfg, A=A, device=device)
            all_results.append(eval_and_tag(
                sc_out["codes_iid"], sc_out["codes_ood"],
                data, "fista_oracle", **tag,
            ))

            save_incremental(all_results, out_path)
            save_incremental(all_diagnostics, diag_path)

    print(f"\nResults saved to {out_path}")
    print(f"Diagnostics saved to {diag_path}")

    # --- Summary ---
    print("\n--- Support Recovery Summary (mean over seeds) ---")
    sae_types = sorted(set(d["sae_type"] for d in all_diagnostics))
    nvals = sorted(set(d["num_latents"] for d in all_diagnostics))
    print("%6s %10s %7s %7s %7s %6s %6s" % (
        "n", "sae_type", "prec", "recall", "f1", "gt_l0", "pred_l0"))
    for n in nvals:
        for st in sae_types:
            rows = [d for d in all_diagnostics if d["num_latents"] == n and d["sae_type"] == st]
            if rows:
                print("%6d %10s %7.3f %7.3f %7.3f %6.1f %6.1f" % (
                    n, st,
                    np.mean([r["precision_ood"] for r in rows]),
                    np.mean([r["recall_ood"] for r in rows]),
                    np.mean([r["f1_ood"] for r in rows]),
                    np.mean([r["gt_l0_ood"] for r in rows]),
                    np.mean([r["pred_total_l0_ood"] for r in rows]),
                ))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output filename, e.g. '_small'")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, out_suffix=args.out_suffix)
