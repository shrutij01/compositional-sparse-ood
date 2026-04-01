"""
Experiment: Dictionary Learning Dynamics.

Question: Why does DL-FISTA learn a better dictionary than SAE training?
Do SAE dictionaries start good and drift, or never converge to the right
directions?

Tracks dictionary quality (cosine similarity to ground truth) during training
for both SAE and DL-FISTA, reporting a per-epoch/per-round learning curve.
This reveals whether the SAE dictionary:
  (a) converges to a good solution then overfits
  (b) converges slowly to a bad local minimum
  (c) never makes progress toward the correct directions

For each (num_latents, sae_type, seed), reports:
  - SAE decoder cosine vs epoch (sampled every N epochs)
  - DL-FISTA dictionary cosine vs outer round

Sweeps num_latents while holding k fixed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data import generate_datasets
from utils.metrics import match_columns
from experiments._common import SAE_TYPES, save_incremental, _lazy_torch_imports
from experiments.param_check import get_frozen_decoder_configs


def track_sae_dynamics(data, A, input_dim, width, k, num_latents, n_samples,
                       epochs, gamma_reg, seed, device, eval_every=10):
    """Train each SAE type and record dictionary cosine at regular intervals."""
    torch_mod, DataLoader, TensorDataset, SAE, SAEConfig, train_sae_fn, _, _, _ = _lazy_torch_imports()

    Y_train_t = torch_mod.tensor(data["Y_train"], dtype=torch_mod.float32, device=device)
    X_iid = torch_mod.tensor(data["Y_val"], dtype=torch_mod.float32, device=device)
    X_ood = torch_mod.tensor(data["Y_ood"], dtype=torch_mod.float32, device=device)

    loader_gen = torch_mod.Generator(device="cpu")
    loader_gen.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(Y_train_t), batch_size=64, shuffle=True, generator=loader_gen,
    )

    results = []
    for sae_type in SAE_TYPES:
        print(f"  Training SAE ({sae_type}) with dynamics tracking...")

        torch_mod.manual_seed(seed)
        sae = SAE(
            input_dim=input_dim, width=width, sae_type=sae_type,
            kval_topk=k if sae_type == "topk" else None,
            mp_kval=k if sae_type == "MP" else None,
        )
        sae.to(device)
        optimizer = torch_mod.optim.AdamW(sae.parameters(), lr=5e-4)

        from models.saes import compute_reg_loss, renorm_decoder_cols_, project_decoder_grads_

        for ep in range(epochs):
            sae.train()
            for (Y_batch,) in train_loader:
                Y_batch = Y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                x_hat, codes = sae(Y_batch, return_hidden=True)
                loss_recon = F.mse_loss(x_hat, Y_batch)
                loss_reg = compute_reg_loss(sae, codes, device)
                loss = loss_recon + gamma_reg * loss_reg
                loss.backward()
                if sae.Ad.grad is not None:
                    with torch_mod.no_grad():
                        project_decoder_grads_(sae.Ad)
                optimizer.step()

            if ep % 50 == 0:
                with torch_mod.no_grad():
                    renorm_decoder_cols_(sae.Ad)

            # Periodic evaluation
            if ep % eval_every == 0 or ep == epochs - 1:
                D_np = sae.Ad.data.detach().cpu().numpy()
                match = match_columns(D_np, A)

                sae.eval()
                with torch_mod.no_grad():
                    _, codes_iid = sae(X_iid, return_hidden=True)
                    mse = F.mse_loss(sae(X_iid)[0] if isinstance(sae(X_iid), tuple) else sae(X_iid), X_iid).item()
                    l0 = (codes_iid.abs() > 0.01).float().sum(dim=-1).mean().item()

                results.append({
                    "method": f"sae_{sae_type}",
                    "epoch": ep,
                    "mean_cosine": match["mean_cosine"],
                    "frac_close": match["frac_close"],
                    "mean_norm_ratio": match["mean_norm_ratio"],
                    "mse_iid": mse,
                    "l0_iid": l0,
                })

                if ep % (eval_every * 5) == 0:
                    print(f"    ep={ep:4d}  cos={match['mean_cosine']:.3f}  "
                          f"frac_close={match['frac_close']:.3f}  mse={mse:.4f}  l0={l0:.1f}")

    return results


def track_dl_fista_dynamics(data, A, input_dim, num_latents, fista_lam,
                            sc_max_steps, seed, device, eval_every_rounds=10):
    """Train DL-FISTA and record dictionary cosine at regular intervals."""
    from models.sparse_coding import fista, update_dictionary, _lipschitz_constant, SparseCodingConfig

    torch_mod = torch

    X_iid = torch_mod.tensor(data["Y_val"], dtype=torch_mod.float32, device=device)
    X_ood = torch_mod.tensor(data["Y_ood"], dtype=torch_mod.float32, device=device)

    torch_mod.manual_seed(seed)
    D = torch_mod.randn(input_dim, num_latents, device=device)
    D = F.normalize(D, dim=0)

    dict_update_every = 50
    n_outer = sc_max_steps // dict_update_every
    n_iter = 100

    Z_iid = None
    Z_ood = None

    results = []
    for outer in range(n_outer):
        with torch_mod.no_grad():
            Z_iid = fista(X_iid, D, fista_lam, n_iter=n_iter, z_init=Z_iid, nonneg=True)
            Z_ood = fista(X_ood, D, fista_lam, n_iter=n_iter, z_init=Z_ood, nonneg=True)

        D_old = D
        D = update_dictionary(X_iid, Z_iid)

        dict_change = (D - D_old).norm().item() / max(D_old.norm().item(), 1e-8)
        if dict_change < 1e-5 and outer > 10:
            print(f"  DL-FISTA early stop at round {outer}")
            break

        if outer % eval_every_rounds == 0 or outer == n_outer - 1:
            D_np = D.detach().cpu().numpy()
            match = match_columns(D_np, A)

            with torch_mod.no_grad():
                mse = F.mse_loss(Z_iid @ D.T, X_iid).item()
                l0 = (Z_iid.abs() > 0.01).float().sum(dim=-1).mean().item()

            results.append({
                "method": "dl_fista",
                "round": outer,
                "mean_cosine": match["mean_cosine"],
                "frac_close": match["frac_close"],
                "mean_norm_ratio": match["mean_norm_ratio"],
                "mse_iid": mse,
                "l0_iid": l0,
            })

            if outer % (eval_every_rounds * 5) == 0:
                print(f"    round={outer:4d}  cos={match['mean_cosine']:.3f}  "
                      f"frac_close={match['frac_close']:.3f}  mse={mse:.4f}  l0={l0:.1f}")

    return results


def run(
    epochs=500,
    gamma_reg=1e-4,
    fista_lam=0.1,
    sc_max_steps=50_000,
    seeds=(0, 1, 2),
    min_n=0,
    max_n=float("inf"),
    eval_every=10,
    out_suffix="",
):
    configs = get_frozen_decoder_configs()
    configs = [c for c in configs if min_n <= c["num_latents"] <= max_n]

    if not configs:
        print(f"No configs match min_n={min_n}, max_n={max_n}")
        return []

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"exp_learning_dynamics{out_suffix}.json"

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

            # SAE learning dynamics
            sae_results = track_sae_dynamics(
                data, A, input_dim_actual, width, k, num_latents, n_samples,
                epochs, gamma_reg, seed, device, eval_every=eval_every,
            )
            for r in sae_results:
                r.update(tag)
            all_results.extend(sae_results)

            # DL-FISTA learning dynamics
            print("  DL-FISTA dynamics...")
            dl_results = track_dl_fista_dynamics(
                data, A, input_dim_actual, num_latents, fista_lam,
                sc_max_steps, seed, device,
            )
            for r in dl_results:
                r.update(tag)
            all_results.extend(dl_results)

            save_incremental(all_results, out_path)

    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=999999)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--out-suffix", type=str, default="")
    args = parser.parse_args()
    run(min_n=args.min_n, max_n=args.max_n, eval_every=args.eval_every,
        out_suffix=args.out_suffix)
