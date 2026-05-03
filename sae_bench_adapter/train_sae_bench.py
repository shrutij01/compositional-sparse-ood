"""
Train dl_firsta or softplus_adam on LLM activations, then run SAEBench sparse probing.

Steps:
  1. Load the LLM and collect token-level activations from a text corpus.
  2. Run train_sparse_coding() to learn a dictionary.
  3. Wrap the dictionary in DLFistaSAE or SoftplusAdamSAE.
  4. Run sparse probing evaluation via SAEBench.

Example
-------
    # From the sparse_ood project root:
    python sae_bench_adapter/train_sae_bench.py \
        --model_name pythia-70m-deduped \
        --hook_layer 4 \
        --method dl_fista \
        --d_sae 2048 \
        --n_train_tokens 50000 \
        --max_steps 5000 \
        --output_dir results/saebench_sparse_probing
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — must come before any project imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SAEBENCH_ROOT = _PROJECT_ROOT / "SAEBench"

for _p in [str(_PROJECT_ROOT), str(_SAEBENCH_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from models.sparse_coding import SparseCodingConfig, train_sparse_coding  # noqa: E402
from sae_bench_adapter.sparse_coding_sae import DLFistaSAE, SoftplusAdamSAE  # noqa: E402

# SAEBench imports
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig  # noqa: E402
from sae_bench.evals.sparse_probing.main import run_eval  # noqa: E402
from sae_bench.sae_bench_utils import activation_collection  # noqa: E402


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def collect_activations(
    model_name: str,
    hook_layer: int,
    n_tokens: int,
    context_len: int,
    llm_batch_size: int,
    device: str,
    llm_dtype: torch.dtype,
    dataset_name: str = "NeelNanda/pile-10k",
) -> torch.Tensor:
    """Collect flat token-level activations from the LLM at a given layer.

    Returns
    -------
    acts : torch.Tensor, shape (n_valid_tokens, d_model)
        Activations for non-padding tokens, on CPU.
    """
    from datasets import load_dataset
    from transformer_lens import HookedTransformer

    print(f"Loading {model_name} ...")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=llm_dtype
    )
    model.eval()

    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    n_sequences = max(1, n_tokens // context_len)

    print(f"Loading dataset '{dataset_name}' ...")
    raw = load_dataset(dataset_name, split="train", trust_remote_code=True)
    texts = [item["text"] for item in raw][:n_sequences]

    print(f"Tokenizing {len(texts)} sequences (context_len={context_len}) ...")
    tokens = model.to_tokens(texts, prepend_bos=True)
    tokens = tokens[:, :context_len]  # truncate to context_len

    print(f"Collecting activations at layer {hook_layer} ({hook_name}) ...")
    acts_BLD = activation_collection.get_llm_activations(
        tokens, model, llm_batch_size, hook_layer, hook_name,
        mask_bos_pad_eos_tokens=True,
    )  # (n_sequences, context_len, d_model)

    # Flatten and drop padding (zeroed by mask)
    acts_flat = acts_BLD.reshape(-1, acts_BLD.shape[-1])
    nonzero_mask = acts_flat.abs().sum(-1) > 0
    acts_flat = acts_flat[nonzero_mask].cpu()

    print(f"Collected {acts_flat.shape[0]:,} valid token activations, d_model={acts_flat.shape[1]}")
    del model
    torch.cuda.empty_cache()
    return acts_flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sparse coding SAE and run sparse probing")

    # LLM settings
    p.add_argument("--model_name", type=str, default="pythia-70m-deduped",
                   help="TransformerLens model name (must be in LLM_NAME_TO_BATCH_SIZE)")
    p.add_argument("--hook_layer", type=int, default=4,
                   help="Which residual-stream layer to hook")
    p.add_argument("--context_len", type=int, default=128,
                   help="Token context length per sequence")

    # Dictionary learning settings
    p.add_argument("--method", type=str, default="dl_fista",
                   choices=["dl_fista", "softplus_adam"],
                   help="dl_fista → alternating FISTA+dict-update; "
                        "softplus_adam → joint Adam optimization")
    p.add_argument("--d_sae", type=int, default=None,
                   help="SAE width (default: 4 × d_model)")
    p.add_argument("--lam", type=float, default=0.1,
                   help="L1 sparsity weight")
    p.add_argument("--n_train_tokens", type=int, default=50_000,
                   help="Number of activation tokens to train on")
    p.add_argument("--max_steps", type=int, default=5_000,
                   help="Training steps / outer iterations")
    p.add_argument("--n_iter", type=int, default=100,
                   help="FISTA inner iterations (dl_fista only)")
    p.add_argument("--dict_update_every", type=int, default=50,
                   help="Dictionary update frequency (dl_fista only)")
    p.add_argument("--seed", type=int, default=42)

    # Inference settings (used during SAEBench encoding)
    p.add_argument("--n_fista_iter", type=int, default=200,
                   help="FISTA iterations at eval time (dl_fista)")
    p.add_argument("--n_encode_steps", type=int, default=300,
                   help="Adam steps at eval time (softplus_adam)")
    p.add_argument("--encode_lr", type=float, default=1e-2,
                   help="Adam learning rate at eval time (softplus_adam)")

    # Eval settings
    p.add_argument("--output_dir", type=str,
                   default="results/saebench_sparse_probing",
                   help="Directory for SAEBench evaluation results")
    p.add_argument("--skip_eval", action="store_true",
                   help="Only train the dictionary, skip SAEBench eval")
    p.add_argument("--force_rerun", action="store_true",
                   help="Re-run even if results already exist")
    p.add_argument("--save_activations", action="store_true",
                   help="Cache LLM activations to disk (reused across runs)")
    p.add_argument("--lower_vram_usage", action="store_true",
                   help="Reduce GPU memory (slower)")

    # Training corpus
    p.add_argument("--train_dataset", type=str, default="NeelNanda/pile-10k",
                   help="HuggingFace dataset to collect training activations from")

    # Per-dataset parallelism: when set, run eval on only this dataset and
    # write the result to OUTPUT_DIR/per_dataset/<safe_name>.json. Multiple
    # such jobs can run concurrently for different datasets without races.
    p.add_argument("--dataset_filter", type=str, default=None,
                   help="If set, eval only this single dataset and write to per_dataset/<name>.json")

    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_dtype_str = activation_collection.LLM_NAME_TO_DTYPE.get(args.model_name, "float32")
    llm_dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[llm_dtype_str]
    llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE.get(
        args.model_name, 64
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_path = output_dir / f"dictionary_{args.method}_layer{args.hook_layer}.npy"

    # -----------------------------------------------------------------------
    # Phase 1: Train the dictionary (or load cached)
    # -----------------------------------------------------------------------
    if dict_path.exists() and not args.force_rerun:
        print(f"Loading cached dictionary from {dict_path}")
        dictionary = np.load(dict_path)
        d_model = dictionary.shape[0]
    else:
        acts_flat = collect_activations(
            model_name=args.model_name,
            hook_layer=args.hook_layer,
            n_tokens=args.n_train_tokens,
            context_len=args.context_len,
            llm_batch_size=llm_batch_size,
            device=device,
            llm_dtype=llm_dtype,
            dataset_name=args.train_dataset,
        )

        d_model = acts_flat.shape[1]
        d_sae = args.d_sae if args.d_sae is not None else 4 * d_model

        # 80 / 20 split — train_sparse_coding needs iid + ood tensors
        n_total = acts_flat.shape[0]
        n_iid = int(n_total * 0.8)
        X_iid = acts_flat[:n_iid].to(device)
        X_ood = acts_flat[n_iid:].to(device)

        sc_method = "fista" if args.method == "dl_fista" else "direct"
        cfg = SparseCodingConfig(
            input_dim=d_model,
            num_latents=d_sae,
            method=sc_method,
            lam=args.lam,
            max_steps=args.max_steps,
            n_iter=args.n_iter,
            dict_update_every=args.dict_update_every,
            supervised=False,
            seed=args.seed,
        )

        print(f"\nTraining dictionary with method={sc_method}, "
              f"d_model={d_model}, d_sae={d_sae}, lam={args.lam} ...")
        result = train_sparse_coding(X_iid, X_ood, cfg, device=torch.device(device))
        dictionary = result["dictionary"]  # (d_model, d_sae)

        try:
            np.save(dict_path, dictionary)
            print(f"Dictionary saved to {dict_path} | shape={dictionary.shape}", flush=True)
        except OSError as _save_err:
            print(f"Warning: could not save dictionary to {dict_path} ({_save_err}); "
                  "continuing with in-memory dictionary.", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Create the SAEBench-compatible adapter
    # -----------------------------------------------------------------------
    d_model, d_sae = dictionary.shape
    hook_name = f"blocks.{args.hook_layer}.hook_resid_post"

    sae_device = torch.device(device)
    sae_dtype = llm_dtype  # match the LLM dtype for SAEBench

    if args.method == "dl_fista":
        sae = DLFistaSAE.from_trained(
            dictionary,
            model_name=args.model_name,
            hook_layer=args.hook_layer,
            device=sae_device,
            dtype=sae_dtype,
            hook_name=hook_name,
            lam=args.lam,
            n_iter=args.n_fista_iter,
            nonneg=True,
        )
        sae_label = f"dl_fista_layer{args.hook_layer}_lam{args.lam}"
    else:
        sae = SoftplusAdamSAE.from_trained(
            dictionary,
            model_name=args.model_name,
            hook_layer=args.hook_layer,
            device=sae_device,
            dtype=sae_dtype,
            hook_name=hook_name,
            lam=args.lam,
            n_encode_steps=args.n_encode_steps,
            encode_lr=args.encode_lr,
        )
        sae_label = f"softplus_adam_layer{args.hook_layer}_lam{args.lam}"

    print(f"\nCreated {type(sae).__name__}: d_in={d_model}, d_sae={d_sae}")
    print(f"SAE label for results: {sae_label}")

    if args.skip_eval:
        print("--skip_eval set, stopping before SAEBench evaluation.")
        return

    # -----------------------------------------------------------------------
    # Phase 3: Run SAEBench sparse probing evaluation
    # -----------------------------------------------------------------------
    config = SparseProbingEvalConfig(model_name=args.model_name)
    config.random_seed = args.seed
    config.llm_batch_size = llm_batch_size
    config.llm_dtype = llm_dtype_str
    config.lower_vram_usage = args.lower_vram_usage

    selected_saes = [(sae_label, sae)]

    # Write eval output to /tmp (local disk) to avoid Lustre write failures on
    # long-running jobs, then copy the JSON to the final Lustre destination.
    import tempfile, shutil, json as _json, sys as _sys
    tmp_eval_dir = tempfile.mkdtemp(prefix="sparse_probing_eval_")
    final_eval_dir = output_dir / "sparse_probing"
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    # Filter to a single dataset for parallel per-(seed, dataset) jobs.
    if args.dataset_filter:
        config.dataset_names = [args.dataset_filter]
        print(f"[filter] Running only dataset: {args.dataset_filter}", flush=True)

    # Per-dataset result files (race-free): each dataset writes to its own
    # file under output_dir/per_dataset/. Aggregation reads the directory.
    per_dataset_dir = output_dir / "per_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    def _safe(name: str) -> str:
        return name.replace("/", "_")

    import sae_bench.evals.sparse_probing.main as _sb_main
    _orig_single_dataset = _sb_main.run_eval_single_dataset

    def _per_ds_single_dataset(dataset_name, *a, **kw):
        pd_path = per_dataset_dir / f"{_safe(dataset_name)}.json"
        if pd_path.exists() and not args.force_rerun:
            print(f"[per_dataset] Loading cached: {pd_path}", flush=True)
            cached = _json.load(open(pd_path))
            return cached["dataset_results"], cached["per_class_dict"]
        print(f"[per_dataset] Running: {dataset_name}", flush=True)
        ds_result, pc_result = _orig_single_dataset(dataset_name, *a, **kw)
        try:
            with open(pd_path, "w") as f:
                _json.dump({"dataset_results": ds_result, "per_class_dict": pc_result},
                           f, default=str, indent=2)
            print(f"[per_dataset] Saved {pd_path}", flush=True)
        except Exception as _e:
            print(f"[per_dataset] Warning: save failed: {_e}", flush=True)
        return ds_result, pc_result

    _sb_main.run_eval_single_dataset = _per_ds_single_dataset

    print(f"\nRunning sparse probing eval on {args.model_name}, layer {args.hook_layer} ...")
    print(f"Eval temp dir: {tmp_eval_dir}", flush=True)
    print(f"Per-dataset dir: {per_dataset_dir}", flush=True)
    results = run_eval(
        config,
        selected_saes,
        device,
        tmp_eval_dir,
        force_rerun=args.force_rerun,
        clean_up_activations=False,
        save_activations=False,  # never cache activations to avoid large Lustre writes
        artifacts_path=tmp_eval_dir,
    )

    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)

    # Print results to stdout so they're captured in the SLURM log even if
    # the file copy to Lustre fails.
    print("\n=== EVAL RESULTS ===", flush=True)
    for key, val in results.items():
        try:
            print(_json.dumps({key: val}, default=str), flush=True)
        except Exception:
            print(f"  {key}: {val}", flush=True)
    print("=== END EVAL RESULTS ===", flush=True)

    # Copy JSON files from /tmp to final Lustre destination.
    copied = []
    for json_file in Path(tmp_eval_dir).glob("*.json"):
        dst = final_eval_dir / json_file.name
        try:
            shutil.copy2(str(json_file), str(dst))
            copied.append(str(dst))
        except OSError as _e:
            print(f"Warning: could not copy {json_file.name} to Lustre ({_e}). "
                  "Results are in stdout above.", flush=True)
    print(f"Results copied to: {final_eval_dir}" if copied else
          f"No JSON files copied (check stdout for results)", flush=True)
    shutil.rmtree(tmp_eval_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
