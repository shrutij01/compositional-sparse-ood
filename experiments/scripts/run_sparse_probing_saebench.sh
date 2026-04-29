#!/bin/bash
# Run SAEBench sparse probing for dl_fista and softplus_adam on the Mila cluster.
#
# Submit both methods in one call:
#   bash experiments/scripts/run_sparse_probing_saebench.sh
#
# Or submit a specific method:
#   sbatch --export=METHOD=dl_fista experiments/scripts/run_sparse_probing_saebench.sh
#   sbatch --export=METHOD=softplus_adam experiments/scripts/run_sparse_probing_saebench.sh
#
# Env vars you can override at sbatch time:
#   METHOD        dl_fista | softplus_adam           (default: dl_fista)
#   MODEL         LLM name                           (default: pythia-70m-deduped)
#   LAYER         hook layer index                   (default: 4)
#   D_SAE         SAE width                          (default: 2048)
#   LAM           L1 sparsity weight                 (default: 0.1)
#   N_TRAIN       tokens for dict training           (default: 50000)
#   MAX_STEPS     training steps / outer iterations  (default: 5000)
#   OUTPUT_DIR    results directory                  (default: results/saebench_sparse_probing)

#SBATCH --job-name=sparse_probing_saebench
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sparse_probing_%j_%x.out
#SBATCH --error=logs/sparse_probing_%j_%x.err

# ---- defaults (override via --export at sbatch time) ----
METHOD="${METHOD:-dl_fista}"
MODEL="${MODEL:-pythia-70m-deduped}"
LAYER="${LAYER:-4}"
D_SAE="${D_SAE:-2048}"
LAM="${LAM:-0.1}"
N_TRAIN="${N_TRAIN:-50000}"
MAX_STEPS="${MAX_STEPS:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/saebench_sparse_probing}"

# ---- environment ----
cd /home/mila/v/vitoria.barin-pacela/sparse_ood
source .venv/bin/activate

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Method      : $METHOD"
echo "  Model       : $MODEL"
echo "  Layer       : $LAYER"
echo "  d_sae       : $D_SAE"
echo "  lam         : $LAM"
echo "  n_train_tok : $N_TRAIN"
echo "  max_steps   : $MAX_STEPS"
echo "  Output dir  : $OUTPUT_DIR"
echo "=============================================="

python sae_bench_adapter/train_sae_bench.py \
    --model_name    "$MODEL"      \
    --hook_layer    "$LAYER"      \
    --method        "$METHOD"     \
    --d_sae         "$D_SAE"      \
    --lam           "$LAM"        \
    --n_train_tokens "$N_TRAIN"   \
    --max_steps     "$MAX_STEPS"  \
    --output_dir    "$OUTPUT_DIR" \
    --save_activations            \
    --force_rerun

echo "Job finished."
