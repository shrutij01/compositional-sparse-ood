#!/bin/bash
# Run SAEBench sparse probing for ONE (seed, dataset) pair.
#
# Submit:
#   sbatch --export=SEED=0,DATASET=fancyzhx/ag_news experiments/scripts/run_sparse_probing_single_dataset.sh
#
# Required env vars:
#   SEED          random seed
#   DATASET       single SAEBench dataset name (e.g. fancyzhx/ag_news)
# Optional (with defaults):
#   METHOD        dl_fista | softplus_adam              (default: dl_fista)
#   MODEL         LLM name                              (default: gemma-2-2b)
#   LAYER         hook layer index                      (default: 12)
#   D_SAE         SAE width                             (default: 16384)
#   LAM           L1 sparsity weight                    (default: 0.1)
#   OUTPUT_BASE   parent of seed{N} dirs                (default: /network/scratch/v/vitoria.barin-pacela/sparse_ood/results/saebench_sparse_probing)
#
# Output: writes OUTPUT_BASE/seed{SEED}/per_dataset/{safe_name}.json

#SBATCH --job-name=sp_single
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-b001,cn-b002,cn-b003,cn-b004,cn-b005,cn-e002,cn-e003
#SBATCH --output=/network/scratch/v/vitoria.barin-pacela/sparse_ood/logs/sp_single_%j.out
#SBATCH --error=/network/scratch/v/vitoria.barin-pacela/sparse_ood/logs/sp_single_%j.err

set -euo pipefail

# ---- defaults ----
METHOD="${METHOD:-dl_fista}"
MODEL="${MODEL:-gemma-2-2b}"
LAYER="${LAYER:-12}"
D_SAE="${D_SAE:-16384}"
LAM="${LAM:-0.1}"
OUTPUT_BASE="${OUTPUT_BASE:-/network/scratch/v/vitoria.barin-pacela/sparse_ood/results/saebench_sparse_probing}"

if [ -z "${SEED:-}" ] || [ -z "${DATASET:-}" ]; then
    echo "ERROR: SEED and DATASET env vars are required."
    exit 1
fi

OUTPUT_DIR="$OUTPUT_BASE/seed${SEED}"

# ---- environment ----
cd /home/mila/v/vitoria.barin-pacela/sparse_ood
source .venv/bin/activate
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

mkdir -p /network/scratch/v/vitoria.barin-pacela/sparse_ood/logs
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Seed        : $SEED"
echo "  Dataset     : $DATASET"
echo "  Method      : $METHOD"
echo "  Model       : $MODEL"
echo "  Layer       : $LAYER"
echo "  d_sae       : $D_SAE"
echo "  lam         : $LAM"
echo "  Output dir  : $OUTPUT_DIR"
echo "=============================================="

python sae_bench_adapter/train_sae_bench.py \
    --model_name      "$MODEL"   \
    --hook_layer      "$LAYER"   \
    --method          "$METHOD"  \
    --d_sae           "$D_SAE"   \
    --lam             "$LAM"     \
    --seed            "$SEED"    \
    --output_dir      "$OUTPUT_DIR" \
    --dataset_filter  "$DATASET"

echo "Job finished."
