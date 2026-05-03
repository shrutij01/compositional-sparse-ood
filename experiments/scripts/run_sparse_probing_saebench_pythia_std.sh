#!/bin/bash
# Run SAEBench sparse probing with a pre-trained standard SAE for pythia-70m-deduped, layer 4.
# Uses trainer_0 (res-4k, ~4096 features) from sae_bench_pythia70m_sweep_standard_ctx128_0712.
#
# Submit with a specific seed:
#   sbatch --export=SEED=42 experiments/scripts/run_sparse_probing_saebench_pythia_std.sh
#
# Env vars you can override at sbatch time:
#   SAE_REGEX   sae-lens release pattern    (default: sae_bench_pythia70m_sweep_standard_ctx128_0712)
#   SAE_BLOCK   block/hook pattern          (default: blocks.4.hook_resid_post__trainer_0$)
#   MODEL       LLM name                   (default: pythia-70m-deduped)
#   SEED        random seed for probes     (default: 42)
#   OUTPUT_DIR  results directory          (default: results/saebench_sparse_probing_std/seed<SEED>)

#SBATCH --job-name=sparse_probing_std
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-b001,cn-b002,cn-b003,cn-b004,cn-b005,cn-e002,cn-e003
#SBATCH --output=/network/scratch/v/vitoria.barin-pacela/sparse_ood/logs/sparse_probing_std_%j_%x.out
#SBATCH --error=/network/scratch/v/vitoria.barin-pacela/sparse_ood/logs/sparse_probing_std_%j_%x.err

SAE_REGEX="${SAE_REGEX:-sae_bench_pythia70m_sweep_standard_ctx128_0712}"
SAE_BLOCK="${SAE_BLOCK:-blocks\\.4\\.hook_resid_post__trainer_0$}"
MODEL="${MODEL:-pythia-70m-deduped}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-results/saebench_sparse_probing_std/seed${SEED}}"

cd /home/mila/v/vitoria.barin-pacela/sparse_ood
source .venv/bin/activate
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  SAE regex   : $SAE_REGEX"
echo "  SAE block   : $SAE_BLOCK"
echo "  Model       : $MODEL"
echo "  Seed        : $SEED"
echo "  Output dir  : $OUTPUT_DIR"
echo "=============================================="

python SAEBench/sae_bench/evals/sparse_probing/main.py \
    --model_name        "$MODEL"      \
    --sae_regex_pattern "$SAE_REGEX"  \
    --sae_block_pattern "$SAE_BLOCK"  \
    --random_seed       "$SEED"       \
    --output_folder     "$OUTPUT_DIR" \
    --force_rerun

echo "Job finished."
