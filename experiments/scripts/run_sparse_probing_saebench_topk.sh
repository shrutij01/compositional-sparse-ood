#!/bin/bash
# Run SAEBench sparse probing with pre-trained SAEBench TopK SAEs on the Mila cluster.
# Uses the native SAEBench sparse_probing eval (not the custom adapter).
#
# Submit with a specific seed:
#   sbatch --export=SEED=0 experiments/scripts/run_sparse_probing_saebench_topk.sh
#
# Env vars you can override at sbatch time:
#   SAE_REGEX      sae-lens release pattern           (default: saebench_gemma-2-2b_topk_width-2pow14_date-1109)
#   SAE_BLOCK      block/hook pattern                 (default: blocks.12.hook_resid_post)
#   MODEL          LLM name                           (default: gemma-2-2b)
#   SEED           random seed for probe training     (default: 0)
#   OUTPUT_DIR     results directory                  (default: results/saebench_sparse_probing_topk)

#SBATCH --job-name=sparse_probing_topk
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sparse_probing_topk_%j_%x.out
#SBATCH --error=logs/sparse_probing_topk_%j_%x.err

# ---- defaults (override via --export at sbatch time) ----
SAE_REGEX="${SAE_REGEX:-sae_bench_gemma-2-2b_topk_width-2pow14_date-1109}"
SAE_BLOCK="${SAE_BLOCK:-blocks\.12\.hook_resid_post__trainer_0$}"
MODEL="${MODEL:-gemma-2-2b}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/saebench_sparse_probing_topk/seed${SEED}}"

# ---- environment ----
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
    --model_name      "$MODEL"      \
    --sae_regex_pattern "$SAE_REGEX" \
    --sae_block_pattern "$SAE_BLOCK" \
    --random_seed     "$SEED"       \
    --output_folder   "$OUTPUT_DIR" \
    --save_activations              \
    --force_rerun

echo "Job finished."
