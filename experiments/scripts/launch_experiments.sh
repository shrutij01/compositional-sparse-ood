#!/bin/bash

# Launch sweep experiments on SLURM.
# Experiments that sweep num_latents are split into two tiers:
#   - fast: num_latents <= 5000 (6h time limit)
#   - heavy: num_latents = 10000 (24h time limit)
#
# Usage:
#   bash experiments/launch_experiments.sh          # submit all experiments
#   bash experiments/launch_experiments.sh samples   # submit only vary_samples
#   bash experiments/launch_experiments.sh sparsity  # submit only vary_sparsity
#   bash experiments/launch_experiments.sh latents   # submit only vary_latents
#   bash experiments/launch_experiments.sh frozen    # submit only frozen_decoder
#   bash experiments/launch_experiments.sh large     # submit only large_latents

# ---- Paths ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv/bin/activate"

# ---- Job settings ----
memory="16Gb"
gpu_req="gpu:a100:1"
cpu_req="cpus-per-task=2"

# ---- Setup directories ----
mkdir -p "${PROJECT_ROOT}/experiments/generated_jobs"
mkdir -p "${PROJECT_ROOT}/experiments/logs"
mkdir -p "${PROJECT_ROOT}/results"

counter=0

submit_job() {
    local job_name="$1"
    local time_limit="$2"
    local cmd="$3"
    local mem="${4:-$memory}"

    local script_name="${PROJECT_ROOT}/experiments/generated_jobs/job_${job_name}.sh"

    cat > "${script_name}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${PROJECT_ROOT}/experiments/logs/${job_name}_%j.out
#SBATCH --error=${PROJECT_ROOT}/experiments/logs/${job_name}_%j.err
#SBATCH --time=${time_limit}
#SBATCH --mem=${mem}
#SBATCH --gres=${gpu_req}
#SBATCH --${cpu_req}

module load python/3.10
module load cuda/12.6.0/cudnn

source ${VENV_PATH}

export PYTHONPATH="${PROJECT_ROOT}:\$PYTHONPATH"
cd ${PROJECT_ROOT}

${cmd}
EOF

    chmod +x "${script_name}"
    echo "Submitting ${job_name} (${time_limit}, ${mem})..."
    sbatch "${script_name}"
    ((counter++))
}

# ---- Determine which experiments to run ----
run_samples=false
run_sparsity=false
run_latents=false
run_frozen=false
run_large=false

if [ -z "$1" ]; then
    run_samples=true
    run_sparsity=true
    run_latents=true
    run_frozen=true
    run_large=true
else
    case "$1" in
        samples)   run_samples=true ;;
        sparsity)  run_sparsity=true ;;
        latents)   run_latents=true ;;
        frozen)    run_frozen=true ;;
        large)     run_large=true ;;
        *)         echo "Unknown: $1. Use: samples, sparsity, latents, frozen, large"; exit 1 ;;
    esac
fi

# ---- Submit jobs ----

# vary_samples: num_latents=100 (default) + num_latents=10000
if $run_samples; then
    submit_job "exp_vary_samples" "12:00:00" \
        "python experiments/sensitivity/exp_vary_samples.py"
    submit_job "exp_vary_samples_10k" "24:00:00" \
        "python experiments/sensitivity/exp_vary_samples.py --num-latents 10000 --out-suffix _10k" "32Gb"
fi

# vary_sparsity: num_latents=1000
if $run_sparsity; then
    submit_job "exp_vary_sparsity" "12:00:00" \
        "python experiments/sensitivity/exp_vary_sparsity.py"
fi

# vary_latents: split into num_latents<=5000 (fast) and num_latents=10000 (heavy)
if $run_latents; then
    submit_job "exp_vary_latents" "12:00:00" \
        "python experiments/sensitivity/exp_vary_latents.py --max-n 5000"
    submit_job "exp_vary_latents_10k" "24:00:00" \
        "python experiments/sensitivity/exp_vary_latents.py --min-n 10000 --out-suffix _10k"
fi

# frozen_decoder: split by num_latents
if $run_frozen; then
    submit_job "exp_frozen_decoder" "12:00:00" \
        "python experiments/sensitivity/exp_frozen_decoder.py --max-n 5000"
    submit_job "exp_frozen_decoder_10k" "24:00:00" \
        "python experiments/sensitivity/exp_frozen_decoder.py --min-n 10000 --out-suffix _10k"
fi

# large_latents: split into three tiers by num_latents
#   - small: 1K-100K (6h, 32GB) — runs all methods up to 50K, fewer above
#   - 500K:  (24h, 48GB) — FISTA oracle + linear baselines only
#   - 1M:    (48h, 64GB) — FISTA oracle + linear baselines only, separate job
if $run_large; then
    submit_job "exp_large_latents_small" "06:00:00" \
        "python experiments/sensitivity/exp_large_latents.py --max-n 100000" "32Gb"
    submit_job "exp_large_latents_500k" "24:00:00" \
        "python experiments/sensitivity/exp_large_latents.py --min-n 500000 --max-n 500000 --out-suffix _500k" "48Gb"
    submit_job "exp_large_latents_1M" "48:00:00" \
        "python experiments/sensitivity/exp_large_latents.py --min-n 1000000 --out-suffix _1M" "64Gb"
fi

echo ""
echo "Submitted ${counter} job(s). Check logs in experiments/logs/"
echo "Results will be saved to ${PROJECT_ROOT}/results/"
echo ""
echo "After jobs complete, merge split results with:"
echo "  python experiments/merge_results.py"
