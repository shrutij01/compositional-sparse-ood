#!/bin/bash

# Updated parameter names to match refactored code
activations=(
    "--activation relu"  #"--activation jumprelu"   # overcompleteness factor
)
latentvars=(
    "--hid 100"
)
obsdims=(
    "--fanin 32" "--fanin 64" #"--fanin 24"
)
sparsities=(
    "--n-concepts 10" # "--n-concepts 20" "--n-concepts 30"
)

use_amp=(
    "--use-amp"             # Enable mixed precision training
)
seeds=(
    "--seed 0" "--seed 1" "--seed 2" "--seed 3" "--seed 4"
)

# Job settings
job_name="sae_synth"
output="job_output_%j.txt"
error="job_error_%j.txt"
time_limit="0:15:00"
memory="32Gb"
gpu_req="gpu:1"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Directory to store generated job scripts
mkdir -p "${SCRIPT_DIR}/generated_jobs"
mkdir -p "${SCRIPT_DIR}/logs"

# Counter for unique job names
counter=0

# Loop through all combinations of hyperparameters
for activation in "${activations[@]}"; do
    for latentvar in "${latentvars[@]}"; do
        for obsdim in "${obsdims[@]}"; do
            for sparsity in "${sparsities[@]}"; do
                for amp in "${use_amp[@]}"; do
                    for seed in "${seeds[@]}"; do
                        # Define a script name
                        script_name="${SCRIPT_DIR}/generated_jobs/job_${counter}.sh"

                        # Create a batch script for each job
                        echo "#!/bin/bash" > "${script_name}"
                        echo "#SBATCH --job-name=${job_name}_${counter}" >> "${script_name}"
                        echo "#SBATCH --error=${SCRIPT_DIR}/logs/job_%j.err" >> "${script_name}"
                        echo "#SBATCH --time=${time_limit}" >> "${script_name}"
                        echo "#SBATCH --mem=${memory}" >> "${script_name}"
                        echo "#SBATCH --gres=${gpu_req}" >> "${script_name}"
                        echo "module load python/3.10" >> "${script_name}"
                        echo "module load cuda/12.6.0/cudnn" >> "${script_name}"
                        echo "source /home/mila/j/joshi.shruti/venvs/agents/bin/activate" >> "${script_name}"
                        echo "export PYTHONPATH=\"/home/mila/j/joshi.shruti/causalrepl_space/sparse_ood:$PYTHONPATH\"" >> "${script_name}"
                        echo "cd /home/mila/j/joshi.shruti/causalrepl_space/sparse_ood/sae" >> "${script_name}"

                        # Updated command with new parameter names and optimizations
                        echo "python train_saes.py synth --model sae --use-lambda ${latentvar} ${sparsity} ${obsdim} ${amp} ${activation} ${seed}" >> "${script_name}"

                        # Make the script executable
                        chmod +x "${script_name}"

                        # Submit the job
                        sbatch "${script_name}"

                        # Increment counter
                        ((counter++))
                    done
                done
            done
        done
    done
done
