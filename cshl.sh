#!/bin/bash
#SBATCH --job-name=pytorch_cuda_eb
#SBATCH --output=pytorch_cuda_eb.out
#SBATCH --error=pytorch_cuda_eb.err
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --qos=fast
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuq

module load EBH100 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load singularity/3.6.3
# module load uge
# module load python37
# module load slurm/slurm/23.11.3
# module load DefaultModules

eval "$(conda shell.bash hook)"
conda activate jhub-py312

# Install required packages
conda install -y scikit-learn pandas tqdm wandb -c conda-forge
pip install wandb
    
python run.py --lambda_p $1 