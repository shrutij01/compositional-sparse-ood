#!/bin/bash
#SBATCH --job-name=pytorch_cuda_eb
#SBATCH --output=pytorch_cuda_eb.out
#SBATCH --error=pytorch_cuda_eb.err
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:2
#SBATCH --qos=fast
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuq

# module load EBH100 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# python pytorch_test.py

conda activate jhub-py312
python run.py
