#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

#Load module
module load anaconda/3

#Load python environment
source activate research

# python run.py --steps 5 --num_seed 5 --lambda_p $1 

# python run.py --num_seed 10 --lambda_p $1 

# python run.py --num_seed 1

python run2.py