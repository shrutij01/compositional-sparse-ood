#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

#Load module
module load anaconda/3

#Load python environment
source activate research

# python david_demo.py --seed $1 --lam $2

# python david_demo.py --seed $1 --n $2

# python david_demo.py --seed $1 --m $2

python david_demo.py --seed $1 --k $2