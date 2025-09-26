#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00

#Load module
module load anaconda/3

#Load python environment
source activate research


# python linear_probe.py --seed $1 --n_points $2

python linear_probe.py --seed $1 --k $2 --m 47

# python linear_probe.py --seed $1 --n $2 --m 47

# python linear_probe.py --seed $1 --m $2