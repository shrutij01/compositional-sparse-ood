import os
import sys
import argparse

seeds = [7012025, 1, 2, 3, 4, 5, 6, 7, 8, 9]
lambdas = [0.001, 0.01, 0.1, 1.0]
# ns = [24, 47, 100, 200, 300, 500]
ns = [47, 100, 200, 300, 500]
ms = [10, 24, 47, 75, 100]
ks = [5, 10, 20, 40]
nsamples = [500, 1000, 5000, 10000, 50000, 100000]

# for seed in seeds:
#     for lam in lambdas:
#         os.system("sbatch mila_slurm.sh " + str(seed) + " " + str(lam))

# for seed in seeds:
#     for n in ns:
#         os.system("sbatch mila_slurm.sh " + str(seed) + " " + str(n))

# for seed in seeds:
#     for m in ms:
#         os.system("sbatch mila_slurm.sh " + str(seed) + " " + str(m))

for seed in seeds:
    for k in ks:
        os.system("sbatch mila_slurm.sh " + str(seed) + " " + str(k))

# for seed in seeds:
#     for samples in nsamples:
#         os.system("sbatch mila_slurm.sh " + str(seed) + " " + str(samples))