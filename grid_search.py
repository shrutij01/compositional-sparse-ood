import os
import sys
import argparse

# lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
lambdas = [1e-2]

for lambda_p in lambdas:
    os.system("sbatch cshl.sh " + str(lambda_p))