import itertools
import argparse
import numpy as np
from sae.evals.metrics import mean_corr_coef
from sae.loaders import load_sae_models


def compute_all_pairwise_mccs(weight_matrices: list) -> list[float]:
    """
    Compute mean correlation coefficients (MCCs) between all model pairs.
    """
    mccs = []
    for i, j in itertools.combinations(range(len(weight_matrices)), 2):
        mcc = mean_corr_coef(
            weight_matrices[i],
            weight_matrices[j],
            method="pearson",
        )
        mccs.append(mcc)
    return mccs


def main(args):
    """Evaluate MCCs between trained SAE models."""
    # Load trained SAE models
    modeldirs = args.modeldirs
    if len(modeldirs) < 2:
        raise ValueError("You must provide at least two model directories.")

    (
        decoder_weight_matrices,
        _,
        _,
        _,
    ) = load_sae_models(modeldirs)

    # Compute pairwise MCCs
    print("Computing pairwise MCCs...")
    mccs = compute_all_pairwise_mccs(decoder_weight_matrices)

    mean_mcc = np.mean(mccs)
    std_mcc = np.std(mccs)

    print("\nPairwise MCCs:")
    for i, (a, b) in enumerate(
        itertools.combinations(range(len(modeldirs)), 2)
    ):
        print(f"Model {a+1} vs Model {b+1}: MCC = {mccs[i]:.4f}")

    print(f"\nMean MCC: {mean_mcc:.4f}")
    print(f"Std  MCC: {std_mcc:.4f}")

    return {"mccs": mccs, "mean_mcc": mean_mcc, "std_mcc": std_mcc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MCCs between SSAE models"
    )
    parser.add_argument(
        "--modeldirs",
        nargs="+",
        type=str,
        required=True,
        help="List of model directories to compare (minimum 2)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()
    main(args)
