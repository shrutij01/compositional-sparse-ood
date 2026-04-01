"""Models package.

Heavy imports (torch-dependent) are lazy to allow lightweight modules
like linear_probe to be imported without torch installed.
"""


def __getattr__(name):
    if name in (
        "SAE", "SAEConfig", "train_sae", "run_sae_experiment",
    ):
        from .saes import SAE, SAEConfig, train_sae, run_sae_experiment
        return locals()[name]

    if name in (
        "SparseCodingConfig", "ista", "fista", "soft_threshold",
        "LISTA", "train_lista", "update_dictionary",
        "train_sparse_coding", "refine_from_sae", "compare_methods",
        "run_sparse_coding_experiment",
    ):
        from .sparse_coding import (
            SparseCodingConfig, ista, fista, soft_threshold,
            LISTA, train_lista, update_dictionary,
            train_sparse_coding, refine_from_sae, compare_methods,
            run_sparse_coding_experiment,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
