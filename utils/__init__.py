"""Utils package.

Torch-dependent helpers (numerics.*_t functions) are loaded lazily
so that numpy-only code (metrics, linear_probe) can import without torch.
"""

# NumPy-only — always available
from utils.metrics import compute_mcc, evaluate_accuracy, evaluate_auc, evaluate_all

# Torch-dependent — loaded on first access
_NUMPY_NAMES = {
    "EPS", "safe_divide", "safe_norm", "safe_normalize_cols", "safe_corrcoef",
    "sanitize_array", "sanitize_mi", "clamp", "warn_nan",
}
_TORCH_NAMES = {
    "safe_norm_t", "safe_normalize_cols_t", "safe_sqrt_t", "safe_divide_t",
    "safe_log_t", "safe_exp_t", "nan_to_num_t", "safe_corrcoef_t", "check_nan",
}


def __getattr__(name):
    if name in _NUMPY_NAMES or name in _TORCH_NAMES:
        from utils import numerics
        return getattr(numerics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
