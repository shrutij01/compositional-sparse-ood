"""Shared numerical constants and helpers for metric stability."""

import numpy as np
import torch
import warnings

EPS = 1e-12
SCALE_EPS_FACTOR = 1e-10


# ============================================================================
# NumPy helpers
# ============================================================================


def safe_entropy_eps(values: np.ndarray) -> float:
    """Scale-aware epsilon for entropy smoothing."""
    return max(EPS, float(np.max(np.abs(values))) * SCALE_EPS_FACTOR)


def safe_divide(numerator, denominator, fallback=0.0):
    """Division with epsilon floor on denominator."""
    if abs(denominator) < EPS:
        return fallback
    return numerator / denominator


def clamp(value, lo=0.0, hi=1.0):
    """Clamp scalar to [lo, hi]."""
    return float(np.clip(value, lo, hi))


def sanitize_mi(mi_value: float) -> float:
    """Clamp MI to [0, inf), replacing NaN/negative with 0."""
    if not np.isfinite(mi_value) or mi_value < 0:
        return 0.0
    return float(mi_value)


def warn_nan(metric_name: str, context: str, count: int):
    """Consistent NaN warning across metrics."""
    if count > 0:
        warnings.warn(
            f"{metric_name}: {count} {context} produced NaN/non-finite values "
            f"and were excluded from the score."
        )


def safe_norm(x: np.ndarray, axis=None, keepdims=False, eps=1e-8) -> np.ndarray:
    """L2 norm with epsilon floor to avoid division by zero."""
    norms = np.linalg.norm(x, axis=axis, keepdims=keepdims)
    return np.where(norms < eps, eps, norms)


def safe_normalize_cols(x: np.ndarray, eps=1e-8) -> np.ndarray:
    """Center and L2-normalize each column, safe for zero-variance columns."""
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    norms = safe_norm(x_centered, axis=0, keepdims=True, eps=eps)
    return x_centered / norms


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation via centering + normalizing, NaN-safe."""
    corr = safe_normalize_cols(a).T @ safe_normalize_cols(b)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_array(x: np.ndarray, fallback=0.0) -> np.ndarray:
    """Replace NaN and Inf values in an array."""
    return np.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback)


# ============================================================================
# Torch helpers
# ============================================================================


def safe_norm_t(x: torch.Tensor, dim=None, keepdim=False, eps=1e-8) -> torch.Tensor:
    """L2 norm with epsilon floor to avoid division by zero."""
    return torch.linalg.norm(x, dim=dim, keepdim=keepdim).clamp(min=eps)


def safe_normalize_cols_t(x: torch.Tensor, dim=0, eps=1e-8) -> torch.Tensor:
    """Normalize along a dimension with epsilon floor."""
    norms = safe_norm_t(x, dim=dim, keepdim=True, eps=eps)
    return x / norms


def safe_sqrt_t(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """Square root with clamping to avoid NaN from negative values."""
    return torch.sqrt(x.clamp(min=eps))


def safe_divide_t(numerator: torch.Tensor, denominator: torch.Tensor,
                  eps=1e-12) -> torch.Tensor:
    """Element-wise division with epsilon floor on denominator."""
    return numerator / denominator.clamp(min=eps)


def safe_log_t(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """Logarithm with clamping to avoid log(0)."""
    return torch.log(x.clamp(min=eps))


def safe_exp_t(x: torch.Tensor, max_val=20.0) -> torch.Tensor:
    """Exponential with clamping to avoid overflow."""
    return torch.exp(x.clamp(max=max_val))


def nan_to_num_t(x: torch.Tensor, fallback=0.0) -> torch.Tensor:
    """Replace NaN and Inf values in a tensor."""
    return torch.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback)


def safe_corrcoef_t(x: torch.Tensor, y: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Pearson correlation matrix between columns of x and y (torch).
    Safe against zero-variance columns.

    x, y: (n_samples, d_x) and (n_samples, d_y)
    Returns: (d_x, d_y) correlation matrix.
    """
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    x_norm = safe_norm_t(x_c, dim=0, keepdim=True, eps=eps)
    y_norm = safe_norm_t(y_c, dim=0, keepdim=True, eps=eps)
    corr = (x_c / x_norm).T @ (y_c / y_norm)
    return nan_to_num_t(corr)


def check_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check for NaN/Inf in a tensor, warn if found. Returns True if clean."""
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    if n_nan > 0 or n_inf > 0:
        warnings.warn(f"{name} contains {n_nan} NaN and {n_inf} Inf values")
        return False
    return True
