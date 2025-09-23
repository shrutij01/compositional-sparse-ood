# sae_plots.py

import numpy as np
import matplotlib.pyplot as plt
import torch

def to_numpy(x):
    """Convert torch.Tensor or np.ndarray to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def plot_mse_vs_epoch(history, num_epochs):
    """
    Plot IID and OOD MSE vs epoch.

    Parameters
    ----------
    history : dict
        Dictionary with keys 'iid_mse' and 'ood_mse', each a list or array of MSE values per epoch.
    num_epochs : int
        Total number of training epochs.
    """
    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history['iid_mse'], 'o-', label='IID MSE')
    plt.plot(epochs, history['ood_mse'], 's--', label='OOD MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Supervised SAE Epoch‐wise MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_latent_comparison(latent_all, Z_all_pred):
    """
    Plot true vs predicted latent representations in 3D.

    Parameters
    ----------
    latent_all : np.ndarray
        Ground truth latent codes, shape (n_samples, 3).
    Z_all_pred : np.ndarray
        Predicted latent codes, shape (n_samples, 3).
    """
    latent_all = to_numpy(latent_all)
    Z_all_pred = to_numpy(Z_all_pred)

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(*latent_all.T, s=5)
    ax1.set_title("True Z")
    ax1.view_init(30, 30)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(*Z_all_pred.T, s=5)
    ax2.set_title("Predicted Z")
    ax2.view_init(30, 30)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(*latent_all.T, s=5, alpha=0.3, label='true')
    ax3.scatter(*Z_all_pred.T, s=5, alpha=0.3, marker='^', label='pred')
    ax3.set_title("Overlay")
    ax3.view_init(30, 30)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_iid_latents(targets_iid, Z_pred_iid):
    """
    Plot true vs predicted IID latents in 3D.

    Parameters
    ----------
    targets_iid : torch.Tensor or np.ndarray
        True IID latent codes, shape (B, 3).
    Z_pred_iid : torch.Tensor or np.ndarray
        Predicted IID latent codes, shape (B, 3).
    """
    t_true_iid = to_numpy(targets_iid)
    t_pred_iid = to_numpy(Z_pred_iid)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(t_true_iid[:, 0], t_true_iid[:, 1], t_true_iid[:, 2],
               s=20, marker='o', alpha=0.25, label='True IID')
    ax.scatter(t_pred_iid[:, 0], t_pred_iid[:, 1], t_pred_iid[:, 2],
               s=20, marker='^', alpha=0.6, label='Predicted IID')

    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.set_zlabel('z₃')
    ax.set_title('True vs. Predicted Latents (IID)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ood_latents(targets_ood, Z_pred_ood):
    """
    Plot true vs predicted OOD latents in 3D.

    Parameters
    ----------
    targets_ood : torch.Tensor or np.ndarray
        True OOD latent codes, shape (B, 3).
    Z_pred_ood : torch.Tensor or np.ndarray
        Predicted OOD latent codes, shape (B, 3).
    """
    t_true_ood = to_numpy(targets_ood)
    t_pred_ood = to_numpy(Z_pred_ood)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(t_true_ood[:, 0], t_true_ood[:, 1], t_true_ood[:, 2],
               s=20, marker='o', alpha=0.25, label='True OOD')
    ax.scatter(t_pred_ood[:, 0], t_pred_ood[:, 1], t_pred_ood[:, 2],
               s=20, marker='^', alpha=0.6, label='Predicted OOD')

    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.set_zlabel('z₃')
    ax.set_title('True vs. Predicted Latents (OOD)')
    ax.legend()
    plt.tight_layout()
    plt.show()

