import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """
    Single-layer encoder: Linear -> Softplus
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEncoder(nn.Module):
    """
    Two-layer encoder: Linear -> ReLU -> Linear -> Softplus
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Simple linear decoder: Linear
    """
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SAE(nn.Module):
    """
    Sparse Autoencoder combining an Encoder and a Decoder
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = None
    ):
        super().__init__()
        # Choose single-layer or MLP encoder
        if hidden_dim is None:
            self.encoder = Encoder(input_dim, latent_dim)
        else:
            self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        rec = self.decoder(z)
        return z, rec


def train_supervised_sae(
    model: SAE,
    train_loader: DataLoader,
    inputs_iid: torch.Tensor,
    targets_iid: torch.Tensor,
    inputs_ood: torch.Tensor,
    targets_ood: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    num_epochs: int = 10
) -> tuple:
    """
    Train SAE to predict latent Z from inputs Y.
    Logs IID & OOD MSE each epoch and returns final reconstructions.
    """
    model.to(device)
    # Reinitialize encoder & decoder parameters
    for module in (model.encoder, model.decoder):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for Y_batch, Z_batch in train_loader:
            Y_batch, Z_batch = Y_batch.to(device), Z_batch.to(device)
            optimizer.zero_grad()
            Z_pred, _ = model(Y_batch)
            loss = loss_fn(Z_pred, Z_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            Z_iid_pred, _ = model(inputs_iid.to(device))
            mse_i = loss_fn(Z_iid_pred, targets_iid.to(device)).item()

            Z_ood_pred, _ = model(inputs_ood.to(device))
            mse_o = loss_fn(Z_ood_pred, targets_ood.to(device)).item()

        print(f"[Sup SAE Epoch {epoch:2d}] IID MSE: {mse_i:.4f}   OOD MSE: {mse_o:.4f}")

    return (
        Z_iid_pred.cpu().numpy(),
        Z_ood_pred.cpu().numpy()
    )


def train_unsupervised_sae(
    model: SAE,
    train_loader: DataLoader,
    inputs_iid: torch.Tensor,
    inputs_ood: torch.Tensor,
    device: torch.device,
    lr: float = 1e-2,
    lambda_l1: float = 1e-1,
    num_epochs: int = 10,
    return_metrics: bool = False
) -> tuple:
    """
    Train SAE to reconstruct inputs with an L1 penalty on latent codes.
    Logs losses each epoch and returns latent codes (and metrics if requested).
    """
    model.to(device)
    # Reinitialize encoder & decoder parameters
    for module in (model.encoder, model.decoder):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            Z_pred, X_rec = model(X_batch)
            loss_rec = loss_fn(X_rec, X_batch)
            l1_pen = torch.mean(torch.abs(Z_pred) * torch.linalg.norm(model.decoder.net.weight, dim=0))
            loss = loss_rec + lambda_l1 * l1_pen
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            Z_i, X_i = model(inputs_iid.to(device))
            rec_i = loss_fn(X_i, inputs_iid.to(device)).item()
            l1_i = torch.mean(torch.abs(Z_i)).item()
            total_i = rec_i + lambda_l1 * l1_i

            Z_o, X_o = model(inputs_ood.to(device))
            rec_o = loss_fn(X_o, inputs_ood.to(device)).item()
            l1_o = torch.mean(torch.abs(Z_o)).item()
            total_o = rec_o + lambda_l1 * l1_o

        print(f"[Unsup SAE Epoch {epoch:2d}] IID Loss: {total_i:.4f} (rec {rec_i:.4f}, l1 {l1_i:.4f})   OOD Loss: {total_o:.4f} (rec {rec_o:.4f}, l1 {l1_o:.4f})")

    Z_unsup_iid = Z_i.cpu().numpy()
    Z_unsup_ood = Z_o.cpu().numpy()

    if return_metrics:
        return (Z_unsup_iid, Z_unsup_ood, rec_i, l1_i, rec_o)
    return (Z_unsup_iid, Z_unsup_ood)

