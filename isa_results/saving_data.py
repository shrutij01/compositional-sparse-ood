import os
import torch
import numpy as np

def save_sae_run(
    A,
    inputs_iid,
    targets_iid,
    inputs_ood,
    targets_ood,
    z_pred_iid,
    z_pred_ood,
    history,
    *,
    mlp: bool,
    epochs: int,
    seed: int,
    hidden_dim: int | None = None,
    accuracy_iid: list[float] | None = None,
    accuracy_ood: list[float] | None = None,
    mcc_iid: list[float] | None = None,
    mcc_ood: list[float] | None = None,
    out_dir: str = '.'
) -> str:
    """
    Save SAE run data (+ optional per-dim accuracy and MCC lists) to a .pt file.

    Parameters
    ----------
    accuracy_iid, accuracy_ood : list of float, optional
        Per-latent-dim classification accuracy on IID / OOD.
    mcc_iid, mcc_ood : list of float, optional
        Per-latent-dim MCC on IID / OOD.
    """
    # Build filename
    model_type = 'mlp' if mlp else 'no_mlp'
    fname = f"sae_{model_type}_epochs_{epochs}_seed_{seed}"
    if mlp and hidden_dim is not None:
        fname += f"_hidden_{hidden_dim}"
    fname += ".pt"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)

    # Convert arrays to CPU tensors
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return x  # leave lists, floats, dicts as-is

    save_dict = {
        'A':           to_tensor(A),
        'inputs_iid':  to_tensor(inputs_iid),
        'targets_iid': to_tensor(targets_iid),
        'inputs_ood':  to_tensor(inputs_ood),
        'targets_ood': to_tensor(targets_ood),
        'z_pred_iid':  to_tensor(z_pred_iid),
        'z_pred_ood':  to_tensor(z_pred_ood),
        'history':     history,
        'config': {
            'mlp':       mlp,
            'epochs':    epochs,
            'seed':      seed,
            'hidden_dim':hidden_dim,
        }
    }

    # Attach the lists if provided
    if accuracy_iid is not None:
        save_dict['accuracy_iid'] = list(accuracy_iid)
    if accuracy_ood is not None:
        save_dict['accuracy_ood'] = list(accuracy_ood)
    if mcc_iid is not None:
        save_dict['mcc_iid'] = list(mcc_iid)
    if mcc_ood is not None:
        save_dict['mcc_ood'] = list(mcc_ood)

    torch.save(save_dict, path)
    return path