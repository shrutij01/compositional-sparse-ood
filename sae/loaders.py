import os

import torch


def load_sae_models(modeldirs):
    """Load SSAE models from multiple directories."""
    decoder_weight_matrices = []
    decoder_bias_vectors = []
    encoder_weight_matrices = []
    encoder_bias_vectors = []

    for modeldir in modeldirs:
        weight_path = os.path.join(modeldir, "weights.pth")

        try:
            state_dict = torch.load(weight_path, map_location="cpu")
            # Check each tensor for corruption
            for key, tensor in state_dict.items():
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN detected in {key}")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf detected in {key}")
                if tensor.numel() == 0:
                    raise ValueError(f"Empty tensor in {key}")

            decoder_weight_matrices.append(
                state_dict["decoder.weight"].clone()
            )
            decoder_bias_vectors.append(state_dict["decoder.bias"].clone())
            encoder_weight_matrices.append(
                state_dict["encoder.weight"].clone()
            )
            encoder_bias_vectors.append(state_dict["encoder.bias"].clone())

        except Exception as e:
            raise ValueError(f"Error loading model from {modeldir}: {e}")

    return (
        decoder_weight_matrices,
        decoder_bias_vectors,
        encoder_weight_matrices,
        encoder_bias_vectors,
    )
