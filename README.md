# Compositional Sparse OOD

Code for **"Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation"** (UAI 2026).

**[Documentation](https://shrutij01.github.io/compositional-sparse-ood/)**

*Vitória Barin Pacela\*, Shruti Joshi\*, Isabela Camacho, Simon Lacoste-Julien, David Klindt*

## Summary

Under superposition, concepts are linearly *represented* in neural network activations but not linearly *accessible*. We show that:

- **Per-sample sparse inference** (FISTA with the ground-truth dictionary) achieves near-perfect OOD compositional generalisation at all scales tested
- **SAEs** fail because their learned dictionaries point in wrong directions — replacing the encoder with FISTA on the same dictionary does not help
- **Linear probes** degrade sharply in identifiability under superposition (MCC < 0.1 at d_z = 10,000)
- **Dictionary learning** is the bottleneck: DL-FISTA beats linear probes when it succeeds but fails at scale, identifying scalable dictionary learning as the open challenge

## Setup

```bash
pip install -e .
```

Requires Python >= 3.10 and PyTorch.

## Reproducing paper figures

All results are pre-computed in `results/`. To regenerate figures:

```bash
# Main text + appendix v2 figures
python experiments/plotting/plot_paper_figures.py --only v2

# All figures (v1 appendix + v2 + phase)
python experiments/plotting/plot_paper_figures.py

# Controlled experiment figures (warmstart, dict quality, support recovery, learning dynamics)
python experiments/plotting/plot_controlled.py
```

Figures are saved to `paper_figures/`.

## Running experiments

### Sensitivity experiments (vary one parameter, hold others fixed)

```bash
# Vary latent dimension d_z
python experiments/sensitivity/exp_vary_latents.py

# Vary number of training samples
python experiments/sensitivity/exp_vary_samples.py

# Vary sparsity level k
python experiments/sensitivity/exp_vary_sparsity.py
```

### Controlled experiments (decompose SAE failure)

```bash
# Frozen decoder: FISTA on SAE-learned dictionary
python experiments/controlled/exp_dict_quality.py

# Warm-start: SAE decoder as initialisation for DL-FISTA
python experiments/controlled/exp_warmstart_decoder.py

# Support recovery diagnostics
python experiments/controlled/exp_support_recovery.py

# Dictionary learning dynamics
python experiments/controlled/exp_learning_dynamics.py

# Lambda sensitivity sweep
python experiments/controlled/exp_lambda_sensitivity.py
```

Results are saved incrementally to `results/`.

## Repository structure

```
src/data.py                  # Synthetic data generation (sparse codes + linear mixing)
models/
  saes.py                    # SAE variants (ReLU, TopK, JumpReLU, MP)
  sparse_coding.py           # FISTA, DL-FISTA, Softplus-Adam, LISTA
  linear_probe.py            # Supervised linear probe baseline
utils/metrics.py             # MCC, accuracy, AUC, support recovery metrics
experiments/
  _common.py                 # Shared training/evaluation helpers
  param_check.py             # Compressed sensing bound validation
  sensitivity/               # Vary latents, samples, sparsity
  controlled/                # Frozen decoder, warmstart, dict quality, etc.
  plotting/                  # Figure generation scripts
results/                     # Pre-computed experiment results (JSON)
paper_figures/               # Generated figures
```

## Citation

```bibtex
@inproceedings{barinpacela2026stop,
  title={Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation},
  author={Barin Pacela, Vit{\'o}ria and Joshi, Shruti and Camacho, Isabela and Lacoste-Julien, Simon and Klindt, David},
  booktitle={Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2026}
}
```

## License

MIT
