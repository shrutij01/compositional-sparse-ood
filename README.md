# Compositional Sparse OOD

Code for **"Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation"** (UAI 2026).

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://shrutij01.github.io/compositional-sparse-ood/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*Vitoria Barin Pacela\*, Shruti Joshi\*, Isabela Camacho, Simon Lacoste-Julien, David Klindt*

## What This Does

Under superposition, concepts are linearly *represented* in neural network activations but not linearly *accessible*. We compare sparse coding, SAEs, and linear probes on compositional OOD generalisation.

| Method | OOD generalisation | Why |
|--------|-------------------|-----|
| FISTA (oracle dictionary) | Near-perfect at all scales | Per-sample sparse inference solves the right problem |
| SAEs (ReLU, TopK, JumpReLU) | Fail | Learned dictionaries point in wrong directions |
| Linear probes | Degrade sharply | MCC < 0.1 at d_z = 10,000 under superposition |
| DL-FISTA | Beats probes, fails at scale | Dictionary learning is the bottleneck |

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

Requires Python >= 3.10 and PyTorch.

## Reproduce Paper Figures

All results are pre-computed in `results/`. To regenerate:

```bash
# Main text + appendix figures
python experiments/plotting/plot_paper_figures.py --only v2

# Controlled experiment figures
python experiments/plotting/plot_controlled.py
```

Figures are saved to `paper_figures/`.

## Run Experiments

**Sensitivity** — vary one parameter, hold others fixed:

```bash
python experiments/sensitivity/exp_vary_latents.py
python experiments/sensitivity/exp_vary_samples.py
python experiments/sensitivity/exp_vary_sparsity.py
```

**Controlled** — decompose SAE failure:

```bash
python experiments/controlled/exp_dict_quality.py       # FISTA on SAE-learned dictionary
python experiments/controlled/exp_warmstart_decoder.py   # SAE decoder as DL-FISTA init
python experiments/controlled/exp_support_recovery.py    # Sparsity pattern recovery
python experiments/controlled/exp_learning_dynamics.py   # Dictionary learning over time
python experiments/controlled/exp_lambda_sensitivity.py  # Regularisation sweep
```

Results are saved incrementally to `results/`.

## Structure

```
src/data.py              # Synthetic data generation (sparse codes + linear mixing)
models/
├── saes.py              # SAE variants (ReLU, TopK, JumpReLU, MP)
├── sparse_coding.py     # FISTA, DL-FISTA, Softplus-Adam, LISTA
└── linear_probe.py      # Supervised linear probe baseline
utils/metrics.py         # MCC, accuracy, AUC, support recovery metrics
experiments/
├── _common.py           # Shared training/evaluation helpers
├── sensitivity/         # Vary latents, samples, sparsity
├── controlled/          # Frozen decoder, warmstart, dict quality, etc.
└── plotting/            # Figure generation scripts
results/                 # Pre-computed experiment results (JSON)
paper_figures/           # Generated figures
```

## Citation

```bibtex
@inproceedings{barinpacela2026stop,
  title={Stop Probing, Start Coding: Why Linear Probes and Sparse
         Autoencoders Fail at Compositional Generalisation},
  author={Barin Pacela, Vit{\'o}ria and Joshi, Shruti and
          Camacho, Isabela and Lacoste-Julien, Simon and Klindt, David},
  booktitle={Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2026}
}
```

## License

MIT
