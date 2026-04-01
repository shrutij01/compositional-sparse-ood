# Compositional Sparse OOD

Code for **"Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation"** (UAI 2026).

*Vitoria Barin Pacela\*, Shruti Joshi\*, Isabela Camacho, Simon Lacoste-Julien, David Klindt*

## Key findings

- **Per-sample sparse inference** (FISTA with the ground-truth dictionary) achieves near-perfect OOD compositional generalisation at all scales tested.
- **SAEs** fail because their learned dictionaries point in wrong directions — replacing the encoder with FISTA on the same dictionary does not help.
- **Linear probes** degrade sharply under superposition (MCC < 0.1 at $d_z$ = 10,000).
- **Dictionary learning** is the bottleneck: DL-FISTA beats linear probes when it succeeds but fails at scale.

## Quick start

```bash
pip install -e .
```

Requires Python >= 3.10 and PyTorch.

## Reproducing paper figures

All results are pre-computed in `results/`. To regenerate figures:

```bash
# Main text + appendix figures
python experiments/plotting/plot_paper_figures.py --only v2

# Controlled experiment figures
python experiments/plotting/plot_controlled.py
```

Figures are saved to `paper_figures/`.

## Running experiments

### Sensitivity (vary one parameter)

```bash
python experiments/sensitivity/exp_vary_latents.py
python experiments/sensitivity/exp_vary_samples.py
python experiments/sensitivity/exp_vary_sparsity.py
```

### Controlled (decompose SAE failure)

```bash
python experiments/controlled/exp_dict_quality.py
python experiments/controlled/exp_warmstart_decoder.py
python experiments/controlled/exp_support_recovery.py
python experiments/controlled/exp_learning_dynamics.py
python experiments/controlled/exp_lambda_sensitivity.py
```

Results are saved incrementally to `results/`.

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
