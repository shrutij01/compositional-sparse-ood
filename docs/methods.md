# Methods

## Problem Setup

We observe $\mathbf{y} = A\mathbf{z}$ where $A$ is an unknown mixing matrix and $\mathbf{z}$ is $k$-sparse. The goal: recover $\mathbf{z}$ from $\mathbf{y}$, including on OOD combinations of active latents never seen during training.

---

## Sparse Coding

### FISTA (oracle)

Fast Iterative Shrinkage-Thresholding Algorithm. Solves:

$$\min_{\mathbf{z}} \frac{1}{2}\|\mathbf{y} - D\mathbf{z}\|^2 + \lambda\|\mathbf{z}\|_1$$

Given a **fixed dictionary** $D$, FISTA iteratively applies gradient descent on the reconstruction term then soft-thresholds to enforce sparsity. Uses Nesterov momentum for $O(1/k^2)$ convergence (vs $O(1/k)$ for plain ISTA). When $D = A$ (the ground-truth), this is the oracle baseline.

### DL-FISTA (dictionary learning)

Extends FISTA to the **unsupervised** setting where $D$ is unknown. Alternates two steps:

1. **Infer codes** $Z$ given current $D$ using FISTA
2. **Update dictionary** $D$ via least-squares: $D = X^\top Z (Z^\top Z)^{-1}$

This is non-convex — results depend on initialisation. The key question the paper asks: can we learn a good $D$ at scale?

### Softplus-Adam

Baseline that directly optimises pre-activation codes with Adam. Codes are parameterised as $\mathbf{z} = \text{softplus}(\tilde{\mathbf{z}})$ to enforce non-negativity, then jointly optimised with the dictionary.

### LISTA

Learned ISTA: unrolls ISTA for a fixed number of steps (e.g., 16), making each step's parameters (weights, thresholds) learnable. Initialised from analytical ISTA, then trained end-to-end. After training, inference is a single forward pass — no iterations needed.

---

## Sparse Autoencoders

All SAE variants use an encoder-decoder architecture: $\mathbf{z} = f(\mathbf{W}_e \mathbf{y} + \mathbf{b}_e)$, $\hat{\mathbf{y}} = \mathbf{W}_d \mathbf{z} + \mathbf{b}_d$. They differ in how the sparsity function $f$ works:

| Variant | Sparsity mechanism | Regularisation |
|---------|-------------------|----------------|
| **ReLU** | $\text{ReLU}(\cdot)$ | L1 penalty on codes |
| **TopK** | Keep $k$ largest activations, zero the rest | None needed (structural) |
| **JumpReLU** | Learnable per-dimension threshold | Penalty on active count |
| **MP** | Matching pursuit: iteratively select decoder columns by correlation with residual | None (exactly $k$ atoms) |

---

## Controlled Experiments ("Surgeries")

These experiments decompose *why* SAEs fail at OOD generalisation by swapping components between methods.

### Frozen Decoder

!!! question "Question: Is the SAE decoder a bad dictionary?"

Take a trained SAE's decoder and use it as a fixed dictionary for FISTA inference (replacing the SAE encoder). Variants:

- **frozen**: raw SAE decoder columns as dictionary
- **renormed**: SAE decoder with columns normalised to unit norm
- **oracle norms**: SAE decoder directions but ground-truth column magnitudes

If FISTA on the frozen decoder still fails, the problem is the **dictionary directions**, not the encoder. If renorming or fixing magnitudes helps, the issue is scaling.

### Warm-Start Decoder

!!! question "Question: Is the SAE decoder at least a useful initialisation?"

Use the SAE decoder to initialise DL-FISTA instead of a random dictionary. Compare convergence:

- **warm-start**: DL-FISTA from SAE decoder
- **cold-start**: DL-FISTA from random initialisation

Since dictionary learning is non-convex, the SAE decoder might lead to better or worse local minima.

### Warm-Start Encoder

!!! question "Question: Does the SAE encoder give FISTA a head start?"

Use SAE encoder output as the initial $\mathbf{z}_0$ for FISTA iterations (instead of starting from zero). Since code inference with a fixed dictionary is convex, both initialisations converge to the same solution — the question is how many iterations the SAE encoder "saves".

### Support Recovery

!!! question "Question: Does the SAE activate the right features?"

Decompose SAE errors into two sources:

- **Support errors**: activating wrong features (wrong sparsity pattern)
- **Magnitude errors**: right features, wrong values

Test by taking the SAE's binary support pattern and re-estimating magnitudes via least-squares. If this helps, the SAE finds the right features but gets magnitudes wrong. If it doesn't help, the SAE picks the wrong features entirely.

### Learning Dynamics

!!! question "Question: When does dictionary learning go wrong?"

Track dictionary quality (cosine similarity to ground truth) throughout training for both SAEs and DL-FISTA. Reveals whether dictionaries converge then drift, plateau at a bad minimum, or never improve.
