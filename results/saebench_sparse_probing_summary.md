# dl_fista vs TopK SAE — Sparse Probing on Gemma-2-2B layer 12

All values are **% test accuracy** (averaged across classes within a dataset, then mean ± std across seeds 0/1/2).

Setup: gemma-2-2b layer 12 residual stream, d_sae=16384, λ=0.1, FISTA inner=200 at eval, 8 SAEBench sparse-probing datasets, K-sweep aggregated to overall sae_test_accuracy.

## Per-dataset results

| Dataset | dl_fista (sae) | TopK (sae) | LLM linear probe | Δ (dl_fista − TopK) |
|---------|----------------|------------|-------------------|---------------------|
| `bias_in_bios_class_set1` | **96.96 ± 0.15** | 95.71 ± 0.51 | 96.86 | **+1.25** |
| `bias_in_bios_class_set2` | **95.53 ± 0.25** | 93.55 ± 0.08 | 95.28 | **+1.99** |
| `bias_in_bios_class_set3` | **93.47 ± 0.08** | 92.21 ± 0.06 | 93.11 | **+1.26** |
| `amazon_reviews_mcauley_1and5` | **91.89 ± 0.40** | 89.65 ± 0.26 | 91.55 | **+2.24** |
| `amazon_reviews_mcauley_1and5_sentiment` | **96.97 ± 0.24** | 95.45 ± 0.80 | 96.80 | **+1.52** |
| `github-code` | **96.71 ± 0.15** | 96.33 ± 0.33 | 96.83 | **+0.39** |
| `ag_news` | **94.93 ± 0.51** | 94.62 ± 0.35 | 94.55 | **+0.32** |
| `europarl` | **99.87 ± 0.09** | 99.50 ± 0.14 | 99.94 | **+0.37** |
| **OVERALL** (mean of 8 datasets) | **95.79 ± 0.11** | 94.63 ± 0.08 | 95.57 | **+1.17** |

## Per-seed overall (mean across 8 datasets)

| Seed | dl_fista (sae) | TopK (sae) | LLM linear probe |
|------|----------------|------------|-------------------|
| 0 | 95.67 | 94.63 | 95.58 |
| 1 | 95.88 | 94.71 | 95.57 |
| 2 | 95.83 | 94.54 | 95.55 |
| **mean ± std** | **95.79 ± 0.11** | 94.63 ± 0.08 | 95.57 ± 0.01 |

## Headline

- **dl_fista beats TopK on all 8 datasets**, by 0.32 – 2.24 pp; overall gap **+1.17 pp**.
- **dl_fista even slightly beats the raw-LLM linear probe overall** (**+0.22 pp**) — its latents preserve essentially all the linearly-decodable label info.
- Cross-seed variance is tight for all three (≤ 0.11 pp std on the overall mean).

## Caveat

dl_fista uses ~1900 active latents per token (out of 16384) vs TopK's fixed K=320 — dl_fista has ~6× more nonzero features available to the linear probe, so this is **not** a sparsity-matched comparison. To make a head-to-head at matched sparsity: either sweep λ in dl_fista to match L0, or compare top-K subset accuracies (the per-dataset JSONs contain `sae_top_K_test_accuracy` for K ∈ {1, 2, 5, 10, 20, 50, 100}).