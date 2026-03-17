# Training Results — EfficientNet-B3

## Final Results

| Metric | Value |
|---|---|
| Best validation kappa | 0.9053 |
| Best validation accuracy | 83.6% |
| Best training kappa | 0.9836 |
| Epochs trained | 15 |
| Dataset | APTOS 2019 — 3,662 images |
| Device | NVIDIA T4 GPU (Google Colab) |

## Epoch History

| Epoch | Train Loss | Train Acc | Train Kappa | Val Loss | Val Acc | Val Kappa |
|---|---|---|---|---|---|---|
| 1 | 1.3350 | 0.576 | 0.5818 | 1.0128 | 0.675 | 0.7930 |
| 2 | 0.8776 | 0.729 | 0.8385 | 0.8256 | 0.756 | 0.8717 |
| 3 | 0.7139 | 0.802 | 0.8931 | 0.8159 | 0.776 | 0.8968 |
| 4 | 0.5934 | 0.818 | 0.9070 | 0.8297 | 0.816 | 0.8946 |
| 5 | 0.5140 | 0.858 | 0.9149 | 0.8489 | 0.795 | 0.8965 |
| 6 | 0.4244 | 0.883 | 0.9364 | 0.9484 | 0.820 | 0.8971 |
| 7 | 0.3466 | 0.900 | 0.9441 | 1.0289 | 0.805 | 0.8854 |
| 8 | 0.2689 | 0.922 | 0.9562 | 1.0473 | 0.809 | 0.8778 |
| 9 | 0.2057 | 0.937 | 0.9642 | 1.2117 | 0.802 | 0.8690 |
| 10 | 0.2029 | 0.943 | 0.9656 | 1.2268 | 0.814 | 0.8874 |
| 11 | 0.1780 | 0.948 | 0.9729 | 1.2420 | 0.821 | 0.9017 |
| 12 | 0.1256 | 0.967 | 0.9834 | 1.2970 | 0.834 | 0.8980 |
| 13 | 0.1114 | 0.968 | 0.9836 | 1.4430 | 0.836 | 0.9053 |
| 14 | 0.1196 | 0.965 | 0.9808 | 1.4192 | 0.819 | 0.8912 |
| 15 | 0.1317 | 0.965 | 0.9794 | 1.5145 | 0.831 | 0.9023 |

## Context

- APTOS 2019 Kaggle competition winner: ~0.93 kappa
- This model: 0.9053 kappa
- Random baseline: 0.0 kappa
- Model architecture: EfficientNet-B3 + custom head
- Training strategy: full fine-tuning, weighted loss for class imbalance
