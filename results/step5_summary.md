# Step 5 results (BloodMNIST, 10 sites, 30 rounds, mean of 3 seeds)

Accuracy is the final-round global test accuracy. Worst site is the lowest per-site accuracy.
Raw per-run values: `step5_bloodmnist_3seeds.json`. Baselines at alpha 0.1 come from Step 4
(FedAvg 0.647, worst site 0.099; central 0.941).

## Client drift fixes
| alpha | method | accuracy | worst site |
|---|---|---|---|
| 0.1 | FedAvg (Step 4) | 0.647 | 0.099 |
| 0.1 | FedProx mu 0.01 | 0.674 | 0.100 |
| 0.1 | FedProx mu 0.1 | 0.686 | 0.285 |
| 0.1 | FedProx mu 1.0 | 0.588 | 0.073 |
| 0.1 | SCAFFOLD (2x upload) | 0.719 | 0.296 |
| 0.5 | FedAvg | 0.888 | 0.761 |
| 0.5 | FedProx mu 0.01 / 0.1 / 1.0 | 0.887 / 0.872 / 0.744 | 0.765 / 0.734 / 0.463 |
| 0.5 | SCAFFOLD (2x upload) | 0.887 | 0.797 |

SCAFFOLD helps most under strong skew (alpha 0.1) but the seed spread there is 0.05-0.10, so
treat the gains as indicative. At alpha 0.5 nothing beats FedAvg by more than seed noise (about 0.01).

## Privacy (alpha 0.5)
| setting | accuracy | guarantee |
|---|---|---|
| no privacy | 0.888 | none |
| simulated secure aggregation | 0.891 | none by itself (simulation, not cryptographic) |
| client-level DP, sigma 0.3 | 0.688 | epsilon 253 (very weak) |
| client-level DP, sigma 1.0 | 0.408 | epsilon 40 |
| client-level DP, sigma 3.0 | 0.144 | epsilon 9.6 |
| sample-level DP-SGD, sigma 1.0 (1 seed) | 0.648 | epsilon 39 per sample |

Meaningful client-level DP with 10 sites destroys accuracy. Membership-inference AUROC is about 0.50-0.52
in every run, including the non-private baseline, so this attack found no leakage even without DP.

## Compression (alpha 0.5)
| scheme | upload per round, all sites | accuracy | worst site |
|---|---|---|---|
| none | 3.79 MB | 0.888 | 0.761 |
| int8 | 0.95 MB (4x) | 0.882 | 0.753 |
| top-k 10% | 0.76 MB (5x) | 0.876 | 0.723 |
| top-k 1% | 0.076 MB (50x) | 0.834 | 0.668 |
