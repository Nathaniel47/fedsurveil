# Same comparison on every dataset (10 sites, 30 rounds, 3 seeds)

Final-round mean ± standard deviation over seeds 0, 1 and 2 (rough: 3 seeds). All datasets use the settings chosen on BloodMNIST; nothing was re-tuned. Worst site is the lowest per-site accuracy. No privacy mechanism is on in any row.

| Setting | bloodmnist acc (%) | bloodmnist worst site (%) | organamnist acc (%) | organamnist worst site (%) | pathmnist acc (%) | pathmnist worst site (%) | ip102 acc (%) | ip102 worst site (%) |
|---|---|---|---|---|---|---|---|---|
| Central (pooled data) | 94.1 ± 0.9 | 92.3 ± 1.8 | 92.9 ± 0.1 | 87.7 ± 1.8 | 86.4 ± 1.0 | 72.7 ± 5.3 | 57.7 ± 0.7 | 43.5 ± 3.8 |
| FedAvg, alpha 100 | 90.8 ± 1.1 | 89.6 ± 1.1 | 91.9 ± 0.7 | 90.2 ± 0.6 | 87.0 ± 1.3 | 80.7 ± 2.2 | 47.0 ± 0.6 | 43.8 ± 1.9 |
| FedAvg, alpha 0.5 | 88.6 ± 1.0 | 76.2 ± 5.4 | 91.1 ± 0.8 | 84.1 ± 4.1 | 86.5 ± 0.3 | 70.7 ± 2.1 | 40.0 ± 1.6 | 24.5 ± 0.9 |
| FedAvg, alpha 0.1 | 64.7 ± 9.6 | 9.9 ± 4.5 | 84.9 ± 5.9 | 62.7 ± 18.5 | 82.5 ± 2.2 | 48.9 ± 3.3 | 39.4 ± 8.4 | 9.1 ± 9.6 |
| FedProx mu 0.1, alpha 0.5 | 87.2 ± 1.2 | 73.4 ± 4.4 | 88.1 ± 0.3 | 78.9 ± 0.4 | 83.2 ± 1.1 | 64.4 ± 2.8 | 39.1 ± 2.0 | 22.3 ± 4.2 |
| FedProx mu 0.1, alpha 0.1 | 68.5 ± 9.9 | 28.5 ± 17.5 | 80.5 ± 3.4 | 55.2 ± 11.4 | 78.6 ± 1.8 | 31.1 ± 2.9 | 38.3 ± 1.9 | 8.1 ± 5.7 |
| SCAFFOLD, alpha 0.5 | 88.6 ± 1.1 | 79.7 ± 5.1 | 91.0 ± 0.4 | 83.3 ± 2.0 | 84.2 ± 1.2 | 65.9 ± 1.6 | 37.4 ± 4.0 | 20.3 ± 10.3 |
| SCAFFOLD, alpha 0.1 | 71.9 ± 4.6 | 29.6 ± 18.1 | 74.0 ± 25.8 | 57.9 ± 40.3 | 75.2 ± 9.6 | 30.1 ± 21.1 | 20.8 ± 6.9 | 0.5 ± 0.3 |
