# Step 4 check: does non-IID data hurt FedAvg? (BloodMNIST, 3 seeds)

Per-run numbers: [step4_bloodmnist_3seeds.json](step4_bloodmnist_3seeds.json); seed 0 per-round curves: [step4_bloodmnist_seed0.json](step4_bloodmnist_seed0.json).
Setup: 10 sites, 30 rounds, 2 local epochs per round, lr 0.02, momentum 0.9, batch 64, small CNN with group norm, Tesla T4. The central baseline gets the same 2 epochs per round over the pooled data. Values are mean ± standard deviation over seeds 0, 1 and 2 (3 seeds is few, so read the spreads as rough).

| Setting | Accuracy, mean of last 5 rounds (%) | Final-round accuracy (%) | Final-round macro-F1 | Worst-site accuracy, final round (%) |
|---|---|---|---|---|
| Central (pooled data) | 94.8 ± 0.3 | 94.1 ± 0.9 | 0.936 ± 0.011 | 92.3 ± 1.8 |
| FedAvg, alpha 100 (near-IID) | 90.3 ± 1.2 | 90.8 ± 1.1 | 0.892 ± 0.014 | 89.6 ± 1.1 |
| FedAvg, alpha 0.5 | 87.5 ± 1.2 | 88.6 ± 1.0 | 0.862 ± 0.018 | 76.2 ± 5.4 |
| FedAvg, alpha 0.1 (strongly non-IID) | 60.2 ± 3.2 | 64.7 ± 9.6 | 0.544 ± 0.085 | 9.9 ± 4.5 |

**Answer to the PLAN.md week-2 check: yes, non-IID data visibly hurts, and it holds across all 3 seeds.** At alpha 0.1 FedAvg is about 35 points below central training on the last-5-round average, its accuracy swings by 10 or more points from round to round, and the worst site is near 10%. At alpha 0.5 the overall number looks fine (87.5%) but the worst site is 76%, so a single global accuracy hides the damage.

**Caveats (read before quoting these numbers)**
- 3 seeds only: the spreads are rough, and no formal confidence interval was computed.
- 30 rounds is not converged: FedAvg at alpha 100 was still improving, so its gap to central training (about 4.5 points) is probably overstated.
- The learning rate, momentum and local epochs were picked once and not tuned.
- Per-site accuracy uses 500 test images per site sampled with replacement from the 3,421 test images, so the worst-site number is noisy. Seeds 1 and 2 use different site splits.
- BloodMNIST's test set has 5 images that also appear in training (see data_report.md).
- Only BloodMNIST was run. PathMNIST and IP102 are not run yet.
