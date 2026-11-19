# Step 4 check: does non-IID data hurt FedAvg? (BloodMNIST, seed 0)

Raw per-round numbers: [step4_bloodmnist_seed0.json](step4_bloodmnist_seed0.json).
Setup: 10 sites, 30 rounds, 2 local epochs per round, lr 0.02, momentum 0.9, batch 64, small CNN with group norm, Tesla T4. The central baseline gets the same 2 epochs per round over the pooled data.

| Setting | Final accuracy | Macro-F1 | Worst-site accuracy | First round at 80% accuracy |
|---|---|---|---|---|
| Central (pooled data) | 94.8% | 0.941 | 94.0% | 1 |
| FedAvg, alpha 100 (near-IID) | 90.3% | 0.884 | 89.8% | 11 |
| FedAvg, alpha 0.5 | 88.1% | 0.852 | 70.8% | 13 |
| FedAvg, alpha 0.1 (strongly non-IID) | 53.6% | 0.448 | 5.6% | never (best 67.9%) |

**Answer to the PLAN.md week-2 check: yes, non-IID data visibly hurts.** At alpha 0.1 FedAvg loses about 41 points of accuracy against central training, its accuracy swings between about 50% and 68% from round to round, and the worst site is close to 0%. At alpha 0.5 the overall number looks fine (88%) but the worst site is 71%, so a single global accuracy hides the damage.

**Caveats (read before quoting these numbers)**
- One seed only. Seeds 1 and 2 are not in this file yet, so there are no confidence intervals.
- 30 rounds is not converged: FedAvg at alpha 100 was still improving (90.3% and rising), so its gap to central training (4.5 points) is probably overstated.
- The learning rate, momentum and local epochs were picked once and not tuned.
- Per-site accuracy uses 500 test images per site sampled with replacement from the 3,421 test images, so the worst-site number is noisy.
- BloodMNIST's test set has 5 images that also appear in training (see data_report.md).
- Only BloodMNIST was run. PathMNIST and IP102 are not run yet.
