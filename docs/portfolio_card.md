# Portfolio card: FedSurveil

**Title:** FedSurveil: privacy-preserving federated learning, simulated

**One line:** Ten simulated clinics train one image classifier without sharing images, and I measured
what skewed data, privacy and low bandwidth each cost.

**Details (3 lines):**
- Built FedAvg, FedProx and SCAFFOLD, client-level DP and DP-SGD, a simulated secure aggregation,
  a membership-inference check and update compression in plain PyTorch, with a Flower demo run as
  separate processes and a Streamlit replay dashboard.
- On BloodMNIST (3 seeds): FedAvg falls from 90.8% to 64.7% as sites get more skewed, with the worst
  site at 10%; SCAFFOLD recovers about 7 points; strong privacy (epsilon 40) cuts accuracy to 41%;
  50x smaller uploads cost about 5 points.
- Honest limits: one dataset, three seeds, a simulation, no clinical claims.

**Tags:** PyTorch, federated learning, differential privacy, Streamlit, Flower
**Links:** GitHub repo (github.com/Nathaniel47/fedsurveil), report (docs/REPORT.md), live dashboard (add after deploy)
