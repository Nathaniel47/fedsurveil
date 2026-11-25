# Portfolio card: FedSurveil

**Title:** FedSurveil: privacy-preserving federated learning, simulated

**One line:** Ten simulated clinics train one image classifier without sharing images, and I measured
what skewed data, privacy and low bandwidth each cost.

**Details (3 lines):**
- Built FedAvg, FedProx and SCAFFOLD, client-level DP and DP-SGD, a simulated secure aggregation,
  a membership-inference check and update compression in plain PyTorch, with a Flower demo run as
  separate processes and a Streamlit replay dashboard.
- On BloodMNIST (3 seeds), FedAvg falls from 90.8% to 64.7% as sites get more skewed (worst site
  10%) and SCAFFOLD recovers about 7 points, a gain that did not repeat on three other datasets.
  Client-level DP at epsilon 40 (delta 1e-5, still a weak bound) cuts accuracy to 41%; 50x smaller
  uploads cost about 5 points.
- Honest limits: three seeds, settings chosen on one dataset, privacy and compression tested on
  BloodMNIST only, secure aggregation simulated (not cryptography), no clinical claims.

**Tags:** PyTorch, federated learning, differential privacy, Streamlit, Flower
**Links:** GitHub repo (github.com/Nathaniel47/fedsurveil), report (docs/REPORT.md),
[live dashboard](https://fedsurveil-smwgy3kxghui5lqha38jve.streamlit.app)
