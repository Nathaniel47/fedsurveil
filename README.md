# FedSurveil: Privacy-Preserving Federated Surveillance

A simulation of federated learning across 5–20 clinics or field sites. It trains one image classifier without pooling the raw images, and compares FedAvg, FedProx and SCAFFOLD against central training. It adds differential privacy, a simulated secure aggregation, a membership-inference check and a communication-cost analysis, with a Streamlit dashboard.

Status: data pipeline, FedAvg/FedProx/SCAFFOLD, differential privacy, simulated secure aggregation and the dashboard are built. BloodMNIST FedAvg vs central is run (3 seeds); FedProx, SCAFFOLD, privacy, compression and PathMNIST/IP102 runs are still to do on a free GPU (notebooks 02-04). See [PLAN.md](PLAN.md) for the design.

This is a single-machine simulation on tiny public images. It makes no clinical claims, and its secure aggregation is simulated masking, not real cryptography.

## Run locally (laptop: tests and the dashboard, no GPU)

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
streamlit run app/dashboard.py
```

macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app/dashboard.py
```

Docker: `docker compose up` serves the dashboard on http://localhost:8501.

## Training (Kaggle, Colab or a local GPU)

Training needs PyTorch, Opacus and Flower: `pip install -r requirements-ml.txt`. Runs write JSONL logs to `runs/`, and the dashboard reads those.

Step 4 runner: [notebooks/01_noniid_fedavg.ipynb](notebooks/01_noniid_fedavg.ipynb) (turn on a GPU, keep `SMOKE = True` for the first run). Or one run from the command line:

```bash
python -m fedsurveil.fl.simulate --dataset bloodmnist --algorithm fedavg --alpha 0.1 --num-sites 10 --rounds 30
python -m fedsurveil.fl.simulate --dataset bloodmnist --algorithm central --rounds 30
```

Other commands: `streamlit run app/dashboard.py` (replay dashboard) and `python -m fedsurveil.eval.comms_report` (writes results/comms_report.md). Options on the simulator: `--algorithm fedavg|fedprox|scaffold|central`, `--dp-clip/--dp-noise` (client-level DP), `--site-dp-noise` (DP-SGD), `--secagg` (simulated), `--compress int8|topk`, `--mia`.

Tests that need torch skip themselves when torch is missing, so CI (no torch) runs the rest.

## Build the data

```bash
python -m fedsurveil.data.fetch     # MedMNIST 28 px, about 280 MB, md5 checked against Zenodo
python -m fedsurveil.data.prepare   # seeded Dirichlet site splits -> data/splits/ (gitignored)
python -m fedsurveil.data.report    # -> results/data_report.md
```

Raw files and split indices are gitignored. The repo holds checksums (`data/manifest.json`), one short hash per split (`data/split_digests.json`) and the report ([results/data_report.md](results/data_report.md)).
IP102 is a manual 3 GB download: get `ip102_v1.1` from the [IP102 repo](https://github.com/xpwu95/IP102), extract it, then run `python -m fedsurveil.data.ip102 --root <folder>` to build the 10-class 64 px subset, and re-run `prepare` and `report`.

## Data and licences (checked 2026-11-18)

| Data / library | Licence | Use here |
|---|---|---|
| MedMNIST v2 data | CC-BY-4.0, except DermaMNIST (CC-BY-NC-4.0, not used); not for clinical use | PathMNIST, BloodMNIST, OrganAMNIST |
| MedMNIST code | Apache-2.0 | `pip install medmnist` (kept below 3.0) |
| IP102 | "Free for academic usage"; other uses need permission from the authors | 10–20 class subset, 64 px; not redistributed, no images in the public dashboard |
| Flower, Opacus | Apache-2.0 (to confirm at install) | Flower demo run; DP-SGD |

## Responsible use

Public benchmark data only. The study measures model accuracy, privacy leakage and communication cost. It is not a surveillance tool: the name refers to disease and pest surveillance by simulated health and field sites.
