# FedSurveil: Privacy-Preserving Federated Surveillance

**Live dashboard:** https://fedsurveil-smwgy3kxghui5lqha38jve.streamlit.app (results only; a sleeping app may take a minute to wake). The replay tabs show the BloodMNIST seed 0 runs; the comparison across datasets is in the "Final results" tab.

A simulation of federated learning across 5–20 clinics or field sites. It trains one image classifier without pooling the raw images, and compares FedAvg, FedProx and SCAFFOLD against central training. It adds differential privacy, a simulated secure aggregation, a membership-inference check and a communication-cost analysis, with a Streamlit dashboard.

Status: everything in the plan is built and run on BloodMNIST (10 sites, 3 seeds): FedAvg, FedProx, SCAFFOLD, client-level DP, DP-SGD, simulated secure aggregation, a membership-inference check, update compression, a dashboard and a Flower end-to-end demo. The central / FedAvg / FedProx / SCAFFOLD comparison was repeated on OrganAMNIST, PathMNIST and a 10-class IP102 insect-pest subset; privacy, compression and the membership-inference check are BloodMNIST only. See [docs/REPORT.md](docs/REPORT.md) for results and limitations, and [PLAN.md](PLAN.md) for the design.

This is a single-machine simulation on tiny public images. It makes no clinical claims, and its secure aggregation is simulated masking, not real cryptography.

## Results at a glance (BloodMNIST, 10 sites, 3 seeds; full table in [results/results_table.md](results/results_table.md))

| Setting | Accuracy | Worst site |
|---|---|---|
| Central training | 94.1 ± 0.9% | 92.3 ± 1.8% |
| FedAvg, near-IID sites (alpha 100) | 90.8 ± 1.1% | 89.6 ± 1.1% |
| FedAvg, alpha 0.5 | 88.6 ± 1.0% | 76.2 ± 5.4% |
| FedAvg, alpha 0.1 | 64.7 ± 9.6% | 9.9 ± 4.5% |
| SCAFFOLD, alpha 0.1 | 71.9 ± 4.6% | 29.6 ± 18.1% |
| Client-level DP (epsilon 40) | 40.8 ± 10.7% | 14.9 ± 8.4% |
| top-k 1% compression (50x less upload) | 83.4 ± 1.1% | 66.8 ± 9.8% |

Under FedAvg, strongly skewed sites leave the worst site far behind on all four datasets; SCAFFOLD helped under strong skew on BloodMNIST only, was unstable on OrganAMNIST and PathMNIST, and did far worse than FedAvg on the IP102 subset ([comparison](results/cross_dataset_table.md)); strong privacy costs most of the accuracy on a model this small; compression is cheap to about 10x. Three seeds is few, and the report lists the limits.

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

Flower end-to-end demo (CPU is enough, about 5 minutes): `python scripts/run_flower_demo.py` starts a SuperLink and three SuperNodes as separate processes on localhost (no TLS), runs 5 rounds of FedAvg on a 3,000-image BloodMNIST subset (no privacy), and writes `results/flower_demo.json`. It shows the real protocol; every experiment in the results uses the plain-PyTorch simulator instead. The first run makes Flower create `~/.flwr/config.toml`, with a connection named `local-deployment` that points at `127.0.0.1:9093`.

Step 4 runner: [notebooks/01_noniid_fedavg.ipynb](notebooks/01_noniid_fedavg.ipynb) (turn on a GPU, keep `SMOKE = True` for the first run). Or one run from the command line:

```bash
python -m fedsurveil.fl.simulate --dataset bloodmnist --algorithm fedavg --alpha 0.1 --num-sites 10 --rounds 30
python -m fedsurveil.fl.simulate --dataset bloodmnist --algorithm central --rounds 30
```

Other datasets: [notebooks/05_other_datasets.ipynb](notebooks/05_other_datasets.ipynb) runs central, FedAvg, FedProx and SCAFFOLD on OrganAMNIST, PathMNIST and the IP102 subset (24 runs each, same settings as BloodMNIST) through `python scripts/run_datasets.py <dataset>`. IP102 is read from the path in `IP102_NPZ_PATH`; it is not downloaded for you.

Other commands: `streamlit run app/dashboard.py` (replay dashboard) and `python -m fedsurveil.eval.comms_report` (writes results/comms_report.md) and `python -m fedsurveil.eval.report` (writes results/results_table.md and the charts). Options on the simulator: `--algorithm fedavg|fedprox|scaffold|central`, `--dp-clip/--dp-noise` (client-level DP), `--site-dp-noise` (DP-SGD), `--secagg` (simulated), `--compress int8|topk`, `--mia`.

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
| IP102 | "Free for academic usage"; other uses need permission from the authors | 10-class subset, 64 px; not redistributed, no images in the public dashboard |
| Flower | Apache-2.0 (confirmed for flwr 1.44.0) | Flower demo run |
| Opacus | Apache-2.0 (to confirm at install) | DP-SGD |

## Responsible use

Public benchmark data only. The study measures model accuracy, privacy leakage and communication cost. It is not a surveillance tool: the name refers to disease and pest surveillance by simulated health and field sites.
