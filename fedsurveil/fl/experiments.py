"""The comparison run on each extra dataset, and a compact summary of one finished run log.

Same protocol as Step 4 and Step 5 on BloodMNIST: 10 sites, 30 rounds, 2 local epochs, lr 0.02,
momentum 0.9. Nothing was re-tuned for the new datasets.
"""

import json
from pathlib import Path

BASE = [
    "--rounds", "30", "--lr", "0.02", "--momentum", "0.9", "--local-epochs", "2",
]  # fmt: skip
ALPHAS = ("0.1", "0.5", "100")
DRIFT_ALPHAS = ("0.1", "0.5")
PROX_MU = "0.1"  # the best FedProx value on BloodMNIST; 0.01 and 1.0 are not repeated here


def dataset_runs(dataset: str, seeds: tuple[int, ...] = (0, 1, 2)) -> list[list[str]]:
    """Simulator arguments for every run: central, FedAvg at 3 skews, FedProx and SCAFFOLD."""
    runs = []
    for seed in seeds:
        common = ["--dataset", dataset, *BASE, "--seed", str(seed)]
        runs.append([*common, "--algorithm", "central"])
        for alpha in ALPHAS:
            runs.append([*common, "--algorithm", "fedavg", "--alpha", alpha])
        for alpha in DRIFT_ALPHAS:
            runs.append([*common, "--algorithm", "fedprox", "--prox-mu", PROX_MU, "--alpha", alpha])
            runs.append([*common, "--algorithm", "scaffold", "--alpha", alpha])
    return runs


def group_name(cfg: dict) -> str:
    """Key shared by the seeds of one setting, e.g. fedavg_a0.5 or fedprox-mu0.1_a0.1."""
    algo = cfg["algorithm"]
    if algo == "central":
        return "central"
    if algo == "fedprox":
        algo = f"fedprox-mu{cfg['prox_mu']:g}"
    return f"{algo}_a{cfg['alpha']:g}"


def summarize_log(path: Path) -> dict:
    """Final-round values of one run log (same fields as results/step4_*_3seeds.json)."""
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    cfg = next(r for r in rows if r["type"] == "meta")["config"]
    rounds = [r for r in rows if r["type"] == "round"]
    last = rounds[-1]
    return {
        "group": group_name(cfg),
        "seed": cfg["seed"],
        "dataset": cfg["dataset"],
        "rounds": last["round"],
        "final_acc": round(last["global_acc"], 4),
        "final_macro_f1": round(last["macro_f1"], 4),
        "final_worst_site_acc": round(last["site_acc_spread"]["min"], 4),
        "last5_mean_acc": round(sum(r["global_acc"] for r in rounds[-5:]) / len(rounds[-5:]), 4),
        "bytes_up_round": last.get("bytes_up", 0),
        "device": rows[0].get("device"),
    }
