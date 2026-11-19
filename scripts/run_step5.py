"""Run the Step 5 GPU experiments (the same runs as notebooks 02, 03 and 04) from one script.

    python scripts/run_step5.py 02     # FedProx and SCAFFOLD vs FedAvg
    python scripts/run_step5.py 03     # differential privacy, secure aggregation, membership check
    python scripts/run_step5.py 04     # update compression

Each part clears runs/, runs its experiments through fedsurveil.fl.simulate, and prints one
compact "RUN" line per run: [round, accuracy, macro_f1, worst_site, mean_site] for every round,
plus the final per-site accuracies and the privacy and byte fields. Needs a GPU.
"""

import glob
import json
import os
import statistics
import subprocess
import sys

BASE = [
    "--dataset", "bloodmnist", "--rounds", "30", "--lr", "0.02", "--momentum", "0.9",
    "--local-epochs", "2",
]  # fmt: skip
SEEDS = [0, 1, 2]


def go(*args: str) -> None:
    cmd = [sys.executable, "-m", "fedsurveil.fl.simulate", *BASE, *args]
    subprocess.run(cmd, check=True, capture_output=True)
    print("done", " ".join(args), flush=True)


def clear_runs() -> None:
    for f in glob.glob("runs/*.jsonl"):
        os.remove(f)


def report() -> None:
    for path in sorted(glob.glob("runs/*.jsonl")):
        rows = [json.loads(line) for line in open(path)]
        rounds = [r for r in rows if r["type"] == "round"]
        summary = next((r for r in rows if r["type"] == "summary"), {})
        curve = [
            [
                r["round"],
                round(r["global_acc"], 4),
                round(r["macro_f1"], 4),
                round(r["site_acc_spread"]["min"], 4),
                round(r["site_acc_spread"]["mean"], 4),
            ]
            for r in rounds
        ]
        extra = {
            "final_site_acc": [round(a, 3) for a in rounds[-1]["per_site_acc"]],
            "bytes_up": rounds[0]["bytes_up"],
            "bytes_down": rounds[0]["bytes_down"],
            "eps_client": rounds[-1].get("eps_client"),
            "eps_sample_max": rounds[-1].get("eps_sample_max"),
            "clipped_fraction": rounds[-1].get("clipped_fraction"),
            "mean_update_norm": rounds[-1].get("mean_update_norm"),
            "mia_auroc": summary.get("mia_auroc"),
            "device": rows[0]["device"],
        }
        print("RUN", os.path.basename(path), json.dumps(curve), json.dumps(extra))


def part_02() -> None:
    for seed in SEEDS:
        for alpha in ["0.1", "0.5"]:
            common = ["--alpha", alpha, "--seed", str(seed)]
            go(*common, "--algorithm", "scaffold")
            for mu in ["0.01", "0.1", "1.0"]:
                go(*common, "--algorithm", "fedprox", "--prox-mu", mu)


def part_03() -> None:
    base = ["--alpha", "0.5", "--mia"]
    # Pilot: clip so loosely that nothing is clipped, only to measure typical update norms.
    go(*base, "--seed", "0", "--rounds", "5", "--dp-clip", "1000000")
    pilot = [json.loads(line) for line in open(glob.glob("runs/*seed0.jsonl")[0])]
    norms = [r["mean_update_norm"] for r in pilot if r["type"] == "round"]
    clip = round(statistics.median(norms), 2)
    print("update norms", [round(n, 2) for n in norms], "-> clip C =", clip, flush=True)
    clear_runs()
    for seed in SEEDS:
        go(*base, "--seed", str(seed))
        go(*base, "--seed", str(seed), "--secagg")
        for sigma in ["0.3", "1.0", "3.0"]:
            go(*base, "--seed", str(seed), "--dp-clip", str(clip), "--dp-noise", sigma)
    go(*base, "--seed", "0", "--site-dp-noise", "1.0", "--site-dp-clip", "1.0")


def part_04() -> None:
    for seed in SEEDS:
        common = ["--alpha", "0.5", "--seed", str(seed)]
        go(*common, "--compress", "int8")
        go(*common, "--compress", "topk", "--topk-fraction", "0.1")
        go(*common, "--compress", "topk", "--topk-fraction", "0.01")


if __name__ == "__main__":
    part = sys.argv[1]
    parts = {"02": part_02, "03": part_03, "04": part_04}
    clear_runs()
    parts[part]()
    report()
