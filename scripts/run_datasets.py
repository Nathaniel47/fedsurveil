"""Run the Step 4 + FedProx/SCAFFOLD comparison on one more dataset (needs a GPU).

    python scripts/run_datasets.py organamnist     # about 35 min on a Colab T4 (estimate)
    python scripts/run_datasets.py pathmnist       # about 90 min (estimate)
    python scripts/run_datasets.py ip102           # needs IP102_NPZ_PATH; time unknown, see below

24 runs per dataset: central, FedAvg at alpha 0.1, 0.5 and 100, FedProx (mu 0.1) and SCAFFOLD at
alpha 0.1 and 0.5, for seeds 0, 1 and 2. After every run it prints one compact line:

    SUMMARY <run name> <json with final accuracy, macro-F1, worst site, ...> seconds=<n>

so a session that disconnects still leaves the finished runs visible in the notebook output.
The elapsed time of each run is printed, so after two or three runs you can judge the total.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedsurveil.fl.experiments import dataset_runs, summarize_log  # noqa: E402

DATASETS = ("organamnist", "pathmnist", "ip102")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        print(__doc__)
        return 2
    dataset = sys.argv[1]
    runs = dataset_runs(dataset)
    for i, args in enumerate(runs, 1):
        started = time.time()
        done = subprocess.run(
            [sys.executable, "-m", "fedsurveil.fl.simulate", *args],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if done.returncode != 0:
            print("FAILED", " ".join(args), "\n", done.stderr[-1500:], flush=True)
            return 1
        log = Path(done.stdout.strip().splitlines()[-1])
        summary = summarize_log(log if log.is_absolute() else ROOT / log)
        elapsed = time.time() - started
        print(
            f"SUMMARY {log.stem} {json.dumps(summary)} seconds={elapsed:.0f} ({i}/{len(runs)})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
