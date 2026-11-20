"""Run the Flower end-to-end demo: a SuperLink and N SuperNodes as separate processes, then one run.

    python scripts/run_flower_demo.py [--sites 3] [--rounds 5]

Starts everything on localhost without TLS (a demo, not a deployment), submits the app in
flower_app/ with `flwr run`, then stops the processes. The result is written to
results/flower_demo.json. Needs torch, flwr and data/raw/bloodmnist.npz (CPU is enough).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "flower_app"
RESULT = ROOT / "results" / "flower_demo.json"
SUPERLINK = "127.0.0.1:9093"
FIRST_NODE_PORT = 9094


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    RESULT.unlink(missing_ok=True)  # `flwr run` exits 0 even if the ServerApp crashed
    procs: list[subprocess.Popen] = []
    try:
        procs.append(
            subprocess.Popen(
                [
                    "flower-superlink",
                    "--insecure",
                    "--disable-runtime-dependency-installation",
                    "--port",
                    SUPERLINK.split(":")[1],
                ],
                env=env,
                cwd=ROOT,
            )
        )
        time.sleep(5)
        for i in range(args.sites):
            procs.append(
                subprocess.Popen(
                    [
                        "flower-supernode",
                        "--insecure",
                        "--superlink",
                        SUPERLINK,
                        "--port",
                        str(FIRST_NODE_PORT + i),
                        "--node-config",
                        f"partition-id={i} num-partitions={args.sites}",
                    ],
                    env=env,
                    cwd=ROOT,
                )
            )
        time.sleep(5)
        run_config = f"num-sites={args.sites} num-server-rounds={args.rounds}"
        done = subprocess.run(
            [
                "flwr",
                "run",
                str(APP),
                "local-deployment",
                "--stream",
                "--run-config",
                run_config,
            ],
            env=env,
            cwd=ROOT,
        )
        time.sleep(2)
        if done.returncode == 0 and not RESULT.exists():
            print("The run did not finish: no results/flower_demo.json was written.")
            return 1
        return done.returncode
    finally:
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
