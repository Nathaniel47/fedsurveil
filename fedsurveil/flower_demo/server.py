"""Flower ServerApp: FedAvg over the SuperNodes, with global test accuracy logged every round.

Writes results/flower_demo.json (not runs/, which holds the simulator's full logs).
"""

import json
import platform
import subprocess
from importlib.metadata import version

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedsurveil import config
from fedsurveil.eval.metrics import accuracy, macro_f1
from fedsurveil.fl.client import predict
from fedsurveil.fl.simulate import load_dataset
from fedsurveil.flower_demo.common import new_model

app = ServerApp()
OUT = config.RESULTS_DIR / "flower_demo.json"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.ROOT,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    dataset, seed = str(cfg["dataset"]), int(cfg["seed"])
    num_sites, rounds = int(cfg["num-sites"]), int(cfg["num-server-rounds"])
    data = load_dataset(dataset)
    k = data.num_classes
    model = new_model(dataset, seed)
    model_bytes = int(sum(v.numel() * v.element_size() for v in model.state_dict().values()))
    history: list[dict] = []

    def evaluate(rnd: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        preds = predict(model, data.x_test, "cpu")
        acc = accuracy(data.y_test, preds)
        sent = model_bytes * num_sites if rnd else 0  # round 0 is the untrained model
        history.append(
            {
                "round": rnd,
                "global_acc": acc,
                "macro_f1": macro_f1(data.y_test, preds, k),
                "bytes_down": sent,
                "bytes_up": sent,
            }
        )
        print(f"round {rnd}: global test accuracy {acc:.4f}", flush=True)
        return MetricRecord({"global_acc": acc})

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=num_sites,
        min_available_nodes=num_sites,
    )
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        num_rounds=rounds,
        train_config=ConfigRecord({"round": 0}),
        evaluate_fn=evaluate,
    )
    meta = {
        "what": "Flower end-to-end demo: SuperLink + SuperNode processes, FedAvg, no privacy",
        "config": {k_: (v if not isinstance(v, np.generic) else v.item()) for k_, v in cfg.items()},
        "model_bytes": model_bytes,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "flwr": version("flwr"),
            "numpy": np.__version__,
        },
        "git_commit": _git_commit(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "rounds": history}, indent=1), encoding="utf-8")
