"""Run FedAvg across simulated non-IID sites, or the central (pooled) baseline.

    python -m fedsurveil.fl.simulate --dataset bloodmnist --algorithm fedavg --alpha 0.1

Writes one JSONL log to runs/: a "meta" line, then one "round" line per evaluated round.
"""

import argparse
import copy
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fedsurveil import config
from fedsurveil.comms.meter import round_bytes, state_bytes
from fedsurveil.data.load import ImageSet, load_ip102_subset, load_medmnist
from fedsurveil.data.partition import class_matrix, dirichlet_partition
from fedsurveil.eval.metrics import accuracy, macro_f1, site_eval_indices, spread
from fedsurveil.fl.client import local_train, num_steps, predict
from fedsurveil.fl.fedavg import fedavg_aggregate, state_delta
from fedsurveil.fl.scaffold import (
    effective_lr,
    update_server_control,
    update_site_control,
    zero_controls,
)
from fedsurveil.models.cnn import build_model, default_model_name


@dataclass
class RunConfig:
    dataset: str = "bloodmnist"
    algorithm: str = "fedavg"  # "fedavg", "fedprox", "scaffold" or "central"
    num_sites: int = 10
    alpha: float = 0.5
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 0.05
    momentum: float = 0.0
    prox_mu: float = 0.0  # FedProx strength; only used when algorithm == "fedprox"
    batch_size: int = 64
    sites_per_round: int | None = None  # None = every site, every round
    site_eval_size: int = 500
    seed: int = 0
    limit: int | None = None  # smoke tests only: use a random subset of the training set
    min_site_size: int = 10


def run_name(cfg: RunConfig) -> str:
    algo = f"fedprox-mu{cfg.prox_mu:g}" if cfg.algorithm == "fedprox" else cfg.algorithm
    return f"{cfg.dataset}_{algo}_sites{cfg.num_sites}_alpha{cfg.alpha:g}_seed{cfg.seed}"


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


def load_dataset(name: str) -> ImageSet:
    return load_ip102_subset() if name == "ip102" else load_medmnist(name)


def _subsample(data: ImageSet, limit: int, seed: int) -> ImageSet:
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(data.x_train), size=min(limit, len(data.x_train)), replace=False))
    return ImageSet(
        data.name,
        data.x_train[keep],
        data.y_train[keep],
        data.x_val,
        data.y_val,
        data.x_test,
        data.y_test,
    )


def _evaluate(model, data, site_idx, device, num_classes) -> dict:
    preds = predict(model, data.x_test, device)
    site_acc = [accuracy(data.y_test[i], preds[i]) for i in site_idx]
    return {
        "global_acc": accuracy(data.y_test, preds),
        "macro_f1": macro_f1(data.y_test, preds, num_classes),
        "per_site_acc": site_acc,
        "site_acc_spread": spread(site_acc),
    }


def run(
    cfg: RunConfig,
    data: ImageSet | None = None,
    device: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    if cfg.algorithm not in config.ALGORITHMS + ("central",):
        raise ValueError(f"unknown algorithm {cfg.algorithm!r}")
    if cfg.algorithm == "fedprox" and cfg.prox_mu <= 0:
        raise ValueError("fedprox needs --prox-mu > 0")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = data or load_dataset(cfg.dataset)
    if cfg.limit:
        data = _subsample(data, cfg.limit, cfg.seed)
    k = data.num_classes
    channels = data.x_train.shape[-1]

    parts = dirichlet_partition(
        data.y_train, cfg.num_sites, cfg.alpha, cfg.seed, min_size=cfg.min_site_size
    )
    counts = class_matrix(data.y_train, parts, k)
    site_idx = site_eval_indices(data.y_test, counts, cfg.site_eval_size, cfg.seed)

    torch.manual_seed(cfg.seed)
    model = build_model(default_model_name(cfg.dataset), channels, k).to(device)
    model_bytes = state_bytes(model.state_dict())
    rng = np.random.default_rng(cfg.seed + 1)
    scaffold = cfg.algorithm == "scaffold"
    if scaffold:
        c_server = zero_controls(model)
        c_sites = [zero_controls(model) for _ in range(cfg.num_sites)]
        lr_eff = effective_lr(cfg.lr, cfg.momentum)

    out_dir = out_dir or config.RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name(cfg)}.jsonl"
    meta = {
        "type": "meta",
        "config": asdict(cfg),
        "model": default_model_name(cfg.dataset),
        "model_bytes": model_bytes,
        "site_sizes": [int(len(p)) for p in parts],
        "site_class_counts": counts.tolist(),
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "git_commit": _git_commit(),
    }
    with open(path, "w", encoding="utf-8") as log:
        log.write(json.dumps(meta) + "\n")
        for rnd in range(1, cfg.rounds + 1):
            if cfg.algorithm == "central":
                seed = cfg.seed * 100003 + rnd
                local_train(
                    model,
                    data.x_train,
                    data.y_train,
                    epochs=cfg.local_epochs,
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                    seed=seed,
                    device=device,
                    momentum=cfg.momentum,
                )
                traffic = {"bytes_down": 0, "bytes_up": 0}
            else:
                n_pick = cfg.sites_per_round or cfg.num_sites
                chosen = np.sort(rng.choice(cfg.num_sites, size=n_pick, replace=False))
                global_state = {
                    k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()
                }
                deltas, sizes, control_deltas = [], [], []
                for site in chosen:
                    local = copy.deepcopy(model)
                    n_site = len(parts[site])
                    local_train(
                        local,
                        data.x_train[parts[site]],
                        data.y_train[parts[site]],
                        epochs=cfg.local_epochs,
                        lr=cfg.lr,
                        batch_size=cfg.batch_size,
                        seed=cfg.seed * 100003 + rnd * 1009 + int(site),
                        device=device,
                        momentum=cfg.momentum,
                        prox_mu=cfg.prox_mu if cfg.algorithm == "fedprox" else 0.0,
                        control=(c_server, c_sites[site]) if scaffold else None,
                    )
                    deltas.append(state_delta(local.state_dict(), global_state))
                    sizes.append(n_site)
                    if scaffold:
                        new_c, delta_c = update_site_control(
                            c_sites[site],
                            c_server,
                            [global_state[k_] for k_ in global_state],
                            [p.detach().cpu() for p in local.parameters()],
                            num_steps(n_site, cfg.batch_size, cfg.local_epochs),
                            lr_eff,
                        )
                        c_sites[site] = new_c
                        control_deltas.append(delta_c)
                model.load_state_dict(fedavg_aggregate(global_state, deltas, sizes))
                if scaffold:
                    c_server = update_server_control(c_server, control_deltas, cfg.num_sites)
                traffic = round_bytes(model_bytes, len(chosen), 2 if scaffold else 1)
            record = {"type": "round", "round": rnd, **traffic}
            record.update(_evaluate(model, data, site_idx, device, k))
            log.write(json.dumps(record) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    defaults = RunConfig()
    for name, value in asdict(defaults).items():
        kind = type(value) if value is not None else int
        parser.add_argument(f"--{name.replace('_', '-')}", type=kind, default=value)
    args = parser.parse_args()
    cfg = RunConfig(**{k_.replace("-", "_"): v for k_, v in vars(args).items()})
    print(run(cfg))


if __name__ == "__main__":
    main()
