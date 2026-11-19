"""Load logged runs for the dashboard and the analysis scripts. No torch needed.

Two sources:
- full logs written by `fedsurveil.fl.simulate` (runs/*.jsonl): every round with per-site accuracy;
- compact result files saved from the Colab printouts (results/step4_*_seed*.json): per-round
  global accuracy, macro-F1, worst-site and mean-site accuracy, and final per-site accuracy.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(
    r"^(?P<dataset>[a-z0-9]+)_(?P<algo>[a-z0-9.\-]+?)(?P<tags>(?:_[a-z]+-[a-z0-9.+\-]+)*)"
    r"_sites(?P<sites>\d+)_alpha(?P<alpha>[\d.]+)_seed(?P<seed>\d+)$"
)


@dataclass
class Run:
    name: str
    dataset: str
    algorithm: str
    alpha: float
    seed: int
    rounds: list[dict] = field(default_factory=list)
    site_sizes: list[int] | None = None
    class_counts: list[list[int]] | None = None
    final_site_acc: list[float] | None = None
    mia_auroc: float | None = None
    model_bytes: int | None = None
    config: dict = field(default_factory=dict)
    source: str = "log"

    @property
    def label(self) -> str:
        return f"{self.algorithm}, alpha {self.alpha:g}, seed {self.seed}"

    @property
    def has_per_site_history(self) -> bool:
        return bool(self.rounds) and "per_site_acc" in self.rounds[0]


def load_log(path: Path) -> Run:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    meta = next(r for r in rows if r["type"] == "meta")
    cfg = meta["config"]
    algo = cfg["algorithm"]
    if algo == "fedprox":
        algo = f"fedprox-mu{cfg.get('prox_mu', 0):g}"
    if cfg.get("dp_clip", 0) > 0:
        algo += f"+clientDP(sigma={cfg.get('dp_noise', 0):g})"
    if cfg.get("site_dp_noise", 0) > 0:
        algo += f"+DP-SGD(sigma={cfg['site_dp_noise']:g})"
    if cfg.get("secagg"):
        algo += "+secagg(sim)"
    rounds = []
    for r in (r for r in rows if r["type"] == "round"):
        row = dict(r)
        row["worst_site"] = r["site_acc_spread"]["min"]
        row["mean_site"] = r["site_acc_spread"]["mean"]
        rounds.append(row)
    summary = next((r for r in rows if r["type"] == "summary"), {})
    return Run(
        name=Path(path).stem,
        dataset=cfg["dataset"],
        algorithm=algo,
        alpha=cfg["alpha"],
        seed=cfg["seed"],
        rounds=rounds,
        site_sizes=meta.get("site_sizes"),
        class_counts=meta.get("site_class_counts"),
        final_site_acc=rounds[-1]["per_site_acc"] if rounds else None,
        mia_auroc=summary.get("mia_auroc"),
        model_bytes=meta.get("model_bytes"),
        config=cfg,
        source="log",
    )


def load_compact(path: Path) -> list[Run]:
    """Runs from a compact results file. Entries without a per-round curve are skipped."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model_bytes = data.get("_model_bytes")
    runs = []
    for name, entry in data.items():
        if name.startswith("_") or not isinstance(entry, dict) or "curve" not in entry:
            continue
        m = NAME.match(name)
        if not m:
            continue
        n_sites = int(m["sites"])
        up = (model_bytes or 0) * n_sites if m["algo"] != "central" else 0
        rounds = [
            {
                "round": c[0],
                "global_acc": c[1],
                "macro_f1": c[2],
                "worst_site": c[3],
                "mean_site": c[4],
                "bytes_up": up,
                "bytes_down": up,
            }
            for c in entry["curve"]
        ]
        runs.append(
            Run(
                name=name,
                dataset=m["dataset"],
                algorithm=m["algo"],
                alpha=float(m["alpha"]),
                seed=int(m["seed"]),
                rounds=rounds,
                site_sizes=entry.get("site_sizes"),
                final_site_acc=entry.get("final_site_acc"),
                model_bytes=model_bytes,
                config={"num_sites": n_sites},
                source="compact",
            )
        )
    return runs


def load_all(runs_dir: Path, results_dir: Path) -> list[Run]:
    runs = []
    for p in sorted(Path(runs_dir).glob("*.jsonl")):
        try:
            runs.append(load_log(p))
        except (StopIteration, KeyError, json.JSONDecodeError):
            continue
    have = {r.name for r in runs}
    for p in sorted(Path(results_dir).glob("step*_seed*.json")):
        runs += [r for r in load_compact(p) if r.name not in have]
    return runs


def curves(runs: list[Run], metric: str = "global_acc") -> pd.DataFrame:
    rows = [
        {"run": r.label, "round": row["round"], "value": row[metric]}
        for r in runs
        for row in r.rounds
        if metric in row
    ]
    return pd.DataFrame(rows, columns=["run", "round", "value"])


def site_matrix(run: Run) -> np.ndarray | None:
    """Rounds x sites accuracy, only for full logs."""
    if not run.has_per_site_history:
        return None
    return np.array([r["per_site_acc"] for r in run.rounds])


def cumulative_mb(run: Run) -> list[float]:
    total, out = 0.0, []
    for r in run.rounds:
        total += (r.get("bytes_up", 0) + r.get("bytes_down", 0)) / 1e6
        out.append(total)
    return out


def summary_table(runs: list[Run], last: int = 5) -> pd.DataFrame:
    rows = []
    for r in runs:
        if not r.rounds:
            continue
        tail = r.rounds[-last:]
        final = r.rounds[-1]
        rows.append(
            {
                "run": r.label,
                "dataset": r.dataset,
                "rounds": len(r.rounds),
                "accuracy (last-5 mean)": float(np.mean([x["global_acc"] for x in tail])),
                "accuracy (final)": final["global_acc"],
                "macro-F1 (final)": final["macro_f1"],
                "worst site (final)": final["worst_site"],
                "total MB sent": cumulative_mb(r)[-1],
                "epsilon (client)": final.get("eps_client"),
                "epsilon (sample, max site)": final.get("eps_sample_max"),
                "membership AUROC": r.mia_auroc,
            }
        )
    return pd.DataFrame(rows)


def rounds_to_accuracy(run: Run, target: float) -> int | None:
    for r in run.rounds:
        if r["global_acc"] >= target:
            return r["round"]
    return None
