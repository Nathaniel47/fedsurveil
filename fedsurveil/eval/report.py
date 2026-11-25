"""Final results table and charts from the saved result files. No torch, no GPU.

    python -m fedsurveil.eval.report

Writes results/results_table.md and three PNG charts in results/charts/. Every number comes from
results/step4_bloodmnist_3seeds.json and results/step5_bloodmnist_3seeds.json (3 seeds, final
round). Nothing is re-run, and nothing here is a new measurement.
"""

import json
import re
import statistics as st

import pandas as pd

from fedsurveil import config

STEP4 = config.RESULTS_DIR / "step4_bloodmnist_3seeds.json"
STEP5 = config.RESULTS_DIR / "step5_bloodmnist_3seeds.json"
TABLE = config.RESULTS_DIR / "results_table.md"
CROSS_TABLE = config.RESULTS_DIR / "cross_dataset_table.md"
OTHER_DATASETS = ("organamnist", "pathmnist", "ip102")
# (label, key in a step8 file, label of the same setting in the BloodMNIST summary)
CROSS_SETTINGS = (
    ("Central (pooled data)", "central", "Central (pooled data)"),
    ("FedAvg, alpha 100", "fedavg_a100", "FedAvg, alpha 100 (near-IID)"),
    ("FedAvg, alpha 0.5", "fedavg_a0.5", "FedAvg, alpha 0.5"),
    ("FedAvg, alpha 0.1", "fedavg_a0.1", "FedAvg, alpha 0.1"),
    ("FedProx mu 0.1, alpha 0.5", "fedprox-mu0.1_a0.5", "FedProx mu 0.1, alpha 0.5"),
    ("FedProx mu 0.1, alpha 0.1", "fedprox-mu0.1_a0.1", "FedProx mu 0.1, alpha 0.1"),
    ("SCAFFOLD, alpha 0.5", "scaffold_a0.5", "SCAFFOLD (2x upload), alpha 0.5"),
    ("SCAFFOLD, alpha 0.1", "scaffold_a0.1", "SCAFFOLD (2x upload), alpha 0.1"),
)
CHARTS = config.RESULTS_DIR / "charts"
MODEL_BYTES = 378_912  # the small CNN, from the run logs
SITES = 10
DELTA = config.DP_DELTA


def _stats(values: list[float]) -> tuple[float, float]:
    return st.mean(values), (st.stdev(values) if len(values) > 1 else float("nan"))


def load_runs() -> pd.DataFrame:
    """One row per (setting, seed) with the final-round metrics."""
    rows = []
    step4 = json.loads(STEP4.read_text(encoding="utf-8"))["runs"]
    names4 = {
        "central": ("Central (pooled data)", "-", 0),
        "fedavg_a100": ("FedAvg, alpha 100 (near-IID)", "100", MODEL_BYTES),
        "fedavg_a0.5": ("FedAvg, alpha 0.5", "0.5", MODEL_BYTES),
        "fedavg_a0.1": ("FedAvg, alpha 0.1", "0.1", MODEL_BYTES),
    }
    for key, seeds in step4.items():
        label, alpha, up = names4[key]
        for seed, v in seeds.items():
            rows.append(
                {
                    "group": "baseline",
                    "setting": label,
                    "key": f"s4_{key}",
                    "alpha": alpha,
                    "seed": seed,
                    "acc": v["final_acc"],
                    "f1": v["final_macro_f1"],
                    "worst": v["final_worst_site_acc"],
                    "upload_per_site": up,
                    "guarantee": "none",
                    "mia": None,
                    "eps": None,
                }  # fmt: skip
            )
    step5 = json.loads(STEP5.read_text(encoding="utf-8"))["runs"]
    for name, v in step5.items():
        m = re.match(r"(.+)_alpha([\d.]+)_seed(\d)$", name)
        base, alpha, seed = m.group(1), m.group(2), f"seed{m.group(3)}"
        label, group, guarantee, eps = _describe(base, v)
        rows.append(
            {
                "group": group, "setting": f"{label}, alpha {alpha}", "key": f"s5_{base}_{alpha}",
                "alpha": alpha, "seed": seed, "acc": v["final_acc"], "f1": v["final_macro_f1"],
                "worst": v["final_worst_site_acc"],
                "upload_per_site": v["bytes_up_round"] / SITES, "guarantee": guarantee,
                "mia": v["mia_auroc"], "eps": eps,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def _describe(base: str, v: dict) -> tuple[str, str, str, float | None]:
    """Readable label, group, and what privacy guarantee (if any) the setting gives."""
    if base.startswith("fedprox"):
        mu = base.split("mu")[1]
        return f"FedProx mu {mu}", "drift", "none", None
    if base == "scaffold":
        return "SCAFFOLD (2x upload)", "drift", "none", None
    if base == "fedavg":
        return "FedAvg", "privacy", "none", None
    if base == "fedavg_secagg":
        return (
            "FedAvg + simulated secure aggregation", "privacy",
            "none (simulation, not cryptographic)", None,
        )  # fmt: skip
    if base.startswith("fedavg_cdp"):
        sigma = re.search(r"-n([\d.]+)$", base).group(1)
        eps = v["eps_client"]
        return (
            f"FedAvg + client-level DP, sigma {sigma}", "privacy",
            f"client-level (eps {eps:.0f}, delta {DELTA:g})", eps,
        )  # fmt: skip
    if base == "fedavg_sdp-n1":
        eps = v["eps_sample_max"]
        return (
            "FedAvg + DP-SGD, sigma 1 (1 seed)", "privacy",
            f"sample-level (eps {eps:.0f}, delta {DELTA:g})", eps,
        )  # fmt: skip
    if base.startswith("fedavg_cmp"):
        return "FedAvg + " + base.split("cmp-")[1].replace("topk", "top-k "), "comms", "none", None
    raise ValueError(base)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, g in runs.groupby("key", sort=False):
        first = g.iloc[0]
        row = {"group": first["group"], "setting": first["setting"], "seeds": len(g)}
        for col in ("acc", "f1", "worst"):
            row[col], row[col + "_sd"] = _stats(g[col].tolist())
        row["upload_per_site"] = first["upload_per_site"]
        row["guarantee"] = first["guarantee"]
        row["mia"] = g["mia"].mean() if g["mia"].notna().any() else None
        row["eps"] = first["eps"]
        row["alpha"] = first["alpha"]
        row["key"] = key
        out.append(row)
    return pd.DataFrame(out)


def _pm(mean: float, sd: float, scale: float = 100, digits: int = 1) -> str:
    if sd != sd:  # a single seed has no spread
        return f"{mean * scale:.{digits}f} (1 seed)"
    return f"{mean * scale:.{digits}f} ± {sd * scale:.{digits}f}"


def write_table(summary: pd.DataFrame) -> str:
    titles = {
        "baseline": "Central training and FedAvg under different data skew (alpha)",
        "drift": "Fixes for client drift (10 sites)",
        "privacy": "Privacy (alpha 0.5)",
        "comms": "Update compression (alpha 0.5)",
    }
    lines = [
        "# Results table (BloodMNIST, 10 sites, 30 rounds)",
        "",
        "Generated by `python -m fedsurveil.eval.report` from the saved result files. Values are "
        "the final-round mean ± standard deviation over seeds 0, 1 and 2 (3 seeds is few: read "
        "the spreads as rough). Worst site is the lowest per-site accuracy. Upload is bytes one "
        "site sends per round. MIA AUROC is a membership-inference attack: 0.5 means it found "
        "no leakage.",
        "",
    ]
    for group, title in titles.items():
        part = summary[summary["group"] == group]
        if part.empty:
            continue
        lines += [f"## {title}", ""]
        lines += [
            "| Setting | Accuracy (%) | Macro-F1 | Worst site (%) | Upload / site / round | "
            "Privacy guarantee | MIA AUROC |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in part.iterrows():
            up = "-" if not r["upload_per_site"] else f"{r['upload_per_site'] / 1000:,.1f} KB"
            mia = "-" if r["mia"] is None or r["mia"] != r["mia"] else f"{r['mia']:.3f}"
            lines.append(
                f"| {r['setting']} | {_pm(r['acc'], r['acc_sd'])} | "
                f"{_pm(r['f1'], r['f1_sd'], 1, 3)} | {_pm(r['worst'], r['worst_sd'])} | "
                f"{up} | {r['guarantee']} | {mia} |"
            )
        lines.append("")
    text = "\n".join(lines)
    TABLE.write_text(text, encoding="utf-8")
    return text


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def chart_noniid(runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Accuracy and worst-site accuracy against data skew, for each method."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    methods = {
        "FedAvg": lambda s: s["setting"].str.startswith("FedAvg, alpha"),
        "FedProx mu 0.1": lambda s: s["setting"].str.startswith("FedProx mu 0.1,"),
        "SCAFFOLD": lambda s: s["setting"].str.startswith("SCAFFOLD"),
    }
    # FedAvg at alpha 0.1/0.5/100 comes from Step 4; alpha 0.5 duplicates exist in Step 5.
    s4 = summary[summary["key"].str.startswith("s4_fedavg")]
    for ax, col, title in (
        (axes[0], "acc", "Overall accuracy"),
        (axes[1], "worst", "Worst site's accuracy"),
    ):
        for name, pick in methods.items():
            part = s4 if name == "FedAvg" else summary[pick(summary)]
            part = part.assign(a=part["alpha"].astype(float)).sort_values("a")
            ax.errorbar(
                part["a"], part[col] * 100, yerr=part[col + "_sd"] * 100, marker="o", capsize=3,
                label=name,
            )  # fmt: skip
        central = summary[summary["key"] == "s4_central"].iloc[0]
        if col == "acc":
            ax.axhline(central[col] * 100, color="grey", ls="--", label="Central (pooled)")
        ax.set_xscale("log")
        ax.set_xlabel("alpha (left = more skewed sites)")
        ax.set_ylabel("%")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Skewed sites hurt the worst-off site most (BloodMNIST, mean ± sd, 3 seeds)")
    fig.tight_layout()
    fig.savefig(CHARTS / "noniid_methods.png", dpi=150)
    plt.close(fig)


def chart_privacy(summary: pd.DataFrame) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    base = summary[summary["key"] == "s5_fedavg_0.5"].iloc[0]
    ax.axhline(base["acc"] * 100, color="grey", ls="--", label="No privacy (FedAvg)")
    cdp = summary[summary["key"].str.contains("cdp")].sort_values("eps")
    ax.errorbar(
        cdp["eps"], cdp["acc"] * 100, yerr=cdp["acc_sd"] * 100, marker="o", capsize=3,
        label="Client-level DP (3 seeds)",
    )  # fmt: skip
    sdp = summary[summary["key"].str.contains("sdp")].iloc[0]
    ax.plot([sdp["eps"]], [sdp["acc"] * 100], "s", color="C3", label="Sample-level DP-SGD (1 seed)")
    ax.set_xscale("log")
    ax.set_xlabel(f"epsilon (delta {DELTA:g}); smaller = stronger privacy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Privacy costs accuracy (BloodMNIST, 10 sites)")
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01, 0.005,
        "Client-level and sample-level epsilon protect different things and are not comparable.",
        fontsize=7, color="grey",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHARTS / "privacy_utility.png", dpi=150)
    plt.close(fig)


def chart_comms(summary: pd.DataFrame) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    keys = {
        "s5_fedavg_0.5": "dense",
        "s5_fedavg_cmp-int8_0.5": "int8",
        "s5_fedavg_cmp-topk0.1_0.5": "top-k 10%",
        "s5_fedavg_cmp-topk0.01_0.5": "top-k 1%",
    }
    for key, name in keys.items():
        r = summary[summary["key"] == key].iloc[0]
        ax.errorbar(
            [r["upload_per_site"] / 1000], [r["acc"] * 100], yerr=[r["acc_sd"] * 100],
            marker="o", capsize=3, label=name,
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("Upload per site per round (KB, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Compression: 50x less upload for about 5 points (3 seeds)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "compression_cost.png", dpi=150)
    plt.close(fig)


def cross_dataset(runs: pd.DataFrame, summary: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Per dataset and setting: per-seed final accuracy and worst-site accuracy.

    BloodMNIST comes from the Step 4 and 5 files; the others from results/step8_<name>_3seeds.json
    (a dataset whose file is missing is skipped).
    """
    out: dict[str, dict[str, dict]] = {"bloodmnist": {}}
    for label, _, blood in CROSS_SETTINGS:
        part = summary[summary["setting"] == blood]
        part = part[part["key"].str.startswith(("s4_", "s5_fedprox", "s5_scaffold"))]
        row = part.iloc[0]
        per_seed = runs[runs["key"] == row["key"]]
        out["bloodmnist"][label] = {
            "acc": per_seed["acc"].tolist(), "worst": per_seed["worst"].tolist(),
            "acc_sd": row["acc_sd"], "worst_sd": row["worst_sd"], "n": int(row["seeds"]),
        }  # fmt: skip
    for name in OTHER_DATASETS:
        path = config.RESULTS_DIR / f"step8_{name}_3seeds.json"
        if not path.exists():
            continue
        runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
        out[name] = {}
        for label, key, _ in CROSS_SETTINGS:
            seeds = runs[key].values()
            accs = [v["final_acc"] for v in seeds]
            worst = [v["final_worst_site_acc"] for v in seeds]
            out[name][label] = {
                "acc": accs, "worst": worst, "acc_sd": _stats(accs)[1],
                "worst_sd": _stats(worst)[1], "n": len(accs),
            }  # fmt: skip
    return out


def write_cross_table(data: dict[str, dict[str, dict]]) -> str:
    lines = [
        "# Same comparison on every dataset (10 sites, 30 rounds, 3 seeds)",
        "",
        "Final-round mean ± standard deviation over seeds 0, 1 and 2 (rough: 3 seeds). "
        "All datasets use the settings chosen on BloodMNIST; nothing was re-tuned. "
        "Worst site is the lowest per-site accuracy. No privacy mechanism is on in any row.",
        "",
        "| Setting | " + " | ".join(f"{d} acc (%) | {d} worst site (%)" for d in data) + " |",
        "|---|" + "---|---|" * len(data),
    ]
    for label, _, _ in CROSS_SETTINGS:
        cells = []
        for d in data:
            r = data[d][label]
            cells += [
                _pm(st.mean(r["acc"]), r["acc_sd"]),
                _pm(st.mean(r["worst"]), r["worst_sd"]),
            ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    text = "\n".join(lines)
    CROSS_TABLE.write_text(text, encoding="utf-8")
    return text


def chart_datasets(data: dict[str, dict[str, dict]]) -> None:
    """Accuracy at alpha 0.1 for FedAvg, FedProx and SCAFFOLD, one panel per dataset."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(data), figsize=(3.4 * len(data), 3.8), sharey=False)
    axes = [axes] if len(data) == 1 else list(axes)
    methods = ("FedAvg, alpha 0.1", "FedProx mu 0.1, alpha 0.1", "SCAFFOLD, alpha 0.1")
    short = ("FedAvg", "FedProx", "SCAFFOLD")
    for ax, (name, d) in zip(axes, data.items(), strict=True):
        for i, m in enumerate(methods):
            r = d[m]
            sd = 0 if r["acc_sd"] != r["acc_sd"] else r["acc_sd"] * 100
            ax.bar(i, st.mean(r["acc"]) * 100, yerr=sd, capsize=3, color=f"C{i}")
            ax.plot([i] * len(r["acc"]), [a * 100 for a in r["acc"]], "k.", ms=4)
        ax.axhline(st.mean(d["Central (pooled data)"]["acc"]) * 100, color="grey", ls="--")
        ax.set_xticks(range(3), short)
        ax.set_title(name)
        ax.set_ylabel("Accuracy (%)")
    fig.suptitle(
        "Strongly skewed sites (alpha 0.1): bars = mean ± sd, dots = seeds, dashed = central"
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "datasets_alpha01.png", dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    summary = summarize(runs)
    write_table(summary)
    chart_noniid(runs, summary)
    chart_privacy(summary)
    chart_comms(summary)
    cross = cross_dataset(runs, summary)
    write_cross_table(cross)
    chart_datasets(cross)
    print(f"wrote {TABLE}, {CROSS_TABLE} and 4 charts in {CHARTS}")


if __name__ == "__main__":
    main()
