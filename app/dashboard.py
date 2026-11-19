"""FedSurveil dashboard: replay logged federated-learning runs round by round.

Reads full logs from runs/ and compact result files from results/. Nothing is trained here, so it
runs on any laptop or free host with no GPU. Raw IP102 images are never shown, only metrics.

    streamlit run app/dashboard.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedsurveil.eval.runs import (  # noqa: E402
    cumulative_mb,
    load_all,
    rounds_to_accuracy,
    site_matrix,
    summary_table,
)

st.set_page_config(page_title="FedSurveil", layout="wide")


@st.cache_data
def get_runs(runs_dir: str, results_dir: str):
    return load_all(Path(runs_dir), Path(results_dir))


def fig_layout(fig: go.Figure, y_title: str, x_title: str = "Round") -> go.Figure:
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.25),
        height=380,
    )
    return fig


def line_chart(runs, metric: str, y_title: str, upto: int | None = None) -> go.Figure:
    fig = go.Figure()
    for r in runs:
        rows = [x for x in r.rounds if metric in x and (upto is None or x["round"] <= upto)]
        fig.add_trace(
            go.Scatter(
                x=[x["round"] for x in rows],
                y=[x[metric] for x in rows],
                mode="lines",
                name=r.label,
            )
        )
    return fig_layout(fig, y_title)


st.title("FedSurveil")
st.caption(
    "Federated learning across simulated sites, replayed from logs. Privacy numbers say which "
    "guarantee they give; secure aggregation here is a simulation, not cryptography."
)

runs = get_runs(str(ROOT / "runs"), str(ROOT / "results"))
if not runs:
    st.info("No runs found yet. Run `python -m fedsurveil.fl.simulate` or add result files.")
    st.stop()

datasets = sorted({r.dataset for r in runs})
dataset = st.sidebar.selectbox("Dataset", datasets)
pool = [r for r in runs if r.dataset == dataset]
labels = [f"{r.label} [{r.source}]" for r in pool]
default = [i for i, r in enumerate(pool) if r.seed == pool[0].seed][:6]
chosen_idx = st.sidebar.multiselect(
    "Runs to compare", range(len(pool)), default=default, format_func=lambda i: labels[i]
)
chosen = [pool[i] for i in chosen_idx]
if not chosen:
    st.warning("Pick at least one run in the sidebar.")
    st.stop()

max_round = max(len(r.rounds) for r in chosen)
st.sidebar.markdown("---")
shown = st.sidebar.slider("Replay up to round", 1, max_round, max_round)
if st.sidebar.button("Play replay"):
    slot = st.empty()
    for upto in range(1, max_round + 1):
        slot.plotly_chart(
            line_chart(chosen, "global_acc", "Global test accuracy", upto),
            width="stretch",
            key=f"replay-{upto}",
        )
        time.sleep(0.12)

tab_train, tab_sites, tab_split, tab_cost, tab_table = st.tabs(
    ["Training", "Per-site accuracy", "Data split", "Privacy and cost", "Summary table"]
)

with tab_train:
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        line_chart(chosen, "global_acc", "Global test accuracy", shown), width="stretch"
    )
    c2.plotly_chart(line_chart(chosen, "macro_f1", "Macro-F1", shown), width="stretch")
    target = st.slider("Target accuracy", 0.5, 0.99, 0.8, 0.01)
    st.write({r.label: (rounds_to_accuracy(r, target) or "not reached") for r in chosen})

with tab_sites:
    st.write(
        "A global accuracy can hide a site the model fails. The worst site is the number to "
        "watch; the mean is shown for comparison."
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(line_chart(chosen, "worst_site", "Worst-site accuracy", shown), width="stretch")
    c2.plotly_chart(line_chart(chosen, "mean_site", "Mean site accuracy", shown), width="stretch")
    pick = st.selectbox(
        "Run for site detail", range(len(chosen)), format_func=lambda i: chosen[i].label
    )
    run = chosen[pick]
    matrix = site_matrix(run)
    if matrix is not None:
        upto = min(shown, len(matrix))
        heat = go.Figure(
            go.Heatmap(
                z=matrix[:upto].T,
                x=list(range(1, upto + 1)),
                y=[f"site {i}" for i in range(matrix.shape[1])],
                zmin=0,
                zmax=1,
                colorscale="RdYlGn",
                colorbar=dict(title="accuracy"),
            )
        )
        st.plotly_chart(fig_layout(heat, "Site"), width="stretch")
    else:
        st.caption("Round-by-round per-site accuracy was not saved for this run (compact result).")
    if run.final_site_acc:
        sizes = run.site_sizes or [None] * len(run.final_site_acc)
        bars = go.Figure(
            go.Bar(
                x=[f"site {i}" for i in range(len(run.final_site_acc))],
                y=run.final_site_acc,
                text=[f"{n} imgs" if n else "" for n in sizes],
            )
        )
        bars.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_layout(bars, "Final accuracy", "Site"), width="stretch")

with tab_split:
    run = chosen[0]
    if run.class_counts:
        counts = np.array(run.class_counts)
        heat = go.Figure(
            go.Heatmap(
                z=counts,
                x=[f"class {c}" for c in range(counts.shape[1])],
                y=[f"site {i}" for i in range(counts.shape[0])],
                colorscale="Blues",
                colorbar=dict(title="images"),
            )
        )
        st.write(f"Class mix per site for: {run.label}")
        st.plotly_chart(fig_layout(heat, "Site", "Class"), width="stretch")
    else:
        st.caption("Class counts per site are only stored in full logs from runs/.")
        if run.site_sizes:
            st.bar_chart({f"site {i}": n for i, n in enumerate(run.site_sizes)})

with tab_cost:
    fig = go.Figure()
    for r in chosen:
        fig.add_trace(
            go.Scatter(
                x=cumulative_mb(r),
                y=[x["global_acc"] for x in r.rounds],
                name=r.label,
                mode="lines",
            )
        )
    st.write("Accuracy against total data sent (up + down), in megabytes.")
    st.plotly_chart(fig_layout(fig, "Global test accuracy", "Total MB sent"), width="stretch")
    eps_runs = [
        r for r in chosen if any("eps_client" in x or "eps_sample_max" in x for x in r.rounds)
    ]
    if eps_runs:
        efig = go.Figure()
        for r in eps_runs:
            key = "eps_client" if "eps_client" in r.rounds[0] else "eps_sample_max"
            efig.add_trace(
                go.Scatter(
                    x=[x["round"] for x in r.rounds],
                    y=[x[key] for x in r.rounds],
                    name=f"{r.label} ({'client-level' if key == 'eps_client' else 'sample-level'})",
                )
            )
        st.write("Privacy budget spent (epsilon, delta = 1e-5). Smaller is more private.")
        st.plotly_chart(fig_layout(efig, "epsilon"), width="stretch")
    else:
        st.caption("None of the selected runs used differential privacy.")

with tab_table:
    table = summary_table(chosen)
    st.dataframe(table, width="stretch", hide_index=True)
