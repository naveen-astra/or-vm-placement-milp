"""Interactive dashboard for energy-aware VM placement.

Run with:
    streamlit run app/streamlit_app.py

The UI is a thin layer over the vmplace package: every number shown here comes
from the same preprocessing, MILP and metrics code that scripts/03 uses, so the
dashboard and the report cannot disagree.
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from vmplace import baselines, milp, preprocess as pp, sensitivity, servers as sv
from vmplace.config import (
    PROCESSED_DIR,
    RAW_TRACE_PATH,
    SERVER_CATALOGUE,
    WEIGHT_PRESETS,
    FleetConfig,
    PowerModelConfig,
    QoSConfig,
    ScenarioConfig,
    SolverConfig,
    WastageConfig,
    Weights,
)
from vmplace.metrics import build_normalisers, compare_results

st.set_page_config(
    page_title="Energy-Aware VM Placement",
    page_icon="server",
    layout="wide",
)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ==========================================================================
# data loading
# ==========================================================================
@st.cache_data(show_spinner=False)
def _load_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _process_raw(path: str, nrows: int | None) -> pd.DataFrame:
    raw = pp.load_raw(path, nrows=nrows)
    return pp.clean_vm_table(pp.build_vm_table(raw))


@st.cache_data(show_spinner=False)
def _process_upload(data: bytes, nrows: int | None) -> pd.DataFrame:
    buf = io.BytesIO(data)
    raw = pd.read_csv(buf, nrows=nrows, low_memory=False)
    missing = [c for c in ("collection_id", "instance_index", "cluster",
                           "resource_request") if c not in raw.columns]
    if missing:
        raise ValueError(
            "uploaded CSV is missing required Borg trace columns: "
            + ", ".join(missing)
        )
    return pp.clean_vm_table(pp.build_vm_table(raw))


def _available_tables() -> list[str]:
    return sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "vm_table_*.parquet"))
        + glob.glob(os.path.join(PROCESSED_DIR, "vm_table_*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )


# ==========================================================================
# sidebar: data source
# ==========================================================================
st.sidebar.title("VM Placement Optimiser")
st.sidebar.caption("Multi-objective MILP over the Google Borg 2019 trace")

st.sidebar.header("1 - Dataset")
source = st.sidebar.radio(
    "Source",
    ["Processed table", "Raw trace", "Upload CSV"],
    help="Processed tables come from scripts/01_preprocess.py and load instantly.",
)

pool: pd.DataFrame | None = None
source_label = ""

if source == "Processed table":
    tables = _available_tables()
    if not tables:
        st.sidebar.warning(
            "No processed table found. Run scripts/01_preprocess.py, "
            "or choose 'Raw trace'."
        )
    else:
        pick = st.sidebar.selectbox(
            "Table", tables, format_func=os.path.basename
        )
        pool = _load_table(pick)
        source_label = os.path.basename(pick)

elif source == "Raw trace":
    path = st.sidebar.text_input("Path to trace CSV", RAW_TRACE_PATH)
    nrows = st.sidebar.number_input(
        "Event rows to read (0 = all 1.32M)",
        min_value=0, max_value=2_000_000, value=200_000, step=50_000,
        help="Rows are EVENTS, not VMs. Several events collapse into one VM.",
    )
    if st.sidebar.button("Load trace", type="primary"):
        st.session_state["_load_raw"] = (path, int(nrows) or None)
    if "_load_raw" in st.session_state:
        p, n = st.session_state["_load_raw"]
        with st.spinner(f"reading and collapsing {p}..."):
            try:
                pool = _process_raw(p, n)
                source_label = f"{os.path.basename(p)} ({n or 'all'} events)"
            except FileNotFoundError:
                st.sidebar.error(f"file not found: {p}")

else:
    up = st.sidebar.file_uploader("Borg trace CSV", type=["csv"])
    nrows = st.sidebar.number_input(
        "Event rows to read (0 = all)", min_value=0, value=200_000, step=50_000
    )
    if up is not None:
        with st.spinner("processing upload..."):
            try:
                pool = _process_upload(up.getvalue(), int(nrows) or None)
                source_label = up.name
            except Exception as exc:
                st.sidebar.error(str(exc))

if pool is None or len(pool) == 0:
    st.title("Energy-Efficient VM Placement")
    st.markdown(
        """
Load a dataset from the sidebar to begin.

**What this tool does.** It reads the Google Borg 2019 cluster trace, collapses
its event stream into one record per VM instance, and solves a multi-objective
mixed-integer linear program that decides which VM runs on which physical
server. The objective trades off three things that genuinely conflict:

| Term | Meaning | Source |
|---|---|---|
| **E** energy | load-proportional power of the servers left switched on | modelled power curve |
| **W** wastage | unused capacity on active servers, plus CPU/memory imbalance | trace demand |
| **Q** QoS risk | risk-weighted peak demand above a safety threshold | trace `maximum_usage`, `failed`, `priority` |

The placement is then compared against First-Fit, Best-Fit,
First-Fit-Decreasing and the trace's own assignment.
        """
    )
    st.stop()

st.sidebar.success(f"{len(pool):,} VM instances loaded")

# ==========================================================================
# sidebar: workload and fleet
# ==========================================================================
st.sidebar.header("2 - Workload")
n_vms = st.sidebar.slider(
    "VMs to optimise", 20, min(600, len(pool)),
    min(200, len(pool)), step=10,
    help="The MILP has n_vms x n_servers binary variables. CBC handles a few "
         "thousand comfortably; beyond that expect the time limit to bind.",
)
strategy = st.sidebar.selectbox(
    "Sampling", ["stratified", "random", "largest", "head"],
    help="'stratified' preserves the priority-band mix of the full workload.",
)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

st.sidebar.header("3 - Server fleet")
auto_fleet = st.sidebar.checkbox("Size fleet automatically", value=True)
if auto_fleet:
    slack = st.sidebar.slider(
        "Capacity slack", 1.1, 4.0, 2.0, 0.1,
        help="Fleet size = ceil(demand / capacity) x slack. "
             "Slack 1.0 leaves nothing to optimise.",
    )
    n_servers_fixed = None
else:
    slack = 2.0
    n_servers_fixed = st.sidebar.slider("Number of servers", 2, 60, 12)

server_type = st.sidebar.selectbox(
    "Server type", list(SERVER_CATALOGUE.keys()),
    help="Capacity is in trace-normalised units: 1.0 == largest machine "
         "in the cell.",
)
_st = SERVER_CATALOGUE[server_type]
c1, c2 = st.sidebar.columns(2)
idle_w = c1.number_input("Idle power (W)", 10.0, 1000.0,
                         float(_st.idle_power_w), 5.0)
max_w = c2.number_input("Max power (W)", 20.0, 2000.0,
                        float(_st.max_power_w), 5.0)
if idle_w >= max_w:
    st.sidebar.error("Idle power must be below max power.")

st.sidebar.header("4 - Objective weights")
preset = st.sidebar.selectbox(
    "Preset", ["Custom"] + list(WEIGHT_PRESETS.keys()), index=2
)
if preset == "Custom":
    alpha = st.sidebar.slider("alpha - energy", 0.0, 1.0, 0.4, 0.05)
    beta = st.sidebar.slider("beta - wastage", 0.0, 1.0, 0.3, 0.05)
    gamma = st.sidebar.slider("gamma - QoS", 0.0, 1.0, 0.3, 0.05)
else:
    w = WEIGHT_PRESETS[preset]
    alpha, beta, gamma = w.alpha, w.beta, w.gamma
    st.sidebar.caption(f"alpha={alpha}, beta={beta}, gamma={gamma}")

if alpha + beta + gamma <= 0:
    st.sidebar.error("Weights must sum to more than zero.")
    st.stop()
_wn = Weights(alpha, beta, gamma).normalised()
st.sidebar.caption(
    f"normalised: {_wn.alpha:.2f} / {_wn.beta:.2f} / {_wn.gamma:.2f}"
)

with st.sidebar.expander("5 - Advanced model settings"):
    theta = st.slider("QoS safety threshold", 0.5, 1.0, 0.85, 0.05,
                      help="Peak demand above this fraction of capacity is "
                           "penalised.")
    qos_mode = st.selectbox(
        "QoS definition", ["peak_overcommit", "failure_proxy"],
        help="'failure_proxy' is the brief's literal Q = sum(failed)/N. It is "
             "CONSTANT under the assignment constraint, so it cannot change "
             "the placement -- selecting it makes gamma inert, which the "
             "results panel will point out.",
    )
    fail_w = st.slider("Failure risk weight", 0.0, 2.0, 0.5, 0.1)
    prio_w = st.slider("Priority risk weight", 0.0, 2.0, 0.5, 0.1)
    lam = st.slider("Wastage imbalance lambda", 0.0, 2.0, 0.5, 0.1)
    horizon = st.number_input("Energy horizon (hours)", 0.1, 24.0, 1.0, 0.5)
    load_prop = st.checkbox("Load-proportional power", value=True,
                            help="Off = flat P_max per active server, the "
                                 "simpler form in section 12 of the brief.")
    power_driver = st.selectbox("Power driven by", ["average", "request"])
    time_limit = st.slider("Solver time limit (s)", 10, 600, 120, 10)
    mip_gap = st.slider("MIP gap", 0.0, 0.20, 0.01, 0.005)
    sym = st.checkbox("Symmetry breaking", value=True)
    vineq = st.checkbox("Strong linking + valid inequalities", value=True)

# ==========================================================================
# build instance
# ==========================================================================
workload = pp.sample_vms(pool, n_vms, strategy=strategy, seed=int(seed))

cfg = ScenarioConfig(
    weights=Weights(alpha, beta, gamma),
    power=PowerModelConfig(horizon_hours=horizon,
                           load_proportional=load_prop,
                           power_driver=power_driver),
    qos=QoSConfig(mode=qos_mode, safety_threshold=theta,
                  failure_weight=fail_w, priority_weight=prio_w),
    wastage=WastageConfig(imbalance_lambda=lam),
    fleet=FleetConfig(n_servers=n_servers_fixed, slack=slack,
                      composition={server_type: 1.0}),
    solver=SolverConfig(time_limit_s=time_limit, mip_gap=mip_gap,
                        symmetry_breaking=sym, valid_inequalities=vineq),
)

fleet = sv.build_fleet(workload, cfg.fleet)
fleet["idle_power_w"] = idle_w
fleet["max_power_w"] = max_w
feas = sv.feasibility_report(workload, fleet)

# ==========================================================================
tab_data, tab_model, tab_run, tab_compare, tab_sens = st.tabs(
    ["Dataset", "Model", "Optimise", "Comparison", "Sensitivity"]
)

# --------------------------------------------------------------------------
with tab_data:
    st.header("Workload")
    st.caption(f"Source: {source_label}")

    stats = pp.dataset_statistics(pool)
    k = st.columns(5)
    k[0].metric("VM instances", f"{stats['n_vms']:,}")
    k[1].metric("Collections", f"{stats['n_collections']:,}")
    k[2].metric("Clusters", stats["n_clusters"])
    k[3].metric("Trace machines", f"{stats['n_original_machines']:,}")
    k[4].metric("Failure rate", f"{stats['failure_rate']:.1%}")

    k = st.columns(5)
    k[0].metric("Total CPU demand", f"{stats['total_cpu_request']:.2f}")
    k[1].metric("Total memory demand", f"{stats['total_mem_request']:.2f}")
    k[2].metric("Mean CPU request", f"{stats['mean_cpu_request']:.4f}")
    k[3].metric("Mean peak CPU", f"{stats['mean_peak_cpu']:.4f}")
    k[4].metric("Median duration", f"{stats['median_duration_s']:.0f} s")

    st.info(
        f"Demand is in trace-normalised units where **1.0 = the largest "
        f"machine in the cell**. The whole pool needs at least "
        f"**{stats['cpu_lower_bound_servers']} servers** on CPU grounds alone, "
        f"which is why the optimiser runs on a sample rather than all "
        f"{stats['n_vms']:,} VMs."
    )

    st.subheader("Preview")
    st.dataframe(
        pool.head(200)[
            ["vm_id", "cpu_request", "mem_request", "avg_cpu", "avg_mem",
             "max_cpu", "max_mem", "priority", "priority_band", "failed",
             "duration_s", "orig_machine_id"]
        ],
        width="stretch", height=280,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Demand distribution")
        res = st.radio("Resource", ["CPU", "Memory"], horizontal=True,
                       key="dist_res")
        col = "cpu_request" if res == "CPU" else "mem_request"
        fig = px.histogram(pool, x=col, nbins=80, log_y=True,
                           color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(height=330, margin=dict(t=30, b=10),
                          yaxis_title="VM count (log)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Priority bands")
        pb = pool["priority_band"].value_counts().reset_index()
        pb.columns = ["band", "count"]
        fig = px.bar(pb, x="band", y="count",
                     color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(height=330, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CPU vs memory demand")
        s = pool.sample(min(6000, len(pool)), random_state=0)
        fig = px.scatter(s, x="cpu_request", y="mem_request",
                         color="priority_band", opacity=0.55,
                         color_discrete_sequence=PALETTE)
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=340, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        r = pool["cpu_request"].corr(pool["mem_request"])
        st.caption(
            f"Pearson r = {r:.3f}. Weak correlation is what makes the "
            "CPU/memory imbalance term in W worth having: a server can run "
            "out of one resource while the other sits idle."
        )
    with c2:
        st.subheader("Failure rate by priority band")
        fb = (pool.groupby("priority_band")
              .agg(failure_rate=("failed", "mean"), n=("vm_id", "size"))
              .reset_index())
        fig = px.bar(fb, x="priority_band", y="failure_rate",
                     hover_data=["n"], color_discrete_sequence=[PALETTE[3]])
        fig.update_layout(height=340, margin=dict(t=30, b=10),
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "`failed` marks instances whose event stream contains a "
            "FAIL/EVICT/KILL/LOST event. The trace has no SLA field, so this "
            "is a reliability **proxy**, not an SLA violation count."
        )

# --------------------------------------------------------------------------
with tab_model:
    st.header("The optimisation model")

    c = st.columns(4)
    c[0].metric("VMs (i)", len(workload))
    c[1].metric("Servers (j)", len(fleet))
    c[2].metric("Binary variables", f"{len(workload)*len(fleet)+len(fleet):,}")
    ms = milp.model_size(len(workload), len(fleet), vineq)
    c[3].metric("Constraints (approx)", f"{ms['approx_constraints']:,}")

    if len(workload) * len(fleet) > 12000:
        st.warning(
            f"{len(workload)*len(fleet):,} binary variables is large for CBC. "
            "Expect the time limit to bind; the run will fall back to the "
            "warm-start placement if no incumbent is proven."
        )

    st.subheader("Formulation")
    st.latex(r"\min\ Z = \alpha E' + \beta W' + \gamma Q'")
    st.markdown("**Decision variables**")
    st.latex(r"x_{ij}\in\{0,1\}\quad y_j\in\{0,1\}\quad d_j\ge 0"
             r"\quad o^{c}_j,\,o^{m}_j\ge 0")
    st.markdown("**Constraints**")
    st.latex(r"\sum_j x_{ij}=1\qquad \forall i")
    st.latex(r"\sum_i \mathrm{cpu}_i x_{ij}\le C_j\,y_j\qquad \forall j")
    st.latex(r"\sum_i \mathrm{mem}_i x_{ij}\le M_j\,y_j\qquad \forall j")
    st.latex(r"x_{ij}\le y_j\qquad \forall i,j")
    st.latex(r"d_j\ \ge\ \pm\left(r^{c}_j-r^{m}_j\right)\qquad \forall j")
    st.latex(r"o^{c}_j\ \ge\ \frac{1}{C_j}\sum_i \rho_i\,\hat{c}_i x_{ij}"
             r"\ -\ \theta y_j\qquad \forall j")

    with st.expander("Why the QoS term is not simply sum(failed)"):
        st.markdown(
            r"""
The project brief defines $q_i = \mathrm{failed}_i$ and $Q=\sum_i q_i$.
Under the assignment constraint $\sum_j x_{ij}=1$, every VM is placed exactly
once, so

$$\sum_i q_i = \sum_i \mathrm{failed}_i \cdot \Big(\sum_j x_{ij}\Big) = \text{constant}$$

It does not depend on $x$ at all. Adding it to the objective shifts $Z$ by a
fixed amount and changes **nothing** about which placement is optimal, so
$\gamma$ would have no effect.

This implementation therefore uses the brief's own section-13 "advanced"
definition as the *active* term: a risk-weighted peak-overcommit penalty built
from `maximum_usage`, with per-VM risk
$\rho_i = 1 + w_f\,\mathrm{failed}_i + w_p\,\mathrm{priority}_i$.
That genuinely depends on the placement and stays linear. The simple failure
rate is still computed and reported.
            """
        )

    with st.expander("Power model and its assumptions"):
        st.markdown(
            f"""
$$P_j(u) = P^{{idle}}_j + (P^{{max}}_j - P^{{idle}}_j)\\,u$$

Currently **{idle_w:.0f} W idle**, **{max_w:.0f} W peak**, horizon
**{horizon} h**.

The trace contains **no power measurements and no machine capacities**. These
figures follow the shape of published SPECpower_ssj2008 results for two-socket
x86 servers and are a documented modelling assumption, not dataset values.
The idle/max ratio is the single most consequential assumption in the project:
it is what makes consolidation worth anything. The Sensitivity tab sweeps it.
            """
        )

    st.subheader("Server fleet")
    st.dataframe(fleet, width="stretch", height=220)

    st.subheader("Feasibility pre-check")
    c = st.columns(4)
    c[0].metric("CPU demand", f"{feas['total_cpu_demand']:.3f}")
    c[1].metric("CPU capacity", f"{feas['total_cpu_capacity']:.3f}")
    c[2].metric("Memory demand", f"{feas['total_mem_demand']:.3f}")
    c[3].metric("Memory capacity", f"{feas['total_mem_capacity']:.3f}")
    lb = max(feas["min_servers_cpu_bound"], feas["min_servers_mem_bound"])
    if feas["feasible"]:
        st.success(
            f"Instance is feasible. No placement can use fewer than "
            f"**{lb}** active servers (bin-packing lower bound), so that is "
            f"the best the optimiser could possibly do."
        )
    else:
        st.error(f"Instance looks infeasible: {feas}")

# --------------------------------------------------------------------------
with tab_run:
    st.header("Run the optimisation")

    if cfg.qos.mode == "failure_proxy":
        st.warning(
            "QoS mode is **failure_proxy**, which is constant with respect to "
            "the decision variables. gamma will have no effect on the "
            "placement in this run -- that is a property of the definition, "
            "not a bug. Switch to `peak_overcommit` for an active QoS term."
        )

    go_btn = st.button("Solve", type="primary", width="stretch")

    if go_btn:
        norm = build_normalisers(workload, fleet, cfg)
        prog = st.progress(0.0, "running baselines...")
        res = baselines.run_all_baselines(workload, fleet, cfg,
                                          normalisers=norm)
        prog.progress(0.4, f"solving MILP ({len(workload)*len(fleet):,} "
                           "binaries)...")
        t0 = time.perf_counter()
        mres = milp.solve(workload, fleet, cfg, normalisers=norm)
        prog.progress(1.0, f"done in {time.perf_counter()-t0:.1f}s")
        res.append(mres)

        st.session_state["results"] = res
        st.session_state["milp"] = mres
        st.session_state["norm"] = norm
        st.session_state["cfg_snapshot"] = cfg.to_dict()

    if "milp" in st.session_state:
        mres = st.session_state["milp"]
        m = mres.metrics

        if m.get("milp_solution_used") == 0:
            st.error(
                f"**No verified MILP solution.** Status `{mres.status}`. "
                f"{mres.notes}"
            )
        elif mres.status != "Optimal":  # incl. "Feasible (time limit)"
            gap_txt = (f" Remaining optimality gap: {mres.mip_gap:.1%}."
                       if mres.mip_gap is not None else "")
            st.warning(
                f"Solver stopped with status `{mres.status}` after "
                f"{mres.runtime_s:.1f}s. The placement below is feasible but "
                f"not proven optimal.{gap_txt}"
            )
        else:
            st.success(
                f"Proven optimal in {mres.runtime_s:.1f}s "
                f"(gap target {cfg.solver.mip_gap:.1%})."
            )

        c = st.columns(5)
        c[0].metric("Active servers", m["n_active_servers"],
                    delta=f"{m['n_inactive_servers']} idle")
        c[1].metric("Energy", f"{m['energy_wh']:.0f} Wh")
        c[2].metric("CPU utilisation",
                    f"{m['cpu_util_active_mean']:.1%}")
        c[3].metric("Memory utilisation",
                    f"{m['mem_util_active_mean']:.1%}")
        c[4].metric("Objective Z", f"{m['weighted_objective']:.5f}")

        c = st.columns(5)
        c[0].metric("Wastage W", f"{m['wastage']:.3f}")
        c[1].metric("QoS penalty Q", f"{m['qos_overcommit']:.3f}")
        c[2].metric("Servers over threshold", m["n_servers_over_threshold"])
        c[3].metric("Capacity violations", m["n_capacity_violations"])
        c[4].metric("Runtime", f"{mres.runtime_s:.1f} s")

        if m.get("objective_mismatch") is not None:
            mm = m["objective_mismatch"]
            if mm > 1e-4:
                st.error(
                    f"Solver objective and evaluated objective differ by "
                    f"{mm:.2e}. The model and the metric have drifted apart."
                )
            else:
                st.caption(
                    f"Self-check: solver objective matches the independently "
                    f"evaluated objective to {mm:.1e}."
                )

        st.subheader("Server loads")
        sl = mres.server_loads
        plot = sl.copy()
        plot["state"] = np.where(plot["active"] == 1, "active", "powered off")
        fig = go.Figure()
        fig.add_bar(x=plot["server_id"], y=plot["cpu_util_request"],
                    name="CPU (requested)", marker_color=PALETTE[0])
        fig.add_bar(x=plot["server_id"], y=plot["mem_util_request"],
                    name="Memory (requested)", marker_color=PALETTE[1])
        fig.add_scatter(x=plot["server_id"], y=plot["cpu_util_peak"],
                        name="CPU (peak)", mode="markers",
                        marker=dict(color=PALETTE[3], size=9, symbol="diamond"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                      annotation_text="capacity")
        fig.add_hline(y=cfg.qos.safety_threshold, line_dash="dot",
                      line_color="orange",
                      annotation_text="QoS threshold")
        fig.update_layout(barmode="group", height=420,
                          yaxis_title="utilisation",
                          margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("VM to server assignment")
            merged = workload.merge(mres.assignment, on="vm_id", how="left")
            st.dataframe(
                merged[["vm_id", "server_id", "cpu_request", "mem_request",
                        "max_cpu", "priority_band", "failed",
                        "orig_machine_id"]],
                width="stretch", height=380,
            )
            st.download_button(
                "Download assignment CSV",
                merged.to_csv(index=False).encode(),
                "optimal_assignment.csv", "text/csv",
            )
        with c2:
            st.subheader("VMs per server")
            act = sl[sl["active"] == 1]
            fig = px.bar(act, x="server_id", y="n_vms",
                         color_discrete_sequence=[PALETTE[2]])
            fig.update_layout(height=380, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
with tab_compare:
    st.header("Baseline versus optimised")
    if "results" not in st.session_state:
        st.info("Run the optimisation first.")
    else:
        res = st.session_state["results"]
        comp = compare_results(res)

        show = [c for c in [
            "method", "n_active_servers", "energy_wh", "wastage",
            "qos_overcommit", "cpu_util_active_mean", "mem_util_active_mean",
            "weighted_objective", "n_capacity_violations", "runtime_s",
        ] if c in comp.columns]
        st.dataframe(
            comp[show].style.format({
                "energy_wh": "{:.1f}", "wastage": "{:.3f}",
                "qos_overcommit": "{:.3f}",
                "cpu_util_active_mean": "{:.1%}",
                "mem_util_active_mean": "{:.1%}",
                "weighted_objective": "{:.5f}", "runtime_s": "{:.2f}",
            }),
            width="stretch",
        )

        st.caption(
            "**Trace-Original** is the trace's own placement remapped onto the "
            "modelled fleet. The real Borg cell had thousands of machines "
            "against this fleet's handful, so it is context rather than a fair "
            "head-to-head; **First-Fit-Decreasing** is the baseline the MILP "
            "genuinely has to beat."
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(comp, x="method", y="energy_wh", color="method",
                         color_discrete_sequence=PALETTE, text_auto=".0f")
            fig.update_layout(height=360, showlegend=False,
                              yaxis_title="Energy (Wh)", margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(comp, x="method", y="n_active_servers", color="method",
                         color_discrete_sequence=PALETTE, text_auto=True)
            fig.update_layout(height=360, showlegend=False,
                              yaxis_title="Active servers", margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalised objective components")
        long = comp.melt(
            id_vars="method",
            value_vars=[c for c in ["energy_normalised", "wastage_normalised",
                                    "qos_normalised"] if c in comp.columns],
            var_name="component", value_name="value",
        )
        fig = px.bar(long, x="method", y="value", color="component",
                     barmode="group", color_discrete_sequence=PALETTE)
        fig.update_layout(height=360, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        ffd = comp[comp["method"] == "First-Fit-Decreasing"]
        mil = comp[comp["method"].str.startswith("MILP")]
        if len(ffd) and len(mil):
            e0, e1 = float(ffd["energy_wh"].iloc[0]), float(mil["energy_wh"].iloc[0])
            z0 = float(ffd["weighted_objective"].iloc[0])
            z1 = float(mil["weighted_objective"].iloc[0])
            s0 = int(ffd["n_active_servers"].iloc[0])
            s1 = int(mil["n_active_servers"].iloc[0])
            st.markdown(
                f"""
**MILP versus First-Fit-Decreasing on this instance**

| | FFD | MILP | change |
|---|---|---|---|
| active servers | {s0} | {s1} | {s1-s0:+d} |
| energy (Wh) | {e0:.1f} | {e1:.1f} | {(e1-e0)/e0*100:+.2f}% |
| objective Z | {z0:.5f} | {z1:.5f} | {(z1-z0)/z0*100:+.2f}% |
                """
            )
            if abs(z1 - z0) < 1e-9:
                st.info(
                    "The MILP matched FFD exactly here. On loosely-packed "
                    "instances the greedy heuristic is already optimal; "
                    "raise the VM count or lower the fleet slack to create an "
                    "instance where the exact method has room to win."
                )

# --------------------------------------------------------------------------
with tab_sens:
    st.header("Sensitivity analysis")
    st.caption(
        "Each sweep re-solves the MILP, so a wide sweep takes "
        "roughly (points x solve time)."
    )

    kind = st.selectbox(
        "Analysis",
        ["Objective weights", "Workload size", "Power assumption (idle/max)",
         "QoS safety threshold", "Fleet slack", "Pareto frontier"],
    )

    if st.button("Run sweep", type="primary"):
        bar = st.progress(0.0)
        status = st.empty()

        def cb(k, total, label):
            bar.progress((k + 1) / total)
            status.text(f"[{k+1}/{total}] {label}")

        with st.spinner("solving..."):
            if kind == "Objective weights":
                df = sensitivity.sweep_weights(workload, cfg, fleet=fleet,
                                               progress=cb)
                xcol, xlab = "scenario", "weight vector"
            elif kind == "Workload size":
                base = len(workload)
                sizes = [max(20, int(base * f)) for f in (0.5, 0.75, 1.0, 1.5)]
                df = sensitivity.sweep_workload_size(
                    pool, cfg, sizes, strategy=strategy, seed=int(seed),
                    progress=cb)
                xcol, xlab = "value", "number of VMs"
            elif kind == "Power assumption (idle/max)":
                df = sensitivity.sweep_power_assumption(workload, cfg,
                                                        progress=cb)
                xcol, xlab = "value", "idle power / max power"
            elif kind == "QoS safety threshold":
                df = sensitivity.sweep_parameter(
                    workload, cfg, "safety_threshold",
                    [0.6, 0.7, 0.8, 0.85, 0.9, 1.0],
                    fixed_fleet=True, progress=cb)
                xcol, xlab = "value", "safety threshold theta"
            elif kind == "Fleet slack":
                df = sensitivity.sweep_parameter(
                    workload, cfg, "fleet_slack",
                    [1.25, 1.5, 2.0, 2.5, 3.0],
                    fixed_fleet=False, progress=cb)
                xcol, xlab = "value", "fleet slack"
            else:
                df = sensitivity.pareto_frontier(workload, cfg, step=0.25,
                                                 fleet=fleet, progress=cb)
                xcol, xlab = "scenario", "weight vector"

        st.session_state["sweep"] = (kind, df, xcol, xlab)
        bar.empty()
        status.empty()

    if "sweep" in st.session_state:
        kind, df, xcol, xlab = st.session_state["sweep"]
        st.subheader(kind)
        st.dataframe(sensitivity.summarise_sweep(df),
                     width="stretch", height=280)

        if kind == "Pareto frontier":
            fig = px.scatter_3d(
                df, x="energy_normalised", y="wastage_normalised",
                z="qos_normalised", color="pareto_optimal",
                hover_name="scenario",
                color_discrete_sequence=[PALETTE[1], PALETTE[0]],
            )
            fig.update_layout(height=600, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                f"{int(df['pareto_optimal'].sum())} of {len(df)} weight "
                "combinations are non-dominated. A weighted sum can only reach "
                "the **convex** part of the Pareto front; non-convex efficient "
                "points need an epsilon-constraint method."
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_scatter(x=df[xcol], y=df["energy_wh"], name="energy",
                                mode="lines+markers",
                                line=dict(color=PALETTE[0], width=3))
                fig.update_layout(height=340, xaxis_title=xlab,
                                  yaxis_title="Energy (Wh)", margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                fig.add_scatter(x=df[xcol], y=df["n_active_servers"],
                                name="active servers", mode="lines+markers",
                                line=dict(color=PALETTE[2], width=3))
                fig.update_layout(height=340, xaxis_title=xlab,
                                  yaxis_title="Active servers",
                                  margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)

            fig = go.Figure()
            for col, nm, cl in [
                ("energy_normalised", "E' energy", PALETTE[0]),
                ("wastage_normalised", "W' wastage", PALETTE[1]),
                ("qos_normalised", "Q' QoS", PALETTE[3]),
            ]:
                if col in df.columns:
                    fig.add_scatter(x=df[xcol], y=df[col], name=nm,
                                    mode="lines+markers",
                                    line=dict(color=cl, width=3))
            fig.update_layout(height=380, xaxis_title=xlab,
                              yaxis_title="normalised component",
                              margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download sweep CSV",
                df.to_csv(index=False).encode(),
                f"sensitivity_{kind.lower().replace(' ', '_')}.csv",
                "text/csv",
            )
