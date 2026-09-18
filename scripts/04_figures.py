"""Stage 4: report figures from the experiment CSVs.

Reads outputs/experiments/*.csv (written by scripts/03_run_experiments.py) and
writes publication-ready PNGs to outputs/figures/.  Keeping this separate from
the solving means figures can be restyled without re-running the optimisation.

Usage
-----
    python scripts/04_figures.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmplace.config import OUTPUT_DIR

EXP = os.path.join(OUTPUT_DIR, "experiments")
FIG = os.path.join(OUTPUT_DIR, "figures")

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _read(name: str) -> pd.DataFrame | None:
    p = os.path.join(EXP, name)
    if not os.path.exists(p):
        print(f"  skip {name} (not found -- run scripts/03 first)")
        return None
    return pd.read_csv(p)


def fig_comparison() -> None:
    df = _read("comparison.csv")
    if df is None:
        return
    df = df.copy()
    df["short"] = df["method"].str.replace("First-Fit-Decreasing", "FFD")
    df["short"] = df["short"].str.replace(" (warm-start fallback)", "*",
                                          regex=False)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]

    ax[0].bar(df["short"], df["energy_wh"], color=colors)
    ax[0].set_ylabel("Energy (Wh)")
    ax[0].set_title("Energy consumption")

    ax[1].bar(df["short"], df["n_active_servers"], color=colors)
    ax[1].set_ylabel("Active servers")
    ax[1].set_title("Servers left powered on")

    ax[2].bar(df["short"], df["cpu_util_active_mean"] * 100, color=colors)
    ax[2].set_ylabel("Mean CPU utilisation of active servers (%)")
    ax[2].set_title("Utilisation")

    for a in ax:
        a.tick_params(axis="x", rotation=30)
        for lbl in a.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle("Baseline versus optimised placement", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "comparison.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_objective_components() -> None:
    df = _read("comparison.csv")
    if df is None:
        return
    cols = ["energy_normalised", "wastage_normalised", "qos_normalised"]
    if not all(c in df.columns for c in cols):
        return
    df = df.copy()
    df["short"] = df["method"].str.replace("First-Fit-Decreasing", "FFD")

    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, (c, lab) in enumerate(zip(cols, ["E' energy", "W' wastage",
                                            "Q' QoS risk"])):
        ax.bar(x + (k - 1) * w, df[c], w, label=lab, color=PALETTE[k])
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=25, ha="right")
    ax.set_ylabel("normalised component")
    ax.set_title("Objective components by method (lower is better)",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(FIG, "objective_components.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_weight_sensitivity() -> None:
    df = _read("sensitivity_weights.csv")
    if df is None:
        return
    # Order by the QoS weight so the plot reads as a trade-off curve rather
    # than a zigzag through an arbitrary scenario ordering.
    if "gamma" in df.columns:
        df = df.sort_values("gamma").reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))

    x = np.arange(len(df))
    ax[0].plot(x, df["energy_wh"], "o-", color=PALETTE[0], lw=2,
               label="energy (Wh)")
    ax[0].set_ylabel("Energy (Wh)", color=PALETTE[0])
    ax[0].tick_params(axis="y", labelcolor=PALETTE[0])
    twin = ax[0].twinx()
    twin.plot(x, df["qos_overcommit"], "s--", color=PALETTE[3], lw=2,
              label="QoS penalty")
    twin.set_ylabel("QoS penalty Q", color=PALETTE[3])
    twin.tick_params(axis="y", labelcolor=PALETTE[3])
    twin.grid(False)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df["scenario"], rotation=40, ha="right", fontsize=8)
    ax[0].set_title("The energy / QoS trade-off", fontweight="bold")

    ax[1].plot(x, df["n_active_servers"], "o-", color=PALETTE[2], lw=2)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(df["scenario"], rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("Active servers")
    ax[1].set_title("Consolidation versus objective weights",
                    fontweight="bold")

    fig.tight_layout()
    p = os.path.join(FIG, "sensitivity_weights.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_power_sensitivity() -> None:
    df = _read("sensitivity_power.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(df["value"], df["n_active_servers"], "o-", color=PALETTE[0],
               lw=2)
    ax[0].set_xlabel("idle power / max power")
    ax[0].set_ylabel("Active servers chosen")
    ax[0].set_title("Does the assumed idle power change the answer?",
                    fontweight="bold")

    ax[1].plot(df["value"], df["energy_wh"], "s-", color=PALETTE[1], lw=2)
    ax[1].set_xlabel("idle power / max power")
    ax[1].set_ylabel("Energy (Wh)")
    ax[1].set_title("Energy under each power assumption", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "sensitivity_power.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_workload_sensitivity() -> None:
    df = _read("sensitivity_workload.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(df["value"], df["n_active_servers"], "o-", color=PALETTE[0],
               lw=2, label="active")
    if "n_servers_in_fleet" in df.columns:
        ax[0].plot(df["value"], df["n_servers_in_fleet"], "s--",
                   color=PALETTE[4], lw=1.5, label="fleet size")
    ax[0].set_xlabel("number of VMs")
    ax[0].set_ylabel("servers")
    ax[0].set_title("Servers used as workload grows")
    ax[0].legend()

    ax[1].plot(df["value"], df["energy_wh"], "o-", color=PALETTE[1], lw=2)
    ax[1].set_xlabel("number of VMs")
    ax[1].set_ylabel("Energy (Wh)")
    ax[1].set_title("Energy as workload grows")

    ax[2].plot(df["value"], df["runtime_s"], "o-", color=PALETTE[3], lw=2)
    ax[2].set_xlabel("number of VMs")
    ax[2].set_ylabel("solve time (s)")
    ax[2].set_title("Solve time -- the practical limit")
    fig.suptitle("Workload-size sensitivity", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "sensitivity_workload.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_threshold_sensitivity() -> None:
    df = _read("sensitivity_threshold.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(df["value"], df["n_active_servers"], "o-", color=PALETTE[0], lw=2,
            label="active servers")
    ax.set_xlabel(r"QoS safety threshold $\theta$")
    ax.set_ylabel("Active servers", color=PALETTE[0])
    ax.tick_params(axis="y", labelcolor=PALETTE[0])
    twin = ax.twinx()
    twin.plot(df["value"], df["energy_wh"], "s--", color=PALETTE[1], lw=2)
    twin.set_ylabel("Energy (Wh)", color=PALETTE[1])
    twin.tick_params(axis="y", labelcolor=PALETTE[1])
    twin.grid(False)
    ax.set_title("Cost of the QoS safety margin", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "sensitivity_threshold.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_scaling() -> None:
    df = _read("scaling.csv")
    if df is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(df["binary_vars"], df["runtime_s"], "o-", color=PALETTE[0], lw=2)
    ax[0].set_xlabel("binary variables")
    ax[0].set_ylabel("solve time (s)")
    ax[0].set_title("Solve time versus model size", fontweight="bold")
    ax[0].set_yscale("log")

    ax[1].plot(df["n_vms"], df["n_active_servers"], "o-", color=PALETTE[2],
               lw=2)
    ax[1].set_xlabel("number of VMs")
    ax[1].set_ylabel("active servers")
    ax[1].set_title("Consolidation across instance sizes", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "scaling.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_server_loads() -> None:
    df = _read("optimal_server_loads.csv")
    if df is None:
        return
    df = df.sort_values("server_id")
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(x - w / 2, df["cpu_util_request"], w, label="CPU (requested)",
           color=PALETTE[0])
    ax.bar(x + w / 2, df["mem_util_request"], w, label="Memory (requested)",
           color=PALETTE[1])
    if "cpu_util_peak" in df.columns:
        ax.plot(x, df["cpu_util_peak"], "D", color=PALETTE[3], ms=6,
                label="CPU (peak)")
    ax.axhline(1.0, ls="--", color="red", lw=1.2, label="capacity")
    ax.set_xticks(x)
    ax.set_xticklabels(df["server_id"], rotation=60, fontsize=7)
    ax.set_ylabel("utilisation")
    ax.set_title("Optimised placement: per-server utilisation",
                 fontweight="bold")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "server_loads.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def fig_pareto() -> None:
    df = _read("pareto.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5))
    dom = df[~df["pareto_optimal"]]
    opt = df[df["pareto_optimal"]]
    ax.scatter(dom["energy_normalised"], dom["qos_normalised"], s=35,
               color="#bbbbbb", label="dominated")
    ax.scatter(opt["energy_normalised"], opt["qos_normalised"], s=70,
               color=PALETTE[3], label="non-dominated", zorder=3)
    ax.set_xlabel("E' normalised energy")
    ax.set_ylabel("Q' normalised QoS risk")
    ax.set_title("Pareto frontier: energy versus QoS", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(FIG, "pareto.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p}")


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    print(f"reading {EXP}")
    print("writing figures:")
    for fn in [
        fig_comparison,
        fig_objective_components,
        fig_weight_sensitivity,
        fig_power_sensitivity,
        fig_workload_sensitivity,
        fig_threshold_sensitivity,
        fig_scaling,
        fig_server_loads,
        fig_pareto,
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  {fn.__name__} failed: {type(exc).__name__}: {exc}")
    print(f"\nfigures in {FIG}")


if __name__ == "__main__":
    main()
