"""Stage 2: exploratory data analysis on the processed VM table.

Produces the statistics and figures that justify the modelling choices, and
writes them to outputs/eda/.

Usage
-----
    python scripts/02_eda.py
    python scripts/02_eda.py --vm-table data/processed/vm_table_full.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmplace import preprocess as pp
from vmplace.config import OUTPUT_DIR, PROCESSED_DIR


def latest_vm_table() -> str:
    cands = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "vm_table_*.parquet"))
        + glob.glob(os.path.join(PROCESSED_DIR, "vm_table_*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not cands:
        raise SystemExit(
            "no processed VM table found -- run scripts/01_preprocess.py first"
        )
    return cands[0]


def load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vm-table", default=None)
    args = ap.parse_args()

    path = args.vm_table or latest_vm_table()
    print(f"loading {path}")
    vms = load(path)

    outdir = os.path.join(OUTPUT_DIR, "eda")
    os.makedirs(outdir, exist_ok=True)

    stats = pp.dataset_statistics(vms)
    with open(os.path.join(outdir, "statistics.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    print("\n--- workload statistics ---")
    for k, v in stats.items():
        print(f"  {k:28} {v}")

    # ---- the numbers that drive the modelling decisions -----------------
    print("\n--- why a sample is needed ---")
    print(f"  a MILP over all {stats['n_vms']:,} VMs and even 20 servers would")
    print(f"  need {stats['n_vms']*20:,} binary variables, which CBC cannot")
    print("  solve.  Experiments therefore run on stratified samples.")

    print("\n--- demand concentration ---")
    for q in [0.5, 0.9, 0.99]:
        print(f"  cpu_request p{int(q*100):<3} {vms['cpu_request'].quantile(q):.5f}"
              f"   mem_request p{int(q*100):<3} {vms['mem_request'].quantile(q):.5f}")
    top1 = vms.nlargest(max(1, len(vms) // 100), "cpu_request")["cpu_request"].sum()
    print(f"  top 1% of VMs hold {top1/vms['cpu_request'].sum():.1%} of all CPU demand")

    print("\n--- request versus actual usage (over-provisioning) ---")
    ratio_cpu = (vms["avg_cpu"] / vms["cpu_request"].replace(0, np.nan)).dropna()
    ratio_mem = (vms["avg_mem"] / vms["mem_request"].replace(0, np.nan)).dropna()
    print(f"  mean avg_cpu / cpu_request = {ratio_cpu.mean():.3f} "
          f"(median {ratio_cpu.median():.3f})")
    print(f"  mean avg_mem / mem_request = {ratio_mem.mean():.3f} "
          f"(median {ratio_mem.median():.3f})")
    print("  Requests exceed actual usage substantially, which is exactly the")
    print("  slack that consolidation exploits -- and the reason the power")
    print("  model is driven by average_usage rather than resource_request.")

    print("\n--- failure rate by priority band ---")
    fb = vms.groupby("priority_band").agg(
        n=("vm_id", "size"), failure_rate=("failed", "mean"),
        mean_cpu=("cpu_request", "mean"),
    ).sort_values("failure_rate", ascending=False)
    print(fb.to_string())

    # ---- figures --------------------------------------------------------
    figs = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(vms["cpu_request"], bins=80, log=True, color="#4C72B0")
    ax[0].set_xlabel("CPU request (normalised)")
    ax[0].set_ylabel("VM count (log)")
    ax[0].set_title("CPU demand distribution")
    ax[1].hist(vms["mem_request"], bins=80, log=True, color="#DD8452")
    ax[1].set_xlabel("Memory request (normalised)")
    ax[1].set_title("Memory demand distribution")
    fig.tight_layout()
    p = os.path.join(outdir, "demand_distributions.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sample = vms.sample(min(20000, len(vms)), random_state=0)
    ax.scatter(sample["cpu_request"], sample["mem_request"], s=4, alpha=0.25,
               color="#55A868")
    ax.set_xlabel("CPU request")
    ax.set_ylabel("Memory request")
    ax.set_title("CPU vs memory demand\n(shape drives the imbalance term in W)")
    corr = vms["cpu_request"].corr(vms["mem_request"])
    ax.annotate(f"pearson r = {corr:.3f}", xy=(0.05, 0.92),
                xycoords="axes fraction")
    fig.tight_layout()
    p = os.path.join(outdir, "cpu_vs_memory.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)
    print(f"\n  cpu/mem request correlation r = {corr:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fb["n"].plot(kind="bar", ax=ax[0], color="#4C72B0")
    ax[0].set_title("VMs per priority band")
    ax[0].set_ylabel("count")
    fb["failure_rate"].plot(kind="bar", ax=ax[1], color="#C44E52")
    ax[1].set_title("Failure rate per priority band")
    ax[1].set_ylabel("rate")
    fig.tight_layout()
    p = os.path.join(outdir, "priority_and_failures.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    peak_ratio = (
        vms["max_cpu"] / vms["avg_cpu"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(peak_ratio.clip(upper=20), bins=60, color="#8172B3")
    ax.set_xlabel("peak CPU / average CPU")
    ax.set_ylabel("VM count")
    ax.set_title("Burstiness\n(why the QoS term uses maximum_usage)")
    fig.tight_layout()
    p = os.path.join(outdir, "burstiness.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)
    print(f"  median peak/avg CPU ratio = {peak_ratio.median():.2f}")

    print("\nfigures written:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
