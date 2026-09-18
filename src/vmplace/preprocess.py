"""Raw Google Borg trace  ->  clean per-VM workload table.

The single most important fact about this dataset is that ROWS ARE EVENTS,
NOT VMs.  A single VM instance emits several rows (ENABLE, SCHEDULE, FINISH,
FAIL, EVICT, ...).  Feeding the raw rows to the optimiser would place the same
VM several times and inflate demand, so the first job of this module is to
collapse the event stream into one record per VM instance.

VM identity
-----------
An instance is identified by the triple

    (collection_id, instance_index, cluster)

collection_id alone is a job, and instance_index alone is only unique inside a
collection, so neither identifies a VM on its own.

Resource units
--------------
resource_request / average_usage / maximum_usage are dict-valued strings such
as  "{'cpus': 0.0206, 'memory': 0.0144}".  Values are normalised so that 1.0 is
the capacity of the largest machine in the cell.  They are parsed with a
vectorised regex rather than ast.literal_eval, which matters at 1.3M rows.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_TRACE_PATH

# Columns we actually need.  Reading a subset keeps a 328 MB CSV manageable.
USE_COLUMNS: List[str] = [
    "time",
    "instance_events_type",
    "collection_id",
    "scheduling_class",
    "collection_type",
    "priority",
    "instance_index",
    "machine_id",
    "resource_request",
    "start_time",
    "end_time",
    "average_usage",
    "maximum_usage",
    "assigned_memory",
    "page_cache_memory",
    "cycles_per_instruction",
    "sample_rate",
    "cluster",
    "event",
    "failed",
]

VM_KEY = ["collection_id", "instance_index", "cluster"]

# Google instance event type codes (trace documentation, 2019 format).
EVENT_SCHEDULE = 6
EVENT_FAIL = 2
EVENT_FINISH = 3
EVENT_EVICT = 1
EVENT_KILL = 4
EVENT_LOST = 5

# Priority bands from the Borg documentation.  Used only for human-readable
# labelling and for the QoS risk weight; the numeric priority is kept as well.
PRIORITY_BANDS = [
    (0, 99, "free"),
    (100, 115, "best_effort"),
    (116, 119, "mid"),
    (120, 359, "production"),
    (360, 10_000, "monitoring"),
]


def _extract_pair(series: pd.Series, key: str) -> pd.Series:
    """Pull one float out of a dict-valued string column, vectorised.

    Returns NaN where the key is absent or literally ``None`` (the trace uses
    ``'memory': None`` in some usage records).
    """
    pattern = r"'" + key + r"':\s*([0-9eE+.\-]+)"
    out = series.astype("string").str.extract(pattern, expand=False)
    return pd.to_numeric(out, errors="coerce")


def priority_band(p: float) -> str:
    for lo, hi, name in PRIORITY_BANDS:
        if lo <= p <= hi:
            return name
    return "unknown"


def load_raw(
    path: str = RAW_TRACE_PATH,
    nrows: Optional[int] = None,
    chunksize: int = 250_000,
    clusters: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Stream the raw CSV and return only the columns we model on.

    Parameters
    ----------
    nrows
        Cap on raw event rows read.  ``None`` reads the whole 1.32M-row file.
    clusters
        Optional filter on the ``cluster`` column (the trace covers 8 cells).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"raw trace not found at {path}")

    frames: List[pd.DataFrame] = []
    seen = 0
    reader = pd.read_csv(
        path,
        usecols=USE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        if clusters is not None:
            chunk = chunk[chunk["cluster"].isin(list(clusters))]
        frames.append(chunk)
        seen += len(chunk)
        if nrows is not None and seen >= nrows:
            break

    df = pd.concat(frames, ignore_index=True)
    if nrows is not None:
        df = df.head(nrows)
    return df


def build_vm_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the event stream into one row per VM instance.

    Aggregation rules, and why each was chosen:

    cpu_request / mem_request
        max over the instance's events.  resource_request can change across
        events when a job is vertically scaled; the max is the capacity the
        placement must actually be able to honour.
    avg_cpu / avg_mem
        mean of average_usage over events that reported usage.
    max_cpu / max_mem
        max of maximum_usage, i.e. the observed peak.  Drives the QoS term.
    priority
        max over events (priority is effectively constant per instance).
    failed
        1 if ANY event for the instance is a failure/eviction/kill/loss.  The
        raw ``failed`` column is 1 exactly on FAIL events, so taking the max
        over the instance turns a per-event flag into a per-VM label.
    orig_machine_id
        machine from the SCHEDULE event when present, else the most frequent
        non-zero machine seen.  This is the trace's own placement and is used
        only as a baseline for comparison, never as an optimiser input.
    duration_s
        (max end_time - min start_time) in seconds; trace times are in
        microseconds since the start of the trace window.
    """
    df = raw.copy()

    # ---- parse the dict-valued resource columns -------------------------
    df["cpu_request"] = _extract_pair(df["resource_request"], "cpus")
    df["mem_request"] = _extract_pair(df["resource_request"], "memory")
    df["avg_cpu_ev"] = _extract_pair(df["average_usage"], "cpus")
    df["avg_mem_ev"] = _extract_pair(df["average_usage"], "memory")
    df["max_cpu_ev"] = _extract_pair(df["maximum_usage"], "cpus")
    df["max_mem_ev"] = _extract_pair(df["maximum_usage"], "memory")

    # ---- per-VM failure label ------------------------------------------
    bad_events = {EVENT_FAIL, EVENT_EVICT, EVENT_KILL, EVENT_LOST}
    df["ev_failed"] = (
        df["failed"].fillna(0).astype(int)
        | df["instance_events_type"].isin(bad_events).astype(int)
    )
    df["ev_hard_fail"] = df["failed"].fillna(0).astype(int)

    # ---- machine actually used, from the SCHEDULE event -----------------
    sched = df["instance_events_type"] == EVENT_SCHEDULE
    df["sched_machine"] = df["machine_id"].where(sched)

    agg = df.groupby(VM_KEY, sort=False).agg(
        cpu_request=("cpu_request", "max"),
        mem_request=("mem_request", "max"),
        avg_cpu=("avg_cpu_ev", "mean"),
        avg_mem=("avg_mem_ev", "mean"),
        max_cpu=("max_cpu_ev", "max"),
        max_mem=("max_mem_ev", "max"),
        priority=("priority", "max"),
        scheduling_class=("scheduling_class", "max"),
        collection_type=("collection_type", "max"),
        failed=("ev_failed", "max"),
        hard_failed=("ev_hard_fail", "max"),
        assigned_memory=("assigned_memory", "max"),
        cycles_per_instruction=("cycles_per_instruction", "mean"),
        n_events=("time", "size"),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        sched_machine=("sched_machine", "max"),
        any_machine=("machine_id", "max"),
    )
    agg = agg.reset_index()

    agg["orig_machine_id"] = (
        agg["sched_machine"].fillna(agg["any_machine"]).astype("Int64")
    )
    agg = agg.drop(columns=["sched_machine", "any_machine"])

    # ---- derived fields -------------------------------------------------
    agg["duration_s"] = (agg["end_time"] - agg["start_time"]) / 1e6
    agg["priority_band"] = agg["priority"].map(priority_band)
    # Normalised priority in [0,1], used by the QoS risk weight.
    pmax = float(agg["priority"].max()) if len(agg) else 1.0
    agg["priority_norm"] = agg["priority"] / pmax if pmax > 0 else 0.0

    agg["vm_id"] = [f"VM{i:05d}" for i in range(1, len(agg) + 1)]

    cols = [
        "vm_id",
        "collection_id",
        "instance_index",
        "cluster",
        "cpu_request",
        "mem_request",
        "avg_cpu",
        "avg_mem",
        "max_cpu",
        "max_mem",
        "priority",
        "priority_band",
        "priority_norm",
        "scheduling_class",
        "collection_type",
        "failed",
        "hard_failed",
        "duration_s",
        "start_time",
        "end_time",
        "n_events",
        "orig_machine_id",
        "assigned_memory",
        "cycles_per_instruction",
    ]
    return agg[cols]


def clean_vm_table(
    vms: pd.DataFrame,
    min_cpu: float = 1e-6,
    min_mem: float = 1e-6,
    max_cpu_request: float = 1.0,
) -> pd.DataFrame:
    """Drop records the optimiser cannot use, and repair the repairable ones.

    Rules
    -----
    * A VM with no CPU or no memory request cannot be placed meaningfully; such
      rows are dropped and counted.
    * A request above the largest machine's capacity (1.0) is infeasible by
      construction and is dropped.
    * Missing usage values fall back to the request, which is the conservative
      assumption (assume it uses what it asked for).
    * max_usage below avg_usage is inconsistent; max is raised to avg.
    """
    df = vms.copy()
    n0 = len(df)

    df["cpu_request"] = pd.to_numeric(df["cpu_request"], errors="coerce")
    df["mem_request"] = pd.to_numeric(df["mem_request"], errors="coerce")

    dropped_missing = int(
        df["cpu_request"].isna().sum() + df["mem_request"].isna().sum()
    )
    df = df.dropna(subset=["cpu_request", "mem_request"])

    too_small = (df["cpu_request"] < min_cpu) | (df["mem_request"] < min_mem)
    df = df[~too_small]

    too_big = (df["cpu_request"] > max_cpu_request) | (
        df["mem_request"] > max_cpu_request
    )
    df = df[~too_big]

    # usage falls back to request
    df["avg_cpu"] = df["avg_cpu"].fillna(df["cpu_request"])
    df["avg_mem"] = df["avg_mem"].fillna(df["mem_request"])
    df["max_cpu"] = df["max_cpu"].fillna(df["cpu_request"])
    df["max_mem"] = df["max_mem"].fillna(df["mem_request"])

    # enforce avg <= max
    df["max_cpu"] = df[["max_cpu", "avg_cpu"]].max(axis=1)
    df["max_mem"] = df[["max_mem", "avg_mem"]].max(axis=1)

    # peak usage can exceed the request in the trace (bursting); cap it at the
    # largest machine so the QoS term stays on a comparable scale.
    df["max_cpu"] = df["max_cpu"].clip(upper=1.0)
    df["max_mem"] = df["max_mem"].clip(upper=1.0)

    df = df.reset_index(drop=True)
    df.attrs["n_input"] = n0
    df.attrs["n_output"] = len(df)
    df.attrs["n_dropped"] = n0 - len(df)
    df.attrs["n_dropped_missing"] = dropped_missing
    df.attrs["n_dropped_too_small"] = int(too_small.sum())
    df.attrs["n_dropped_too_big"] = int(too_big.sum())
    return df


def sample_vms(
    vms: pd.DataFrame,
    n: int,
    strategy: str = "stratified",
    seed: int = 42,
    band: tuple = (0.20, 0.45),
) -> pd.DataFrame:
    """Take a workload of n VMs for a tractable MILP instance.

    A MILP over 238k VMs is not solvable, so experiments run on a sample.  The
    sampling strategy is part of the experimental design and is recorded with
    every result.

    strategy
        "random"      uniform random sample.
        "stratified"  preserves the priority-band mix of the full workload, so
                      the sample keeps the real production/best-effort ratio.
                      This is the representative choice and the default.
        "largest"     the n largest CPU requests.
        "band"        VMs whose CPU request falls in ``band``.  A deliberate
                      STRESS instance rather than a representative one, and
                      worth explaining.  In this trace the median VM asks for
                      under 1% of a machine, and bin packing with items that
                      tiny is easy -- First-Fit-Decreasing reaches the
                      bin-packing lower bound and an exact method has nothing
                      left to win.  Items sized at roughly a third of a machine
                      are where packing decisions actually bite, so this
                      strategy isolates the regime in which exact optimisation
                      earns its cost.
        "head"        the first n rows, for reproducible debugging.
    """
    n = int(min(n, len(vms)))
    if strategy == "head":
        out = vms.head(n)
    elif strategy == "largest":
        out = vms.nlargest(n, "cpu_request")
    elif strategy == "band":
        lo, hi = band
        sub = vms[(vms["cpu_request"] >= lo) & (vms["cpu_request"] < hi)]
        if len(sub) == 0:
            raise ValueError(
                f"no VMs have cpu_request in [{lo}, {hi}); widen the band"
            )
        if len(sub) < n:
            out = sub
        else:
            out = sub.sample(n=n, random_state=seed)
    elif strategy == "random":
        out = vms.sample(n=n, random_state=seed)
    elif strategy == "stratified":
        # Sample index labels per band rather than using groupby.apply, which
        # keeps the grouping column in the result without tripping pandas'
        # deprecation of apply-over-grouping-columns.
        frac = n / len(vms)
        picks = []
        for _, idx in vms.groupby("priority_band", sort=False).groups.items():
            take = max(1, int(round(len(idx) * frac)))
            take = min(take, len(idx))
            picks.append(
                pd.Index(idx).to_series().sample(n=take, random_state=seed)
            )
        chosen = pd.Index(pd.concat(picks).values) if picks else vms.index[:0]
        out = vms.loc[chosen]

        if len(out) > n:
            out = out.sample(n=n, random_state=seed)
        elif len(out) < n:
            rest = vms.drop(out.index)
            extra = rest.sample(n=min(n - len(out), len(rest)), random_state=seed)
            out = pd.concat([out, extra])
    else:
        raise ValueError(f"unknown sampling strategy: {strategy}")

    return out.reset_index(drop=True)


def run_pipeline(
    path: str = RAW_TRACE_PATH,
    nrows: Optional[int] = None,
    clusters: Optional[Iterable[int]] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """RAW TRACE -> select -> parse -> collapse -> clean -> VM table."""
    raw = load_raw(path, nrows=nrows, clusters=clusters)
    vms = build_vm_table(raw)
    vms = clean_vm_table(vms)

    if cache:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        tag = "full" if nrows is None else f"n{nrows}"
        out = os.path.join(PROCESSED_DIR, f"vm_table_{tag}.parquet")
        try:
            vms.to_parquet(out, index=False)
        except Exception:
            vms.to_csv(out.replace(".parquet", ".csv"), index=False)
    return vms


def dataset_statistics(vms: pd.DataFrame) -> dict:
    """The statistics the UI exposes (section 26 of the project brief)."""
    if len(vms) == 0:
        return {}
    total_cpu = float(vms["cpu_request"].sum())
    total_mem = float(vms["mem_request"].sum())
    return {
        "n_vms": int(len(vms)),
        "n_collections": int(vms["collection_id"].nunique()),
        "n_clusters": int(vms["cluster"].nunique()),
        "n_original_machines": int(vms["orig_machine_id"].nunique()),
        "total_cpu_request": total_cpu,
        "total_mem_request": total_mem,
        "mean_cpu_request": float(vms["cpu_request"].mean()),
        "mean_mem_request": float(vms["mem_request"].mean()),
        "median_cpu_request": float(vms["cpu_request"].median()),
        "median_mem_request": float(vms["mem_request"].median()),
        "max_cpu_request": float(vms["cpu_request"].max()),
        "max_mem_request": float(vms["mem_request"].max()),
        "mean_avg_cpu": float(vms["avg_cpu"].mean()),
        "mean_avg_mem": float(vms["avg_mem"].mean()),
        "mean_peak_cpu": float(vms["max_cpu"].mean()),
        "mean_peak_mem": float(vms["max_mem"].mean()),
        "failure_count": int(vms["failed"].sum()),
        "failure_rate": float(vms["failed"].mean()),
        "hard_failure_rate": float(vms["hard_failed"].mean()),
        "mean_duration_s": float(vms["duration_s"].mean()),
        "median_duration_s": float(vms["duration_s"].median()),
        "priority_distribution": vms["priority_band"]
        .value_counts()
        .to_dict(),
        # Lower bound on servers of capacity 1.0, ignoring integrality.
        "cpu_lower_bound_servers": int(np.ceil(total_cpu)),
        "mem_lower_bound_servers": int(np.ceil(total_mem)),
    }
