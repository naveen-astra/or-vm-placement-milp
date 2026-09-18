"""Baseline placement policies, for the comparison in section 22 of the brief.

An optimisation result means nothing without something to beat.  Four baselines
are provided, in increasing order of how hard they are to beat:

trace_original
    The placement Borg itself chose, read from machine_id.  This is the only
    baseline that is real rather than simulated, but it is not directly
    comparable to the others: the trace's machines are a different, much larger
    fleet than our modelled one, so its "active server" count is the number of
    distinct machines the sampled VMs happened to land on.  Reported for
    context, and flagged as such, rather than used as the headline comparison.

first_fit
    Place each VM on the first server with room.  The naive policy.

best_fit
    Place each VM on the server that will have the least leftover room.  The
    standard greedy consolidation heuristic.

first_fit_decreasing
    Sort VMs by descending size, then first-fit.  Has a known 11/9 OPT + 1
    approximation guarantee for one-dimensional bin packing, and is the
    genuinely strong baseline the MILP has to justify itself against.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import ScenarioConfig
from .metrics import Normalisers, PlacementResult, build_normalisers, evaluate_placement


def _greedy(
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    order: List[int],
    rule: str,
) -> pd.DataFrame:
    """Shared greedy loop.  ``rule`` is 'first' or 'best'."""
    cap_c = servers["cpu_capacity"].to_numpy(dtype=float).copy()
    cap_m = servers["mem_capacity"].to_numpy(dtype=float).copy()
    rem_c = cap_c.copy()
    rem_m = cap_m.copy()
    sids = servers["server_id"].tolist()
    opened = np.zeros(len(sids), dtype=bool)

    cpu = vms["cpu_request"].to_numpy(dtype=float)
    mem = vms["mem_request"].to_numpy(dtype=float)
    vids = vms["vm_id"].tolist()

    out: List[dict] = []
    for i in order:
        fits = (rem_c >= cpu[i] - 1e-12) & (rem_m >= mem[i] - 1e-12)
        if not fits.any():
            out.append({"vm_id": vids[i], "server_id": None})
            continue

        if rule == "first":
            # Prefer an already-open server, which is what makes first-fit a
            # consolidating policy rather than a scattering one.
            cand = np.where(fits & opened)[0]
            j = int(cand[0]) if len(cand) else int(np.where(fits)[0][0])
        else:  # best fit: minimise leftover room, normalised across resources
            leftover = np.full(len(sids), np.inf)
            idx = np.where(fits)[0]
            leftover[idx] = (
                (rem_c[idx] - cpu[i]) / cap_c[idx]
                + (rem_m[idx] - mem[i]) / cap_m[idx]
            )
            # break ties toward already-open servers
            leftover[idx] -= 1e-6 * opened[idx]
            j = int(np.argmin(leftover))

        rem_c[j] -= cpu[i]
        rem_m[j] -= mem[i]
        opened[j] = True
        out.append({"vm_id": vids[i], "server_id": sids[j]})

    return pd.DataFrame(out)


def first_fit(vms: pd.DataFrame, servers: pd.DataFrame) -> pd.DataFrame:
    return _greedy(vms, servers, list(range(len(vms))), "first")


def best_fit(vms: pd.DataFrame, servers: pd.DataFrame) -> pd.DataFrame:
    return _greedy(vms, servers, list(range(len(vms))), "best")


def first_fit_decreasing(vms: pd.DataFrame, servers: pd.DataFrame) -> pd.DataFrame:
    # Order by the larger of the two normalised demands, which is the usual
    # generalisation of FFD to vector bin packing.
    size = np.maximum(
        vms["cpu_request"].to_numpy(dtype=float)
        / float(servers["cpu_capacity"].max()),
        vms["mem_request"].to_numpy(dtype=float)
        / float(servers["mem_capacity"].max()),
    )
    order = list(np.argsort(-size))
    return _greedy(vms, servers, order, "first")


def trace_original(vms: pd.DataFrame, servers: pd.DataFrame) -> pd.DataFrame:
    """The trace's own placement, remapped onto our modelled fleet.

    The trace used thousands of distinct machines; our fleet has tens.  We map
    each distinct original machine to a modelled server round-robin, which
    preserves the *grouping* Borg chose (VMs that shared a machine still share
    a server) while fitting the modelled fleet.  This can and does overflow
    capacity, and metrics.py reports those violations rather than hiding them.
    """
    machines = vms["orig_machine_id"].astype("Int64")
    uniq = machines.dropna().unique().tolist()
    sids = servers["server_id"].tolist()
    mapping = {m: sids[k % len(sids)] for k, m in enumerate(uniq)}
    return pd.DataFrame(
        {
            "vm_id": vms["vm_id"],
            "server_id": [
                mapping.get(m) if pd.notna(m) else None for m in machines
            ],
        }
    )


BASELINES = {
    "First-Fit": first_fit,
    "Best-Fit": best_fit,
    "First-Fit-Decreasing": first_fit_decreasing,
    "Trace-Original": trace_original,
}


def run_baseline(
    name: str,
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    cfg: ScenarioConfig,
    normalisers: Optional[Normalisers] = None,
) -> PlacementResult:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'")
    norm = normalisers or build_normalisers(vms, servers, cfg)
    t0 = time.perf_counter()
    assignment = BASELINES[name](vms, servers)
    dt = time.perf_counter() - t0
    return evaluate_placement(
        vms, servers, assignment, cfg,
        method=name, normalisers=norm, runtime_s=dt,
        notes="heuristic baseline" if name != "Trace-Original"
        else "trace placement remapped onto the modelled fleet",
    )


def run_all_baselines(
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    cfg: ScenarioConfig,
    names: Optional[List[str]] = None,
    normalisers: Optional[Normalisers] = None,
) -> List[PlacementResult]:
    norm = normalisers or build_normalisers(vms, servers, cfg)
    names = names or list(BASELINES.keys())
    return [run_baseline(n, vms, servers, cfg, norm) for n in names]
