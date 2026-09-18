"""Construction of the physical server fleet.

IMPORTANT, and stated plainly because the project brief insists on it: the
Google 2019 trace subset used here contains machine *identifiers* but neither
machine *capacities* nor machine *power draw*.  The fleet in this module is
therefore a documented model, not a dataset extract.  What the trace does fix
is the unit system: all demand is normalised so that 1.0 equals the capacity of
the largest machine in the cell, so a server with cpu_capacity 1.0 is exactly
that largest machine.  Every power figure comes from config.SERVER_CATALOGUE
and is flagged as an assumption there.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd

from .config import SERVER_CATALOGUE, FleetConfig, ServerType


def size_fleet(vms: pd.DataFrame, cfg: FleetConfig, driver: str = "request") -> int:
    """Choose how many servers the fleet should contain.

    The optimiser can only switch servers off, never conjure them, so the fleet
    must be large enough to be feasible and loose enough that consolidation is
    a real choice.  We take the bin-packing lower bound on both resources and
    multiply by a slack factor.

    A fleet sized exactly at the lower bound would make the problem a pure
    feasibility question with nothing to optimise; a fleet many times too large
    just adds symmetric, never-used binaries.  slack = 2.0 is the default.
    """
    if cfg.n_servers is not None:
        return int(max(1, cfg.n_servers))

    cpu_col = "cpu_request" if driver == "request" else "avg_cpu"
    mem_col = "mem_request" if driver == "request" else "avg_mem"

    ref = SERVER_CATALOGUE[next(iter(cfg.composition))]
    cpu_lb = vms[cpu_col].sum() / ref.cpu_capacity
    mem_lb = vms[mem_col].sum() / ref.mem_capacity
    lb = max(cpu_lb, mem_lb)

    n = int(math.ceil(lb * cfg.slack))
    n = max(n, cfg.min_servers)

    # The max_servers cap must never silently manufacture an infeasible
    # instance.  If even the bin-packing lower bound exceeds the cap, no
    # placement exists and the caller needs to hear that now, with the numbers,
    # rather than discover it as a mysterious "Infeasible" from the solver.
    need = int(math.ceil(lb))
    if need > cfg.max_servers:
        raise ValueError(
            f"workload needs at least {need} servers "
            f"(cpu bound {cpu_lb:.2f}, memory bound {mem_lb:.2f}) but "
            f"FleetConfig.max_servers is {cfg.max_servers}. "
            f"Raise max_servers, use a larger server type, or reduce the "
            f"number of VMs."
        )
    return min(n, cfg.max_servers)


def build_fleet(
    vms: pd.DataFrame,
    cfg: Optional[FleetConfig] = None,
    driver: str = "request",
) -> pd.DataFrame:
    """Return the server table the optimiser consumes.

    Columns: server_id, type, cpu_capacity, mem_capacity, idle_power_w,
    max_power_w.  Servers are emitted grouped by type so that the symmetry
    breaking constraint in milp.py can be applied within each identical block.
    """
    cfg = cfg or FleetConfig()
    n = size_fleet(vms, cfg, driver=driver)

    # Normalise the composition into integer counts summing to n.
    total_w = sum(cfg.composition.values())
    if total_w <= 0:
        raise ValueError("fleet composition weights must be positive")

    counts: Dict[str, int] = {}
    assigned = 0
    names = list(cfg.composition.keys())
    for name in names[:-1]:
        c = int(round(n * cfg.composition[name] / total_w))
        counts[name] = c
        assigned += c
    counts[names[-1]] = max(0, n - assigned)

    rows: List[dict] = []
    idx = 1
    for name in names:
        if name not in SERVER_CATALOGUE:
            raise KeyError(f"unknown server type '{name}'")
        st: ServerType = SERVER_CATALOGUE[name]
        for _ in range(counts[name]):
            rows.append(
                {
                    "server_id": f"S{idx:03d}",
                    "type": st.name,
                    "cpu_capacity": st.cpu_capacity,
                    "mem_capacity": st.mem_capacity,
                    "idle_power_w": st.idle_power_w,
                    "max_power_w": st.max_power_w,
                }
            )
            idx += 1

    fleet = pd.DataFrame(rows)
    if fleet.empty:
        raise ValueError("fleet construction produced zero servers")
    return fleet


def feasibility_report(vms: pd.DataFrame, servers: pd.DataFrame,
                       driver: str = "request") -> dict:
    """Cheap pre-solve checks, so an infeasible model is explained not just failed.

    Catches the two ways this model goes infeasible:
      1. Some VM is larger than every server (no bin can hold it).
      2. Aggregate demand exceeds aggregate capacity.
    """
    cpu_col = "cpu_request" if driver == "request" else "avg_cpu"
    mem_col = "mem_request" if driver == "request" else "avg_mem"

    max_cpu_cap = float(servers["cpu_capacity"].max())
    max_mem_cap = float(servers["mem_capacity"].max())

    unplaceable = vms[
        (vms[cpu_col] > max_cpu_cap) | (vms[mem_col] > max_mem_cap)
    ]

    tot_cpu = float(vms[cpu_col].sum())
    tot_mem = float(vms[mem_col].sum())
    cap_cpu = float(servers["cpu_capacity"].sum())
    cap_mem = float(servers["mem_capacity"].sum())

    return {
        "feasible": len(unplaceable) == 0
        and tot_cpu <= cap_cpu
        and tot_mem <= cap_mem,
        "n_unplaceable_vms": int(len(unplaceable)),
        "unplaceable_vm_ids": unplaceable["vm_id"].tolist()[:20],
        "total_cpu_demand": tot_cpu,
        "total_mem_demand": tot_mem,
        "total_cpu_capacity": cap_cpu,
        "total_mem_capacity": cap_mem,
        "cpu_headroom": cap_cpu - tot_cpu,
        "mem_headroom": cap_mem - tot_mem,
        "min_servers_cpu_bound": int(math.ceil(tot_cpu / max_cpu_cap))
        if max_cpu_cap > 0
        else 0,
        "min_servers_mem_bound": int(math.ceil(tot_mem / max_mem_cap))
        if max_mem_cap > 0
        else 0,
    }
