"""The multi-objective MILP for energy-aware VM placement.

MODEL
=====

Sets
    I   VM instances (from the Borg trace)
    J   physical servers (from servers.build_fleet)

Decision variables
    x[i,j] in {0,1}   VM i is placed on server j
    y[j]   in {0,1}   server j is powered on
    d[j]   >= 0       |residual CPU - residual memory| on server j   (aux)
    oc[j]  >= 0       risk-weighted CPU peak overcommit on server j  (aux)
    om[j]  >= 0       risk-weighted memory peak overcommit           (aux)

Objective
    min Z = alpha * E' + beta * W' + gamma * Q'

    with each component normalised by an instance-level worst case so the
    three are commensurable and the weights mean what they appear to mean.

    E  = sum_j [ Pidle_j * y[j] + (Pmax_j - Pidle_j) * cpuload_j / C_j ] * T
         load-proportional and linear in x and y.
    W  = sum_j [ rc_j + rm_j + lambda * d[j] ]
         where rc_j = y[j] - cpu_j/C_j  and  rm_j = y[j] - mem_j/M_j.
    Q  = sum_j ( oc[j] + om[j] )
         the risk-weighted peak demand in excess of a safety threshold, kept as
         two SEPARATE slacks so that Q equals
             sum_j [ max(0, cpupeak_j/C_j - theta) + max(0, mempeak_j/M_j - theta) ]
         A single combined slack on (cpupeak + mempeak - 2*theta) would be a
         strictly weaker penalty: it lets a CPU overshoot be cancelled by spare
         memory headroom, so a server running at 100% CPU and 10% memory would
         score zero QoS risk.  The separate form is also exactly what metrics.py
         measures, and model and evaluator MUST agree or the solver optimises
         one function while the report grades another.

Constraints
    (C1) sum_j x[i,j] = 1                          for all i   assignment
    (C2) sum_i cpu_i x[i,j] <= C_j * y[j]          for all j   CPU capacity
    (C3) sum_i mem_i x[i,j] <= M_j * y[j]          for all j   memory capacity
    (C4) x[i,j] <= y[j]                            for all i,j activation
    (C5) d[j] >= rc_j - rm_j,  d[j] >= rm_j - rc_j for all j   |.| linearisation
    (C6) oc[j] >= cpupeak_j/C_j - theta * y[j]     for all j   QoS overcommit
         om[j] >= mempeak_j/M_j - theta * y[j]     for all j
    (C7) y[j] >= y[j+1] within identical server blocks         symmetry break
    (C8) sum_j y[j] >= ceil(total demand / max capacity)       valid inequality

Note on (C2)/(C4).  Writing the capacity constraint with C_j * y[j] on the
right already implies x[i,j] <= y[j] whenever every VM has strictly positive
demand, so (C4) is redundant *as a feasibility constraint*.  It is kept because
it is a much tighter LP relaxation than (C2) alone -- the classic
strong-versus-weak formulation trade-off in facility location -- and it is what
section 16.4 of the brief specifies.  It can be switched off to demonstrate the
difference in solve time, which is a worthwhile experiment in itself.

Note on the QoS term.  The brief's section 12 definition Q = sum_i q_i with
q_i = failed_i is a constant under (C1): every VM is placed exactly once, so
the sum does not depend on x at all and gamma would have no effect on the
solution.  This module therefore implements the section 13 "advanced"
definition as the active objective term, and metrics.py still reports the
simple failure proxy alongside it.
"""
from __future__ import annotations

import contextlib
import math
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

from .config import ScenarioConfig
from .metrics import Normalisers, build_normalisers, evaluate_placement, risk_weights
from .metrics import PlacementResult


def _solver(cfg: ScenarioConfig, warm: bool = False):
    """CBC, configured from SolverConfig.

    CBC on Windows only honours a warm start when keepFiles=True, which makes
    PuLP write its .lp/.sol files into the CURRENT WORKING DIRECTORY and leave
    them there.  :func:`_scratch_cwd` contains that so the project tree stays
    clean.
    """
    kwargs = dict(
        msg=bool(cfg.solver.msg),
        timeLimit=int(cfg.solver.time_limit_s),
        gapRel=float(cfg.solver.mip_gap),
        warmStart=bool(warm),
        keepFiles=bool(warm),
    )
    if cfg.solver.threads and cfg.solver.threads > 0:
        kwargs["threads"] = int(cfg.solver.threads)
    return pulp.PULP_CBC_CMD(**kwargs)


@contextlib.contextmanager
def _scratch_cwd(enabled: bool):
    """Run the solve inside a throwaway directory, then remove it."""
    if not enabled:
        yield
        return
    prev = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="vmplace_cbc_")
    try:
        os.chdir(tmp)
        yield
    finally:
        os.chdir(prev)
        shutil.rmtree(tmp, ignore_errors=True)


def build_model(
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    cfg: ScenarioConfig,
    normalisers: Optional[Normalisers] = None,
) -> Tuple[pulp.LpProblem, dict]:
    """Construct the MILP.  Returns the problem and a handle on its variables.

    Kept separate from :func:`solve` so the model can be inspected, exported to
    LP format for the report, or re-solved with a different solver.
    """
    norm = normalisers or build_normalisers(vms, servers, cfg)
    w = cfg.weights.normalised()

    I = vms["vm_id"].tolist()
    J = servers["server_id"].tolist()

    # ---- data vectors ---------------------------------------------------
    cpu_req = dict(zip(I, vms["cpu_request"].astype(float)))
    mem_req = dict(zip(I, vms["mem_request"].astype(float)))
    cpu_avg = dict(zip(I, vms["avg_cpu"].astype(float)))
    mem_avg = dict(zip(I, vms["avg_mem"].astype(float)))

    rho = risk_weights(vms, cfg.qos)
    peak_cpu = dict(zip(I, rho * vms["max_cpu"].to_numpy(dtype=float)))
    peak_mem = dict(zip(I, rho * vms["max_mem"].to_numpy(dtype=float)))

    srv = servers.set_index("server_id")
    C = srv["cpu_capacity"].to_dict()
    M = srv["mem_capacity"].to_dict()
    Pidle = srv["idle_power_w"].to_dict()
    Pmax = srv["max_power_w"].to_dict()

    # Which demand vector drives each term.
    pcpu = cpu_avg if cfg.power.power_driver == "average" else cpu_req
    wcpu = cpu_req if cfg.wastage.wastage_driver == "request" else cpu_avg
    wmem = mem_req if cfg.wastage.wastage_driver == "request" else mem_avg

    T = cfg.power.horizon_hours
    lam = cfg.wastage.imbalance_lambda
    theta = cfg.qos.safety_threshold

    prob = pulp.LpProblem("EnergyAware_VM_Placement", pulp.LpMinimize)

    # ---- variables ------------------------------------------------------
    x = pulp.LpVariable.dicts("x", (I, J), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", J, cat=pulp.LpBinary)
    d = pulp.LpVariable.dicts("d", J, lowBound=0, cat=pulp.LpContinuous)
    oc = pulp.LpVariable.dicts("oc", J, lowBound=0, cat=pulp.LpContinuous)
    om = pulp.LpVariable.dicts("om", J, lowBound=0, cat=pulp.LpContinuous)

    # ---- objective components ------------------------------------------
    # E: load-proportional energy.
    if cfg.power.load_proportional:
        energy = pulp.lpSum(
            Pidle[j] * y[j]
            + (Pmax[j] - Pidle[j])
            * pulp.lpSum(pcpu[i] * x[i][j] for i in I)
            / C[j]
            for j in J
        ) * T
    else:
        energy = pulp.lpSum(Pmax[j] * y[j] for j in J) * T

    # W: residual capacity on active servers plus imbalance.
    wastage = pulp.lpSum(
        (
            y[j]
            - pulp.lpSum(wcpu[i] * x[i][j] for i in I) / C[j]
        )
        + (
            y[j]
            - pulp.lpSum(wmem[i] * x[i][j] for i in I) / M[j]
        )
        + lam * d[j]
        for j in J
    )

    # Q: risk-weighted peak overcommit, per resource.
    qos = pulp.lpSum(oc[j] + om[j] for j in J)

    prob += (
        w.alpha * energy / norm.energy_max
        + w.beta * wastage / norm.wastage_max
        + w.gamma * qos / norm.qos_max
    ), "weighted_normalised_objective"

    # ---- (C1) every VM placed exactly once ------------------------------
    for i in I:
        prob += pulp.lpSum(x[i][j] for j in J) == 1, f"assign_{i}"

    for j in J:
        xcpu = pulp.lpSum(cpu_req[i] * x[i][j] for i in I)
        xmem = pulp.lpSum(mem_req[i] * x[i][j] for i in I)

        # ---- (C2)(C3) capacity, linked to activation --------------------
        prob += xcpu <= C[j] * y[j], f"cpu_cap_{j}"
        prob += xmem <= M[j] * y[j], f"mem_cap_{j}"

        # ---- (C5) |rc - rm| linearisation -------------------------------
        rc = y[j] - pulp.lpSum(wcpu[i] * x[i][j] for i in I) / C[j]
        rm = y[j] - pulp.lpSum(wmem[i] * x[i][j] for i in I) / M[j]
        prob += d[j] >= rc - rm, f"imbal_pos_{j}"
        prob += d[j] >= rm - rc, f"imbal_neg_{j}"

        # ---- (C6) QoS overcommit, one slack per resource ----------------
        prob += (
            oc[j]
            >= pulp.lpSum(peak_cpu[i] * x[i][j] for i in I) / C[j]
            - theta * y[j]
        ), f"qos_cpu_{j}"
        prob += (
            om[j]
            >= pulp.lpSum(peak_mem[i] * x[i][j] for i in I) / M[j]
            - theta * y[j]
        ), f"qos_mem_{j}"

    # ---- (C4) strong activation linking ---------------------------------
    if cfg.solver.valid_inequalities:
        for i in I:
            for j in J:
                prob += x[i][j] <= y[j], f"link_{i}_{j}"

    # ---- (C7) symmetry breaking within identical server blocks ----------
    if cfg.solver.symmetry_breaking:
        for _, block in servers.groupby("type", sort=False):
            ids = block["server_id"].tolist()
            for a, b in zip(ids, ids[1:]):
                prob += y[a] >= y[b], f"sym_{a}_{b}"

    # ---- (C8) lower bound on the number of active servers ---------------
    if cfg.solver.valid_inequalities:
        tot_cpu = float(vms["cpu_request"].sum())
        tot_mem = float(vms["mem_request"].sum())
        max_c = float(servers["cpu_capacity"].max())
        max_m = float(servers["mem_capacity"].max())
        lb = max(
            math.ceil(tot_cpu / max_c - 1e-9),
            math.ceil(tot_mem / max_m - 1e-9),
            1,
        )
        lb = min(lb, len(J))
        prob += pulp.lpSum(y[j] for j in J) >= lb, "min_active_servers"

    handle = {
        "x": x, "y": y, "d": d, "oc": oc, "om": om,
        "I": I, "J": J,
        "normalisers": norm,
        "energy_expr": energy,
        "wastage_expr": wastage,
        "qos_expr": qos,
    }
    return prob, handle


def _warm_start(prob, h, vms, servers, cfg) -> Optional[pd.DataFrame]:
    """Seed CBC with a First-Fit-Decreasing incumbent.

    Two reasons this matters.  It usually cuts solve time sharply, because CBC
    can prune against a good bound immediately instead of wandering.  More
    importantly it guarantees that a run which hits the time limit still returns
    a FEASIBLE placement at least as good as FFD, rather than nothing at all.
    """
    from .baselines import first_fit_decreasing

    try:
        fallback = first_fit_decreasing(vms, servers)
    except Exception:
        return None
    if fallback["server_id"].isna().any():
        return None  # heuristic itself failed; nothing useful to seed with

    x, y, I, J = h["x"], h["y"], h["I"], h["J"]
    chosen = dict(zip(fallback["vm_id"], fallback["server_id"]))
    used = set(chosen.values())

    for i in I:
        for j in J:
            x[i][j].setInitialValue(1 if chosen.get(i) == j else 0)
    for j in J:
        y[j].setInitialValue(1 if j in used else 0)
    return fallback


def solve(
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    cfg: Optional[ScenarioConfig] = None,
    normalisers: Optional[Normalisers] = None,
    method_name: str = "MILP",
    warm_start: bool = True,
) -> PlacementResult:
    """Build, solve, and evaluate.  Never raises, and never reports a placement
    it cannot verify.

    Status handling is deliberately strict.  CBC can return LpStatus "Optimal"
    alongside variable values that are None when it stopped before finding any
    incumbent; scoring those as if they were a placement produces nonsense such
    as "2 active servers hold 7.9 CPU of demand".  We therefore validate the
    extracted solution before trusting it, and fall back to the warm-start
    placement (clearly relabelled) when the solver gave us nothing usable.
    """
    cfg = cfg or ScenarioConfig()
    norm = normalisers or build_normalisers(vms, servers, cfg)

    t0 = time.perf_counter()
    prob, h = build_model(vms, servers, cfg, normalisers=norm)
    fallback = _warm_start(prob, h, vms, servers, cfg) if warm_start else None
    build_s = time.perf_counter() - t0

    warm = fallback is not None
    t1 = time.perf_counter()
    solver_error: Optional[str] = None
    try:
        with _scratch_cwd(warm):
            status_code = prob.solve(_solver(cfg, warm=warm))
        status = pulp.LpStatus[status_code]
    except Exception as exc:
        # CBC is an external process and can fail outright rather than return
        # a status: it dies on very large models, on a malformed warm-start
        # file, or when the OS kills it.  A crashed solver must not take the
        # caller down with it -- the dashboard in particular needs a result
        # object it can render.
        solver_error = f"{type(exc).__name__}: {exc}"
        status = "Solver error"
    solve_s = time.perf_counter() - t1
    runtime = build_s + solve_s

    x, y, I, J = h["x"], h["y"], h["I"], h["J"]

    def _fail(note: str, st: str) -> PlacementResult:
        """No trustworthy MILP solution: report the fallback, or nothing."""
        if fallback is not None:
            r = evaluate_placement(
                vms, servers, fallback, cfg,
                method=f"{method_name} (warm-start fallback)",
                normalisers=norm, runtime_s=runtime, status=st,
                notes=note + "; reporting the FFD warm-start placement instead",
            )
            r.metrics["milp_solution_used"] = 0
            return r
        empty = pd.DataFrame({"vm_id": I, "server_id": [None] * len(I)})
        r = evaluate_placement(
            vms, servers, empty, cfg, method=method_name,
            normalisers=norm, runtime_s=runtime, status=st, notes=note,
        )
        r.metrics["milp_solution_used"] = 0
        return r

    if solver_error is not None:
        return _fail(f"solver crashed -- {solver_error}", status)

    if status in ("Infeasible", "Unbounded"):
        return _fail(f"solver status {status}", status)

    # ---- extract the assignment -----------------------------------------
    rows = []
    n_missing = 0
    for i in I:
        chosen = None
        best = 0.5
        for j in J:
            val = x[i][j].value()
            if val is None:
                n_missing += 1
                continue
            if val > best:
                best = val
                chosen = j
        rows.append({"vm_id": i, "server_id": chosen})
    assignment = pd.DataFrame(rows)

    # ---- validate before trusting ---------------------------------------
    n_unplaced = int(assignment["server_id"].isna().sum())
    if n_missing or n_unplaced:
        return _fail(
            f"solver returned no usable incumbent "
            f"({n_unplaced} VMs unassigned, {n_missing} null variables)",
            "No solution found",
        )

    gap = None
    try:
        obj = pulp.value(prob.objective)
        bound = getattr(prob, "bestBound", None)
        if bound is not None and obj is not None:
            gap = abs(obj - bound) / max(abs(obj), 1e-12)
    except Exception:
        gap = None

    res = evaluate_placement(
        vms, servers, assignment, cfg,
        method=method_name,
        normalisers=norm,
        runtime_s=runtime,
        status=status,
        objective_value=pulp.value(prob.objective),
        mip_gap=gap,
        notes=f"build {build_s:.2f}s, solve {solve_s:.2f}s"
        + ("" if fallback is None else ", warm-started from FFD"),
    )
    res.metrics["build_time_s"] = build_s
    res.metrics["solve_time_s"] = solve_s
    res.metrics["n_binary_vars"] = len(I) * len(J) + len(J)
    res.metrics["n_constraints"] = len(prob.constraints)
    res.metrics["milp_solution_used"] = 1

    # Cross-check: the objective CBC reports and the objective metrics.py
    # measures must agree, or model and evaluator have drifted apart.
    z_model = pulp.value(prob.objective)
    z_eval = res.metrics["weighted_objective"]
    if z_model is not None:
        res.metrics["objective_mismatch"] = abs(z_model - z_eval)
        if abs(z_model - z_eval) > 1e-4:
            res.notes += (
                f" | WARNING: solver objective {z_model:.6f} != "
                f"evaluated {z_eval:.6f}"
            )
    return res


def model_size(n_vms: int, n_servers: int, valid_inequalities: bool = True) -> dict:
    """Report model dimensions without building it, for UI warnings."""
    binaries = n_vms * n_servers + n_servers
    cons = n_vms + 5 * n_servers + 1
    if valid_inequalities:
        cons += n_vms * n_servers + 1
    return {
        "binary_variables": binaries,
        "continuous_variables": 2 * n_servers,
        "approx_constraints": cons,
    }
