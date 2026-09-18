"""Sensitivity analysis and Pareto exploration (sections 20 and 36F of the brief).

Sensitivity analysis is the part of an OR study that turns a single number into
an argument.  One optimal placement tells you what to do under one set of
assumptions; a sensitivity sweep tells you which assumptions actually matter.

Three sweep families are provided:

sweep_weights
    Vary alpha/beta/gamma.  Answers "how much energy do we give up per unit of
    QoS protection?"  This is the trade-off the multi-objective formulation
    exists to expose.

sweep_parameter
    Vary one scalar input (workload size, fleet slack, safety threshold, idle
    power, horizon) while holding everything else fixed.  Answers "is the
    recommendation robust to this assumption?" -- which matters most for the
    power figures, since those are modelled rather than measured.

pareto_frontier
    Sweep a two-way weight simplex and keep the non-dominated solutions, giving
    the trade-off surface rather than one point on it.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import baselines, milp, servers as sv
from .config import ScenarioConfig, Weights
from .metrics import PlacementResult, build_normalisers


# --------------------------------------------------------------------------
def _run_one(
    vms: pd.DataFrame,
    cfg: ScenarioConfig,
    label: str,
    rebuild_fleet: bool = True,
    fleet: Optional[pd.DataFrame] = None,
) -> dict:
    """Solve one scenario and flatten it into a row."""
    f = fleet if fleet is not None and not rebuild_fleet else sv.build_fleet(vms, cfg.fleet)
    norm = build_normalisers(vms, f, cfg)
    res = milp.solve(vms, f, cfg, normalisers=norm, method_name=label)
    row = {"scenario": label, "n_servers_in_fleet": len(f)}
    row.update(res.summary_row())
    row["alpha"] = cfg.weights.alpha
    row["beta"] = cfg.weights.beta
    row["gamma"] = cfg.weights.gamma
    return row


def sweep_weights(
    vms: pd.DataFrame,
    base_cfg: ScenarioConfig,
    grid: Optional[Sequence[Weights]] = None,
    fleet: Optional[pd.DataFrame] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Solve the model under several weight vectors.

    The fleet is held FIXED across the sweep (passed in, or built once from the
    base config) so that differences in the results come from the objective
    weights alone and not from a silently resized fleet.
    """
    if grid is None:
        grid = [
            Weights(1.0, 0.0, 0.0),
            Weights(0.7, 0.2, 0.1),
            Weights(0.4, 0.3, 0.3),
            Weights(0.2, 0.6, 0.2),
            Weights(0.2, 0.2, 0.6),
            Weights(0.0, 0.0, 1.0),
        ]
    f = fleet if fleet is not None else sv.build_fleet(vms, base_cfg.fleet)

    rows: List[dict] = []
    for k, w in enumerate(grid):
        cfg = copy.deepcopy(base_cfg)
        cfg.weights = w
        label = f"a={w.alpha:g},b={w.beta:g},g={w.gamma:g}"
        if progress:
            progress(k, len(grid), label)
        rows.append(_run_one(vms, cfg, label, rebuild_fleet=False, fleet=f))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Single-parameter sweeps
# --------------------------------------------------------------------------
def _apply_param(cfg: ScenarioConfig, param: str, value) -> ScenarioConfig:
    c = copy.deepcopy(cfg)
    if param == "fleet_slack":
        c.fleet.slack = float(value)
    elif param == "n_servers":
        c.fleet.n_servers = int(value)
    elif param == "safety_threshold":
        c.qos.safety_threshold = float(value)
    elif param == "idle_power_fraction":
        # Scales idle power as a fraction of max power, across the catalogue.
        # Handled by the caller via a custom fleet; see sweep_idle_power.
        pass
    elif param == "horizon_hours":
        c.power.horizon_hours = float(value)
    elif param == "imbalance_lambda":
        c.wastage.imbalance_lambda = float(value)
    elif param == "failure_weight":
        c.qos.failure_weight = float(value)
    elif param == "priority_weight":
        c.qos.priority_weight = float(value)
    elif param == "power_driver":
        c.power.power_driver = str(value)
    elif param == "load_proportional":
        c.power.load_proportional = bool(value)
    else:
        raise KeyError(f"unknown sweep parameter '{param}'")
    return c


def sweep_parameter(
    vms: pd.DataFrame,
    base_cfg: ScenarioConfig,
    param: str,
    values: Iterable,
    fixed_fleet: bool = False,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Vary one input and record how the optimum moves.

    fixed_fleet
        Keep the fleet from the base config.  Set False for parameters that
        legitimately change the fleet (slack, n_servers), True for parameters
        that should not (safety_threshold, horizon, lambda).
    """
    vals = list(values)
    base_fleet = sv.build_fleet(vms, base_cfg.fleet) if fixed_fleet else None

    rows: List[dict] = []
    for k, v in enumerate(vals):
        cfg = _apply_param(base_cfg, param, v)
        label = f"{param}={v}"
        if progress:
            progress(k, len(vals), label)
        row = _run_one(
            vms, cfg, label,
            rebuild_fleet=not fixed_fleet,
            fleet=base_fleet,
        )
        row["parameter"] = param
        row["value"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_workload_size(
    vm_pool: pd.DataFrame,
    base_cfg: ScenarioConfig,
    sizes: Sequence[int],
    strategy: str = "stratified",
    seed: int = 42,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Scenario D of the brief: how does the optimum scale with workload?

    Each size gets its own fleet, since a larger workload genuinely needs more
    machines.  Also records solve time, which is the practical limit on how far
    an exact MILP can be pushed.
    """
    from .preprocess import sample_vms

    rows: List[dict] = []
    for k, n in enumerate(sizes):
        w = sample_vms(vm_pool, n, strategy=strategy, seed=seed)
        cfg = copy.deepcopy(base_cfg)
        label = f"n_vms={n}"
        if progress:
            progress(k, len(sizes), label)
        row = _run_one(w, cfg, label, rebuild_fleet=True)
        row["parameter"] = "n_vms"
        row["value"] = n
        row["total_cpu_demand"] = float(w["cpu_request"].sum())
        row["total_mem_demand"] = float(w["mem_request"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_power_assumption(
    vms: pd.DataFrame,
    base_cfg: ScenarioConfig,
    idle_fractions: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """How sensitive is the recommendation to the ASSUMED idle power?

    This is the most important sweep in the project, because idle power is the
    one number that is modelled rather than measured.  The idle/max ratio is
    what makes consolidation worthwhile at all: at idle = max, switching a
    server off saves everything and consolidation is free; at idle = 0, an
    empty server costs nothing and there is no reason to consolidate.
    """
    rows: List[dict] = []
    for k, frac in enumerate(idle_fractions):
        cfg = copy.deepcopy(base_cfg)
        fleet = sv.build_fleet(vms, cfg.fleet)
        fleet = fleet.copy()
        fleet["idle_power_w"] = fleet["max_power_w"] * float(frac)
        label = f"idle={frac:.0%} of max"
        if progress:
            progress(k, len(idle_fractions), label)
        norm = build_normalisers(vms, fleet, cfg)
        res = milp.solve(vms, fleet, cfg, normalisers=norm, method_name=label)
        row = {"scenario": label, "parameter": "idle_power_fraction",
               "value": frac, "n_servers_in_fleet": len(fleet)}
        row.update(res.summary_row())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pareto frontier
# --------------------------------------------------------------------------
def _non_dominated(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Boolean mask of Pareto-optimal rows, all objectives minimised."""
    pts = df[list(cols)].to_numpy(dtype=float)
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j is <= on all and < on at least one
        dominated = np.all(pts <= pts[i] + 1e-12, axis=1) & np.any(
            pts < pts[i] - 1e-12, axis=1
        )
        if dominated.any():
            keep[i] = False
    return pd.Series(keep, index=df.index)


def pareto_frontier(
    vms: pd.DataFrame,
    base_cfg: ScenarioConfig,
    step: float = 0.2,
    fleet: Optional[pd.DataFrame] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Weighted-sum scalarisation over a simplex grid of (alpha, beta, gamma).

    Caveat worth stating in the report: a weighted sum can only ever recover
    solutions on the CONVEX hull of the Pareto front.  Non-convex ("unsupported")
    efficient points are unreachable by this method no matter how fine the grid.
    Recovering those needs epsilon-constraint or a lexicographic method; the
    convex portion is enough to show the trade-off shape here.
    """
    f = fleet if fleet is not None else sv.build_fleet(vms, base_cfg.fleet)

    grid = []
    ticks = int(round(1.0 / step))
    for ai in range(ticks + 1):
        for bi in range(ticks + 1 - ai):
            gi = ticks - ai - bi
            grid.append(
                Weights(ai / ticks, bi / ticks, gi / ticks)
            )

    rows: List[dict] = []
    for k, w in enumerate(grid):
        cfg = copy.deepcopy(base_cfg)
        cfg.weights = w
        label = f"a={w.alpha:.2f},b={w.beta:.2f},g={w.gamma:.2f}"
        if progress:
            progress(k, len(grid), label)
        rows.append(_run_one(vms, cfg, label, rebuild_fleet=False, fleet=f))

    df = pd.DataFrame(rows)
    df["pareto_optimal"] = _non_dominated(
        df, ["energy_normalised", "wastage_normalised", "qos_normalised"]
    )
    return df


# --------------------------------------------------------------------------
def summarise_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Trim a sweep result to the columns worth putting in a report table."""
    cols = [
        c
        for c in [
            "scenario", "parameter", "value", "alpha", "beta", "gamma",
            "n_active_servers", "n_servers_in_fleet", "energy_wh",
            "wastage", "qos_overcommit", "cpu_util_active_mean",
            "mem_util_active_mean", "weighted_objective", "status",
            "runtime_s", "n_capacity_violations",
        ]
        if c in df.columns
    ]
    return df[cols]
