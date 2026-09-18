"""Evaluation of a VM-to-server assignment.

Every placement produced anywhere in this project -- by the MILP, by a
heuristic baseline, or read from the trace itself -- is scored by the functions
here.  Keeping one evaluator means the comparison in section 22 of the brief is
genuinely like for like: the MILP is never allowed to grade its own homework on
a different scale from the baselines.

The normalisation constants are computed from the instance (VMs + fleet) alone,
never from any particular solution, so E', W' and Q' are comparable across
methods and across scenarios with the same instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import PowerModelConfig, QoSConfig, ScenarioConfig, WastageConfig


# --------------------------------------------------------------------------
# Per-VM risk weight used by the QoS term
# --------------------------------------------------------------------------
def risk_weights(vms: pd.DataFrame, qos: QoSConfig) -> np.ndarray:
    """rho_i = 1 + w_f * failed_i + w_p * priority_norm_i.

    Both inputs are real trace fields.  rho_i >= 1 always, so a VM never
    *reduces* the measured risk of a server it lands on.
    """
    return (
        1.0
        + qos.failure_weight * vms["failed"].to_numpy(dtype=float)
        + qos.priority_weight * vms["priority_norm"].to_numpy(dtype=float)
    )


# --------------------------------------------------------------------------
# Normalisation constants
# --------------------------------------------------------------------------
@dataclass
class Normalisers:
    """Worst-case denominators that map E, W, Q onto roughly [0, 1].

    Using instance-level worst cases (rather than min-max over observed
    solutions) keeps the objective well defined before any solution exists,
    which is what a MILP objective requires.
    """

    energy_max: float
    wastage_max: float
    qos_max: float

    def as_dict(self) -> dict:
        return {
            "energy_max_wh": self.energy_max,
            "wastage_max": self.wastage_max,
            "qos_max": self.qos_max,
        }


def build_normalisers(
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    cfg: ScenarioConfig,
) -> Normalisers:
    """Worst case for each objective component on this instance."""
    T = cfg.power.horizon_hours

    # Energy: every server powered on at full load for the horizon.
    energy_max = float(servers["max_power_w"].sum()) * T
    energy_max = max(energy_max, 1e-9)

    # Wastage: every server on and completely empty, including the imbalance
    # term at its maximum of 0 (empty servers are perfectly balanced), so the
    # bound is 2 units of residual per server.
    n_srv = len(servers)
    wastage_max = 2.0 * n_srv
    wastage_max = max(wastage_max, 1e-9)

    # QoS: all risk-weighted peak demand landing on a single server, i.e. the
    # largest overshoot the instance can physically produce.
    rho = risk_weights(vms, cfg.qos)
    peak_cpu = float(np.sum(rho * vms["max_cpu"].to_numpy(dtype=float)))
    peak_mem = float(np.sum(rho * vms["max_mem"].to_numpy(dtype=float)))
    min_cpu_cap = float(servers["cpu_capacity"].min())
    min_mem_cap = float(servers["mem_capacity"].min())
    qos_max = peak_cpu / max(min_cpu_cap, 1e-9) + peak_mem / max(min_mem_cap, 1e-9)
    qos_max = max(qos_max, 1e-9)

    return Normalisers(energy_max, wastage_max, qos_max)


# --------------------------------------------------------------------------
# Placement evaluation
# --------------------------------------------------------------------------
@dataclass
class PlacementResult:
    """Everything section 21 of the brief asks the system to produce."""

    method: str
    assignment: pd.DataFrame          # vm_id -> server_id
    server_loads: pd.DataFrame        # per-server utilisation and power
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    runtime_s: float = 0.0
    mip_gap: Optional[float] = None
    objective_value: Optional[float] = None
    notes: str = ""

    def summary_row(self) -> dict:
        row = {"method": self.method, "status": self.status,
               "runtime_s": self.runtime_s}
        row.update(self.metrics)
        if self.objective_value is not None:
            row["objective_value"] = self.objective_value
        if self.mip_gap is not None:
            row["mip_gap"] = self.mip_gap
        return row


def evaluate_placement(
    vms: pd.DataFrame,
    servers: pd.DataFrame,
    assignment: pd.DataFrame,
    cfg: ScenarioConfig,
    method: str = "unknown",
    normalisers: Optional[Normalisers] = None,
    runtime_s: float = 0.0,
    status: str = "ok",
    objective_value: Optional[float] = None,
    mip_gap: Optional[float] = None,
    notes: str = "",
) -> PlacementResult:
    """Score an assignment.  ``assignment`` needs columns vm_id, server_id.

    Unplaced VMs (server_id null) are permitted so that a heuristic baseline
    which fails to place everything can still be reported honestly rather than
    crashing; they are counted in ``n_unplaced``.
    """
    norm = normalisers or build_normalisers(vms, servers, cfg)
    T = cfg.power.horizon_hours

    a = assignment[["vm_id", "server_id"]].copy()
    placed = a.dropna(subset=["server_id"])
    n_unplaced = len(a) - len(placed)

    v = vms.merge(placed, on="vm_id", how="left")
    v["rho"] = risk_weights(vms, cfg.qos)

    # ---- which demand vector feeds which term ---------------------------
    pw_cpu = "avg_cpu" if cfg.power.power_driver == "average" else "cpu_request"
    pw_mem = "avg_mem" if cfg.power.power_driver == "average" else "mem_request"
    wa_cpu = "cpu_request" if cfg.wastage.wastage_driver == "request" else "avg_cpu"
    wa_mem = "mem_request" if cfg.wastage.wastage_driver == "request" else "avg_mem"

    v["rho_max_cpu"] = v["rho"] * v["max_cpu"]
    v["rho_max_mem"] = v["rho"] * v["max_mem"]

    grouped = v.dropna(subset=["server_id"]).groupby("server_id")
    loads = grouped.agg(
        n_vms=("vm_id", "size"),
        cpu_request=("cpu_request", "sum"),
        mem_request=("mem_request", "sum"),
        cpu_avg=("avg_cpu", "sum"),
        mem_avg=("avg_mem", "sum"),
        cpu_peak=("max_cpu", "sum"),
        mem_peak=("max_mem", "sum"),
        rho_peak_cpu=("rho_max_cpu", "sum"),
        rho_peak_mem=("rho_max_mem", "sum"),
        n_failed=("failed", "sum"),
    )

    s = servers.set_index("server_id").join(loads, how="left")
    fill = {
        "n_vms": 0, "cpu_request": 0.0, "mem_request": 0.0,
        "cpu_avg": 0.0, "mem_avg": 0.0, "cpu_peak": 0.0, "mem_peak": 0.0,
        "rho_peak_cpu": 0.0, "rho_peak_mem": 0.0, "n_failed": 0,
    }
    s = s.fillna(fill)
    s["active"] = (s["n_vms"] > 0).astype(int)

    # ---- utilisation ----------------------------------------------------
    s["cpu_util_request"] = s["cpu_request"] / s["cpu_capacity"]
    s["mem_util_request"] = s["mem_request"] / s["mem_capacity"]
    s["cpu_util_avg"] = s["cpu_avg"] / s["cpu_capacity"]
    s["mem_util_avg"] = s["mem_avg"] / s["mem_capacity"]
    s["cpu_util_peak"] = s["cpu_peak"] / s["cpu_capacity"]
    s["mem_util_peak"] = s["mem_peak"] / s["mem_capacity"]

    # ---- capacity violations (a baseline may produce them) --------------
    s["cpu_overflow"] = (s["cpu_request"] - s["cpu_capacity"]).clip(lower=0)
    s["mem_overflow"] = (s["mem_request"] - s["mem_capacity"]).clip(lower=0)

    # ---- energy ---------------------------------------------------------
    pw_load_cpu = (
        s["cpu_avg"] if cfg.power.power_driver == "average" else s["cpu_request"]
    ) / s["cpu_capacity"]
    pw_load_cpu = pw_load_cpu.clip(0, 1)

    if cfg.power.load_proportional:
        s["power_w"] = s["active"] * (
            s["idle_power_w"]
            + (s["max_power_w"] - s["idle_power_w"]) * pw_load_cpu
        )
    else:
        # Degenerate form from section 12 of the brief: a flat per-server cost.
        s["power_w"] = s["active"] * s["max_power_w"]

    energy_wh = float(s["power_w"].sum()) * T

    # ---- wastage --------------------------------------------------------
    wc = (
        s["cpu_request"] if cfg.wastage.wastage_driver == "request" else s["cpu_avg"]
    ) / s["cpu_capacity"]
    wm = (
        s["mem_request"] if cfg.wastage.wastage_driver == "request" else s["mem_avg"]
    ) / s["mem_capacity"]
    rc = (s["active"] - wc).clip(lower=0)
    rm = (s["active"] - wm).clip(lower=0)
    imbalance = (rc - rm).abs()
    s["wastage"] = rc + rm + cfg.wastage.imbalance_lambda * imbalance
    wastage = float(s["wastage"].sum())

    # ---- QoS ------------------------------------------------------------
    theta = cfg.qos.safety_threshold
    over_cpu = (
        s["rho_peak_cpu"] / s["cpu_capacity"] - theta * s["active"]
    ).clip(lower=0)
    over_mem = (
        s["rho_peak_mem"] / s["mem_capacity"] - theta * s["active"]
    ).clip(lower=0)
    s["qos_overcommit"] = over_cpu + over_mem
    qos_overcommit = float(s["qos_overcommit"].sum())

    # The literal brief definition, reported but placement-invariant.
    qos_failure_proxy = float(vms["failed"].mean()) if len(vms) else 0.0

    qos = (
        qos_overcommit
        if cfg.qos.mode == "peak_overcommit"
        else qos_failure_proxy
    )

    # ---- normalised components and weighted objective -------------------
    w = cfg.weights.normalised()
    e_n = energy_wh / norm.energy_max
    w_n = wastage / norm.wastage_max
    q_n = qos / norm.qos_max if cfg.qos.mode == "peak_overcommit" else qos
    z = w.alpha * e_n + w.beta * w_n + w.gamma * q_n

    # An incomplete placement must never look like a good one.  Every term in
    # Z rewards servers being off, so an assignment that places nothing scores
    # a perfect Z = 0 -- which would let a failed or infeasible run appear to
    # beat every real solution in the comparison table.  Mark it invalid
    # instead, so it propagates as NaN rather than as a spurious win.
    placement_valid = n_unplaced == 0
    if not placement_valid:
        z = float("nan")

    active = s[s["active"] == 1]
    n_active = int(s["active"].sum())

    metrics = {
        "placement_valid": int(placement_valid),
        "n_vms": int(len(vms)),
        "n_servers_available": int(len(servers)),
        "n_active_servers": n_active,
        "n_inactive_servers": int(len(servers) - n_active),
        "n_unplaced_vms": int(n_unplaced),
        "energy_wh": energy_wh,
        "energy_normalised": e_n,
        "mean_power_w": float(s["power_w"].sum()),
        "wastage": wastage,
        "wastage_normalised": w_n,
        "qos_overcommit": qos_overcommit,
        "qos_normalised": q_n,
        "qos_failure_proxy": qos_failure_proxy,
        "weighted_objective": z,
        "cpu_util_active_mean": float(active["cpu_util_request"].mean())
        if n_active
        else 0.0,
        "mem_util_active_mean": float(active["mem_util_request"].mean())
        if n_active
        else 0.0,
        "cpu_util_peak_active_mean": float(active["cpu_util_peak"].mean())
        if n_active
        else 0.0,
        "mem_util_peak_active_mean": float(active["mem_util_peak"].mean())
        if n_active
        else 0.0,
        "cpu_util_fleet": float(
            s["cpu_request"].sum() / s["cpu_capacity"].sum()
        ),
        "mem_util_fleet": float(
            s["mem_request"].sum() / s["mem_capacity"].sum()
        ),
        "wastage_pct_active": float(
            1.0 - active["cpu_util_request"].mean()
        ) * 100.0
        if n_active
        else 0.0,
        "n_capacity_violations": int(
            ((s["cpu_overflow"] > 1e-9) | (s["mem_overflow"] > 1e-9)).sum()
        ),
        "total_cpu_overflow": float(s["cpu_overflow"].sum()),
        "total_mem_overflow": float(s["mem_overflow"].sum()),
        "n_servers_over_threshold": int((s["qos_overcommit"] > 1e-9).sum()),
        "failed_vms": int(vms["failed"].sum()),
    }

    return PlacementResult(
        method=method,
        assignment=a,
        server_loads=s.reset_index(),
        metrics=metrics,
        status=status,
        runtime_s=runtime_s,
        mip_gap=mip_gap,
        objective_value=objective_value if objective_value is not None else z,
        notes=notes,
    )


def compare_results(results: list[PlacementResult]) -> pd.DataFrame:
    """Side-by-side table for the baseline-versus-optimised comparison."""
    rows = [r.summary_row() for r in results]
    df = pd.DataFrame(rows)
    if "energy_wh" in df.columns and len(df):
        base = df["energy_wh"].iloc[0]
        if base > 0:
            df["energy_vs_first_pct"] = df["energy_wh"] / base * 100.0
    return df
