"""Invariant tests for the placement model.

These are not unit tests of Python plumbing; they are checks on the properties
the OR model is supposed to have.  Two of them encode bugs that were actually
found during development and would otherwise have silently corrupted every
reported result:

  * test_model_matches_evaluator
        The MILP once minimised a QoS term that used one combined slack while
        the evaluator measured two separate ones.  The solver optimised one
        function and the report graded another, and the symptom was an
        "optimal" MILP scoring WORSE than a greedy heuristic.

  * test_no_phantom_placements
        A timed-out CBC returns None-valued variables.  Reading those as
        "unplaced" and scoring them produced a placement that fit 7.9 units of
        CPU demand onto 2 servers of capacity 1.0.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_model.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vmplace import baselines, milp, preprocess as pp, servers as sv
from vmplace.config import (
    FleetConfig,
    PowerModelConfig,
    QoSConfig,
    ScenarioConfig,
    SolverConfig,
    WastageConfig,
    Weights,
)
from vmplace.metrics import build_normalisers, evaluate_placement


# --------------------------------------------------------------------------
# synthetic fixtures -- deterministic, no dependence on the trace file
# --------------------------------------------------------------------------
def make_vms(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cpu = rng.uniform(0.04, 0.22, n)
    mem = rng.uniform(0.03, 0.18, n)
    return pd.DataFrame({
        "vm_id": [f"VM{i:04d}" for i in range(n)],
        "cpu_request": cpu,
        "mem_request": mem,
        "avg_cpu": cpu * rng.uniform(0.3, 0.9, n),
        "avg_mem": mem * rng.uniform(0.3, 0.9, n),
        "max_cpu": np.minimum(cpu * rng.uniform(1.0, 2.2, n), 1.0),
        "max_mem": np.minimum(mem * rng.uniform(1.0, 2.0, n), 1.0),
        "priority": rng.integers(0, 450, n),
        "priority_norm": rng.uniform(0, 1, n),
        "priority_band": "production",
        "failed": rng.integers(0, 2, n),
        "hard_failed": 0,
        "orig_machine_id": rng.integers(1, 6, n),
        "duration_s": 300.0,
    })


def make_cfg(**kw) -> ScenarioConfig:
    cfg = ScenarioConfig(
        weights=Weights(0.4, 0.3, 0.3),
        power=PowerModelConfig(),
        qos=QoSConfig(),
        wastage=WastageConfig(),
        fleet=FleetConfig(slack=2.0),
        solver=SolverConfig(time_limit_s=60, mip_gap=0.0, msg=False),
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def instance():
    vms = make_vms()
    cfg = make_cfg()
    fleet = sv.build_fleet(vms, cfg.fleet)
    return vms, fleet, cfg


# ==========================================================================
# Constraint satisfaction
# ==========================================================================
def test_every_vm_placed_exactly_once(instance):
    """(C1): sum_j x_ij = 1."""
    vms, fleet, cfg = instance
    res = milp.solve(vms, fleet, cfg)
    assert res.metrics["milp_solution_used"] == 1, res.notes
    a = res.assignment
    assert len(a) == len(vms)
    assert a["server_id"].notna().all(), "some VM was left unplaced"
    assert a["vm_id"].nunique() == len(vms), "a VM appears more than once"


def test_capacity_constraints_respected(instance):
    """(C2)/(C3): no server exceeds its CPU or memory capacity."""
    vms, fleet, cfg = instance
    res = milp.solve(vms, fleet, cfg)
    assert res.metrics["n_capacity_violations"] == 0
    loads = res.server_loads
    assert (loads["cpu_request"] <= loads["cpu_capacity"] + 1e-6).all()
    assert (loads["mem_request"] <= loads["mem_capacity"] + 1e-6).all()


def test_activation_linking(instance):
    """(C4): a server carrying VMs must be marked active, and vice versa."""
    vms, fleet, cfg = instance
    res = milp.solve(vms, fleet, cfg)
    loads = res.server_loads
    assert ((loads["n_vms"] > 0) == (loads["active"] == 1)).all()
    inactive = loads[loads["active"] == 0]
    assert (inactive["cpu_request"] < 1e-9).all()
    assert (inactive["power_w"] < 1e-9).all(), "an off server drew power"


# ==========================================================================
# Objective integrity -- the bugs that actually happened
# ==========================================================================
def test_model_matches_evaluator(instance):
    """The solver's objective must equal the independently evaluated one.

    If these drift apart the solver optimises one function while every reported
    metric grades another, and the results are silently meaningless.
    """
    vms, fleet, cfg = instance
    res = milp.solve(vms, fleet, cfg)
    mismatch = res.metrics.get("objective_mismatch")
    assert mismatch is not None
    assert mismatch < 1e-6, (
        f"solver and evaluator disagree by {mismatch:.2e} -- "
        "the MILP objective and metrics.py have drifted apart"
    )


def test_milp_beats_or_matches_heuristics(instance):
    """An exact method must never lose to a greedy one on its own objective."""
    vms, fleet, cfg = instance
    norm = build_normalisers(vms, fleet, cfg)
    res = milp.solve(vms, fleet, cfg, normalisers=norm)
    assert res.metrics["milp_solution_used"] == 1

    z_milp = res.metrics["weighted_objective"]
    for name in ["First-Fit", "Best-Fit", "First-Fit-Decreasing"]:
        r = baselines.run_baseline(name, vms, fleet, cfg, norm)
        assert z_milp <= r.metrics["weighted_objective"] + 1e-6, (
            f"MILP ({z_milp:.6f}) lost to {name} "
            f"({r.metrics['weighted_objective']:.6f})"
        )


def test_no_phantom_placements(instance):
    """Reported demand must be physically consistent with reported capacity."""
    vms, fleet, cfg = instance
    res = milp.solve(vms, fleet, cfg)
    if res.metrics["milp_solution_used"] != 1:
        pytest.skip("no verified MILP solution to check")
    loads = res.server_loads
    active = loads[loads["active"] == 1]
    assert active["cpu_capacity"].sum() >= vms["cpu_request"].sum() - 1e-6, (
        "active servers cannot physically hold the assigned demand"
    )
    assert loads["n_vms"].sum() == len(vms)


def test_incomplete_placement_scores_nan_not_zero(instance):
    """An unplaced VM must invalidate the objective, never flatter it.

    Every term in Z rewards servers being switched off, so an assignment that
    places nothing scores a perfect Z = 0. Left unguarded, an infeasible or
    crashed run would appear to beat every genuine solution in the comparison
    table -- which is exactly what happened before this check existed.
    """
    vms, fleet, cfg = instance
    norm = build_normalisers(vms, fleet, cfg)

    empty = pd.DataFrame({"vm_id": vms["vm_id"], "server_id": [None] * len(vms)})
    r = evaluate_placement(vms, fleet, empty, cfg, normalisers=norm)
    assert r.metrics["placement_valid"] == 0
    assert np.isnan(r.metrics["weighted_objective"]), (
        "an empty placement reported a finite objective"
    )

    partial = baselines.first_fit_decreasing(vms, fleet)
    partial.loc[0, "server_id"] = None
    r2 = evaluate_placement(vms, fleet, partial, cfg, normalisers=norm)
    assert r2.metrics["placement_valid"] == 0
    assert np.isnan(r2.metrics["weighted_objective"])

    good = baselines.first_fit_decreasing(vms, fleet)
    r3 = evaluate_placement(vms, fleet, good, cfg, normalisers=norm)
    assert r3.metrics["placement_valid"] == 1
    assert np.isfinite(r3.metrics["weighted_objective"])


def test_fleet_refuses_to_build_an_impossible_instance():
    """Capping at max_servers must not silently create an infeasible model."""
    vms = make_vms(40)
    vms["cpu_request"] = 0.9          # 40 x 0.9 = 36 units of demand
    vms["mem_request"] = 0.9
    with pytest.raises(ValueError, match="max_servers"):
        sv.build_fleet(vms, FleetConfig(max_servers=5))


def test_solver_never_raises():
    """solve() must return a result object even when the solver fails."""
    vms = make_vms(8)
    cfg = make_cfg(solver=SolverConfig(time_limit_s=1, mip_gap=0.5))
    fleet = sv.build_fleet(vms, cfg.fleet)
    res = milp.solve(vms, fleet, cfg)  # must not raise
    assert res is not None
    assert "weighted_objective" in res.metrics


# ==========================================================================
# Multi-objective behaviour
# ==========================================================================
def test_weights_change_the_solution(instance):
    """Energy-focused and QoS-focused weights must not give the same answer.

    If they do, the multi-objective formulation is not doing anything and the
    weights are decorative.
    """
    vms, fleet, cfg = instance

    e_cfg = make_cfg(weights=Weights(1.0, 0.0, 0.0))
    q_cfg = make_cfg(weights=Weights(0.0, 0.0, 1.0))

    e_res = milp.solve(vms, fleet, e_cfg)
    q_res = milp.solve(vms, fleet, q_cfg)

    assert e_res.metrics["milp_solution_used"] == 1
    assert q_res.metrics["milp_solution_used"] == 1

    # Energy-first should consolidate at least as hard as QoS-first.
    assert e_res.metrics["n_active_servers"] <= q_res.metrics["n_active_servers"]
    # And QoS-first should not be worse on the QoS measure.
    assert (
        q_res.metrics["qos_overcommit"]
        <= e_res.metrics["qos_overcommit"] + 1e-6
    )


def test_energy_first_minimises_energy(instance):
    """With alpha=1, no other weighting may achieve lower energy."""
    vms, fleet, cfg = instance
    e_res = milp.solve(vms, fleet, make_cfg(weights=Weights(1.0, 0.0, 0.0)))
    b_res = milp.solve(vms, fleet, make_cfg(weights=Weights(0.4, 0.3, 0.3)))
    assert e_res.metrics["energy_wh"] <= b_res.metrics["energy_wh"] + 1e-6


def test_failure_proxy_is_placement_invariant(instance):
    """The brief's Q = sum(failed)/N genuinely cannot depend on the placement.

    This is the mathematical claim the project makes; it deserves a test.
    """
    vms, fleet, cfg = instance
    norm = build_normalisers(vms, fleet, cfg)
    values = set()
    for name in ["First-Fit", "Best-Fit", "First-Fit-Decreasing"]:
        r = baselines.run_baseline(name, vms, fleet, cfg, norm)
        values.add(round(r.metrics["qos_failure_proxy"], 12))
    assert len(values) == 1, (
        "the failure proxy varied across placements, which contradicts "
        "the claim that it is constant under the assignment constraint"
    )


# ==========================================================================
# Baselines and evaluation
# ==========================================================================
def test_baselines_place_everything(instance):
    vms, fleet, cfg = instance
    norm = build_normalisers(vms, fleet, cfg)
    for name in ["First-Fit", "Best-Fit", "First-Fit-Decreasing"]:
        r = baselines.run_baseline(name, vms, fleet, cfg, norm)
        assert r.metrics["n_unplaced_vms"] == 0, f"{name} left VMs unplaced"
        assert r.metrics["n_capacity_violations"] == 0, f"{name} overfilled"


def test_evaluator_is_deterministic(instance):
    vms, fleet, cfg = instance
    norm = build_normalisers(vms, fleet, cfg)
    a = baselines.first_fit_decreasing(vms, fleet)
    r1 = evaluate_placement(vms, fleet, a, cfg, normalisers=norm)
    r2 = evaluate_placement(vms, fleet, a, cfg, normalisers=norm)
    assert r1.metrics == r2.metrics


def test_normalisers_are_solution_independent(instance):
    """Normalisers must depend only on the instance, never on a solution."""
    vms, fleet, cfg = instance
    n1 = build_normalisers(vms, fleet, cfg)
    milp.solve(vms, fleet, cfg)
    n2 = build_normalisers(vms, fleet, cfg)
    assert n1.as_dict() == n2.as_dict()


# ==========================================================================
# Preprocessing
# ==========================================================================
def test_resource_request_parsing():
    s = pd.Series([
        "{'cpus': 0.0206, 'memory': 0.0144}",
        "{'cpus': 1.5e-05, 'memory': None}",
        None,
        "garbage",
    ])
    cpu = pp._extract_pair(s, "cpus")
    mem = pp._extract_pair(s, "memory")
    assert cpu.iloc[0] == pytest.approx(0.0206)
    assert cpu.iloc[1] == pytest.approx(1.5e-05)
    assert pd.isna(cpu.iloc[2]) and pd.isna(cpu.iloc[3])
    assert mem.iloc[0] == pytest.approx(0.0144)
    assert pd.isna(mem.iloc[1]), "literal None must parse as NaN, not 0"


def test_event_stream_collapses_to_one_row_per_vm():
    """Several events for one instance must yield exactly one VM record."""
    raw = pd.DataFrame({
        "time": [0, 1, 2],
        "instance_events_type": [0, 6, 2],   # ENABLE, SCHEDULE, FAIL
        "collection_id": [7, 7, 7],
        "scheduling_class": [2, 2, 2],
        "collection_type": [0, 0, 0],
        "priority": [103, 103, 103],
        "instance_index": [5, 5, 5],
        "machine_id": [0, 999, 999],
        "resource_request": ["{'cpus': 0.1, 'memory': 0.05}"] * 3,
        "start_time": [10, 10, 10],
        "end_time": [20, 20, 30],
        "average_usage": ["{'cpus': 0.02, 'memory': 0.01}"] * 3,
        "maximum_usage": ["{'cpus': 0.08, 'memory': 0.04}"] * 3,
        "assigned_memory": [0.05] * 3,
        "page_cache_memory": [0.0] * 3,
        "cycles_per_instruction": [1.0] * 3,
        "sample_rate": [1.0] * 3,
        "cluster": [3, 3, 3],
        "event": ["ENABLE", "SCHEDULE", "FAIL"],
        "failed": [0, 0, 1],
    })
    out = pp.build_vm_table(raw)
    assert len(out) == 1, "three events for one instance produced != 1 VM"
    row = out.iloc[0]
    assert row["failed"] == 1, "a FAIL event must mark the instance failed"
    assert row["cpu_request"] == pytest.approx(0.1)
    assert row["max_cpu"] == pytest.approx(0.08)
    assert row["orig_machine_id"] == 999, "machine must come from SCHEDULE"
    assert row["n_events"] == 3


def test_clean_drops_unplaceable_and_fills_usage():
    vms = pd.DataFrame({
        "vm_id": ["a", "b", "c", "d"],
        "collection_id": [1, 2, 3, 4],
        "instance_index": [0, 0, 0, 0],
        "cluster": [1, 1, 1, 1],
        "cpu_request": [0.1, 0.0, 1.5, 0.2],      # zero and oversized dropped
        "mem_request": [0.1, 0.1, 0.1, 0.1],
        "avg_cpu": [np.nan, 0.0, 0.1, 0.05],      # NaN falls back to request
        "avg_mem": [0.05, 0.05, 0.05, 0.05],
        "max_cpu": [np.nan, 0.0, 0.1, 0.01],      # max < avg gets repaired
        "max_mem": [0.05, 0.05, 0.05, 0.05],
        "priority": [100, 100, 100, 100],
        "priority_band": ["x"] * 4,
        "priority_norm": [0.5] * 4,
        "scheduling_class": [1] * 4,
        "collection_type": [0] * 4,
        "failed": [0] * 4,
        "hard_failed": [0] * 4,
        "duration_s": [1.0] * 4,
        "start_time": [0] * 4,
        "end_time": [1] * 4,
        "n_events": [1] * 4,
        "orig_machine_id": [1] * 4,
        "assigned_memory": [0.1] * 4,
        "cycles_per_instruction": [1.0] * 4,
    })
    out = pp.clean_vm_table(vms)
    assert set(out["vm_id"]) == {"a", "d"}
    a = out[out["vm_id"] == "a"].iloc[0]
    assert a["avg_cpu"] == pytest.approx(0.1), "NaN usage must fall back to request"
    d = out[out["vm_id"] == "d"].iloc[0]
    assert d["max_cpu"] >= d["avg_cpu"], "max_cpu below avg_cpu was not repaired"


def test_stratified_sampling_preserves_priority_mix():
    rng = np.random.default_rng(0)
    n = 4000
    bands = rng.choice(["free", "best_effort", "production"], n,
                       p=[0.2, 0.5, 0.3])
    pool = make_vms(n)
    pool["priority_band"] = bands
    s = pp.sample_vms(pool, 400, strategy="stratified", seed=1)
    assert len(s) == 400
    want = pool["priority_band"].value_counts(normalize=True)
    got = s["priority_band"].value_counts(normalize=True)
    for b in want.index:
        assert abs(want[b] - got.get(b, 0)) < 0.05, f"band {b} mix drifted"


# ==========================================================================
# Fleet and feasibility
# ==========================================================================
def test_fleet_is_large_enough_to_be_feasible(instance):
    vms, fleet, cfg = instance
    f = sv.feasibility_report(vms, fleet)
    assert f["feasible"]
    assert f["total_cpu_capacity"] >= f["total_cpu_demand"]


def test_feasibility_flags_an_oversized_vm():
    vms = make_vms(5)
    vms.loc[0, "cpu_request"] = 5.0          # larger than any server
    fleet = sv.build_fleet(make_vms(5), FleetConfig(n_servers=4))
    rep = sv.feasibility_report(vms, fleet)
    assert not rep["feasible"]
    assert rep["n_unplaceable_vms"] == 1


def test_power_model_is_monotone_and_bounded():
    from vmplace.config import SERVER_CATALOGUE
    st = SERVER_CATALOGUE["standard"]
    assert st.power_at(0.0) == pytest.approx(st.idle_power_w)
    assert st.power_at(1.0) == pytest.approx(st.max_power_w)
    assert st.power_at(0.5) == pytest.approx(
        (st.idle_power_w + st.max_power_w) / 2
    )
    assert st.power_at(-1.0) == pytest.approx(st.idle_power_w)   # clipped
    assert st.power_at(99.0) == pytest.approx(st.max_power_w)    # clipped


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
