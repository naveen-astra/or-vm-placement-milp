"""Stage 3: the full experimental run.

Produces every result the report needs:
  * baseline versus optimised comparison          (brief section 22)
  * objective-weight sensitivity                  (brief section 20, A/B/C)
  * workload-size sensitivity                     (brief section 20, D)
  * power-assumption sensitivity                  (the assumption that matters most)
  * MILP scaling behaviour
  * optional Pareto frontier                      (brief section 36F)

Everything is written to outputs/experiments/ as CSV, so the report and the
dashboard read the same numbers.

Usage
-----
    python scripts/03_run_experiments.py
    python scripts/03_run_experiments.py --n-vms 300 --time-limit 180
    python scripts/03_run_experiments.py --pareto
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vmplace import baselines, milp, preprocess as pp, sensitivity, servers as sv
from vmplace.config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    WEIGHT_PRESETS,
    FleetConfig,
    ScenarioConfig,
    SolverConfig,
    Weights,
)
from vmplace.metrics import build_normalisers, compare_results


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


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _progress(prefix: str):
    def cb(k: int, total: int, label: str) -> None:
        print(f"    [{k+1}/{total}] {prefix} {label}", flush=True)
    return cb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vm-table", default=None)
    ap.add_argument("--n-vms", type=int, default=250)
    ap.add_argument("--strategy", default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slack", type=float, default=2.0)
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--mip-gap", type=float, default=0.01)
    ap.add_argument("--pareto", action="store_true",
                    help="also compute the Pareto frontier (slow)")
    ap.add_argument("--skip-scaling", action="store_true")
    args = ap.parse_args()

    outdir = os.path.join(OUTPUT_DIR, "experiments")
    os.makedirs(outdir, exist_ok=True)

    path = args.vm_table or latest_vm_table()
    print(f"loading {path}")
    pool = _load(path)
    print(f"  pool: {len(pool):,} VMs")

    base_cfg = ScenarioConfig(
        weights=Weights(0.4, 0.3, 0.3),
        fleet=FleetConfig(slack=args.slack),
        solver=SolverConfig(time_limit_s=args.time_limit, mip_gap=args.mip_gap),
        label="base",
    )

    workload = pp.sample_vms(pool, args.n_vms, strategy=args.strategy,
                             seed=args.seed)
    fleet = sv.build_fleet(workload, base_cfg.fleet)
    norm = build_normalisers(workload, fleet, base_cfg)

    print(f"\ninstance: {len(workload)} VMs, {len(fleet)} servers, "
          f"{len(workload)*len(fleet):,} binary x-variables")
    feas = sv.feasibility_report(workload, fleet)
    print(f"  demand cpu={feas['total_cpu_demand']:.2f} "
          f"mem={feas['total_mem_demand']:.2f}  "
          f"capacity cpu={feas['total_cpu_capacity']:.2f} "
          f"mem={feas['total_mem_capacity']:.2f}")
    print(f"  bin-packing lower bound on active servers: "
          f"{max(feas['min_servers_cpu_bound'], feas['min_servers_mem_bound'])}")
    if not feas["feasible"]:
        print(f"  WARNING: instance looks infeasible -- {feas}")

    workload.to_csv(os.path.join(outdir, "workload.csv"), index=False)
    fleet.to_csv(os.path.join(outdir, "fleet.csv"), index=False)

    # ================================================================
    # 1. baseline versus optimised
    # ================================================================
    print("\n=== 1. baseline versus optimised ===")
    results = baselines.run_all_baselines(workload, fleet, base_cfg,
                                          normalisers=norm)
    t0 = time.perf_counter()
    milp_res = milp.solve(workload, fleet, base_cfg, normalisers=norm)
    print(f"  MILP: {milp_res.status} in {time.perf_counter()-t0:.1f}s")
    if milp_res.metrics.get("milp_solution_used") == 0:
        print(f"  NOTE: {milp_res.notes}")
    results.append(milp_res)

    comp = compare_results(results)
    comp.to_csv(os.path.join(outdir, "comparison.csv"), index=False)

    show = [
        "method", "n_active_servers", "energy_wh", "wastage",
        "qos_overcommit", "cpu_util_active_mean", "mem_util_active_mean",
        "weighted_objective", "n_capacity_violations", "runtime_s",
    ]
    print(comp[[c for c in show if c in comp.columns]].to_string(index=False))

    # headline numbers, computed rather than asserted
    ffd = comp[comp["method"] == "First-Fit-Decreasing"]
    mil = comp[comp["method"].str.startswith("MILP")]
    if len(ffd) and len(mil):
        e0 = float(ffd["energy_wh"].iloc[0])
        e1 = float(mil["energy_wh"].iloc[0])
        s0 = int(ffd["n_active_servers"].iloc[0])
        s1 = int(mil["n_active_servers"].iloc[0])
        print("\n  MILP vs First-Fit-Decreasing:")
        print(f"    active servers {s0} -> {s1}")
        print(f"    energy {e0:.1f} Wh -> {e1:.1f} Wh "
              f"({(e1-e0)/e0*100:+.2f}%)")
        print(f"    objective {float(ffd['weighted_objective'].iloc[0]):.5f} -> "
              f"{float(mil['weighted_objective'].iloc[0]):.5f}")

    milp_res.assignment.to_csv(
        os.path.join(outdir, "optimal_assignment.csv"), index=False
    )
    milp_res.server_loads.to_csv(
        os.path.join(outdir, "optimal_server_loads.csv"), index=False
    )

    # ================================================================
    # 2. objective-weight sensitivity
    # ================================================================
    print("\n=== 2. objective-weight sensitivity ===")
    grid = list(WEIGHT_PRESETS.values()) + [
        Weights(1.0, 0.0, 0.0), Weights(0.0, 1.0, 0.0), Weights(0.0, 0.0, 1.0),
    ]
    sw = sensitivity.sweep_weights(workload, base_cfg, grid=grid, fleet=fleet,
                                   progress=_progress("weights"))
    sw.to_csv(os.path.join(outdir, "sensitivity_weights.csv"), index=False)
    print(sensitivity.summarise_sweep(sw)[
        ["scenario", "n_active_servers", "energy_wh", "wastage",
         "qos_overcommit", "cpu_util_active_mean", "runtime_s"]
    ].to_string(index=False))

    # ================================================================
    # 3. workload-size sensitivity
    # ================================================================
    print("\n=== 3. workload-size sensitivity ===")
    sizes = [int(args.n_vms * f) for f in (0.5, 0.75, 1.0, 1.5)]
    ws = sensitivity.sweep_workload_size(pool, base_cfg, sizes,
                                         strategy=args.strategy, seed=args.seed,
                                         progress=_progress("size"))
    ws.to_csv(os.path.join(outdir, "sensitivity_workload.csv"), index=False)
    print(sensitivity.summarise_sweep(ws)[
        ["scenario", "n_servers_in_fleet", "n_active_servers", "energy_wh",
         "cpu_util_active_mean", "status", "runtime_s"]
    ].to_string(index=False))

    # ================================================================
    # 4. power-assumption sensitivity
    # ================================================================
    print("\n=== 4. power-assumption sensitivity (idle/max ratio) ===")
    ps = sensitivity.sweep_power_assumption(
        workload, base_cfg, progress=_progress("idle")
    )
    ps.to_csv(os.path.join(outdir, "sensitivity_power.csv"), index=False)
    print(sensitivity.summarise_sweep(ps)[
        ["scenario", "n_active_servers", "energy_wh",
         "cpu_util_active_mean", "runtime_s"]
    ].to_string(index=False))

    # ================================================================
    # 5. QoS safety-threshold sensitivity
    # ================================================================
    print("\n=== 5. QoS safety-threshold sensitivity ===")
    qs = sensitivity.sweep_parameter(
        workload, base_cfg, "safety_threshold",
        [0.60, 0.70, 0.80, 0.85, 0.90, 1.00],
        fixed_fleet=True, progress=_progress("theta"),
    )
    qs.to_csv(os.path.join(outdir, "sensitivity_threshold.csv"), index=False)
    print(sensitivity.summarise_sweep(qs)[
        ["scenario", "n_active_servers", "energy_wh", "qos_overcommit",
         "cpu_util_active_mean"]
    ].to_string(index=False))

    # ================================================================
    # 6. MILP scaling
    # ================================================================
    if not args.skip_scaling:
        print("\n=== 6. MILP scaling behaviour ===")
        rows = []
        for n in [50, 100, 150, 200, 300]:
            w = pp.sample_vms(pool, n, strategy=args.strategy, seed=args.seed)
            cfg = ScenarioConfig(
                weights=Weights(0.4, 0.3, 0.3),
                fleet=FleetConfig(slack=args.slack),
                solver=SolverConfig(time_limit_s=args.time_limit,
                                    mip_gap=args.mip_gap),
            )
            f = sv.build_fleet(w, cfg.fleet)
            r = milp.solve(w, f, cfg)
            rows.append({
                "n_vms": n,
                "n_servers": len(f),
                "binary_vars": n * len(f) + len(f),
                "constraints": r.metrics.get("n_constraints"),
                "status": r.status,
                "runtime_s": r.runtime_s,
                "build_time_s": r.metrics.get("build_time_s"),
                "solve_time_s": r.metrics.get("solve_time_s"),
                "n_active_servers": r.metrics["n_active_servers"],
                "objective": r.metrics["weighted_objective"],
                "milp_solution_used": r.metrics.get("milp_solution_used"),
            })
            print(f"    n={n:<4} servers={len(f):<3} "
                  f"binaries={n*len(f):<6} {r.status:<12} "
                  f"{r.runtime_s:6.1f}s", flush=True)
        sc = pd.DataFrame(rows)
        sc.to_csv(os.path.join(outdir, "scaling.csv"), index=False)

    # ================================================================
    # 7. Pareto frontier (optional, slow)
    # ================================================================
    if args.pareto:
        print("\n=== 7. Pareto frontier ===")
        pf = sensitivity.pareto_frontier(workload, base_cfg, step=0.2,
                                         fleet=fleet,
                                         progress=_progress("pareto"))
        pf.to_csv(os.path.join(outdir, "pareto.csv"), index=False)
        n_opt = int(pf["pareto_optimal"].sum())
        print(f"  {n_opt} non-dominated of {len(pf)} weight combinations")

    # ================================================================
    meta = {
        "vm_table": path,
        "n_vms": int(len(workload)),
        "n_servers": int(len(fleet)),
        "sampling_strategy": args.strategy,
        "seed": args.seed,
        "config": base_cfg.to_dict(),
        "feasibility": feas,
        "normalisers": norm.as_dict(),
    }
    with open(os.path.join(outdir, "run_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"\nall results written to {outdir}")


if __name__ == "__main__":
    main()
