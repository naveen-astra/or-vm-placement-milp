"""Stage 1: raw Google Borg trace -> clean per-VM workload table.

Usage
-----
    python scripts/01_preprocess.py                  # whole 1.32M-row trace
    python scripts/01_preprocess.py --nrows 200000   # a subset, for speed
    python scripts/01_preprocess.py --clusters 1 2   # only cells 1 and 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vmplace import preprocess as pp
from vmplace.config import PROCESSED_DIR, RAW_TRACE_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=RAW_TRACE_PATH)
    ap.add_argument("--nrows", type=int, default=None,
                    help="cap on raw EVENT rows (not VMs)")
    ap.add_argument("--clusters", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"reading {args.path}")
    t0 = time.perf_counter()
    raw = pp.load_raw(args.path, nrows=args.nrows, clusters=args.clusters)
    print(f"  {len(raw):,} event rows in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    vms = pp.build_vm_table(raw)
    print(f"  collapsed to {len(vms):,} VM instances "
          f"in {time.perf_counter()-t1:.1f}s")

    vms = pp.clean_vm_table(vms)
    print(f"  after cleaning: {len(vms):,} VMs "
          f"({vms.attrs['n_dropped']:,} dropped -- "
          f"{vms.attrs['n_dropped_missing']:,} missing resources, "
          f"{vms.attrs['n_dropped_too_small']:,} zero demand, "
          f"{vms.attrs['n_dropped_too_big']:,} larger than any machine)")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    tag = "full" if args.nrows is None else f"n{args.nrows}"
    out = args.out or os.path.join(PROCESSED_DIR, f"vm_table_{tag}.parquet")
    try:
        vms.to_parquet(out, index=False)
    except Exception:
        out = out.replace(".parquet", ".csv")
        vms.to_csv(out, index=False)
    print(f"wrote {out}")

    stats = pp.dataset_statistics(vms)
    spath = os.path.join(PROCESSED_DIR, f"stats_{tag}.json")
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"wrote {spath}\n")

    print("--- workload summary ---")
    for k in ["n_vms", "n_collections", "n_clusters", "n_original_machines",
              "total_cpu_request", "total_mem_request", "mean_cpu_request",
              "mean_mem_request", "failure_rate", "cpu_lower_bound_servers"]:
        print(f"  {k:28} {stats[k]}")
    print(f"  priority_distribution        {stats['priority_distribution']}")


if __name__ == "__main__":
    main()
