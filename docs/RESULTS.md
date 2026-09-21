# Experimental Results

Every number in this document comes from `outputs/experiments/*.csv`, which are
written by `scripts/03_run_experiments.py`. Nothing is illustrative, estimated
or carried over from the literature. Figures are in `outputs/figures/`; the
full console log of the run is `outputs/logs/run03.log`.

**Reproduce:**

```bash
python scripts/01_preprocess.py
python scripts/03_run_experiments.py --n-vms 250 --time-limit 200 --stress-time-limit 60 --pareto
python scripts/04_figures.py
```

Environment: CBC 2.10.3 via PuLP 3.3.2, one machine, Windows 11, Python 3.13.
Solves ran strictly one at a time (concurrent CBC processes were observed to
crash each other on this machine).

**How solver status is reported.** PuLP labels a solve "Optimal" even when CBC
merely stopped at its time limit with a feasible solution. This project reads
the real outcome from CBC's own log, so in every table below:

- **Optimal** = CBC proved optimality (gap within tolerance).
- **Feasible (time limit)** = a valid placement was found and improved, but
  optimality is *not* proven; the remaining gap is given where relevant.

An improvement over a heuristic is real in both cases (the MILP solution is a
verified feasible placement with a lower objective); only "Optimal" means no
better placement exists.

---

## 1. The dataset

| | |
|---|---|
| file | `borg_traces_data.csv`, 328 MB |
| lines | 1,324,695 |
| **records** | **405,894** |
| **VM instances after collapsing events** | **237,839** |
| collections (jobs) | 3,952 |
| clusters (cells) | 8 |
| distinct machines in the trace | 90,109 |
| dropped in cleaning | 5,107 (1,540 missing resources, 4,337 zero demand) |

Line count and record count differ because `cpu_usage_distribution` fields
contain newlines inside quoted values. Records are **events**, not VMs;
collapsing by `(collection_id, instance_index, cluster)` reduces 405,894 events
to 237,839 instances. Skipping that step would inflate demand roughly 1.7x.

| Measurement | Value | Why it matters |
|---|---|---|
| median CPU request | 0.0085 (0.85 % of a machine) | items are tiny relative to servers |
| top 1 % of VMs | 12.5 % of all CPU demand | skewed demand |
| median `avg_cpu / cpu_request` | 0.282 | requests ~3.5x actual usage: the slack consolidation exploits |
| median `max_cpu / avg_cpu` | 3.84 | bursty, so the QoS term uses peak usage |
| CPU vs memory request correlation | r = 0.399 | weak, so the wastage term carries an imbalance penalty |
| failure rate (any FAIL/EVICT/KILL/LOST) | 49.9 % | reliability proxy only; the trace has no SLA field |

Failure rate by priority band: monitoring 84.2 %, best_effort 47.5 %,
production 46.3 %, free 44.4 %, mid 30.8 %.

## 2. Reference instance: baseline versus optimised

250 VMs, stratified by priority band (seed 42); 11 identical servers (capacity
1.0, 120 W idle / 400 W peak, 1 h horizon); weights alpha=0.4, beta=0.3,
gamma=0.3. Total demand: 5.01 CPU, 3.08 memory. Bin-packing lower bound: **6
active servers**.

| Method | Active servers | Energy (Wh) | Wastage W | QoS penalty Q | CPU util | Objective Z | Runtime |
|---|---|---|---|---|---|---|---|
| First-Fit | 6 | 1410.4 | 4.868 | 7.769 | 83.5 % | 0.35662 | 0.002 s |
| Best-Fit | 6 | 1410.4 | 4.868 | 7.769 | 83.5 % | 0.35662 | 0.004 s |
| First-Fit-Decreasing | 6 | 1410.4 | 4.868 | 7.690 | 83.5 % | 0.35497 | 0.002 s |
| Trace-Original (remapped) | 11 | 2010.4 | 14.986 | 3.777 | 45.6 % | 0.46589 | 0.001 s |
| **MILP** | **6** | **1410.4** | **4.868** | **6.985** | **83.5 %** | **0.34028** | 20.7 s |

The MILP is **proven optimal** here (gap 0). Against First-Fit-Decreasing it
improves the objective by **4.1 %** (0.35497 to 0.34028).

**Where the gain comes from.** All three greedy methods and the MILP use the
same 6 servers, so energy is identical (1410.4 Wh) and wastage is identical
(4.868). 6 is the bin-packing lower bound, so no method can use fewer. The
entire improvement is in the **QoS term: 7.690 to 6.985 (-9.2 %)**. FFD packs
by size only and is blind to which VMs are bursty or failure-prone; the MILP
chooses, among the many 6-server packings, the one that spreads peak risk best.
That is something a size-only heuristic cannot do.

Against the trace's own placement the MILP halves the active servers (11 to 6)
and cuts energy 29.8 %. **Do not read that as a headline result:** the real
Borg cell spread these VMs across thousands of machines and solved a different
problem (constraints, locality, failure domains), so remapping onto an
11-server fleet mostly measures the fleet size. First-Fit-Decreasing is the
honest baseline.

## 3. The multi-objective trade-off

Same instance, fleet fixed, weights varied. Every row below is proven optimal.

| alpha | beta | gamma | Active servers | Energy (Wh) | Wastage | QoS penalty | CPU util |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 6 | 1410.4 | 4.868 | 7.690 | 83.5 % |
| 1 | 0 | 0 | 6 | 1410.4 | 4.868 | 7.690 | 83.5 % |
| 0.7 | 0.2 | 0.1 | 6 | 1410.4 | 4.951 | 6.938 | 83.5 % |
| 0.2 | 0.6 | 0.2 | 6 | 1410.4 | 4.868 | 6.938 | 83.5 % |
| 0.4 | 0.3 | 0.3 | 6 | 1410.4 | 4.868 | 6.985 | 83.5 % |
| **0.2** | **0.2** | **0.6** | **11** | **2010.4** | **14.868** | **2.688** | **45.6 %** |
| 0 | 0 | 1 | 11 | 2010.4 | 14.891 | 2.712 | 45.6 % |

Findings:

1. **The objectives genuinely conflict and the weights genuinely steer the
   answer.** Moving from balanced weights (gamma=0.3) to QoS-focused
   (gamma=0.6) makes the optimiser switch on every server: **6 to 11 servers,
   +42.5 % energy, wastage x3.1, in exchange for QoS penalty -61.5%**
   (6.985 to 2.688).
2. **The change is a step, not a curve.** Placement can only change by whole
   servers, so the response to gamma is discontinuous, characteristic of an
   integer program.
3. **Energy and wastage are not in conflict with each other** on this instance
   (alpha=1 and beta=1 give the same placement): both reward consolidation. The
   real tension is consolidation versus QoS, and only gamma expresses it.
4. Even at the low-QoS end there is room to move without spending energy: at the
   same 6 servers the QoS penalty ranges from 6.938 to 7.690 depending on
   weights, i.e. QoS can be improved ~10 % for free before energy has to be
   paid.

### Pareto frontier

A weighted-sum sweep over a 0.2-step simplex grid (21 weight vectors, all
solved). 5 of 21 are reported non-dominated, but they collapse to **only two
distinct points**, consistent with the step behaviour above:

| Point | Active servers | E' | W' | Q' |
|---|---|---|---|---|
| consolidated | 6 | 0.3206 | 0.2213 | 0.4823 |
| spread | 11 | 0.4569 | 0.6758 | 0.1869 |

Weighted sums can only reach the **convex** part of a Pareto front; any
non-convex efficient placements between these two would need an
epsilon-constraint method to find. Whether such points exist on this instance
was not tested.

## 4. Stress instances: where exact optimisation has room to win

On the representative sample above, servers are the binding constraint and FFD
already reaches the lower bound, so the MILP can only win through the QoS term.
To see how the MILP behaves when packing itself is hard, instances were built
from VMs with CPU request in a band (the `band` sampling strategy): items
around a quarter to a third of a machine. These are **deliberately hard
instances, not representative ones.** Five random instances per configuration
(seeds 100-104), 60 s limit, fleet slack 1.4.

**Band [0.30, 0.45), 40 VMs**

| Seed | FFD servers | MILP servers | Lower bound | Z (FFD) | Z (MILP) | Improvement | Status |
|---|---|---|---|---|---|---|---|
| 100 | 15 | 15 | 14 | 0.4579 | 0.4415 | 3.58 % | time limit, gap 3 % |
| 101 | 14 | 14 | 14 | 0.4485 | 0.4469 | 0.36 % | Optimal |
| 102 | 14 | 14 | 13 | 0.4531 | 0.4511 | 0.44 % | Optimal |
| 103 | 14 | 14 | 14 | 0.4548 | 0.4476 | 1.60 % | time limit, gap ~0 |
| 104 | 14 | 14 | 14 | 0.4489 | 0.4422 | 1.50 % | Optimal |

Mean improvement **1.50 %** (0.36-3.58 %); 3 of 5 proven optimal.

**Band [0.20, 0.45), 60 VMs**

| Seed | FFD servers | MILP servers | Lower bound | Z (FFD) | Z (MILP) | Improvement | Status |
|---|---|---|---|---|---|---|---|
| 100 | 20 | 20 | 17 | 0.4133 | 0.3582 | 13.33 % | time limit, gap 15 % |
| 101 | 20 | 20 | 17 | 0.4049 | 0.3302 | 18.46 % | time limit, gap 14 % |
| 102 | 19 | 19 | 17 | 0.4035 | 0.3616 | 10.39 % | time limit, gap 13 % |
| 103 | 19 | 19 | 17 | 0.4081 | 0.3434 | 15.86 % | time limit, gap 16 % |
| 104 | 20 | 20 | 17 | 0.4030 | 0.3093 | 23.24 % | time limit, gap 10 % |

Mean improvement **16.25 %** (10.4-23.2 %). **None of these five were proven
optimal** within 60 s; remaining gaps are 10-16 %. The improvements are real
(each is a verified feasible placement beating FFD) but the true optimum could
be better still, and no claim about optimality is made for them.

In **every** stress instance the MILP used the *same number of servers as FFD*.
The gains come from arranging the same machines better across wastage and QoS,
not from consolidating further. Note also that on the harder instances the
bin-packing lower bound (17) is below what either method achieves (19-20);
whether a 17-18 server packing exists was not resolved.

## 5. Sensitivity to the assumed power model

The idle/peak ratio is the one number in the objective that is assumed rather
than measured (the trace has no power data).

| idle / max | Active servers | Energy (Wh) |
|---|---|---|
| 20 % | 6 | 1269.1 |
| 30 % | 6 | 1410.4 |
| 40 % | 6 | 1551.8 |
| 50 % | 6 | 1693.2 |
| 60 % | 6 | 1834.5 |
| 70 % | 6 | 1975.9 |

**The chosen placement is identical across the entire range**; only the Wh
figure attached to it changes (linearly). On this instance the decision is
therefore robust to the power assumption. The reason is structural: 6 is the
bin-packing lower bound, so no power model could justify fewer servers, and
none makes more servers attractive at gamma=0.3. The *absolute* Wh figures
remain conditional on the assumed model; the recommendation is not.

## 6. Sensitivity to the QoS threshold

| theta | Active servers | Energy (Wh) | QoS penalty |
|---|---|---|---|
| 0.60 | 6 | 1410.4 | 8.438 |
| 0.70 | 6 | 1410.4 | 7.838 |
| 0.80 | 6 | 1410.4 | 7.238 |
| 0.85 | 6 | 1410.4 | 6.985 |
| 0.90 | 6 | 1410.4 | 6.638 |
| 1.00 | 6 | 1410.4 | 6.038 |

At gamma=0.3 the number of active servers does not change with theta. The
measured penalty falls by about 0.6 per 0.1 of threshold on average, consistent
with the CPU overshoot term binding on each of the 6 active servers; the fall is
not exactly linear (0.5 to 0.7 per 0.1 step) because the optimiser re-arranges
VMs among the same servers as theta moves. Theta begins to change the number of
servers only when gamma is large enough to cross the step in Section 3.

## 7. Sensitivity to workload size

Fleet re-sized for each workload (slack 2.0). All four points proven optimal.

| VMs | Fleet | Active servers | Energy (Wh) | CPU util | Solve time |
|---|---|---|---|---|---|
| 125 | 5 | 3 | 729.1 | 80.3 % | 0.4 s |
| 187 | 8 | 4 | 978.3 | 90.0 % | 1.1 s |
| 250 | 11 | 6 | 1410.4 | 83.5 % | 23.3 s |
| 375 | 16 | 8 | 1956.6 | 94.8 % | 120.5 s |

A 50 % larger workload (250 to 375 VMs, scenario D of the brief) needs 2 more
servers (6 to 8), and energy rises 38.7 %, less than proportionally because the
servers run fuller. Active servers scale roughly with demand and utilisation
rises with load, as expected.

## 8. Solver scaling

Homogeneous fleet, stratified samples, weights (0.4, 0.3, 0.3), 1 % gap target,
200 s limit, warm-started from FFD.

| VMs | Servers | Binary vars | Status | Gap | Runtime |
|---|---|---|---|---|---|
| 50 | 3 | 153 | Optimal | 0 | 0.2 s |
| 100 | 4 | 404 | Optimal | 0 | 0.3 s |
| 150 | 6 | 906 | Optimal | 0 | 0.6 s |
| 200 | 9 | 1,809 | Optimal | 1 % | 1.8 s |
| 300 | 13 | 3,913 | Optimal | 0 | 85.0 s |
| 375 | 16 | 6,016 | Optimal | 0 | 151.6 s |
| 400 | 17 | 6,817 | **Feasible (time limit)** | 2 % | 213.5 s |

Runtime grows sharply: 1,809 to 3,913 binaries (2.2x) costs 46x the time. At
6,817 binaries the 200 s limit is reached with a 2 % gap. Solving all 237,839
VMs exactly is out of reach by orders of magnitude (against even 20 servers it
would need ~4.8 million binaries), which is why the study samples.

Warm-starting from FFD matters at the upper end. In an earlier diagnostic at 375
VMs, a run with the warm start finished, while one without it found **no
feasible solution** in 120 s. (That diagnostic was run before the solver-status
fix and used a different sample; it is reported as a qualitative observation,
not a reproducible table row.)

## 9. Correctness safeguards (found and fixed during development)

Each of these would have silently corrupted reported numbers; each is now
covered by a test in `tests/test_model.py` (24 tests, all passing).

1. **The model optimised a different function from the one reported.** The MILP
   used one combined QoS slack; the evaluator computed two separate ones, so a
   server at 100 % CPU and 10 % memory scored zero risk. The symptom was an
   "optimal" MILP losing to a greedy heuristic. Every solve now cross-checks the
   solver's objective against the independent evaluation (agreement ~1e-9).
2. **A timed-out solver produced phantom placements.** CBC can return a status
   of "Optimal" with no variable values; scoring those reported 7.9 CPU units
   on 2 unit-capacity servers. Extracted solutions are validated before use.
3. **An infeasible run scored best.** Every objective term rewards servers being
   off, so an empty placement scored a perfect Z = 0. Incomplete placements now
   score NaN.
4. **"Optimal" did not mean optimal.** PuLP reports "Optimal" for time-limited
   solves. Discovered while preparing this document: an earlier draft claimed
   proven optimality for results that had actually hit the time limit. Status,
   lower bound and gap are now parsed from CBC's log, and a solve that stops on
   time is reported as `Feasible (time limit)`.

## 10. Honest summary

**What the study shows**

1. A multi-objective MILP over real Borg workload data produces provably
   optimal placements up to ~6,000 binary variables, and good feasible ones
   beyond (2 % gap at ~6,800).
2. On the representative reference instance the MILP beats First-Fit-Decreasing
   by 4.1 % (proven optimal), entirely through lower QoS risk at the same server
   count and energy.
3. The three objectives genuinely conflict. Raising the QoS weight from 0.3 to
   0.6 doubles the powered-on servers (6 to 11) and trades +42.5 % energy for
   -61.5 % QoS penalty. The frontier consists of discrete steps.
4. The recommended placement is robust to the assumed idle-power ratio across
   20-70 %.
5. On deliberately hard packing instances the MILP improves on FFD by 1.5 %
   (band [0.30, 0.45)) to 16 % (band [0.20, 0.45)) on average, always with the
   same server count.

**What it does not show**

1. **No energy saving over a competent heuristic on the representative
   instance.** FFD already reaches the server lower bound; all of the MILP's gain
   is QoS. Any claim of large energy savings over FFD would be false.
2. **The 16 % stress-instance improvements are not proven optimal.** They are
   verified feasible improvements with 10-16 % remaining gaps.
3. **Absolute energy figures depend on an assumed power model.** Ratios and
   decisions are meaningful; Wh values are conditional.
4. **The QoS term is a modelling proxy**, not a measured SLA quantity.
5. **Results come from samples** of up to 400 VMs, not the full 237,839.
6. **Single reference instance for the weight, power and threshold sweeps**
   (seed 42); the stress study uses five seeds per configuration, the other
   sweeps do not.
7. **Static single-period placement**: no migration, arrivals, network topology
   or failure domains.
