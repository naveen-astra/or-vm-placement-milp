# Experimental Results

Every number in this document was produced by running the code in this
repository. Nothing here is illustrative, estimated, or carried over from the
literature. Figures are in `outputs/figures/`, raw CSVs in
`outputs/experiments/`.

**Reproduce with:**

```bash
python scripts/01_preprocess.py
python scripts/03_run_experiments.py --n-vms 250 --time-limit 90
python scripts/04_figures.py
```

Solver: CBC 2.10 via PuLP 3.3.2, single machine, Windows 11, Python 3.13.

---

## 1. The dataset as it actually is

| | |
|---|---|
| file | `borg_traces_data.csv`, 328 MB |
| lines | 1,324,695 |
| **records** | **405,894** |
| **VM instances after collapsing** | **237,839** |
| collections (jobs) | 3,952 |
| cells (clusters) | 8 |
| distinct machines in the trace | 90,109 |
| records dropped in cleaning | 5,107 (1,540 missing resources, 4,337 zero demand) |

The line/record gap is not a parsing error: `cpu_usage_distribution` and
`tail_cpu_usage_distribution` contain newlines inside quoted fields, so one
record spans ~3.3 lines. And records are **events**, not VMs — collapsing the
event stream by `(collection_id, instance_index, cluster)` reduces 405,894
events to 237,839 instances. Skipping that step would inflate total demand by
roughly 1.7x.

### Workload characteristics that drive the model

| Measurement | Value | Consequence for the model |
|---|---|---|
| median CPU request | **0.0085** (0.85 % of a machine) | items are tiny relative to bins — see §3 |
| median memory request | 0.0036 | |
| top 1 % of VMs | hold **12.5 %** of all CPU demand | demand is skewed but not pathologically |
| median `avg_cpu / cpu_request` | **0.282** | requests are ~3.5x actual usage; this over-provisioning is the slack consolidation exploits |
| median `max_cpu / avg_cpu` | **3.84** | VMs are bursty, so packing to averages overcommits badly at peak — this is why the QoS term uses `maximum_usage` |
| CPU/memory request correlation | **r = 0.399** | only weakly correlated, so lopsided residuals are common — this is why W carries an imbalance term |
| failure rate | **49.9 %** | high, because `failed` counts FAIL/EVICT/KILL/LOST across the instance's whole event stream |

Failure rate by priority band:

| Band | VMs | Failure rate |
|---|---|---|
| monitoring | 30,156 | 84.2 % |
| best_effort | 97,958 | 47.5 % |
| production | 53,230 | 46.3 % |
| free | 33,984 | 44.4 % |
| mid | 22,511 | 30.8 % |

## 2. Baseline versus optimised — representative workload

250 VMs sampled stratified by priority band; fleet of 10 homogeneous servers
(capacity 1.0, 120 W idle / 400 W peak, 1 h horizon); weights
α=0.4, β=0.3, γ=0.3.

Total demand: 4.976 CPU, 3.079 memory. **Bin-packing lower bound: 5 servers.**

| Method | Active servers | Energy (Wh) | Wastage | QoS penalty | CPU util | Objective Z | Runtime |
|---|---|---|---|---|---|---|---|
| First-Fit | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % | 0.33467 | 0.002 s |
| Best-Fit | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % | 0.33467 | 0.003 s |
| First-Fit-Decreasing | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % | 0.33467 | 0.001 s |
| Trace-Original | 10 | 1890.4 | 12.893 | 3.586 | 49.8 % | 0.45730 | 0.001 s |
| **MILP** | **5** | **1290.4** | **2.893** | **7.770** | **99.5 %** | **0.33467** | 4.5 s |

### The MILP ties the heuristics here, and that is the finding

The MILP is provably optimal and every greedy heuristic already reaches the
same objective. This is not a defect in the model — it is a property of the
workload, and it is worth stating plainly rather than hiding:

The median VM requests **0.85 % of a machine**. Packing items that small into
bins is easy. First-Fit-Decreasing carries an $11/9\,\mathrm{OPT}+1$ guarantee
for one-dimensional bin packing, and with items this tiny the additive term
dominates and FFD lands exactly on the bin-packing lower bound of 5 servers.
No method can do better, so the exact method has nothing left to win.

**What the MILP contributes here is a proof, not a better answer.** Before
solving, nobody knew 5 was achievable; the heuristic produces a number, the MILP
produces a number *and* a certificate that nothing beats it. That certificate is
what a greedy method structurally cannot provide.

Against the trace's own placement the optimised solution halves the active
servers (10 → 5) and cuts energy 31.7 %, but see the caveat in §7.

## 3. Where exact optimisation does pay — stress instances

To find the regime where the MILP earns its cost, instances were built from VMs
whose CPU request falls in a band (the `band` sampling strategy). These are
deliberately **stress instances, not representative ones**.

| Instance | VMs | Fleet | LB | FFD servers | Z (FFD) | MILP servers | Z (MILP) | **Improvement** |
|---|---|---|---|---|---|---|---|---|
| band [0.30, 0.45) | 40 | 19 | 14 | 14 | 0.44912 | 14 | 0.44185 | **1.62 %** |
| band [0.30, 0.45) | 60 | 28 | 20 | 21 | 0.45450 | 22 | 0.45392 | 0.13 % |
| **band [0.20, 0.45)** | **60** | **24** | **17** | **20** | **0.37764** | **20** | **0.30016** | **20.52 %** |
| stratified | 120 | 5 | 3 | 3 | 0.32879 | 3 | 0.32649 | 0.70 % |
| stratified | 400 | 14 | 7 | 7 | 0.33193 | 7 | 0.32702 | 1.48 % |

The 20.5 % case is instructive: **both methods use 20 servers**, so the gain is
not from consolidation at all. The MILP found a *rearrangement* across the same
number of machines that is far better balanced on wastage and peak risk. FFD
optimises one thing — how few bins it can use — and is blind to the other two
objectives by construction. That is the structural advantage of the
multi-objective formulation, and it does not show up in a server count.

## 4. The multi-objective trade-off

Same 250-VM instance, fleet held fixed, weights swept.

| α | β | γ | Active servers | Energy (Wh) | Wastage | QoS penalty | CPU util |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % |
| 1 | 0 | 0 | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % |
| 0.7 | 0.2 | 0.1 | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % |
| 0.2 | 0.6 | 0.2 | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % |
| 0.4 | 0.3 | 0.3 | 5 | 1290.4 | 2.893 | 7.770 | 99.5 % |
| **0.2** | **0.2** | **0.6** | **10** | **1890.4** | **12.893** | **3.552** | **49.8 %** |
| 0 | 0 | 1 | 10 | 1890.4 | 13.157 | 3.521 | 49.8 % |

**This is the central result of the project.** The objectives genuinely
conflict, and the weights genuinely steer the answer:

- Crossing γ ≈ 0.6 the optimiser **doubles the active servers, 5 → 10**, accepting
  **+46.5 % energy** to buy **−54.3 % QoS risk**.
- Energy and wastage move together (both reward consolidation); QoS moves
  against both.
- The transition is a **sharp step, not a smooth curve** — characteristic of an
  integer program, where the answer can only change by whole servers. There is
  no placement "between" 5 and 10 machines.

Note that α=1,β=0,γ=0 and α=0,β=1,γ=0 give identical placements. On this
instance energy and wastage are not in conflict with each other: both are
minimised by packing into as few servers as possible. The real tension is
**consolidation versus QoS**, and only γ expresses it.

## 5. Sensitivity to the assumed power model

This is the most important robustness check in the study, because the idle/peak
power ratio is the one number in the objective that is assumed rather than
measured.

| idle / max | Active servers | Energy (Wh) |
|---|---|---|
| 20 % | 5 | 1189.0 |
| 30 % | 5 | 1290.4 |
| 40 % | 5 | 1391.8 |
| 50 % | 5 | 1493.1 |
| 60 % | 5 | 1594.5 |
| 70 % | 5 | 1695.9 |

**The chosen placement does not change at all** across the whole sweep — only
the energy figure attached to it does, and that scales linearly as expected.

The recommendation is therefore **robust to the power assumption** on this
instance: one would deploy the same placement whatever the true idle draw. The
absolute Wh figures are only as trustworthy as the assumption, but the
*decision* is not contingent on it. This is the distinction between a result
that survives its assumptions and one that is an artefact of them.

The reason the placement is insensitive here is that 5 servers is the
bin-packing lower bound, so no power model could justify using fewer. On an
instance with slack above the bound, the idle ratio would be expected to bite.

## 6. Sensitivity to the QoS threshold θ

| θ | Active servers | Energy (Wh) | QoS penalty |
|---|---|---|---|
| 0.60 | 5 | 1290.4 | 9.026 |
| 0.70 | 5 | 1290.4 | 8.520 |
| 0.80 | 5 | 1290.4 | 8.020 |
| 0.85 | 5 | 1290.4 | 7.770 |
| 0.90 | 5 | 1290.4 | 7.520 |
| 1.00 | 5 | 1290.4 | 7.020 |

At γ=0.3 the placement is unchanged; θ shifts the *measured* penalty by a
constant (0.5 per 0.1 of threshold, i.e. exactly the 5 active servers times the
threshold change) without altering the optimal decision. The threshold only
starts to bind on the placement once γ is large enough to cross the step
identified in §4.

## 7. Caveat on the Trace-Original baseline

Trace-Original is reported for context and **should not be read as a headline
comparison**. The real Borg cell spread these VMs across thousands of distinct
machines; the modelled fleet has ten. Remapping that placement onto ten servers
round-robin preserves Borg's *grouping* but not its scale, so its "10 active
servers" is an artefact of the fleet size, not a measurement of Borg's
efficiency. Borg was also solving a different problem — with constraints,
locality, failure domains and preemption that this model does not represent.

**First-Fit-Decreasing is the honest baseline**, and the MILP ties it on
representative workloads (§2) and beats it on tight ones (§3).

## 8. Solver scaling

| VMs | Servers | Binary vars | Status | Runtime |
|---|---|---|---|---|
| 50 | 3 | 153 | Optimal | 0.16 s |
| 100 | 4 | 404 | Optimal | 0.24 s |
| 150 | 6 | 906 | Optimal | 0.50 s |
| 200 | 9 | 1,809 | Optimal | 1.60 s |
| 300 | 13 | 3,913 | Optimal | 58.2 s |
| 375 | 16 | 6,016 | Optimal | 132 s |

Growth is sharply superlinear — a ~2x increase in binaries from 200 to 300 VMs
costs ~36x the runtime. Solving all 237,839 VMs exactly is out of reach by many
orders of magnitude, which is why the design samples and reports the sample.

### Warm starting is what makes the larger instances solvable

At 375 VMs / 6,016 binaries, with a 120 s budget:

| | Result |
|---|---|
| **with** FFD warm start | **Optimal in 132 s** |
| **without** warm start | **no feasible solution found at all** (11 VMs still unassigned) |

Seeding CBC with the First-Fit-Decreasing solution is the difference between a
proven optimum and nothing. It also guarantees that any timed-out run still
returns a placement at least as good as the heuristic.

CBC was also observed to crash outright (non-zero exit, no solution file) when
several solves ran concurrently on a loaded machine. `milp.solve` catches that,
falls back to the warm-start placement, and labels the result; it does not
propagate the exception.

## 9. Correctness safeguards

Three defects were found during development that would each have silently
corrupted every reported number. All three are now covered by tests in
`tests/test_model.py` (22 tests, all passing).

**The model optimised a different function from the one reported.** The MILP
used a single combined QoS slack $o_j \ge \mathrm{cpu} + \mathrm{mem} - 2\theta y_j$
while the evaluator computed $\max(0,\mathrm{cpu}-\theta) + \max(0,\mathrm{mem}-\theta)$
separately. The combined form lets a CPU overshoot be cancelled by spare memory,
so a server at 100 % CPU and 10 % memory scored zero QoS risk. The symptom was a
provably "Optimal" MILP scoring **worse** than a greedy heuristic — which is
impossible and was the clue. Every solve now carries an `objective_mismatch`
self-check comparing the solver's objective against the independently evaluated
one; it currently agrees to ~1e-9.

**A timed-out solver produced phantom placements.** CBC can return status
`Optimal` with `None`-valued variables when it stops before finding an
incumbent. Reading those as "unplaced" and scoring them yielded a reported
solution fitting 7.9 units of CPU demand onto 2 servers of capacity 1.0. Results
are now validated before being trusted.

**An infeasible run scored better than every feasible one.** Every term in Z
rewards servers being switched off, so an assignment that places nothing scores
a perfect Z = 0 — and in the comparison table an infeasible MILP appeared to
win by "100 %". Incomplete placements are now marked invalid and propagate as
NaN.

## 10. Honest summary

**What the study shows.**

1. A multi-objective MILP over real Borg workload data produces placements that
   are provably optimal on instances up to ~6,000 binary variables.
2. The three objectives genuinely conflict, and the weights genuinely control
   the outcome: crossing γ ≈ 0.6 trades +46.5 % energy for −54.3 % QoS risk by
   doubling the active servers.
3. The recommendation is robust to the assumed power model — the same placement
   is chosen whether idle draw is 20 % or 70 % of peak.
4. On tight instances the exact method improves the objective by up to 20.5 %
   over First-Fit-Decreasing, mostly through better balance rather than fewer
   servers.

**What it does not show.**

1. **No energy saving over a competent heuristic on representative workloads.**
   Google's VMs are tiny relative to machines, and greedy bin packing is already
   optimal there. Any claim of large savings over FFD would be false.
2. **The absolute energy figures are only as good as the assumed power model.**
   The trace has no power data. Ratios and comparisons are meaningful; the Wh
   numbers are conditional.
3. **The QoS term is a proxy.** The trace has no SLA field. "QoS risk" here means
   risk-weighted peak demand above a threshold, which is a modelling decision,
   not a measurement.
4. **Results are from samples**, not the full 237,839-VM workload, which is not
   exactly solvable.
5. **Static, single-period placement.** No migration cost, no arrivals or
   departures, no network topology, no failure domains.
