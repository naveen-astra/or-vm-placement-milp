# Energy-Efficient VM Placement via Multi-Objective Integer Programming

An Operations Research study of virtual machine placement in cloud data
centres, formulated as a multi-objective mixed-integer linear program and
instantiated on the **Google Borg 2019 cluster trace**.

The model decides which VM runs on which physical server so as to trade off
three objectives that genuinely conflict: **energy consumption**, **resource
wastage**, and **QoS/reliability risk**.

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/01_preprocess.py          # trace -> per-VM table  (~10 s)
python scripts/02_eda.py                 # statistics and figures
python scripts/03_run_experiments.py --pareto   # full experimental run (~40 min)
python scripts/04_figures.py             # report figures

streamlit run app/streamlit_app.py       # interactive dashboard
python -m pytest tests/ -q               # 24 model-invariant tests
```

`scripts/01_preprocess.py` must run first; everything else reads its output
from `data/processed/`.

**Start with the technical report:** [docs/REPORT.md](docs/REPORT.md).
Measured result tables are in [docs/RESULTS.md](docs/RESULTS.md), the model is
in [docs/formulation.md](docs/formulation.md), and the literature review is
[docs/literature_review.md](docs/literature_review.md).

## Layout

```
src/vmplace/
  config.py       parameters, power model, and every stated ASSUMPTION
  preprocess.py   raw event stream -> one clean record per VM instance
  servers.py      physical fleet construction and feasibility pre-checks
  milp.py         the MILP: variables, objective, constraints, solve
  baselines.py    First-Fit, Best-Fit, First-Fit-Decreasing, trace placement
  metrics.py      one evaluator shared by the MILP and every baseline
  sensitivity.py  weight / parameter sweeps and Pareto frontier
scripts/          01 preprocess, 02 EDA, 03 experiments, 04 figures
app/              Streamlit dashboard
tests/            24 model-invariant tests
docs/             REPORT, RESULTS, formulation, literature review
outputs/          generated figures, result CSVs and run logs
```

## The model in brief

$$\min\ Z = \alpha E' + \beta W' + \gamma Q'$$

subject to every VM being placed exactly once, CPU and memory capacity on each
server, and a server being powered on if anything runs on it.

| | |
|---|---|
| $x_{ij}\in\{0,1\}$ | VM $i$ runs on server $j$ |
| $y_j\in\{0,1\}$ | server $j$ is powered on |
| $E$ | load-proportional energy of the servers left on |
| $W$ | unused capacity on active servers, plus CPU/memory imbalance |
| $Q$ | risk-weighted peak demand above a safety threshold |

See [docs/formulation.md](docs/formulation.md) for the complete statement,
including the linearisations and the reasoning behind each term.

## Things worth knowing before reading the results

**The trace rows are events, not VMs.** A single VM instance emits several rows
(`ENABLE`, `SCHEDULE`, `FINISH`, `FAIL`, …). The identity of an instance is the
triple `(collection_id, instance_index, cluster)`. The full trace file holds
1,324,695 *lines* but only **405,894 records** — `cpu_usage_distribution` fields
contain newlines inside quotes — which collapse to **237,839 VM instances**.
Treating rows as VMs would inflate demand by roughly 1.7x.

**Resource units are the trace's own.** CPU and memory are normalised so that
$1.0$ is the largest machine in the cell. A modelled server of capacity $1.0$
*is* that machine. Nothing is converted.

**The power model is an assumption, not data.** This trace subset contains
machine identifiers but no machine capacities and no power measurements. The
idle/peak figures in `config.SERVER_CATALOGUE` follow the shape of published
SPECpower_ssj2008 results and are flagged as assumptions in the code. Because
the idle/max ratio is exactly what makes consolidation worthwhile,
`scripts/03` sweeps it — a conclusion that does not survive that sweep is an
artefact of the assumption rather than a finding.

**The QoS term is a stated proxy.** The trace has no SLA field of any kind. The
brief's literal definition $Q=\sum_i \mathrm{failed}_i$ turns out to be
**constant** under the assignment constraint $\sum_j x_{ij}=1$, so it cannot
influence the optimal placement and $\gamma$ would be inert. The active term is
therefore a risk-weighted peak-overcommit penalty built from `maximum_usage`,
`failed` and `priority`; the plain failure rate is still reported. The
dashboard warns when the inert mode is selected.

**Results here are computed, not illustrative.** Every number in `outputs/`
comes from an actual solve. Nothing in this repository contains mock figures.

## Solver notes

The bundled solver is CBC, via PuLP.

- Model size grows as $|I|\cdot|J|$ binaries. On this machine CBC proves
  optimality through about 6,000 binaries (152 s); at about 6,800 the time limit
  binds and the result is reported as `Feasible (time limit)` with its gap.
- PuLP labels time-limited solves "Optimal". This project parses CBC's own log
  instead, so `Optimal` means proven and `Feasible (time limit)` means not.
- Every solve is **warm-started from First-Fit-Decreasing**, which both speeds
  up the search and guarantees that a timed-out run still returns a feasible
  placement at least as good as the heuristic.
- CBC can fail as an external process rather than return a status. `milp.solve`
  catches that, falls back to the warm-start placement, and labels it clearly;
  it never raises and never reports a placement it could not verify.
- Every MILP result carries an `objective_mismatch` self-check comparing the
  solver's objective against the independently computed one. A non-trivial
  mismatch means the model and the evaluator have drifted apart, which is the
  single most dangerous failure mode in a study like this.

## Baselines

| Baseline | Role |
|---|---|
| First-Fit | the naive policy |
| Best-Fit | standard greedy consolidation |
| **First-Fit-Decreasing** | the serious baseline; has an $11/9\,\mathrm{OPT}+1$ guarantee for 1-D bin packing |
| Trace-Original | Borg's own placement, remapped onto the modelled fleet |

Trace-Original is **context, not a fair head-to-head**: the real cell had tens
of thousands of machines against this fleet's handful, so its active-server
count is not comparable. First-Fit-Decreasing is the baseline the MILP has to
justify itself against.

### The headline findings

On the 250-VM reference instance the MILP is **proven optimal** and beats
First-Fit-Decreasing by **4.1 %**, but at the *same* server count and energy:
6 servers is the bin-packing lower bound, so nothing can use fewer. The whole
gain is lower QoS risk, because a size-only heuristic cannot see which VMs are
bursty or failure-prone. **No energy saving over a good heuristic is claimed.**

- **Trade-off.** Raising the QoS weight from 0.3 to 0.6 switches on every server
  (6 to 11), trading +42.5 % energy for a 61.5 % lower QoS penalty. The response
  is a step, not a curve, because placement changes in whole servers.
- **Robustness.** The chosen placement is identical for assumed idle-power
  ratios from 20 % to 70 %.
- **Hard instances.** For VMs sized at a quarter to a third of a machine the
  MILP improves on FFD by 1.5 % to 16 % on average, always at the same server
  count. The larger gains (10-23 %) are **not proven optimal** (gaps 10-16 %).

Full numbers, caveats and what the study does not show:
[docs/RESULTS.md](docs/RESULTS.md).

## Dataset

Google cluster-usage traces v3 (2019), `borg_traces_data.csv`, 328 MB.
Not redistributed here; place it at `borg_traces_data.csv/borg_traces_data.csv`
(a folder named `borg_traces_data.csv` containing the file of the same name)
or point `--path` at your copy.
