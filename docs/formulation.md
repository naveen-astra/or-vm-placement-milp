# Mathematical Formulation

Energy-efficient virtual machine placement as a multi-objective mixed-integer
linear program, instantiated on the Google Borg 2019 cluster trace.

---

## 1. Sets and indices

| Symbol | Meaning |
|---|---|
| $i \in I$ | VM instances, derived from the trace |
| $j \in J$ | physical servers in the modelled fleet |

A VM instance is identified in the trace by the triple
`(collection_id, instance_index, cluster)`. This matters: the raw trace rows are
**events**, not VMs, and a single instance emits several rows (`ENABLE`,
`SCHEDULE`, `FINISH`, `FAIL`, …). Treating rows as VMs would place the same
workload many times and inflate demand by roughly 1.7x.

## 2. Parameters

### From the trace

| Symbol | Source column | Meaning |
|---|---|---|
| $c_i$ | `resource_request.cpus` | CPU requested by VM $i$ |
| $m_i$ | `resource_request.memory` | memory requested by VM $i$ |
| $\bar c_i, \bar m_i$ | `average_usage` | mean observed usage |
| $\hat c_i, \hat m_i$ | `maximum_usage` | peak observed usage |
| $f_i \in \{0,1\}$ | `failed` / event type | instance ever failed, was evicted, killed or lost |
| $p_i$ | `priority` | Borg scheduling priority, normalised to $\tilde p_i \in [0,1]$ |

All resource values are **normalised by the trace itself**, so $1.0$ is the
capacity of the largest machine in the cell (Tirmazi et al., 2020). No unit
conversion is performed anywhere in this project.

### Modelled, not measured

| Symbol | Meaning | Status |
|---|---|---|
| $C_j, M_j$ | CPU and memory capacity of server $j$ | **assumption** |
| $P^{\text{idle}}_j, P^{\text{max}}_j$ | idle and peak power draw | **assumption** |
| $T$ | energy integration horizon (hours) | modelling choice |
| $\theta$ | QoS safety threshold (default $0.85$) | modelling choice |
| $\lambda$ | wastage imbalance coefficient (default $0.5$) | modelling choice |

> **The trace subset used here contains no machine capacities and no power
> measurements.** The power figures follow the shape of published
> SPECpower_ssj2008 results for two-socket x86 servers, where idle draw is
> typically 45–60 % of peak. They are representative values, not dataset
> values, and every energy number in this project inherits that assumption.
> Section 8 sweeps it precisely because it is the load-bearing assumption.

## 3. Decision variables

$$
x_{ij} \in \{0,1\} \qquad
\text{VM } i \text{ is placed on server } j
$$

$$
y_j \in \{0,1\} \qquad
\text{server } j \text{ is powered on}
$$

$$
d_j \ge 0 \qquad
\text{auxiliary: } |r^c_j - r^m_j|, \text{ the residual imbalance}
$$

$$
o^c_j,\ o^m_j \ge 0 \qquad
\text{auxiliary: risk-weighted peak overcommit, per resource}
$$

Placement is a genuinely discrete decision — a VM runs on one machine, not 40 %
on one and 60 % on another — which is why this is an integer program and not an
LP. The LP relaxation is nonetheless useful as a bound and is what the
branch-and-bound search prunes against.

## 4. Objective

$$
\min\ Z \;=\; \alpha E' + \beta W' + \gamma Q'
$$

where $E'$, $W'$ and $Q'$ are the three components normalised onto a common
scale (Section 6), and $\alpha + \beta + \gamma = 1$ after normalisation.

### 4.1 Energy

$$
E \;=\; T \sum_{j \in J}
\left[
P^{\text{idle}}_j\, y_j
\;+\;
\bigl(P^{\text{max}}_j - P^{\text{idle}}_j\bigr)
\frac{1}{C_j}\sum_{i \in I} \bar c_i\, x_{ij}
\right]
$$

This is the standard linear power model $P(u) = P^{\text{idle}} + (P^{\text{max}} -
P^{\text{idle}})\,u$ of Beloglazov and Buyya (2012), deliberately kept linear so
it embeds directly in the MILP with no piecewise reformulation.

Two properties are worth noting. First, the term is driven by
$\bar c_i$ (**observed average usage**), not $c_i$ (requested), because power is
consumed by work actually done; requests are used for the capacity constraints,
where the reservation is what must be honoured. Second, an idle-but-on server
still costs $P^{\text{idle}}_j$ — that fixed cost is the entire reason
consolidation pays, and if $P^{\text{idle}} = 0$ the energy term would be
indifferent to how many servers are on.

Setting `load_proportional = False` degenerates this to $E = T\sum_j P^{\max}_j y_j$,
the simpler flat-cost form, which is available for comparison.

### 4.2 Resource wastage

Define the normalised residual capacity of server $j$:

$$
r^c_j = y_j - \frac{1}{C_j}\sum_i c_i x_{ij},
\qquad
r^m_j = y_j - \frac{1}{M_j}\sum_i m_i x_{ij}
$$

Both are zero when the server is off, and lie in $[0,1]$ when it is on. Then

$$
W \;=\; \sum_{j \in J}\Bigl[\, r^c_j + r^m_j + \lambda\, d_j \,\Bigr]
$$

The $\lambda d_j$ term penalises **imbalance** between the two residuals. A
server with 60 % spare CPU but 2 % spare memory is effectively full: that CPU
can never be sold. This is a linearised form of the wastage metric of Gao et
al. (2013), and it matters empirically here because CPU and memory requests in
this trace correlate only weakly ($r = 0.399$), so lopsided residuals are
common rather than hypothetical.

### 4.3 QoS / reliability penalty

**The trace has no SLA field** — no response-time target, no latency bound, no
availability figure. Any QoS term must therefore be a stated proxy.

Define a per-VM risk weight from two real trace fields:

$$
\rho_i \;=\; 1 + w_f\, f_i + w_p\, \tilde p_i
$$

so an instance that failed, or one running at high priority, contributes more
measured risk. Then

$$
Q \;=\; \sum_{j\in J}\bigl(o^c_j + o^m_j\bigr)
\;=\;
\sum_{j\in J}
\left[
\max\!\left(0,\ \frac{1}{C_j}\sum_i \rho_i \hat c_i x_{ij} - \theta y_j\right)
+
\max\!\left(0,\ \frac{1}{M_j}\sum_i \rho_i \hat m_i x_{ij} - \theta y_j\right)
\right]
$$

This measures how far the risk-weighted **peak** demand on each server exceeds a
safety threshold $\theta$. It uses `maximum_usage`, which the data justifies:
the median VM peaks at **3.84x** its own average CPU, so a placement that packs
to averages will be badly overcommitted at peak.

#### Why not $Q = \sum_i f_i$?

The project brief's first definition is $q_i = f_i$, $Q = \sum_i q_i$. Under the
assignment constraint $\sum_j x_{ij} = 1$,

$$
\sum_i q_i \;=\; \sum_i f_i \Bigl(\textstyle\sum_j x_{ij}\Bigr) \;=\; \sum_i f_i
\;=\; \text{constant}
$$

It does not depend on $x$ at all. Adding it to the objective shifts $Z$ by a
fixed offset and changes nothing about which placement is optimal — $\gamma$
would be inert. The failure rate is therefore **reported as a descriptive
metric** but the overcommit form above is what the objective actually optimises.
The implementation still offers `qos.mode = "failure_proxy"`, and the UI warns
explicitly that $\gamma$ has no effect in that mode.

## 5. Constraints

**(C1) Assignment.** Every VM is placed on exactly one server.

$$\sum_{j \in J} x_{ij} = 1 \qquad \forall i \in I$$

**(C2) CPU capacity.**

$$\sum_{i \in I} c_i\, x_{ij} \le C_j\, y_j \qquad \forall j \in J$$

**(C3) Memory capacity.**

$$\sum_{i \in I} m_i\, x_{ij} \le M_j\, y_j \qquad \forall j \in J$$

**(C4) Activation linking.**

$$x_{ij} \le y_j \qquad \forall i \in I,\ j \in J$$

**(C5) Imbalance linearisation.** Encodes $d_j = |r^c_j - r^m_j|$, valid because
the objective minimises $d_j$:

$$d_j \ge r^c_j - r^m_j, \qquad d_j \ge r^m_j - r^c_j \qquad \forall j$$

**(C6) QoS overcommit.** Encodes the $\max(0,\cdot)$ above, again valid because
the objective pushes $o$ down:

$$
o^c_j \ge \frac{1}{C_j}\sum_i \rho_i \hat c_i x_{ij} - \theta y_j,
\qquad
o^m_j \ge \frac{1}{M_j}\sum_i \rho_i \hat m_i x_{ij} - \theta y_j
$$

**(C7) Symmetry breaking.** Within a block of identical servers:

$$y_j \ge y_{j+1}$$

**(C8) Valid inequality.** A bin-packing lower bound on active servers:

$$
\sum_j y_j \;\ge\;
\left\lceil \max\left(
\frac{\sum_i c_i}{\max_j C_j},\ \frac{\sum_i m_i}{\max_j M_j}
\right)\right\rceil
$$

### Notes on the formulation

**(C4) is redundant for feasibility but not for speed.** Since every VM has
$c_i > 0$, constraint (C2) already forces $y_j = 1$ whenever any $x_{ij} = 1$.
(C4) is kept because it gives a substantially tighter LP relaxation — the
classic strong-versus-weak formulation trade-off in facility location. It adds
$|I|\cdot|J|$ rows, so on large instances the tightening and the added size pull
against each other; `solver.valid_inequalities` toggles it so the effect can be
measured rather than assumed.

**(C7) matters more than it looks.** With a homogeneous fleet, any solution can
be permuted across identical servers into $n!$ equivalent solutions, and
branch-and-bound will happily explore them all. Forcing the active servers to be
a prefix of the list collapses that symmetry.

**Why a single $o_j$ would be wrong.** An earlier version of this model used one
combined slack $o_j \ge \text{cpu} + \text{mem} - 2\theta y_j$. That is strictly
weaker: a server at 100 % CPU and 10 % memory scores zero QoS risk, because the
spare memory cancels the CPU overshoot. Separating the slacks per resource makes
$Q$ equal the sum of two independent $\max(0,\cdot)$ terms, which is both the
correct semantics and exactly what the evaluator measures.

## 6. Normalisation

$E$, $W$ and $Q$ have incompatible units — watt-hours, dimensionless residual,
dimensionless overcommit. Adding them directly would make $\alpha,\beta,\gamma$
meaningless, since whichever term happened to be numerically largest would
dominate regardless of its weight. Each is therefore divided by an
**instance-level worst case**:

$$
E'=\frac{E}{T\sum_j P^{\max}_j},\qquad
W'=\frac{W}{2|J|},\qquad
Q'=\frac{Q}{\sum_i \rho_i \hat c_i / \min_j C_j + \sum_i \rho_i \hat m_i / \min_j M_j}
$$

These denominators depend only on the instance, never on a particular solution,
which is what a MILP objective requires — a normaliser computed from observed
solutions would change the objective every time a new solution was found. It
also means the same normalisers apply to the MILP and to every baseline, so the
comparison is like for like.

## 7. Model size

| Quantity | Count |
|---|---|
| binary variables | $\lvert I\rvert \cdot \lvert J\rvert + \lvert J\rvert$ |
| continuous variables | $3\lvert J\rvert$ |
| constraints | $\lvert I\rvert + 6\lvert J\rvert + \lvert I\rvert\lvert J\rvert + 1$ |

The $|I||J|$ growth in binaries is what limits exact solution. 237,839 VMs
against even 20 servers would need ~4.76 million binaries, far beyond CBC.
Experiments therefore run on stratified samples that preserve the priority-band
mix of the full workload, and the sample size is reported with every result.

## 8. Sensitivity analysis

| Sweep | Question it answers |
|---|---|
| $\alpha,\beta,\gamma$ | How much energy is given up per unit of QoS protection? |
| workload size | Does the advantage hold as the instance grows, and where does solve time break down? |
| $P^{\text{idle}}/P^{\text{max}}$ | **Is the conclusion an artefact of the assumed power model?** |
| $\theta$ | How much does the safety margin cost? |
| fleet slack | Does spare capacity change the recommendation? |
| Pareto frontier | What does the whole trade-off surface look like? |

The idle-power sweep is the most important one in the study. The idle/max ratio
is the only number in the objective that is assumed rather than measured, and it
is precisely what determines whether consolidation is worth anything: at
$P^{\text{idle}} = P^{\text{max}}$ switching a server off saves everything, and at
$P^{\text{idle}} = 0$ an idle server is free and there is no reason to consolidate
at all. A result that survives that sweep is robust; one that does not is an
artefact of the assumption.

### On the weighted-sum method

Scalarising with weights can only recover solutions on the **convex hull** of
the Pareto front. Non-convex ("unsupported") efficient points are unreachable by
any choice of $\alpha,\beta,\gamma$, however fine the grid. Recovering those
needs an $\varepsilon$-constraint or lexicographic method. The convex portion is
sufficient to show the trade-off shape here, and the limitation is stated rather
than glossed.

## 9. References

- Beloglazov, A., & Buyya, R. (2012). Optimal online deterministic algorithms
  and adaptive heuristics for energy and performance efficient dynamic
  consolidation of virtual machines. *CCPE*, 24(13).
- Gao, Y., Guan, H., Qi, Z., Hou, Y., & Liu, L. (2013). A multi-objective ant
  colony system algorithm for virtual machine placement in cloud computing.
  *JCSS*, 79(8).
- Regaieg, R., Koubàa, M., Ales, Z., & Aguili, T. (2021). Multi-objective
  optimization for VM placement in homogeneous and heterogeneous cloud service
  provider data centers. *Computing*, 103.
- Tirmazi, M., et al. (2020). Borg: the Next Generation. *EuroSys '20*.
- Google (2020). Google cluster-usage traces v3.
