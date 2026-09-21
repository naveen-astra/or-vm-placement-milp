# Literature Review

Scope: virtual machine (VM) placement in cloud data centres, with emphasis on
energy-aware and multi-objective formulations, and on the workload trace used
in this project. Each source below was checked against its publisher or
indexing page; where a detail could not be confirmed it is marked.

## 1. The problem and why it is hard

VM placement maps a set of VMs, each with a multi-dimensional resource demand
(CPU, memory, ...), onto a set of physical machines (PMs) with finite capacity.
With a single resource and a single objective it is the classical bin-packing
problem, which is NP-hard; with several resources and objectives it is a
vector bin-packing problem with conflicting criteria. Exact methods therefore
scale poorly, which explains why most of the literature is heuristic or
metaheuristic, and why the results of this project (Section 5 of the report)
show exact MILP solution becoming expensive at a few thousand binary variables.

## 2. Energy-aware consolidation

Beloglazov and Buyya (2012) study dynamic consolidation of VMs using live
migration and switching idle hosts to a low-power state, proposing adaptive
heuristics driven by historical utilisation data and evaluating both energy
and adherence to service-level agreements. Their work is the standard
reference for two modelling conventions adopted here: energy modelled as a
function of CPU utilisation with a substantial fixed idle component, and the
explicit treatment of performance/SLA as a competing objective to energy.

*Relevance to this project:* the load-proportional power model in
`config.py` follows this convention. The literature commonly assumes a large
idle fraction, which is why this project sweeps the idle/peak ratio from 20 % to
70 % instead of trusting a single value.

## 3. Multi-objective VM placement

**Integer programming.** Regaieg, Koubàa, Ales and Aguili (2021) formulate VM
placement as a multi-objective integer linear program that jointly optimises
the number of hosted VMs, resource wastage and the number of active PMs (as a
proxy for power), for both homogeneous and heterogeneous data centres. This is
the closest prior work to the present project: it establishes that
multi-objective integer programming for VM placement is an existing approach,
so this project does **not** claim methodological novelty in the formulation.

**Ant-colony metaheuristics.** Gao, Guan, Qi, Hou and Liu (2013) propose a
multi-objective ant colony system for VM placement, minimising power
consumption and resource wastage simultaneously, and report it as the first
application of ant colony system to the multi-objective VM placement problem.
Their resource-wastage measure, which rewards balanced use of CPU and memory on
each host, is the basis of the wastage-with-imbalance term used here.

**Evolutionary methods.** Torre et al. (2020, *Information and Software
Technology* 128, 106390) present a dynamic evolutionary multi-objective VM
placement heuristic for cloud data centres, representing the population-based
alternative to exact methods.

**Communication-aware placement.** A 2020 paper in *Sustainable Computing:
Informatics and Systems* (vol. 28), "Multi-objective communication-aware
optimization for virtual machine placement in cloud datacenters", adds network
bandwidth usage as a third objective alongside power and resource wastage.
*Authorship could not be confirmed from the sources consulted and should be
verified before citing by name.* Network-aware placement is outside the scope of
this project (listed as a future extension).

**JAYA-based placement.** A 2023 study in the same journal applies a
multi-objective discrete JAYA algorithm to energy-efficient, topology-aware
placement ("An energy-efficient topology-aware virtual machine placement in
Cloud Datacenters: A multi-objective discrete JAYA optimization"; authors not
confirmed). It is representative of metaheuristic energy/wastage optimisation
and is deliberately not reproduced here, since the brief scopes the project to
exact MILP.

## 4. Workload data

Tirmazi et al. (2020, EuroSys '20) describe the 2019 Google Borg cluster trace
covering eight clusters over May 2019 and compare it longitudinally with the
2011 trace. It is the source of all workload data in this project. Two
properties of the trace shape the modelling and are documented in the report:
resource values are normalised (1.0 = the largest machine), and the trace
carries no machine power measurements and no SLA field, so the power and QoS
terms of the model are stated assumptions rather than data.

## 5. Positioning of this project

| Aspect | Prior work above | This project |
|---|---|---|
| Multi-objective integer programming for VM placement | exists (Regaieg et al.) | reused as the formulation family; not claimed as novel |
| Real workload trace | mostly simulated or synthetic workloads in the metaheuristic papers | Google Borg 2019, 237,839 VM instances |
| QoS objective | SLA-violation counts from simulators | trace has no SLA data; risk-weighted peak-overcommit proxy, and a proof that the naive failure count cannot influence the placement |
| Baseline | often other metaheuristics | First-Fit / Best-Fit / First-Fit-Decreasing, plus the trace's own placement |
| Delivery | algorithm and simulation | reproducible pipeline, solver-honest reporting (bound and gap), test suite, interactive dashboard |
| Sensitivity analysis | varies | weights, workload size, assumed idle power, QoS threshold, Pareto frontier |

The contribution is therefore an applied, reproducible, dataset-driven study
with explicit handling of what the data does and does not support, not a new
optimisation model.

## References

1. A. Beloglazov and R. Buyya. Optimal online deterministic algorithms and
   adaptive heuristics for energy and performance efficient dynamic
   consolidation of virtual machines in Cloud data centers. *Concurrency and
   Computation: Practice and Experience*, 24(13):1397-1420, 2012.
2. Y. Gao, H. Guan, Z. Qi, Y. Hou and L. Liu. A multi-objective ant colony
   system algorithm for virtual machine placement in cloud computing. *Journal
   of Computer and System Sciences*, 79(8):1230-1242, 2013.
3. R. Regaieg, M. Koubàa, Z. Ales and T. Aguili. Multi-objective optimization
   for VM placement in homogeneous and heterogeneous cloud service provider data
   centers. *Computing*, 103:1255-1279, 2021. doi:10.1007/s00607-021-00915-z.
4. E. Torre, J. J. Durillo, V. de Maio, P. Agrawal, S. Benedict, N. Saurabh and
   R. Prodan. A dynamic evolutionary multi-objective virtual machine placement
   heuristic for cloud data centers. *Information and Software Technology*,
   128:106390, 2020.
5. M. Tirmazi, A. Barker, N. Deng, M. E. Haque, Z. G. Qin, S. Hand,
   M. Harchol-Balter and J. Wilkes. Borg: the next generation. *EuroSys '20*,
   2020. doi:10.1145/3342195.3387517.
6. *Multi-objective communication-aware optimization for virtual machine
   placement in cloud datacenters.* Sustainable Computing: Informatics and
   Systems, vol. 28, 2020. (authors to be confirmed)
7. *An energy-efficient topology-aware virtual machine placement in Cloud
   Datacenters: A multi-objective discrete JAYA optimization.* Sustainable
   Computing: Informatics and Systems, 2023. (authors to be confirmed)
