"""Configuration objects and documented modelling assumptions.

Every value in this module that is NOT derived from the Google Borg trace is
marked with an ASSUMPTION note.  This separation is deliberate: the trace gives
us workload demand, but it does not give us server capacities or power draw, so
those must come from a stated model rather than be silently invented.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

# --------------------------------------------------------------------------
# Resource units
# --------------------------------------------------------------------------
# The Google 2019 trace reports CPU and memory in *normalised* units where 1.0
# is the capacity of the largest machine in the cell (Tirmazi et al., 2020,
# "Borg: the Next Generation", EuroSys).  We express server capacity in the
# same normalised units, so a server of capacity 1.0 == one largest machine.
# No unit conversion is required or attempted anywhere in this project.
LARGEST_MACHINE = 1.0


@dataclass(frozen=True)
class ServerType:
    """A physical machine archetype.

    ASSUMPTION (power).  The trace contains no power measurements.  The
    idle_power_w and max_power_w values follow the shape of published
    SPECpower_ssj2008 results for commodity two-socket x86 servers, where idle
    draw is typically 45-60 percent of peak draw.  They are representative
    figures, not measured values for the machines in the trace.  Change them
    here or in the UI and every downstream energy number changes with them.
    """

    name: str
    cpu_capacity: float
    mem_capacity: float
    idle_power_w: float
    max_power_w: float

    def power_at(self, utilisation: float) -> float:
        """Linear power model P(u) = P_idle + (P_max - P_idle) * u.

        This is the standard linear approximation used in the energy-aware VM
        placement literature (Beloglazov and Buyya, 2012).  It is deliberately
        linear so it can be embedded directly in a MILP objective with no
        piecewise or nonlinear reformulation.
        """
        u = max(0.0, min(1.0, utilisation))
        return self.idle_power_w + (self.max_power_w - self.idle_power_w) * u


# ASSUMPTION.  A small heterogeneous catalogue.  "standard" is the reference
# machine (capacity 1.0 == largest machine in the cell).  The other two are
# scaled variants so heterogeneous scenarios can be explored; their capacity
# ratios are modelling choices, not trace facts.
SERVER_CATALOGUE: Dict[str, ServerType] = {
    "standard": ServerType("standard", 1.00, 1.00, 120.0, 400.0),
    "high_density": ServerType("high_density", 1.50, 1.25, 150.0, 520.0),
    "low_power": ServerType("low_power", 0.75, 0.80, 70.0, 250.0),
}


@dataclass
class PowerModelConfig:
    """How energy is turned into a number."""

    # Window over which energy is integrated, in hours.  Energy = power x time.
    horizon_hours: float = 1.0
    # If True, energy is load proportional:
    #     P_idle * y_j + (P_max - P_idle) * load_j
    # If False, energy degenerates to the simpler form sum_j P_j * y_j that
    # section 12 of the project brief describes.
    load_proportional: bool = True
    # Which demand vector drives the power curve: "request" or "average".
    # "average" uses observed average_usage and is the more physically honest
    # choice; "request" uses resource_request and is the conservative choice.
    power_driver: str = "average"


@dataclass
class QoSConfig:
    """Definition of the QoS / reliability penalty.

    The trace has no SLA field.  Two definitions are supported.

    "failure_proxy"
        q_i = failed_i and Q = sum_i q_i / N.  This is the literal definition
        in section 12 of the project brief.  NOTE: because the assignment
        constraint forces sum_j x_ij = 1 for every VM, this quantity is a
        CONSTANT with respect to the decision variables and therefore cannot
        influence the optimal placement.  It is still computed and reported,
        but using it as the objective's Q term makes gamma inert.

    "peak_overcommit"  (default)
        A placement-dependent penalty.  For each server we measure how far the
        risk-weighted PEAK demand of its assigned VMs exceeds a safety
        threshold, using maximum_usage from the trace.  The per-VM risk weight
        rho_i is raised for instances that failed and for high-priority
        instances, so the model avoids piling risky or important work onto an
        already peaky server.  Fully linear; see milp.py.
    """

    mode: str = "peak_overcommit"
    # Fraction of capacity above which peak demand counts as a QoS risk.
    safety_threshold: float = 0.85
    # rho_i = 1 + failure_weight * failed_i + priority_weight * priority_norm_i
    failure_weight: float = 0.50
    priority_weight: float = 0.50


@dataclass
class WastageConfig:
    """Definition of the resource-wastage term.

    Wastage on an ACTIVE server j is the normalised unused capacity

        rc_j = y_j - cpu_load_j / C_j^cpu
        rm_j = y_j - mem_load_j / C_j^mem

    plus an imbalance penalty lambda * |rc_j - rm_j| which discourages lopsided
    residuals, since a server with spare CPU but no spare memory is effectively
    full.  This is a linearised form of the wastage metric of Gao et al.
    (2013).  Inactive servers contribute zero.
    """

    imbalance_lambda: float = 0.50
    # Which demand vector defines "used" capacity: "request" or "average".
    wastage_driver: str = "request"


@dataclass
class Weights:
    """Multi-objective weights over the NORMALISED components."""

    alpha: float = 0.4  # energy
    beta: float = 0.3   # wastage
    gamma: float = 0.3  # QoS / reliability

    def normalised(self) -> "Weights":
        s = self.alpha + self.beta + self.gamma
        if s <= 0:
            raise ValueError("objective weights must sum to a positive number")
        return Weights(self.alpha / s, self.beta / s, self.gamma / s)


# Named weight presets from section 14 of the project brief.
WEIGHT_PRESETS: Dict[str, Weights] = {
    "Energy-focused": Weights(0.7, 0.2, 0.1),
    "Balanced": Weights(0.4, 0.3, 0.3),
    "QoS-focused": Weights(0.2, 0.2, 0.6),
    "Wastage-focused": Weights(0.2, 0.6, 0.2),
}


@dataclass
class FleetConfig:
    """How the physical server fleet is constructed."""

    # Explicit server count.  If None, the fleet is sized from demand.
    n_servers: Optional[int] = None
    # When sizing from demand: n = ceil(total_demand / capacity) * slack.
    slack: float = 2.0
    min_servers: int = 3
    max_servers: int = 60
    # Mix of server types.  Homogeneous by default, matching the brief's table.
    composition: Dict[str, float] = field(
        default_factory=lambda: {"standard": 1.0}
    )


@dataclass
class SolverConfig:
    time_limit_s: int = 120
    mip_gap: float = 0.01            # 1 percent relative optimality gap
    threads: int = 0                 # 0 = solver default
    msg: bool = False
    symmetry_breaking: bool = True   # y_j >= y_{j+1} within identical types
    valid_inequalities: bool = True  # lower bound on active server count


@dataclass
class ScenarioConfig:
    """Everything needed to reproduce one optimisation run."""

    weights: Weights = field(default_factory=Weights)
    power: PowerModelConfig = field(default_factory=PowerModelConfig)
    qos: QoSConfig = field(default_factory=QoSConfig)
    wastage: WastageConfig = field(default_factory=WastageConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    label: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Dataset locations
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RAW_TRACE_PATH = os.path.join(
    PROJECT_ROOT, "borg_traces_data.csv", "borg_traces_data.csv"
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
