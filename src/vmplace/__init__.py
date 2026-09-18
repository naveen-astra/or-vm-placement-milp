"""Energy-efficient VM placement via multi-objective integer programming.

Pipeline:  preprocess -> servers -> (milp | baselines) -> metrics -> sensitivity
"""
from .config import (
    SERVER_CATALOGUE,
    WEIGHT_PRESETS,
    FleetConfig,
    PowerModelConfig,
    QoSConfig,
    ScenarioConfig,
    ServerType,
    SolverConfig,
    WastageConfig,
    Weights,
)

__all__ = [
    "SERVER_CATALOGUE",
    "WEIGHT_PRESETS",
    "FleetConfig",
    "PowerModelConfig",
    "QoSConfig",
    "ScenarioConfig",
    "ServerType",
    "SolverConfig",
    "WastageConfig",
    "Weights",
]
__version__ = "0.1.0"
