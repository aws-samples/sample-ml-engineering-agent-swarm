"""Experiment Runner package — Feature 3.

Provides:
  - ExperimentConfig: Configuration for a single experiment
  - ExperimentPlan: Collection of experiment configs
  - ExperimentRunner: Orchestrator for running experiments
  - DEFAULT_CONFIGS: Default hyperparameter grids
"""

from .config import ExperimentConfig, ExperimentPlan
from .hyperparams import DEFAULT_CONFIGS
from .runner import ExperimentRunner

__all__ = [
    "DEFAULT_CONFIGS",
    "ExperimentConfig",
    "ExperimentPlan",
    "ExperimentRunner",
]
