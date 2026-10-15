"""Evaluation module — experiment tracking, leaderboard, and comparison.

Pure deterministic Python (no LLM). Provides:
  - ExperimentResult: canonical result schema
  - EvaluationStore: JSON-based persistence of experiment results
  - Leaderboard: ranking, comparison, and improvement tracking
  - Integration helpers: training_result_to_experiment, arbiter_report_to_experiment
"""

from .leaderboard import Leaderboard
from .store import (
    EvaluationStore,
    ExperimentResult,
    arbiter_report_to_experiment,
    training_result_to_experiment,
)

__all__ = [
    "EvaluationStore",
    "ExperimentResult",
    "Leaderboard",
    "arbiter_report_to_experiment",
    "training_result_to_experiment",
]
