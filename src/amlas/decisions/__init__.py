"""Decision Support package — rules-based ML experiment recommendation engine.

Provides:
  - Decision: Structured recommendation dataclass
  - DecisionEngine: Analyzes experiment results and produces recommendations
  - Rules: Deterministic checks (target met, overfitting, diversity, etc.)
  - Model Card: Standard ML model documentation generator
"""

from .engine import Decision, DecisionEngine
from .model_card import generate_model_card
from .rules import (
    check_all_poor,
    check_diminishing_returns,
    check_model_diversity,
    check_models_close,
    check_overfitting,
    check_target_met,
    suggest_scaling_strategy,
)

__all__ = [
    "Decision",
    "DecisionEngine",
    "check_all_poor",
    "check_diminishing_returns",
    "check_model_diversity",
    "check_models_close",
    "check_overfitting",
    "check_target_met",
    "generate_model_card",
    "suggest_scaling_strategy",
]
