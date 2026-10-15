"""Task specification — parses task_spec.md + acceptance.yaml into pydantic models."""

from .loader import SpecLoadError, load_spec_from_path, load_spec_from_text
from .schema import (
    AcceptanceCriteria,
    Baseline,
    ComputeBudget,
    Constraints,
    SpecModel,
    Splits,
    TaskType,
)

__all__ = [
    "AcceptanceCriteria",
    "Baseline",
    "ComputeBudget",
    "Constraints",
    "SpecLoadError",
    "SpecModel",
    "Splits",
    "TaskType",
    "load_spec_from_path",
    "load_spec_from_text",
]
