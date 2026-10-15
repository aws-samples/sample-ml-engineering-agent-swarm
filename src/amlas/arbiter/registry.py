"""Task-type → Grader registry. Populated by importing grader modules."""

from __future__ import annotations

from .schema import Grader

REGISTRY: dict[str, Grader] = {}


def register(grader: Grader) -> Grader:
    """Register a grader. Duplicate task_type registrations raise ValueError."""
    if grader.task_type in REGISTRY:
        raise ValueError(
            f"Grader already registered for task_type={grader.task_type!r}. "
            f"Existing: {REGISTRY[grader.task_type]!r}"
        )
    REGISTRY[grader.task_type] = grader
    return grader


def get_grader(task_type: str) -> Grader:
    """Return the registered grader for a task_type; raises KeyError if missing."""
    if task_type not in REGISTRY:
        raise KeyError(
            f"No grader for task_type={task_type!r}. Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[task_type]


# Auto-register built-in graders by importing grader modules
from .graders import classification, detection, regression, segmentation  # noqa: E402,F401
