"""Benchmark Arbiter — deterministic Python grader. NO LLM in this path.

See PLAN.md §7. Shape derived from `openai/mle-bench` `mlebench/grade.py`:
    - `Grader` dataclass binds task_type → grade_fn + direction + optional slice_fn
    - `InvalidSubmissionError` for submission-side errors (fixable by agent, reinjected)
    - Regular AssertionError / KeyError for answer-side bugs (programmer errors)

Every grader must raise `InvalidSubmissionError` on shape/value issues the agent can fix
by re-trying with different output; anything else is a code bug and should propagate.
"""

from .arbiter import grade
from .registry import REGISTRY, get_grader, register
from .schema import ArbiterReport, Grader, InvalidSubmissionError

__all__ = [
    "REGISTRY",
    "ArbiterReport",
    "Grader",
    "InvalidSubmissionError",
    "get_grader",
    "grade",
    "register",
]
