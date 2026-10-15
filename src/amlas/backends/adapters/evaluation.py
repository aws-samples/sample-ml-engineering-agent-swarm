"""EvaluationBackend adapter — the deterministic Arbiter (ADR-003).

``DeterministicArbiter`` wraps the EXISTING ``arbiter/`` grader VERBATIM. The
``grade`` decision is **never delegated** to an LLM or a managed service for
predictive tasks — it is core IP and the deterministic accept/reject gate. This
adapter is BOTH the default and the local adapter (it IS deterministic/local by
design); it returns the rich ``arbiter.schema.ArbiterReport`` unchanged (wrap,
don't reshape).

The ``BedrockJudgeBackend`` alt adapter (generative groundedness/hallucination
ONLY) is a P2 add (doc 06 Part 3 row 13) — NOT implemented at M0.
"""

from __future__ import annotations

from typing import Any

from ..ports import EvaluationBackend
from ..registry import register_adapter


class DeterministicArbiter:
    """EvaluationBackend default+local — delegates to EXISTING ``arbiter.grade``.

    The call signature mirrors the existing ``arbiter/arbiter.py`` ``grade(...)``
    exactly, so a tool calling ``EvaluationBackend.grade(spec=…, y_true=…,
    y_pred=…)`` gets the IDENTICAL ``ArbiterReport`` it would get calling the
    Arbiter directly (the behavior-preservation test asserts equality).
    """

    def grade(
        self,
        *,
        spec: Any,
        y_true: Any,
        y_pred: Any,
        baseline_score: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Score ``y_pred`` vs ``y_true`` via the EXISTING deterministic grader.

        Forwards ``baseline_slices`` / ``phase_log`` / ``llm_total_tokens`` through
        ``**kwargs`` so the full Arbiter contract is preserved unchanged.
        """
        from ...arbiter.arbiter import grade as _grade

        return _grade(
            spec=spec,
            y_true=y_true,
            y_pred=y_pred,
            baseline_score=baseline_score,
            baseline_slices=kwargs.get("baseline_slices"),
            phase_log=kwargs.get("phase_log"),
            llm_total_tokens=kwargs.get("llm_total_tokens", 0),
        )


# Both default and local resolve to the same deterministic grader (ADR-003): the
# decision is identical offline and "in cloud" — there is no managed alternative.
register_adapter(EvaluationBackend, "arbiter", lambda: DeterministicArbiter(), is_default=True)
register_adapter(EvaluationBackend, "local", lambda: DeterministicArbiter(), is_local=True)
