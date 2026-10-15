"""Hooks-as-guardrails layer — the deterministic ``PreToolUse``/``PostToolUse`` seam.

The ML analog of Claude Code's lifecycle hooks (doc 05 §8.9; ADR-015). A hook
*always* fires at its event and can return a structured **deny** the model cannot
override — turning probabilistic agent behavior into deterministic guarantees.

M0 seeds the mechanism with ONE concrete guardrail — the **bronze-write deny**
(ADR-009): the lake already rejects bronze writes inside the tool, but the
architecture routes that rule through a declarative ``PreToolUse`` hook so the
*mechanism* exists and later milestones (M2) can add cost/leakage/quota guardrails
as more hooks without touching adapters. Keep it minimal — the full hook set lands
in M2 (doc 05 §8.9 table).

Design:
  * A ``Hook`` is ``(event, matcher, action)``: the ``matcher`` decides whether the
    hook applies to a ``HookContext`` (tool name + action + args), and ``action``
    returns a :class:`~amlas.backends.types.Verdict`.
  * The :class:`GuardrailHooks` registry runs the matching hooks for an event and
    returns the FIRST deny (fail-closed) or an allow if none deny.
  * Pure-Python, no boto3 / strands — the guardrail decisions are deterministic and
    centrally auditable (ADR-015), exactly like the Arbiter's determinism.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .types import Verdict

__all__ = [
    "GuardrailHooks",
    "Hook",
    "HookContext",
    "HookEvent",
    "bronze_write_deny_hook",
    "default_guardrail_hooks",
]


class HookEvent(StrEnum):
    """The two lifecycle events a guardrail may bind to (doc 05 §8.9)."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"


@dataclass(frozen=True)
class HookContext:
    """The context a hook matcher/action inspects.

    Args:
        tool: The tool / port-method name about to run (or that just ran).
        action: An optional finer action label (e.g. ``"write"``).
        args: The keyword args of the call (the hook reads e.g. ``tier``/``dest_uri``).
        result: The tool result (PostToolUse only).
    """

    tool: str
    action: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None


# A matcher decides whether a hook applies; an action returns the Verdict.
Matcher = Callable[[HookContext], bool]
Action = Callable[[HookContext], Verdict]


@dataclass(frozen=True)
class Hook:
    """A single declarative guardrail: ``event`` + ``matcher`` + ``action``."""

    name: str
    event: HookEvent
    matcher: Matcher
    action: Action


class GuardrailHooks:
    """Registry that fires matching hooks for an event and returns the verdict.

    Fail-closed: the FIRST hook that denies wins (an expensive/irreversible action
    is blocked the moment any guardrail objects). If no matching hook denies, the
    result is an allow. The model cannot override a deny (ADR-015).
    """

    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self._hooks: list[Hook] = list(hooks or [])

    def register(self, hook: Hook) -> Hook:
        """Register a hook (later milestones add cost/leakage/quota hooks here)."""
        self._hooks.append(hook)
        return hook

    def hooks_for(self, event: HookEvent) -> list[Hook]:
        """Return the registered hooks bound to ``event`` (registration order)."""
        return [h for h in self._hooks if h.event == event]

    def run(self, event: HookEvent, ctx: HookContext) -> Verdict:
        """Run every matching hook for ``event``; return the first deny, else allow.

        Returns an allowing :class:`Verdict` when no matching hook denies. The
        ``detail`` of the returned verdict names which hook fired (for the audit
        trail / trace span, doc 05 §8.1).
        """
        for hook in self._hooks:
            if hook.event != event:
                continue
            try:
                if not hook.matcher(ctx):
                    continue
            except Exception:  # nosec B112
                continue
            verdict = hook.action(ctx)
            if not verdict.allowed:
                # First deny wins (fail-closed); stamp which hook denied.
                return Verdict(
                    allowed=False,
                    reason=verdict.reason,
                    detail={**verdict.detail, "hook": hook.name, "event": event.value},
                )
        return Verdict(allowed=True, reason="no guardrail denied", detail={"event": event.value})

    def check_pre(self, ctx: HookContext) -> Verdict:
        """Convenience: run the ``PreToolUse`` guardrails for ``ctx``."""
        return self.run(HookEvent.PRE_TOOL_USE, ctx)

    def check_post(self, ctx: HookContext) -> Verdict:
        """Convenience: run the ``PostToolUse`` guardrails for ``ctx``."""
        return self.run(HookEvent.POST_TOOL_USE, ctx)


# ──────────────────────────────────────────────────────────────────────────────
# The first concrete guardrail — bronze-write deny (ADR-009 → ADR-015)
# ──────────────────────────────────────────────────────────────────────────────
_BRONZE_TIERS = {"bronze", "raw"}


def _is_lake_write(ctx: HookContext) -> bool:
    """Match a lake write-dataset call (by tool name or explicit action)."""
    tool = (ctx.tool or "").lower()
    return "write_lake_dataset" in tool or ctx.action == "write_lake_dataset" or ctx.action == "lake_write"


def _bronze_tier_of(ctx: HookContext) -> bool:
    """Detect a bronze/raw target from the call args (tier arg OR dest-uri segment).

    Mirrors the tool's own derivation (``tools/lake`` GOVERNANCE FIX): the
    destination URI is authoritative when it encodes a tier; the explicit ``tier``
    arg is the fallback. The hook denies if EITHER names bronze/raw.
    """
    tier = str(ctx.args.get("tier", "")).strip().lower()
    if tier in _BRONZE_TIERS:
        return True
    dest = str(ctx.args.get("dest_uri", "")).strip().lower()
    if dest:
        # a /bronze/ or /raw/ path segment, or a bronze:///raw:// scheme
        segments = {seg for seg in dest.replace("://", "/").split("/") if seg}
        if segments & _BRONZE_TIERS:
            return True
    return False


def bronze_write_deny_hook() -> Hook:
    """The bronze-write rejection as a declarative ``PreToolUse`` guardrail.

    Restates ADR-009 (the medallion bronze tier is read-only) as a deterministic
    hook so the guardrail MECHANISM exists at M0. The lake tool still enforces the
    rule internally (defense in depth) — this hook makes the deny declarative,
    composable, and centrally auditable (ADR-015), and is the seam M2's
    cost/leakage/quota guardrails plug into.
    """

    def _deny(ctx: HookContext) -> Verdict:
        tier = str(ctx.args.get("tier", "") or ctx.args.get("dest_uri", "")).lower()
        return Verdict(
            allowed=False,
            reason="bronze/raw is read-only (medallion write-back targets silver or gold)",
            detail={"rule": "ADR-009", "tier": tier},
        )

    return Hook(
        name="bronze_write_deny",
        event=HookEvent.PRE_TOOL_USE,
        matcher=lambda ctx: _is_lake_write(ctx) and _bronze_tier_of(ctx),
        action=_deny,
    )


def default_guardrail_hooks() -> GuardrailHooks:
    """The M0 default hook set — just the bronze-write deny (doc 05 §8.9).

    Later milestones extend this: M2 adds cost/leakage/quota ``PreToolUse`` +
    post-impute distribution-validation ``PostToolUse`` hooks; M1/M4 wire the
    approval gate (T-CTL-03) as a ``PreToolUse`` guardrail here.
    """
    return GuardrailHooks([bronze_write_deny_hook()])
