"""DP-1 — ToolBindingBackend: the TUI's embedded, tool-bound chat agent.

Upgrades the TUI's natural-language brain from suggestion-only chat to **real
function-calling** by wrapping a Strands ``Agent`` built via the proven
``_build.build_agent`` factory with ``SAFE_LLM_TOOLS`` — the identical
pattern that powers ``pipeline.agentic_runner.build_pipeline_agent``
(see ``docs/audit/phase7-road-to-100-design.md`` §2).

Graceful degradation (design §2.3): compatibility validation may replace
``strands`` with an inert identity shim. There, ``is_tool_capable`` is ``False``, so
the engine skips this tier. It used to fall back to the raw-converse
``BedrockChatBackend``; issue #35 deleted that second brain, so the fallback is now
the engine's DETERMINISTIC tier-5 keyword responder — still no LLM under the shim.

Confirmation gate (safety default #3): read-only tools auto-execute; state-changing
tools surface a ``needs_confirmation`` prompt before running.

Phase 11 (DP-2) handoff: ``agent_tools.LONG_RUNNING_TOOLS`` enumerates the bound
tools that should become background jobs; for Phase 10 they execute synchronously.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ...config.settings import Settings
from ...control_plane.approvals import ApprovalPolicy
from .agent_tools import (
    LONG_RUNNING_TOOLS,
    READ_ONLY_TOOLS,
    TUI_SYSTEM_PROMPT,
    get_safe_llm_tools,
    tool_display_name,
)
from .thought_trace import ThoughtEvent, ThoughtTrace, _make_trace_callback

logger = logging.getLogger(__name__)

# Execution status values for a single tool invocation.
ExecStatus = str  # "executed" | "needs_confirmation" | "error" | "job_submitted"


@dataclass
class ToolExecution:
    """The outcome of dispatching one tool_use through the confirmation gate."""

    tool_name: str
    args: dict[str, Any]
    status: ExecStatus
    result: Any = None
    result_summary: str = ""


@dataclass
class AgentTurn:
    """A complete tool-bound agent turn: final assistant text + tool trace.

    ``ChatEngine`` renders this back into the chat pane (a compact "🔧 used: …"
    affordance line above the final text) so users see which REAL tools fired.
    """

    final_text: str
    executions: list[ToolExecution] = field(default_factory=list)


def _resolve_callable(tool: Any) -> Callable[..., Any]:
    """Return the underlying plain function for a (possibly decorated) tool.

    Under the confcompatibility shim ``@tool`` is the identity, so ``tool`` is already the
    raw function. With real Strands, the ``DecoratedFunctionTool`` exposes the
    original via one of these attributes — we prefer it so direct invocation in
    ``execute_tool_use`` calls the real pure-Python implementation.
    """
    for attr in ("_tool_func", "func", "__wrapped__", "_func", "original"):
        candidate = getattr(tool, attr, None)
        if callable(candidate):
            resolved: Callable[..., Any] = candidate
            return resolved
    fallback: Callable[..., Any] = tool
    return fallback


def _summarize(result: Any) -> str:
    """One-line, chat-friendly summary of a tool result for the trace affordance."""
    if isinstance(result, dict):
        parts: list[str] = []
        for k, v in result.items():
            if isinstance(v, (list, dict)):
                continue
            parts.append(f"{k}={v}")
            if len(parts) >= 4:
                break
        return ", ".join(parts)[:200] or "ok"
    return str(result)[:200]


def render_agent_turn(turn: AgentTurn) -> str:
    """Render an ``AgentTurn`` into chat markdown: tool trace + final text.

    - No tool calls → the plain final text (indistinguishable from normal chat).
    - With tool calls → a "🔧 used: …" line for executed tools, any pending
      confirmation prompts, any errors, then the final assistant text.
    """
    if not turn.executions:
        return turn.final_text

    executed = [e for e in turn.executions if e.status == "executed"]
    pending = [e for e in turn.executions if e.status == "needs_confirmation"]
    errored = [e for e in turn.executions if e.status == "error"]

    lines: list[str] = []
    if executed:
        lines.append("🔧 used: " + ", ".join(e.tool_name for e in executed))
    for e in pending:
        lines.append(f"⏸️ {e.result_summary}")
    for e in errored:
        lines.append(f"⚠️ {e.tool_name}: {e.result_summary}")
    if turn.final_text:
        lines.append("")
        lines.append(turn.final_text)
    return "\n".join(lines)


class ToolBindingBackend:
    """Builds/owns the Strands tool-bound agent and the confirmation-gated dispatch.

    The deterministic, shim-safe seam is :meth:`execute_tool_use` — it invokes the
    SAME real ``@tool`` callable a live Strands agent would, so offline validation prove the
    NL→tool→real-effect wiring without a live LLM. :meth:`chat_with_tools` is the
    full LLM path, exercised live (``agents`` extra + creds) or via an injected
    model/agent in tests.
    """

    def __init__(
        self,
        settings: Settings,
        model: Any | None = None,
        job_manager: Any | None = None,
        *,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.approval_policy = approval_policy
        self._model = model  # optional pre-built model; callers may inject a real BedrockModel
        # Name → real callable registry (the tools the LLM may invoke). Built lazily
        # because the tool modules import ``strands`` (optional dep) — constructing
        # the backend (e.g. at TUI launch without ``strands``) must NOT require it.
        self._tool_registry_cache: dict[str, Callable[..., Any]] | None = None
        # Per-instance read-only set (copy so callers can extend it deterministically).
        self._readonly_names: set[str] = set(READ_ONLY_TOOLS)
        # DP-2 (Phase 11): when set, a CONFIRMED long-running tool enqueues a
        # background job instead of blocking the agent turn (design §2.2 / §3). When
        # None (the default / tests), confirmed long ops execute inline as before.
        self.job_manager = job_manager
        # Feature 2 (ephemeral thought ticker): the per-turn ``ThoughtTrace`` for
        # the coarse ``execute_tool_use`` fallback. ``chat_with_tools`` sets this at
        # turn start (and clears it at turn end) so the deterministic tool seam can
        # emit tool_start/tool_end ticker content even when the granular Strands
        # streaming callback shape is unavailable (design §3.2b). ``None`` (the
        # default) → NO emission, so ``execute_tool_use`` is byte-identical when no
        # ticker is wired (every existing caller / headless path).
        self._thought_trace: ThoughtTrace | None = None

    @property
    def _tool_registry(self) -> dict[str, Callable[..., Any]]:
        """Lazily build (and cache) the name→callable registry of bound tools.

        Returns an empty registry if the tool modules can't be imported (no
        ``strands``), so the backend stays constructible/usable for the plain-chat
        fallback path without crashing the TUI at launch.
        """
        if self._tool_registry_cache is None:
            try:
                self._tool_registry_cache = {
                    tool_display_name(t): _resolve_callable(t)
                    for t in get_safe_llm_tools()
                }
            except Exception as exc:
                logger.warning("Tool registry unavailable (no strands?): %s", exc)
                self._tool_registry_cache = {}
        return self._tool_registry_cache

    # ─── Capability detection (graceful degradation) ──────────────────────────

    @property
    def is_tool_capable(self) -> bool:
        """True only when a REAL Strands package is importable (not the shim).

        The hermetic compatibility shim registers ``strands`` as a bare ``types.ModuleType``
        with no ``__file__``; an installed package always has one. We use that as the
        cheap, reliable marker that distinguishes the inert shim from real Strands.
        """
        try:
            import strands
        except Exception:
            return False
        return getattr(strands, "__file__", None) is not None

    def build_agent(
        self,
        thought_trace: ThoughtTrace | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Construct the tool-bound Strands ``Agent`` via the proven factory.

        Raises ``RuntimeError`` when not tool-capable (the shim path) so callers
        never accidentally invoke the inert shim Agent.

        Feature 2 (additive): when ``thought_trace`` is provided, a
        ``callback_handler`` mapping Strands streaming events → ``ThoughtEvent``s is
        attached to the built Agent so the ephemeral ticker sees live steps. The
        canonical ``_build.build_agent`` factory has a fixed signature (it forwards
        only the kwargs the installed ``Agent.__init__`` accepts,
        [`_build.py:27-60`](../../_build.py:27)), so we attach the handler
        DEFENSIVELY on the returned agent — never breaking the build if the installed
        Strands ``Agent`` doesn't expose ``callback_handler``. Default ``None`` keeps
        the build byte-identical for every existing caller/test.

        Multi-turn history (additive): when ``messages`` is provided it is the PRIOR
        conversation history (Strands ``Message`` dicts) replayed into the Agent's
        ``.messages`` at construction, so an affirmative like "yes to both" resolves
        against the assistant's previous question. ``None`` (the default) builds a
        fresh single-turn Agent exactly as before.
        """
        if not self.is_tool_capable:  # nosemgrep
            raise RuntimeError(
                "ToolBindingBackend.build_agent called without real Strands "
                "(install the 'agents' extra); use the plain-chat fallback instead."
            )

        from ..._build import build_agent
        from ...config.chat_model import chat_max_tokens, chat_model_id

        # TUI NL chat runs on the chat tier (Sonnet 4.6 by default, overridable
        # via AMLAS_CHAT_MODEL in .env.tui) — NOT the swarm's Opus 4.8 planner
        # tier. The explicit ``model_id`` wins over ``role`` in build_agent,
        # so the conversational TUI agent stays on Sonnet while the leads/sub-agents
        # keep Opus 4.8 for ML work (user directive 2026-06-04).
        #
        # Issue #35 criterion 4.1: the model id is resolved by
        # [`config.chat_model.chat_model_id`](../../config/chat_model.py:53) — the ONE
        # definition — instead of this call site re-reading ``AMLAS_CHAT_MODEL``
        # itself. It used to be resolved here, in the raw ``bedrock_backend``, and a
        # third time in ``engine._chat_model_label``, so the activity panel could
        # report a different tier than the agent was actually built on.
        #
        # ``max_tokens`` comes from the same source
        # ([`chat_max_tokens`](../../config/chat_model.py:65)), which defaults to
        # ``cfg.bedrock.max_tokens`` (64000) — the value this path already got from
        # the factory — so the consolidation does NOT re-cap chat to the deleted raw
        # path's hardcoded 2048 (criterion 4.2, design Property 7). Passing it
        # explicitly is what makes ``AMLAS_CHAT_MAX_TOKENS`` able to cap chat alone,
        # leaving the swarm tier on ``cfg.bedrock.max_tokens``.
        agent = build_agent(
            name="tui_chat_agent",
            system_prompt=TUI_SYSTEM_PROMPT,
            tools=get_safe_llm_tools(),
            model=self._model,
            role="planner",
            model_id=chat_model_id(),
            max_tokens=chat_max_tokens(),
            description="Embedded TUI chat agent with real ML tool-calling.",
            messages=messages,
        )
        if thought_trace is not None:
            # Best-effort: set the streaming callback if the installed Agent supports
            # it. Non-fatal on version drift (the coarse execute_tool_use fallback
            # still guarantees tool-level ticker content).
            try:
                agent.callback_handler = _make_trace_callback(thought_trace)
            except Exception:  # pragma: no cover — version-drift defensive
                logger.debug("could not attach thought-trace callback_handler")
        return agent

    # ─── Confirmation gate + real tool dispatch (shim-safe seam) ──────────────

    def confirmation_prompt(self, tool_name: str, args: dict[str, Any]) -> str:
        """Human-readable y/n confirmation line for a state-changing tool."""
        arg_hint = ""
        if args:
            shown = ", ".join(
                f"{k}={v}" for k, v in list(args.items())[:3]
                if not isinstance(v, (list, dict))
            )
            if shown:
                arg_hint = f" ({shown})"
        return f"Run `{tool_name}`{arg_hint}? [y/n]"

    def _emit_trace(self, kind: str, text: str, tool: str | None = None) -> None:
        """Coarse ticker emission for the ephemeral thought trace (Feature 2).

        A no-op unless a per-turn ``ThoughtTrace`` is active (set by
        ``chat_with_tools`` for the current turn). This is the offline-safe fallback
        that guarantees tool-level ticker content even when the granular Strands
        streaming callback shape is unavailable (design §3.2b). Never raises — the
        ``ThoughtTrace.emit`` sink discipline already swallows sink errors.
        """
        trace = self._thought_trace
        if trace is None:
            return
        try:
            trace.emit(ThoughtEvent(kind=kind, text=text, tool=tool))
        except Exception:  # pragma: no cover — defensive; emit never raises anyway  # nosec B110
            pass

    def execute_tool_use(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        confirmed: bool = False,
    ) -> ToolExecution:
        """Dispatch one tool_use: gate by class FIRST, then invoke the REAL tool.

        Order matters (and is the correct design): the confirmation gate is a
        pure NAME-classification decision, so it fires BEFORE callable resolution.
        A state-changing tool must surface ``needs_confirmation`` even before we
        bother resolving its callable — and this keeps the gate robust when a test
        patches the underlying tool symbol (which would change the registry key).

        - State-changing tool without ``confirmed`` → ``needs_confirmation`` (the
          real tool is NOT run, callable not even resolved).
        - Read-only tool, or confirmed state-changing tool → resolve + execute.
        - Unknown/unresolvable tool (once past the gate) → ``error`` (never raises).

        Feature 2 (additive, no behavior change): when a per-turn ``ThoughtTrace`` is
        active, coarse ``tool_start``/``tool_end`` ticker events are emitted around a
        successful execution (design §3.2b). With no trace (the default / every
        existing caller), ``_emit_trace`` is a no-op and this method is byte-identical.
        """
        is_read_only = tool_name in self._readonly_names
        if not is_read_only and not confirmed:
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="needs_confirmation",
                result=None,
                result_summary=self.confirmation_prompt(tool_name, args),
            )

        fn = self._tool_registry.get(tool_name)
        if fn is None:
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="error",
                result=None,
                result_summary=f"Unknown tool: {tool_name}",
            )

        # Coarse ticker: announce the tool is starting (the gate has passed).
        self._emit_trace("tool_start", f"running {tool_name}", tool=tool_name)

        # DP-2 (Phase 11): a CONFIRMED long-running tool is handed to the
        # JobManager so the agent turn completes fast and the work runs in the
        # background (design §2.2/§3). The confirmation gate has ALREADY passed
        # above (LONG_RUNNING_TOOLS ⊆ STATE_CHANGING_TOOLS), so backgrounding
        # layers cleanly on top. Only when a JobManager is wired (the app); when
        # None (validation coverage) the tool executes inline below, unchanged.
        if self.job_manager is not None and tool_name in LONG_RUNNING_TOOLS:
            job = self.job_manager.submit(
                tool_name,
                lambda progress_cb, cancel_token, _fn=fn, _args=args: _fn(**_args),
                args=dict(args),
            )
            self._emit_trace("tool_end", f"{tool_name} submitted", tool=tool_name)
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="job_submitted",
                result={"job_id": job.id},
                result_summary=f"{tool_name} submitted as background job {job.id}",
            )

        try:
            result = fn(**args)
        except Exception as exc:
            logger.warning("Tool %s raised: %s", tool_name, exc)
            self._emit_trace("tool_end", f"{tool_name} failed", tool=tool_name)
            return ToolExecution(
                tool_name=tool_name,
                args=args,
                status="error",
                result=None,
                result_summary=f"{tool_name} failed: {exc}",
            )

        self._emit_trace("tool_end", f"{tool_name} done", tool=tool_name)
        return ToolExecution(
            tool_name=tool_name,
            args=args,
            status="executed",
            result=result,
            result_summary=_summarize(result),
        )

    # ─── Full LLM tool-calling turn (live path / injected model in tests) ─────

    def chat_with_tools(
        self,
        message: str,
        context: str = "",
        *,
        thought_trace: ThoughtTrace | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentTurn | None:
        """Run a full tool-bound agent turn; return an ``AgentTurn`` or ``None``.

        Returns ``None`` when not tool-capable, which sends ``ChatEngine`` to its
        DETERMINISTIC tier-5 keyword fallback — the raw-converse brain that used to
        sit between them was deleted by issue #35 (6.1/6.2), so ``None`` now means
        "no LLM answers this turn", not "a different LLM answers it". On the live
        path, builds the agent, invokes it, and parses the final assistant text
        (+ any tool trace) into an ``AgentTurn``.

        Feature 2 (additive): when ``thought_trace`` is provided, it is (a) attached
        as the Strands ``callback_handler`` for granular streaming events AND (b) set
        as the per-turn coarse trace so ``execute_tool_use`` emits tool-level ticker
        content offline. The coarse trace is ALWAYS cleared in a ``finally`` so it
        never leaks into a later turn. Default ``None`` keeps the turn byte-identical
        for every existing caller/test.

        Multi-turn history (additive, the stateful-chat fix): ``history`` is the
        PRIOR conversation (Strands ``Message`` dicts, user+assistant, already
        bounded by the engine) replayed into the freshly-built Agent's ``.messages``
        so the model sees the earlier turns — e.g. an affirmative "yes to both"
        resolves against the two questions the assistant asked last turn. The CURRENT
        ``message`` is still the prompt (never duplicated into ``history``). ``None``
        (the default) preserves the exact single-turn build for every existing caller.
        """
        if not self.is_tool_capable:  # nosemgrep
            return None

        prompt = message if not context else f"{message}\n\n## Context\n{context}"
        # Activate the coarse fallback trace for THIS turn only (cleared in finally).
        self._thought_trace = thought_trace
        try:
            try:
                agent = self.build_agent(thought_trace=thought_trace, messages=history)
                if self.approval_policy is not None:
                    if not isinstance(self.approval_policy, ApprovalPolicy):
                        raise TypeError("approval_policy must be a caller-owned ApprovalPolicy")
                    policy = replace(
                        self.approval_policy, allowlist=list(self.approval_policy.allowlist),
                    )
                    result = agent(
                        prompt, invocation_state={"_amlas_approval_policy": policy},
                    )
                else:
                    result = agent(prompt)
            except Exception as exc:
                logger.error("Tool-bound agent error: %s", exc)
                # Surface an expired-SSO / missing-creds failure as a clear, actionable
                # re-auth message (the shared CREDS-01 diagnostic) instead of a cryptic
                # botocore trace; otherwise keep the raw tool-agent error.
                from ...config.bedrock_creds import diagnose_aws_error

                creds_msg = diagnose_aws_error(exc, self.settings.aws.effective_profile)
                if creds_msg is not None:
                    return AgentTurn(final_text=creds_msg, executions=[])
                return AgentTurn(final_text=f"⚠️ Tool-agent error: {exc}", executions=[])

            return AgentTurn(
                final_text=self._extract_final_text(result),
                executions=self._extract_executions(result),
            )
        finally:
            # The per-turn trace must never bleed into a subsequent turn.
            self._thought_trace = None

    @staticmethod
    def _extract_final_text(result: Any) -> str:
        """Pull the assistant text out of a Strands AgentResult (or stringify)."""
        message = getattr(result, "message", None)
        if isinstance(message, dict):
            content = message.get("content", [])
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and "text" in block
            ]
            if texts:
                return "\n".join(texts)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, list):
                texts = [
                    getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                    for b in content
                ]
                texts = [t for t in texts if t]
                if texts:
                    return "\n".join(texts)
        return str(result)

    def _extract_executions(self, result: Any) -> list[ToolExecution]:
        """Best-effort extraction of executed tool calls from an AgentResult.

        Two complementary sources are merged (deduped by tool name, first-seen
        order preserved):

        1. ``EventLoopMetrics.tool_metrics`` — the AUTHORITATIVE cross-cycle record
           of every tool the agent invoked during the turn. The final assistant
           ``message`` only carries ``toolUse`` blocks for the LAST cycle, so a
           multi-cycle tool loop (e.g. ``list_directory`` in cycle 1, then the text
           answer in cycle 2) would otherwise leave the trace EMPTY even though a
           real tool fired — exactly the filesystem user-story turn (design doc 06).
           ``metrics.tool_metrics`` records ALL of them, so it is the reliable signal
           the engine surfaces as ``data["tools_used"]`` and the "🔧 used: …" line.
        2. The final ``message`` ``toolUse`` blocks — kept for the (common) single-
           cycle case and to capture the concrete call ``input`` args.

        Kept defensive — never raises; an unparseable result yields an empty trace
        (the final text still renders).
        """
        executions: list[ToolExecution] = []
        seen: set[str] = set()

        # 1. Authoritative cross-cycle record: EventLoopMetrics.tool_metrics.
        metrics = getattr(result, "metrics", None)
        tool_metrics = getattr(metrics, "tool_metrics", None)
        if isinstance(tool_metrics, dict):
            for tm in tool_metrics.values():
                name, args = self._tool_use_name_args(getattr(tm, "tool", None))
                if not name or name in seen:
                    continue
                seen.add(name)
                executions.append(
                    ToolExecution(
                        tool_name=name,
                        args=args,
                        status="executed",
                        result=None,
                        result_summary=name,
                    )
                )

        # 2. Final-message toolUse blocks (single-cycle case / concrete args).
        message = getattr(result, "message", None)
        content = message.get("content", []) if isinstance(message, dict) else []
        for block in content:
            if isinstance(block, dict) and "toolUse" in block:
                tu = block["toolUse"]
                name = tu.get("name", "tool")
                if not name or name in seen:
                    continue
                seen.add(name)
                executions.append(
                    ToolExecution(
                        tool_name=name,
                        args=tu.get("input", {}) or {},
                        status="executed",
                        result=None,
                        result_summary=name,
                    )
                )
        return executions

    @staticmethod
    def _tool_use_name_args(tool_use: Any) -> tuple[str, dict[str, Any]]:
        """Extract ``(name, input-args)`` from a Strands ``ToolUse`` (dict or object).

        ``ToolUse`` is a TypedDict (``{"toolUseId","name","input"}``) in practice,
        but attribute-style access is accepted too so this stays robust across
        Strands versions. Returns ``("", {})`` when nothing usable is present.
        """
        if tool_use is None:
            return "", {}
        if isinstance(tool_use, dict):
            name = tool_use.get("name") or ""
            args = tool_use.get("input") or {}
        else:
            name = getattr(tool_use, "name", "") or ""
            args = getattr(tool_use, "input", {}) or {}
        if not isinstance(args, dict):
            args = {}
        return name, args
