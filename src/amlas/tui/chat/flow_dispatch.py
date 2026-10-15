"""Chat → capability dispatch — the chat surface over the M7 funnel (ADR-010; roadmap M7).

The thin path that lets the conversational chat surface reach the SAME one-capability
funnel (``control_plane.capability.invoke_capability``) the CLI / Python-API / TUI use.
No flow logic lives here (ADR-010 / ADR-013: surfaces are renderers): each helper builds a
params dict + reads the result, then renders a :class:`ChatResponse`. The chat engine wires
three slash commands to these helpers — ``/flows`` (discover), ``/flow`` (run/plan),
``/mention`` (``@``-context) — so the chat user reaches the registry-backed capabilities
identically to every other surface.

Pure-Python, import-cheap: the capability/runtime/mention modules are imported lazily inside
each helper so importing this module never drags the control plane into the chat hot path.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .models import ChatResponse


def _summarize_flow_result(obj: Any) -> list[str]:
    """Best-effort compact summary of a flow's result for chat display.

    Scalars render as ``key: value``; lists/dicts render as counts
    (e.g. ``proposed: 7 items``). Never raises — returns ``[]`` on anything odd,
    so the completion path always falls back cleanly.
    """
    try:
        if obj is None:
            return []
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            data = dataclasses.asdict(obj)
        elif isinstance(obj, dict):
            data = obj
        elif hasattr(obj, "__dict__"):
            data = dict(vars(obj))
        else:
            return []
        if not isinstance(data, dict) or not data:
            return []
        lines: list[str] = []
        for key, val in list(data.items())[:12]:
            if isinstance(val, (bool, int, float, str)):
                sval = str(val)
                if len(sval) > 80:
                    sval = sval[:77] + "..."
                lines.append(f"  {key}: {sval}")
            elif isinstance(val, list):
                lines.append(f"  {key}: {len(val)} items")
            elif isinstance(val, dict):
                lines.append(f"  {key}: {len(val)} fields")
        return lines
    except Exception:
        return []

__all__ = [
    "dispatch_flow",
    "dispatch_mention",
    "format_capabilities",
    "parse_param_pairs",
]


def _coerce(value: str) -> Any:
    """Coerce a ``key=value`` string value to bool/None/int/float where possible."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except (TypeError, ValueError):
            pass
    return value


def parse_param_pairs(args: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` chat-command tokens into a flow params dict.

    Non ``key=value`` tokens are ignored (the leading token is the flow id, handled by the
    caller). Values are coerced to bool/None/int/float where possible.
    """
    out: dict[str, Any] = {}
    for tok in args:
        if "=" in tok:
            key, _, val = tok.partition("=")
            out[key.strip()] = _coerce(val)
    return out


def format_capabilities() -> ChatResponse:
    """``/flows`` — list every registered capability from the shared registry.

    Renders the SAME ``list_capabilities`` view the CLI ``capability list`` + the TUI pane
    show, so discovery is identical across surfaces (the ADR-010 parity contract).
    """
    from ...control_plane.capability import list_capabilities

    caps = list_capabilities()
    lines = [f"🧩 **Capabilities** ({len(caps)} registered):", ""]
    for c in caps:
        gate = "🔒 gated" if c["gated"] else "📖 read "
        run = "" if c["runnable"] else " (no executor)"
        lines.append(f"  • `{c['capability_id']}` [{gate}|{c['milestone']}]{run}")
        lines.append(f"      {c['description'][:96]}")
    lines.append("")
    lines.append("💡 `/flow <id> [k=v ...]` to run (gated flows prompt for approval);")
    lines.append("   `/flow <id> --plan` to preview without running.")
    return ChatResponse(
        message="\n".join(lines),
        action="capabilities_listed",
        data={"capabilities": caps},
    )


def _format_flow_usage(flow_id: str, required: list[str], optional: list[str]) -> ChatResponse:
    """``/flow <id>`` with no params — show what the flow needs.

    Discoverability half of the ask: the required and optional params are printed with a
    copy-pasteable example, so the user never has to read the executor's signature.
    """
    req_line = " ".join(f"{p}=..." for p in required)
    lines = [f"🧩 **`{flow_id}`** — parameters:", ""]
    lines.append(f"  **required:** {', '.join(f'`{p}`' for p in required) or '(none)'}")
    if optional:
        shown = ", ".join(f"`{p}`" for p in optional[:12])
        more = f" (+{len(optional) - 12} more)" if len(optional) > 12 else ""
        lines.append(f"  **optional:** {shown}{more}")
    lines.append("")
    lines.append(f"💡 `/flow {flow_id} {req_line}`".rstrip())
    lines.append(f"   `/flow {flow_id} {req_line} --plan` to preview without running.".rstrip())
    return ChatResponse(
        message="\n".join(lines),
        action="capability_usage",
        data={"capability_id": flow_id, "required": required, "optional": optional},
    )


def dispatch_flow(flow_id: str, args: list[str]) -> ChatResponse:
    """``/flow <id> [k=v ...] [--plan]`` — run (or plan) a capability via the M7 funnel.

    Builds a params dict from ``k=v`` tokens and calls ``invoke_capability`` with
    ``surface="chat"`` — the SAME funnel every surface uses. A trailing ``--plan`` returns
    the read-only Plan-Mode preview (no execution). Gated (spend/irreversible) capabilities
    route through the approval gate exactly as on every other surface.

    Required params are validated against the executor's signature BEFORE the funnel is
    called, so a missing param renders a friendly message naming it instead of
    leaking the raw ``TypeError`` from the call site. ``/flow <id>`` with no params prints
    the flow's usage.
    """
    from ...control_plane.capability import invoke_capability, required_params_for

    plan_only = "--plan" in args
    params = parse_param_pairs([a for a in args if a != "--plan"])

    # Signature-derived required params. An unresolvable capability yields
    # ([], []) and falls through to the funnel, which owns the "unknown capability" error.
    required, optional = required_params_for(flow_id)

    # ``/flow <id>`` with no params at all, for a flow that needs some → show usage.
    if required and not params:
        return _format_flow_usage(flow_id, required, optional)

    missing = [p for p in required if p not in params]
    if missing:
        req_line = " ".join(f"{p}=..." for p in required)
        plural = "s" if len(missing) > 1 else ""
        named = ", ".join(f"`{p}`" for p in missing)
        return ChatResponse(
            message=(
                f"⚠️ **`{flow_id}`** needs missing param{plural}: {named}\n\n"
                f"💡 Try: `/flow {flow_id} {req_line}`\n"
                f"   `/flow {flow_id}` (no params) lists every parameter."
            ),
            action="capability_missing_params",
            data={"capability_id": flow_id, "missing": missing,
                  "required": required, "optional": optional, "provided": sorted(params)},
        )

    result = invoke_capability(
        capability_id=flow_id, params=params, surface="chat", plan_only=plan_only
    )

    if result.status == "error":
        return ChatResponse(
            message=f"❌ Capability `{flow_id}` error: {result.error}",
            action="capability_error",
            data=result.to_dict(),
        )
    if result.status == "planned":
        plan = result.plan or {}
        steps = plan.get("steps", []) or []
        step_lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps)) or "  (no steps)"
        gate = "GATED (approval required)" if result.gated else "cheap read (no gate)"
        return ChatResponse(
            message=(
                f"📋 **Plan: `{flow_id}`** — {gate}\n\n"
                f"{step_lines}\n\n"
                f"  Est. cost: {plan.get('cost_estimate', 'n/a')} | "
                f"spend: {plan.get('spend_incurring', False)} | "
                f"irreversible: {plan.get('irreversible', False)}\n\n"
                f"💡 `/flow {flow_id} ...` (without `--plan`) to execute."
            ),
            action="capability_planned",
            data=result.to_dict(),
        )
    if result.status == "denied":
        return ChatResponse(
            message=(
                f"🚫 **`{flow_id}` denied** by the approval gate (safe-by-default).\n"
                f"  Spend/irreversible capabilities require approval — re-run with an "
                f"approving policy, or `/flow {flow_id} --plan` to preview."
            ),
            action="capability_denied",
            data=result.to_dict(),
        )
    # completed — surface a short summary of what the flow produced
    gate = "gated→approved" if result.gated else "cheap read"
    summary = _summarize_flow_result(result.result)
    body = "\n".join(summary) if summary else "  Result available in the response data."
    return ChatResponse(
        message=f"✅ **`{flow_id}` complete** ({gate}).\n{body}",
        action="capability_complete",
        data=result.to_dict(),
    )


def dispatch_mention(ref: str) -> ChatResponse:
    """``/mention @<kind>:<ref>`` — resolve an ``@`` mention into injectable context (T-LAKE-07).

    Renders the resolved dataset/spec/run/artifact context the control plane would inject
    into the turn — the chat-surface entry to the same ``mention_context`` the API/CLI expose.
    """
    from ...control_plane.mention import mention_context

    ctx = mention_context(ref)
    if not ctx.get("resolved"):
        return ChatResponse(
            message=f"❓ Could not resolve mention `{ref}`: {ctx.get('error', 'not found')}",
            action="mention_unresolved",
            data=ctx,
        )
    return ChatResponse(
        message=(
            f"📎 **@-mention resolved** — {ctx.get('summary', ref)}\n"
            f"  kind: `{ctx.get('kind')}` | source: `{ctx.get('adapter')}`\n"
            f"  (context injected into the conversation)"
        ),
        action="mention_resolved",
        data=ctx,
    )
