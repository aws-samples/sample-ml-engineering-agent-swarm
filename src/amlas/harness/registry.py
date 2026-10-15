"""Which Leads are harness-hosted — the single enumeration (T-HARN-02.4 / T-HARN-15.3).

Phase 1 hosts the seven Leads and NOT the Orchestrator. That exclusion is load-bearing, not
stylistic: the Orchestrator owns a deterministic multi-iteration retreat loop, the
non-delegated Arbiter grade (ADR-0003), and the streaming-invariants contract (heartbeats +
a hard terminal event) that no managed loop exposes. It stays a code-defined agent on
AgentCore Runtime, unchanged.

This module deliberately delegates the membership list to
:mod:`amlas.leads.registry` rather than restating it. Two lists that "should" agree are a
drift bug waiting to happen; the parity checks and the provisioner read the same tuple.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..leads.registry import LEADS_BY_KEY, lead_keys

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands import Agent

    from ..leads.declarations import LeadDeclaration
    from ..subagents.declarations import SubagentDeclaration

#: The harness-hosted lead keys, in registry order. NEVER contains ``"orchestrator"``.
HARNESS_HOSTED_LEADS: tuple[str, ...] = lead_keys()

#: Sub-agents to PROVISION a dedicated harness for. **Deliberately empty.**
#:
#: A spec is derivable for any sub-agent (see :func:`harness_spec_for_subagent`) — ADR-0007
#: established the true harness leaves ARE the sub-agents, and T-HARN-01 Part A measured one
#: (`framing`) running on a harness successfully. But phase 1's architecture does not give each
#: sub-agent its own harness: a Lead on a harness declares its sub-agents as ``inline_function``,
#: so the harness pauses and OUR process runs the sub-agent exactly as it does today. That pause
#: is also what makes the approval gate work (T-HARN-18).
#:
#: Provisioning all 30 anyway would mean 30 extra harnesses each with managed Memory, and
#: harness deletion is asynchronous at >5 minutes apiece — real cost and real teardown time for
#: capability phase 1 does not use. So this stays empty until a specific sub-agent is chosen for
#: server-side hosting, and adding one is then a deliberate, reviewable line.
HARNESS_HOSTED_SUBAGENTS: tuple[str, ...] = ()

#: Identities explicitly barred from a harness, with the reason, so a future contributor
#: adding one hits a named refusal instead of a puzzling ``KeyError``.
NEVER_HARNESS_HOSTED: dict[str, str] = {
    "orchestrator": (
        "owns the retreat loop, the non-delegated Arbiter grade (ADR-0003) and the "
        "streaming-invariants contract; stays code-defined on AgentCore Runtime (T-HARN-15)"
    ),
}


def _require_harness_hosted(name: str) -> str:
    if name in NEVER_HARNESS_HOSTED:
        raise KeyError(f"{name!r} is never harness-hosted: {NEVER_HARNESS_HOSTED[name]}")
    if name not in HARNESS_HOSTED_LEADS:
        raise KeyError(
            f"unknown harness-hosted lead {name!r}; known: {list(HARNESS_HOSTED_LEADS)}"
        )
    return name


def harness_declaration(name: str) -> LeadDeclaration:
    """The :class:`~amlas.leads.declarations.LeadDeclaration` behind harness lead ``name``."""
    return LEADS_BY_KEY[_require_harness_hosted(name)].declaration


def local_peer_builder(name: str) -> Callable[..., Agent]:
    """The public ``build_*_lead`` factory for ``name`` — the equivalence oracle.

    Retained per T-HARN-04.3: ``_build.build_agent`` survives for the TUI regardless, so
    keeping the local construction path costs nothing and gives every harness spec a peer to
    be compared against.

    Resolved LATE, through the defining module's public attribute, rather than returning the
    function object :data:`~amlas.leads.registry.LEADS_BY_KEY` captured at import time. The
    two are normally the same object, and the indirection exists because they can diverge:
    ``LeadEntry(RESEARCH_LEAD, build_research_lead)`` binds the function at module-import, so
    a caller that rebinds the documented public factory —
    ``amlas.leads.research_lead.build_research_lead`` — changed a name the registry no longer
    consults. That made one function reachable through two bindings, with the registry's copy
    winning silently for every consumer that goes through here.

    That is not hypothetical: it is how ``app.py`` losing its direct ``build_*_lead`` imports
    (in favour of ``harness.leads.lead_for``) also cost the entrypoint's offline run isolation
    — ten patch sites across two suites kept patching the module attribute and quietly built
    real Bedrock-backed Leads instead. Late-binding collapses it back to one seam: the public
    attribute is the factory, for every caller.
    """
    builder = LEADS_BY_KEY[_require_harness_hosted(name)].builder
    module = sys.modules.get(getattr(builder, "__module__", ""))
    if module is None:
        return builder
    resolved: Callable[..., Agent] = getattr(module, getattr(builder, "__name__", ""), builder)
    return resolved


def subagent_harness_declaration(key: str) -> SubagentDeclaration:
    """The declaration behind sub-agent ``key``, for deriving its harness spec.

    Deliberately NOT gated on :data:`HARNESS_HOSTED_SUBAGENTS`: deriving a spec is free and
    offline, and being able to inspect any leaf's harness configuration is useful well before
    anything is provisioned. Provisioning is what the allowlist gates.
    """
    from ..subagents.registry import subagent_declaration

    return subagent_declaration(key)


def subagent_local_peer_builder(key: str) -> Callable[..., Agent]:
    """The public ``build_*`` factory for sub-agent ``key`` — its equivalence oracle."""
    from ..subagents.registry import subagent_entry

    return subagent_entry(key).builder


__all__ = [
    "HARNESS_HOSTED_LEADS",
    "HARNESS_HOSTED_SUBAGENTS",
    "NEVER_HARNESS_HOSTED",
    "harness_declaration",
    "local_peer_builder",
    "subagent_harness_declaration",
    "subagent_local_peer_builder",
]
