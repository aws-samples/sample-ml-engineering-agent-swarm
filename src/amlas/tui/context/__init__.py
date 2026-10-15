"""Phase 2 — the TUI shared-context package (the ``AppContext`` "context bus").

Re-exports the Textual-free data model (:mod:`.models`) and the App-owned pub/sub
manager (:mod:`.bus`) described in ``docs/tui-context-integration-design.md`` §E/§F.

The ``ContextBus`` is the single source of truth for the shared session state (active
spec/dataset/job/run/model/problem/steering/agent-activity) and an in-process
``subscribe``/``publish`` modeled EXACTLY on ``JobManager.on_event`` — a plain,
Textual-free object held as ``app.context_bus`` alongside ``app.job_manager``. It is
additive and null-safe: it mirrors relevant fields into ``engine.context`` so the
existing callers stay green.
"""

from __future__ import annotations

from .bus import (
    ContextBus,
    ContextEvent,
    get_active_bus,
    get_active_ledger,
    set_active_bus,
    set_active_ledger,
)
from .models import (
    ActiveDataset,
    ActiveJob,
    ActiveModel,
    ActiveProblem,
    ActiveRun,
    ActiveSpec,
    ActiveView,
    AgentActivity,
    AppContextState,
    ProcessActivity,
)
from .session_ledger import LedgerEntry, SessionLedger, SessionState
from .trace_reader import TraceTailReader, aws_services_for_tool

__all__ = [
    # models
    "ActiveDataset",
    "ActiveJob",
    "ActiveModel",
    "ActiveProblem",
    "ActiveRun",
    "ActiveSpec",
    # Plan §B2 — the requested dashboard view (chat-driven re-render)
    "ActiveView",
    "AgentActivity",
    "AppContextState",
    # bus
    "ContextBus",
    "ContextEvent",
    # Phase C2 — session-wide append-only ledger (timeline layered on the bus)
    "LedgerEntry",
    # Phase 5 §2d — local-process surface
    "ProcessActivity",
    "SessionLedger",
    "SessionState",
    # trace reader (Phase 4 — §E.8)
    "TraceTailReader",
    "aws_services_for_tool",
    "get_active_bus",
    "get_active_ledger",
    "set_active_bus",
    "set_active_ledger",
]
