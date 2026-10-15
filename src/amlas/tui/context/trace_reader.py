"""Phase 4 — ``TraceTailReader``: fold agent trace spans into ``AgentActivity`` (§E.8).

This is the ``src/amlas/tui/context/trace_reader.py`` module named in the design
(``docs/tui-context-integration-design.md`` §E.8 / §F Phase 4). It is **Textual-free**
(directly verifiable headless, exactly like :mod:`.models` / :mod:`.bus`), so the Dashboard's
``@work(thread=True)`` poller can call it off the UI thread and publish the folded
records back through ``ContextBus.set_agent_activity`` (marshalled by the app's
``call_from_thread``).

It surfaces WHAT the AgentCore swarm (and the local chat agent) are doing — the
transparency gap GO-3/GO-4/GO-5 — WITHOUT any new instrumentation: it reads the
per-span trace that already exists.

**Sources (design §E.8 table):**

* **Offline** — ``runs/<run_id>/trace.jsonl`` written by
  :class:`~amlas.telemetry.trace.TraceLogger` (each line a :class:`TraceSpan` with
  ``kind ∈ tool|model|phase|agent``), plus ``runs/<run_id>/kanban.json`` for the
  current :class:`~amlas.telemetry.kanban.RunKanban` ``current_phase``.
* **Cloud** — when there is no local file, an injected ``log_reader`` (the
  ``get_recent_logs`` tool) reads the durable CloudWatch trace log group
  (``AMLAS_TRACE_LOG_GROUP``, default ``/aws/amlas/agent-traces``) filtered by the
  AgentCore ``session_id`` → the correlated log stream (design §E.8 / GO-5). The
  log-event messages are the same span JSON, so they fold identically.

**Fold (design §E.8):** spans are grouped per ``agent``-kind span tree into one
:class:`~amlas.tui.context.models.AgentActivity` each: ``last_tool`` = latest
``kind=="tool"`` span, ``last_model`` = latest ``kind=="model"`` span (its
``attributes["model"]`` tier, else the span name), ``phase`` = Kanban
``current_phase``, ``aws_services`` = :func:`aws_services_for_tool` over the tool
spans (+ ``Bedrock`` for any model span), ``status`` from the agent span. When no
``agent`` span exists (e.g. a flat tool trace) all tool/model spans fold into a
single record so the panel still shows activity.

Everything is offline-safe: a missing/None ``run_id`` or an absent trace yields an
empty list — never a crash (the panel then shows an empty table).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import AgentActivity

logger = logging.getLogger(__name__)

__all__ = ["TraceTailReader", "aws_services_for_tool"]

# The durable CloudWatch trace log group (design §E.8; matches
# ``telemetry.trace.resolve_trace_sink`` default). Read from the env so a deployed
# runtime can point the reader at the same group the CloudWatchTraceLogger writes.
_DEFAULT_TRACE_LOG_GROUP = os.environ.get("AMLAS_TRACE_LOG_GROUP", "/aws/amlas/agent-traces")


# ─── tool → AWS-service map (design §E.8: derived from span name) ───────────────
#
# The design's worked examples: ``train_model``→SageMaker, ``read_lake_slice``→S3/Glue,
# model spans→Bedrock, ``get_recent_logs``→CloudWatch. Rather than an exhaustive
# enumeration of all ~34 tools, keyword heuristics map a span/tool name to the AWS
# services it exercises; an UNKNOWN tool degrades to an empty list (never guesses).
_SERVICE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("train", "tune", "hpo", "sagemaker", "deploy", "endpoint", "estimator", "batch_transform"), "SageMaker"),
    (("lake", "s3", "upload", "download", "dataset", "papers", "slice", "bucket", "object"), "S3"),
    (("lake", "glue", "catalog", "crawler"), "Glue"),
    (("logs", "metric", "alarm", "alert", "monitor", "cloudwatch", "spend", "dashboard"), "CloudWatch"),
    (("bedrock", "invoke_agent", "converse"), "Bedrock"),
    (("registry", "model_package", "approve"), "SageMaker"),
)


def aws_services_for_tool(tool_name: str | None) -> list[str]:
    """Map a tool/span ``name`` to the AWS services it touches (design §E.8).

    Keyword-based so it covers the whole tool surface without an exhaustive table;
    returns a de-duplicated, order-preserving list. An empty/unknown name returns
    ``[]`` (never guesses a service).
    """
    if not tool_name:
        return []
    name = tool_name.lower()
    services: list[str] = []
    for keywords, service in _SERVICE_KEYWORDS:
        if service in services:
            continue
        if any(kw in name for kw in keywords):
            services.append(service)
    return services


# ─── swarm span-name → roster-lead normalizer ──────────────────────
#
# The swarm orchestrator names its agent-kind spans ``research_iter1`` / ``fe_iter1`` /
# ``ml_iter1`` / ``orchestrator_run:<id>`` — none of which equal the seven canonical roster
# identities in ``ML_AUTOMATION_AGENTS``. So a folded record landed in the AgentLogsPane's
# de-emphasized "Other agents" tail instead of lighting up the named lead. This normalizer
# maps a raw span/agent name onto its roster lead by keyword; an unrecognized name is
# returned unchanged (so nothing is dropped — it still shows in the tail).
_LEAD_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("research",), "research_lead"),
    (("data_eng", "dataeng", "ingest", "dq_", "data_quality"), "data_engineering_lead"),
    (("fe_", "feature", "feat_"), "fe_lead"),
    (("ml_", "train", "tune", "hpo", "experiment", "model_"), "ml_execution_lead"),
    (("mlops", "monitor_setup", "registry"), "mlops_lead"),
    (("deploy", "serving", "endpoint", "readiness", "arbiter", "decision"), "deploy_serving_lead"),
    (("monitor", "drift", "retrain"), "monitoring_lead"),
)


def normalize_lead_name(span_name: str | None) -> str | None:
    """Map a raw swarm span/agent name to its canonical roster lead.

    ``research_iter1`` → ``research_lead``, ``fe_iter2`` → ``fe_lead``, ``ml_iter1`` →
    ``ml_execution_lead``, etc. Returns the input unchanged when it already IS a roster
    lead or matches no keyword (never drops the record — it just won't map to a named row).
    ``None``/empty → ``None``.
    """
    if not span_name:
        return None
    name = span_name.lower()
    # Already a canonical lead → keep as-is.
    from ...control_plane.token_cost import ML_AUTOMATION_AGENTS

    if span_name in ML_AUTOMATION_AGENTS:
        return span_name
    for keywords, lead in _LEAD_KEYWORDS:
        if any(kw in name for kw in keywords):
            return lead
    return span_name


class TraceTailReader:
    """Folds ``runs/<run_id>/trace.jsonl`` (+ ``kanban.json``) into ``AgentActivity``.

    Offline it reads the JSONL trace file; when that is absent and a ``log_reader``
    (the ``get_recent_logs`` tool) is injected it reads the durable CloudWatch trace
    log group filtered by ``session_id``. Both paths yield the SAME span dicts, folded
    identically. Textual-free and side-effect-free — safe to call from a worker thread.
    """

    def __init__(
        self,
        run_id: str | None,
        *,
        runs_dir: str | Path = "runs",
        session_id: str | None = None,
        log_reader: Callable[..., dict[str, Any]] | None = None,
        log_group: str | None = None,
        since: str = "1h",
    ) -> None:
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)
        self.session_id = session_id
        self._log_reader = log_reader
        self.log_group = log_group or _DEFAULT_TRACE_LOG_GROUP
        self.since = since

    # ─── Public API ─────────────────────────────────────────────────────────────

    def read(self) -> list[AgentActivity]:
        """Return one :class:`AgentActivity` per agent span-tree (or ``[]``, offline-safe)."""
        if not self.run_id:
            return []
        try:
            spans = self._load_spans()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("TraceTailReader failed to load spans for %s: %s", self.run_id, exc)
            return []
        if not spans:
            return []
        phase, project_id = self._load_kanban()
        return self._fold(spans, phase=phase, project_id=project_id)

    # ─── Span loading (offline JSONL first, then injected CloudWatch) ───────────

    def _run_dir(self) -> Path:
        return self.runs_dir / str(self.run_id)

    def _load_spans(self) -> list[dict[str, Any]]:
        """Load span dicts from the local trace file, else the injected cloud reader."""
        trace_path = self._run_dir() / "trace.jsonl"
        if trace_path.exists():
            spans: list[dict[str, Any]] = []
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    spans.append(json.loads(line))
                except Exception:  # nosec B112
                    continue
            return spans
        # No local file → cloud path (design §E.8 / GO-5): read the durable trace log
        # group filtered by session_id (the correlated CloudWatch stream).
        if self._log_reader is not None:
            try:
                out = self._log_reader(
                    log_group=self.log_group,
                    filter=self.session_id or "",
                    since=self.since,
                )
            except Exception as exc:
                logger.warning("TraceTailReader cloud log_reader raised: %s", exc)
                return []
            events = (out or {}).get("events") or []
            spans = []
            for ev in events:
                msg = ev.get("message") if isinstance(ev, dict) else None
                if not msg:
                    continue
                try:
                    spans.append(json.loads(msg))
                except Exception:  # nosec B112
                    continue
            return spans
        return []

    def _load_kanban(self) -> tuple[str | None, str | None]:
        """Return ``(current_phase, project_id)`` from ``kanban.json`` (or ``(None, None)``)."""
        kanban_path = self._run_dir() / "kanban.json"
        if not kanban_path.exists():
            return None, None
        try:
            data = json.loads(kanban_path.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        if not isinstance(data, dict):
            return None, None
        return data.get("current_phase"), data.get("project_id")

    # ─── Fold spans → AgentActivity (design §E.8) ───────────────────────────────

    def _fold(
        self, spans: list[dict[str, Any]], *, phase: str | None, project_id: str | None
    ) -> list[AgentActivity]:
        by_id = {s.get("span_id"): s for s in spans if s.get("span_id")}
        agent_spans = [s for s in spans if s.get("kind") == "agent"]

        def nearest_agent(span: dict[str, Any]) -> str | None:
            """Walk parent_ids to the nearest ``agent``-kind ancestor's span_id."""
            seen: set[str] = set()
            cur: dict[str, Any] | None = span
            while cur is not None:
                if cur.get("kind") == "agent":
                    return cur.get("span_id")
                pid = cur.get("parent_id")
                if not pid or pid in seen:
                    return None
                seen.add(pid)
                cur = by_id.get(pid)
            return None

        # Group every span under its owning agent span (or None when no agent tree).
        groups: dict[str | None, list[dict[str, Any]]] = {}
        for span in spans:
            if span.get("kind") == "agent":
                root = span.get("span_id")
            else:
                root = nearest_agent(span)
            groups.setdefault(root, []).append(span)

        activities: list[AgentActivity] = []
        if agent_spans:
            for agent_span in agent_spans:
                members = groups.get(agent_span.get("span_id"), [])
                activities.append(
                    self._build_activity(agent_span, members, phase=phase, project_id=project_id)
                )
        else:
            # Flat trace (no agent span) → one record aggregating all spans.
            activities.append(
                self._build_activity(None, spans, phase=phase, project_id=project_id)
            )
        return activities

    def _build_activity(
        self,
        agent_span: dict[str, Any] | None,
        members: list[dict[str, Any]],
        *,
        phase: str | None,
        project_id: str | None,
    ) -> AgentActivity:
        def _end(span: dict[str, Any]) -> float:
            try:
                return float(span.get("end_ts") or span.get("start_ts") or 0.0)
            except Exception:
                return 0.0

        tools = sorted((m for m in members if m.get("kind") == "tool"), key=_end)
        models = sorted((m for m in members if m.get("kind") == "model"), key=_end)

        last_tool = tools[-1].get("name") if tools else None
        last_model: str | None = None
        if models:
            latest_model = models[-1]
            attrs = latest_model.get("attributes") or {}
            last_model = attrs.get("model") or attrs.get("model_id") or latest_model.get("name")

        aws_services: list[str] = []
        for tool_span in tools:
            for svc in aws_services_for_tool(tool_span.get("name")):
                if svc not in aws_services:
                    aws_services.append(svc)
        if models and "Bedrock" not in aws_services:
            aws_services.append("Bedrock")

        ordered = sorted(members, key=_end)
        current_span = ordered[-1].get("name") if ordered else None

        if agent_span is not None:
            attrs = agent_span.get("attributes") or {}
            status = attrs.get("status") or ("failed" if agent_span.get("status") == "error" else "running")
            agent_name = agent_span.get("name") or project_id or "agent"
        else:
            status = "failed" if any(m.get("status") == "error" for m in members) else "running"
            agent_name = project_id or "agent"

        # : normalize a swarm span name (``research_iter1`` …) onto its canonical
        # roster lead so the AgentLogsPane lights up the named lead instead of dropping the
        # record into the de-emphasized "Other agents" tail. Unrecognized names pass through.
        agent_name = normalize_lead_name(str(agent_name)) or str(agent_name)

        return AgentActivity(
            agent_name=str(agent_name),
            run_id=self.run_id,
            session_id=self.session_id,
            phase=phase,
            current_span=current_span,
            last_tool=last_tool,
            last_model=last_model,
            aws_services=aws_services,
            status=str(status),
            updated_at=datetime.now(UTC).isoformat(),
        )
