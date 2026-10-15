"""Phase 2 — the shared-context data model (Textual-free, dataclasses only).

This is the ``src/amlas/tui/context/models.py`` module named in the design
(``docs/tui-context-integration-design.md`` §E.2 / §F). It defines the single
source-of-truth *record* the TUI shares between every pane and the chat/agent
brain: one ``AppContextState`` snapshot composed of per-domain dataclasses
(active spec / dataset / job / run / model / problem / agent-activity / selection).

Design constraints honored here:

* **Textual-free** — no ``textual`` import, so this is directly verifiable headless exactly
  like ``capability_surface.py`` and the offline offline tier that builds a bare
  ``ChatEngine()`` without a running App.
* **All fields optional** — nothing is set at launch (mirrors ``session_steering=None``
  and ``current_spec=None`` today), so the bus is *additive* and null-safe.
* **No flow logic** — these are pure records; panes and the engine mutate them via the
  ``ContextBus`` typed mutators, capabilities still run through the ADR-010 funnel.

Only the Phase-2 fields are wired end-to-end (spec/dataset/job + the engine mirror);
the remaining dataclasses (``ActiveRun``/``ActiveModel``/``ActiveProblem``/
``AgentActivity``) are defined now — exactly as the design enumerates — so later phases
(3–6: cross-tab sync, dashboard monitoring, agent-work propagation) extend the SAME
record without a schema change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ActiveDataset",
    "ActiveJob",
    "ActiveModel",
    "ActiveProblem",
    "ActiveRun",
    "ActiveSpec",
    "ActiveView",
    "AgentActivity",
    "AppContextState",
    "ProcessActivity",
]


@dataclass
class ActiveSpec:
    """The spec the user is working on (Specs tab / chat ``/spec`` / agent)."""

    path: str | None = None            # disk path (mirrors engine.context["current_spec"])
    spec_dict: dict[str, Any] = field(default_factory=dict)  # parsed front-matter
    project_id: str | None = None      # from SpecModel.project_id
    task_type: str | None = None       # classification|regression|segmentation|detection
    target: str | None = None
    primary_metric: str | None = None
    golden_dataset: str | None = None
    valid: bool | None = None          # last validate_spec result
    source: str = ""                   # "specs_tab" | "chat" | "agent" | "pipeline_tab"


@dataclass
class ActiveDataset:
    """The dataset in focus (Data/Lake tab / chat ``/data`` / agent)."""

    path: str | None = None            # mirrors engine.context["data_summary"]["path"]
    rows: int | None = None
    cols: int | None = None
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    dq_score: float | None = None      # last /dqscan or data_quality_flow overall_score
    source: str = ""                   # "data_tab" | "lake_tab" | "chat" | "agent"


@dataclass
class ActiveJob:
    """The single background job being monitored (JobManager ``Job`` mirror)."""

    job_id: str | None = None          # the JobManager Job.id being monitored
    kind: str | None = None            # "run" | "train" | "tune" | flow_id | tool name
    status: str = "idle"               # mirrors Job.status
    progress: float = 0.0
    stage: str = ""
    run_id: str | None = None          # pipeline run id when kind=="run"
    started_by: str = ""               # "pipeline_tab" | "chat" | "agent"


@dataclass
class ActiveRun:
    """A COMPLETED run for Results scoping (populated in a later phase)."""

    run_id: str | None = None
    spec_id: str | None = None
    best_algorithm: str | None = None
    primary_metric: str | None = None
    best_score: float | None = None
    decision: str | None = None        # deploy|iterate|reject
    artifacts_dir: str | None = None
    # Phase B1 (additive): the run's lifecycle status so tabs can distinguish a
    # completed vs partial vs FAILED run (not just an empty placeholder). Defaults
    # to None so every existing ``set_run`` caller/test stays green.
    status: str | None = None          # "complete" | "partial" | "failed" | None


@dataclass
class ActiveModel:
    """For MLOps/deploy/monitor (populated in a later phase)."""

    model_ref: str | None = None
    version: str | None = None
    approval: str | None = None
    endpoint: str | None = None        # deployed endpoint → CloudWatch namespace/dims


@dataclass
class AgentActivity:
    """One live/recent AgentCore (or local chat) agent (populated in a later phase)."""

    agent_name: str = ""               # e.g. "research_lead", "tui_chat_agent", "fine_tune"
    run_id: str | None = None          # correlates to runs/<run_id>/trace.jsonl + Kanban
    session_id: str | None = None      # AgentCore runtimeSessionId → CloudWatch log stream
    phase: str | None = None           # Kanban Phase (R1_FRAMING … D3_ARBITER) when known
    current_span: str | None = None    # last span name (kind=tool|model|phase|agent)
    last_tool: str | None = None       # most recent tool span the agent invoked
    last_model: str | None = None      # most recent model span (which LLM tier)
    aws_services: list[str] = field(default_factory=list)  # AWS services touched
    status: str = "idle"               # idle|running|complete|failed
    updated_at: str = ""
    # Phase 5 §2e (D-5): a "roster" flag — an agent explicitly signed on for this run
    # (a declared lead/subagent). Defaults True so the fold producers that already
    # publish activity keep their agents in the roster without any change; the panel
    # groups ``signed_on and status∈{running,submitted}`` as the roster vs. all agents
    # by ``updated_at`` as last-seen. Purely additive (spec §5.2 authorizes the +1 line
    # in ``test_agent_activity_defaults``).
    signed_on: bool = True


@dataclass
class ProcessActivity:
    """One local process/thread/subprocess if running (Phase 5 §2d / D-3).

    Textual-free, all fields optional — mirrors :class:`AgentActivity`. Surfaces an
    OS/subprocess/thread inventory DISTINCT from an ``ActiveJob`` (e.g. a local
    training subprocess, an Optuna HPO thread, a SageMaker submit-poll thread). Keyed
    by ``process_id`` in ``AppContextState.processes`` and published via the bus
    ``set_process_activity`` mutator (the ``"processes"`` event).
    """

    process_id: str = ""               # stable key (pid, thread name, or job_id)
    kind: str = ""                     # "subprocess" | "thread" | "sagemaker_poll" | "optuna" | ...
    pid: int | None = None             # OS pid when a real subprocess
    status: str = "running"            # running | complete | failed
    started_by: str = ""               # "pipeline_tab" | "chat" | "agent" | tool name
    progress: float = 0.0
    detail: str = ""                   # short human label (command / stage)
    updated_at: str = ""


@dataclass
class ActiveView:
    """A REQUESTED dashboard view — "show me X as a Y" (plan §B2).

    Every other record in this module answers *what the session state is*. This one is the
    only record that expresses *what the user asked to SEE*, which is what makes a
    chat-driven dashboard re-render possible: before this existed the agent could publish
    data to the bus but had no way to say "as a heat map", so the Dashboard's primary pane
    picked its content from a job-kind keyword table with no agent input.

    Written by :func:`~amlas.tools.dashboard_tools.render_dashboard_view` (the agent's
    verb) and read by :class:`~amlas.tui.widgets.dynamic_primary.DynamicPrimaryPane`. All
    fields optional, so an unset view is null-safe and the pane keeps its prior
    keyword-driven behaviour.

    The payload fields are deliberately chart-shaped (``series`` / ``values`` / ``matrix``
    / ``x`` / ``y``) rather than one opaque blob, so a pane can tell a bar request from a
    heat-map request without parsing.
    """

    view_id: str | None = None         # logical view name, e.g. "correlation" | "scores"
    chart: str = ""                    # "line"|"bar"|"heatmap"|"scatter"|"histogram"|"confusion"
    title: str = ""                    # human title rendered as the pane header
    series: dict[str, list[float]] = field(default_factory=dict)   # line: name → values
    values: dict[str, float] = field(default_factory=dict)         # bar: label → value
    matrix: list[list[float]] = field(default_factory=list)        # heatmap/confusion
    x: list[float] = field(default_factory=list)                   # scatter x / histogram
    y: list[float] = field(default_factory=list)                   # scatter y
    labels: list[str] = field(default_factory=list)                # axis / class labels
    source: str = ""                   # "agent" | "chat" | "dashboard_tab"
    detail: str = ""                   # optional caption (data provenance)
    # WHICH tab the chart is addressed to ("results"|"pipeline"|"mlops"|"dashboard").
    # Empty = the Dashboard, which keeps ADR-0006's original single-pane behaviour as the
    # back-compatible default. A ``ChartViewPanel`` renders only views addressed to its own
    # tab, so one bus event cannot paint the same chart onto every surface at once.
    target: str = ""


@dataclass
class ActiveProblem:
    """business_framing output (populated in a later phase)."""

    problem_id: str | None = None
    kpi: str | None = None
    problem_card_path: str | None = None


@dataclass
class AppContextState:
    """The single source-of-truth snapshot shared across every pane + the agent."""

    spec: ActiveSpec = field(default_factory=ActiveSpec)
    dataset: ActiveDataset = field(default_factory=ActiveDataset)
    job: ActiveJob = field(default_factory=ActiveJob)
    # Phase 5 §2a (D-1): ALL concurrent jobs keyed by ``job_id`` (the multi-job
    # console). ``job`` above stays the SINGLE focused/latest job (the back-compat
    # alias every existing ``set_job`` test asserts); ``jobs`` is the superset the
    # Active-Jobs panel renders. Populated inside the same ``set_job`` mutation.
    jobs: dict[str, ActiveJob] = field(default_factory=dict)  # job_id → job
    run: ActiveRun = field(default_factory=ActiveRun)
    model: ActiveModel = field(default_factory=ActiveModel)
    problem: ActiveProblem = field(default_factory=ActiveProblem)
    steering: Any | None = None        # the SteeringProfile (unifies session_steering)
    agents: dict[str, AgentActivity] = field(default_factory=dict)  # agent_name → activity
    # Phase 5 §2d (D-3): local processes/threads/subprocesses keyed by ``process_id``,
    # published via ``set_process_activity`` (the ``"processes"`` event).
    processes: dict[str, ProcessActivity] = field(default_factory=dict)  # process_id → activity
    selection: dict[str, str] = field(default_factory=dict)  # transient UI selection
    # Plan §B2: the REQUESTED dashboard view (the agent's "show me X as a Y"). Unset by
    # default, so the Dashboard keeps its job-kind-driven default until something asks.
    view: ActiveView = field(default_factory=ActiveView)

    def to_dict(self) -> dict[str, Any]:
        """Render the whole record as nested plain data (for the agent tool / logging).

        ``steering`` is coerced to a lightweight repr when it is a non-serializable
        object (a ``SteeringProfile``), so this never raises on export.
        """
        data: dict[str, Any] = {
            "spec": asdict(self.spec),
            "dataset": asdict(self.dataset),
            "job": asdict(self.job),
            # Phase 5 §2a: emit the multi-job collection (mirrors the ``agents`` line).
            "jobs": {jid: asdict(job) for jid, job in self.jobs.items()},
            "run": asdict(self.run),
            "model": asdict(self.model),
            "problem": asdict(self.problem),
            "agents": {name: asdict(act) for name, act in self.agents.items()},
            # Phase 5 §2d: emit the local-processes collection (mirrors ``agents``).
            "processes": {pid: asdict(proc) for pid, proc in self.processes.items()},
            "selection": dict(self.selection),
            # Plan §B2: the requested view, so ``get_tui_context`` can tell the agent what
            # the user is currently looking at (and it stays JSON-serializable).
            "view": asdict(self.view),
        }
        if self.steering is None:
            data["steering"] = None
        elif hasattr(self.steering, "as_profile_dict"):
            try:
                data["steering"] = self.steering.as_profile_dict()
            except Exception:
                data["steering"] = str(self.steering)
        else:
            data["steering"] = str(self.steering)
        return data
