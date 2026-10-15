"""Main Textual App — Agentic-ML full-screen Terminal UI.

Provides a tabbed interface for chat, dashboard, specs, pipeline control,
results/leaderboard, data explorer, and (Phase 11, master plan §10) the new
Steering "dial", MLOps, and Lake screens. The app also carries the session
``SteeringProfile`` (built on the Steering screen) so the chat ``/impute``,
``/synth`` and ``/dqscan`` commands can honor the domain dial.

DP-2 (Phase 11): the app also owns a single THREAD-backed ``JobManager`` (so long
ops run off the event loop and the UI stays responsive), injects it into the chat
engine, mounts a :class:`JobsIndicator`, and refreshes that indicator on every
job event — marshalled onto the UI thread via ``call_from_thread`` (the
``status_bar.py`` discipline). See ``docs/audit/phase11-background-and-routing.md``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Literal

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Select, Static, TabbedContent, TabPane

from .context import (
    AppContextState,
    ContextBus,
    ContextEvent,
    ProcessActivity,
    SessionLedger,
    set_active_bus,
    set_active_ledger,
)
from .feature_flags import is_tab_enabled
from .jobs import Job, JobManager, ThreadExecutor
from .markup import escape_markup
from .screens.chat import ChatPane
from .screens.dashboard import DashboardPane
from .screens.data import DataPane
from .screens.lake import LakePane
from .screens.mlops import MLOpsPane
from .screens.pipeline import PipelinePane
from .screens.results import ResultsPane
from .screens.specs import SpecsPane
from .screens.steering import SteeringPane
from .widgets.jobs_indicator import JobsIndicator

logger = logging.getLogger(__name__)

# Phase B2.2 (design §B) — the 3-way runtime mode. Offline = no AWS (local adapters
# everywhere, Bedrock chat disabled → regex path); Local (recommended default) = local
# execution/pipeline but Bedrock inference ENABLED; Full AWS = managed backends +
# Bedrock (offline flag cleared, auto→AWS defaults).
RuntimeMode = Literal["offline", "local", "full_aws"]

# Ordered cycle for the ``f2`` keybinding + human labels for the notify/select.
_RUNTIME_MODE_ORDER: tuple[RuntimeMode, ...] = ("offline", "local", "full_aws")
# Notify labels — each states WHAT the switch changed, matching the design semantics
# (§A.6): the ``/run`` pipeline is ALWAYS local; the mode only flips AWS calls, the
# Bedrock chat gate, and the cloud tabs. Kept short so the toast fits one line.
_RUNTIME_MODE_LABELS: dict[RuntimeMode, str] = {
    "offline": "Offline — no AWS; local models, local (regex) chat",
    "local": "Local — local execution, Bedrock inference on",
    "full_aws": "Full AWS — managed backends, Bedrock inference on",
}
# The Select option list: (visible label, mode value). Labels are self-explanatory;
# the value strings (offline/local/full_aws) are the stable contract tests assert on.
_RUNTIME_MODE_OPTIONS: list[tuple[str, str]] = [
    ("Offline", "offline"),
    ("Local (Bedrock)", "local"),
    ("Full AWS", "full_aws"),
]

# Truthy tokens for the offline env switch (mirrors ``registry.py`` semantics).
_TRUTHY_TOKENS = ("1", "true", "yes", "on")


def _env_truthy(name: str) -> bool:
    """True when env var ``name`` holds a truthy token (``1/true/yes/on``)."""
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in _TRUTHY_TOKENS


def _force_aws_enabled() -> bool:
    """Whether the AWS-first affirmation is on (env ``AMLAS_TUI_FORCE_AWS``, default ON).

    User directive (2026-07-12): *"Ensure everything our TUI does kicks off via the
    AWS account by default, not locally."* The flag defaults ON so the AWS-first
    posture is affirmed by default; set it to ``0``/``false``/``no``/``off`` to skip
    the affirmation (an escape hatch — it does NOT flip the TUI to local, it only
    stops the explicit check).
    """
    val = os.environ.get("AMLAS_TUI_FORCE_AWS")
    if val is None:
        return True  # default ON
    return val.strip().lower() not in ("0", "false", "no", "off")


def _affirm_aws_first_posture() -> Any | None:
    """Explicit, logged AWS-first affirmation for the TUI process (design §5.2).

    Turns the previously implicit AWS-default backend posture into an explicit
    startup check. When :func:`_force_aws_enabled` is on and the process is not
    running offline, it
    verifies (via :func:`amlas.backends.registry.affirm_aws_first`) that the backend
    auto-selection resolves to the AWS defaults (SageMaker/Clarify/S3), not local, and
    logs the posture. Under ``AMLAS_OFFLINE`` the affirmation is an explicit no-op
    that stays local. Returns:

    - the :class:`AwsFirstPosture` when the flag is on (whether offline or not),
    - ``None`` when the flag is off (affirmation skipped) OR the helper hits any error
      (best-effort — a diagnostics/affirmation step must NEVER block the TUI launch).

    It NEVER mutates ``os.environ`` or pins a ``*_BACKEND`` selector — verification only.
    """
    if not _force_aws_enabled():
        logger.debug("AMLAS_TUI_FORCE_AWS off — skipping AWS-first affirmation")
        return None
    try:
        from ..backends.registry import affirm_aws_first

        posture = affirm_aws_first()
        if posture.offline:
            logger.info("TUI AWS-first: %s", posture.reason)
        elif posture.aws_first:
            logger.info("TUI AWS-first affirmed: %s", posture.checked)
        else:
            logger.warning("TUI AWS-first NOT satisfied: %s", posture.reason)
        return posture
    except Exception:
        # Best-effort — never let the affirmation block the TUI from launching.
        return None


class AgenticMLApp(App[None]):
    """Agentic-ML Terminal UI — Interactive ML Engineering Platform."""

    CSS_PATH = "styles.css"
    TITLE = "Agentic-ML"
    SUB_TITLE = "ML Engineering Platform"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("c", "show_tab('chat')", "Chat", show=True),
        Binding("d", "show_tab('dashboard')", "Dashboard", show=True),
        Binding("s", "show_tab('specs')", "Specs", show=True),
        Binding("p", "show_tab('pipeline')", "Pipeline", show=True),
        Binding("r", "show_tab('results')", "Results", show=True),
        Binding("e", "show_tab('data')", "Data", show=True),
        Binding("g", "show_tab('steering')", "Steering", show=True),
        Binding("m", "show_tab('mlops')", "MLOps", show=True),
        Binding("l", "show_tab('lake')", "Lake", show=True),
        # Phase B2.2 (design §C.5): cycle Offline → Local → Full AWS. ``m`` is taken
        # by MLOps, so ``f2`` is the runtime-mode cycle key.
        Binding("f2", "cycle_runtime_mode", "Mode", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The session domain-steering "dial" (master plan §3.1). Built on the
        # Steering screen / `/steer set`, honored by `/impute`,`/synth`,`/dqscan`.
        self.session_steering: Any | None = None
        # DP-2: one thread-backed JobManager for the whole app. ThreadExecutor →
        # ``submit`` returns immediately (job on a daemon thread) so long ops never
        # block the event loop; on_event refreshes the jobs indicator.
        self.job_manager: JobManager = JobManager(
            executor=ThreadExecutor(),
            on_event=self._on_job_event,
        )
        # Phase 2 (TUI context integration, design §E.1/§E.3): the shared
        # ``ContextBus`` — a plain, Textual-free store held as an App attribute
        # alongside ``job_manager``/``session_steering``. It carries the single
        # source-of-truth session state (active spec/dataset/job/…) and an
        # observer callback marshalled onto the UI thread by ``_on_context_event``
        # (the identical ``call_from_thread`` discipline as ``_on_job_event``). The
        # engine + panes are wired to it in ``on_mount``.
        self.context_bus: ContextBus = ContextBus(on_event=self._on_context_event)
        # Phase C2 (session-wide context store, design §D.2/§D.6): the App-owned
        # append-only ``SessionLedger`` that LAYERS ON TOP of ``context_bus`` — the
        # session TIMELINE (turns / tools / commands / bus transitions / artifacts / mode
        # switches) that accumulates in ALL runtime modes INCLUDING Offline. Held as an App
        # attribute alongside ``context_bus``/``job_manager``/``session_steering``. In
        # ``on_mount`` it is (a) subscribed to the bus (state transitions), (b) injected
        # into the engine, and (c) registered process-locally via ``set_active_ledger`` so
        # the read-only ``get_tui_context`` tool can merge its bounded ``"session"`` key.
        # Persistence is OFF by default here (fresh in-memory ledger per launch, §D.4);
        # resume is an additive follow-on gated by ``AMLAS_TUI_RESUME_SESSION``.
        # Phase D2 (resume UX): records the id of a NON-empty prior session actually
        # reloaded by ``_build_session_ledger`` (else ``None``) so ``on_mount`` can surface
        # a visible "Resumed session <id>" indicator. Set BEFORE building the ledger.
        self._resumed_session_id: str | None = None
        self.session_ledger: SessionLedger = self._build_session_ledger()
        # Phase B2.2 (design §B.1): the runtime mode (Offline / Local / Full AWS). The
        # default only LABELS the current env — it does NOT mutate anything at mount
        # (strangler-safe, §B.3). ``set_runtime_mode`` is what mutates env, and it is
        # invoked ONLY by the user (the ``#mode-select`` control or the ``f2`` binding).
        self.runtime_mode: RuntimeMode = self._derive_default_mode()

    # ───  — markup-safe notifications (the single App-level chokepoint) ──

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: Literal["information", "warning", "error"] = "information",
        timeout: float | None = None,
        markup: bool = True,
    ) -> None:
        """Markup-safe override of :meth:`textual.app.App.notify`.

        Every ``self.notify(...)`` call from any pane routes here — ``Widget.notify``
        delegates to ``self.app.notify`` — so this ONE override protects every
        notification site in the TUI at once. The toast render path
        (``Toast.render`` → ``Content.from_markup(message)``) raises
        ``MarkupError`` on dynamic text containing bracket-shaped tokens (a path,
        a ``dict``/``list`` repr, or the  pydantic ``ValidationError`` repr
        ``[type=..., input_value='…', input_type=str]``). Because that render runs
        ASYNC in the message pump AFTER ``notify()`` returns, a caller-side
        try/except cannot catch it — the message must be made safe BEFORE it
        reaches the toast.

        Fix: when ``markup`` is on (the default), escape the message via
        :func:`~amlas.tui.markup.escape_markup` (a no-op for bracket-free text, so
        the vast majority of toasts are byte-for-byte unchanged; display-transparent
        for the rest). When a caller explicitly passes ``markup=False`` the message
        is already literal, so it is left untouched. ``title`` is assembled by
        Textual as a literal styled span (never through ``from_markup``), so it does
        not need escaping. Signature mirrors the base method exactly.
        """
        safe_message = escape_markup(message) if markup else message
        if timeout is None:
            super().notify(
                safe_message, title=title, severity=severity, markup=markup
            )
        else:
            super().notify(
                safe_message,
                title=title,
                severity=severity,
                timeout=timeout,
                markup=markup,
            )

    @staticmethod
    def _derive_default_mode() -> RuntimeMode:
        """Label the CURRENT process env as one of the 3 runtime modes (design §B.1).

        Pure/read-only — it never mutates env. If the offline switch is truthy the
        process is already hermetic-offline → ``offline``; otherwise the recommended
        default ``local`` (local execution + Bedrock inference), which is the honest
        description of what ships today (pure-local ``/run`` pipeline, Bedrock chat).
        """
        if _env_truthy("AMLAS_BACKENDS_OFFLINE") or _env_truthy("AMLAS_OFFLINE"):
            return "offline"
        return "local"

    def _build_session_ledger(self) -> SessionLedger:
        """Build the App's session ledger (Phase C2 §D.4; Phase D2 resume UX).

        A FRESH in-memory ledger per launch is the DEFAULT (strangler-safe: byte-identical
        launch, offline suite unaffected). Persistence to
        ``output/sessions/<session_id>/ledger.json`` (atomic write, turn-boundary debounced)
        is enabled, and load-on-launch RESUME is gated behind ``AMLAS_TUI_RESUME_SESSION``
        (default OFF, additive follow-on): only when that flag is truthy do we look for the
        most-recent persisted ledger and reload it, so the default path never reads disk.

        Phase D2 (resume UX): an instance method (was a ``@staticmethod``) so it can record
        ``self._resumed_session_id`` when a NON-empty prior ledger is actually reloaded — the
        signal ``on_mount`` uses to surface a visible resumed-session indicator. An EMPTY
        reload does NOT count as a resume (no history to announce). ``_latest_session_id``
        stays a staticmethod. Never raises (a bad cache degrades to a fresh session).
        """
        output_root = os.path.join("output", "sessions")
        # Resume is opt-in (default OFF): reload the most-recent persisted ledger.
        if _env_truthy("AMLAS_TUI_RESUME_SESSION"):
            try:
                latest = AgenticMLApp._latest_session_id(output_root)
                if latest is not None:
                    led = SessionLedger.load(latest, persist_dir=output_root)
                    # Only a NON-empty reload counts as a resume worth announcing.
                    if led.entries():
                        self._resumed_session_id = latest
                    return led
            except Exception:
                logger.debug("session-ledger resume failed (fresh)", exc_info=True)
        # Default: a fresh ledger that persists (turn-boundary) under output/sessions/.
        return SessionLedger(persist_dir=output_root)

    @staticmethod
    def _latest_session_id(output_root: str) -> str | None:
        """Return the most-recent ``session-*`` id under ``output_root`` (or ``None``).

        Used only by the opt-in resume path (``AMLAS_TUI_RESUME_SESSION``). Best-effort:
        any FS error → ``None`` (a clean fresh session), never raises.
        """
        try:
            if not os.path.isdir(output_root):
                return None
            sessions = [
                name
                for name in os.listdir(output_root)
                if name.startswith("session-")
                and os.path.exists(os.path.join(output_root, name, "ledger.json"))
            ]
            if not sessions:
                return None
            return sorted(sessions)[-1]
        except OSError:
            return None

    def compose(self) -> ComposeResult:
        """Create the app layout with header, tabs, jobs indicator, and footer."""
        # Mode selector IN the Header row, immediately LEFT of the clock (user
        # directive 2026-07-15 — one top bar, not two). The ``#mode-bar`` Horizontal
        # is nested INSIDE the stock Header and docked right with a 10-cell right
        # margin (the exact width of Textual 8.2.7's right-docked ``HeaderClock``),
        # so it renders on the SAME row as the time, just to its left. Uses only
        # PUBLIC widgets (Horizontal/Static/Select) + CSS — never subclassing the
        # version-fragile private ``Header`` internals. ``compact=True`` keeps the
        # Select 1 cell tall (no border) so it fits the height-1 Header; its
        # dropdown overlay escapes the Header via ``overlay: screen`` (verified
        # against Textual 8.2.7: overlay opens below and picks still fire
        # ``Select.Changed``).
        with Header(show_clock=True):
            with Horizontal(id="mode-bar"):
                yield Static("Mode:", id="mode-label")
                yield Select(
                    _RUNTIME_MODE_OPTIONS,
                    value=self.runtime_mode,
                    allow_blank=False,
                    compact=True,
                    id="mode-select",
                )
        # : the delivered surface ships with ONLY chat + dashboard enabled;
        # every other tab is gated behind the feature-flag system
        # (:mod:`amlas.tui.feature_flags`). ``AMLAS_TUI_ENABLE_ALL_TABS=1`` (or a
        # per-tab ``AMLAS_TUI_TAB_<ID>``) restores the developer surface. The
        # optional-tab compose bodies are declared once here and only YIELDED when
        # ``is_tab_enabled`` says so — chat + dashboard are always composed (the two
        # core tabs), and the initial tab is chat.
        with TabbedContent(initial="chat"):
            yield ChatPane("Chat", id="chat")
            with TabPane("Dashboard", id="dashboard"):
                yield DashboardPane()
            if is_tab_enabled("specs"):
                with TabPane("Specs", id="specs"):
                    yield SpecsPane()
            if is_tab_enabled("pipeline"):
                with TabPane("Pipeline", id="pipeline"):
                    yield PipelinePane()
            if is_tab_enabled("results"):
                with TabPane("Results", id="results"):
                    yield ResultsPane()
            if is_tab_enabled("data"):
                with TabPane("Data", id="data"):
                    yield DataPane()
            if is_tab_enabled("steering"):
                with TabPane("Steering", id="steering"):
                    yield SteeringPane()
            if is_tab_enabled("mlops"):
                with TabPane("MLOps", id="mlops"):
                    yield MLOpsPane()
            if is_tab_enabled("lake"):
                with TabPane("Lake", id="lake"):
                    yield LakePane()
        yield JobsIndicator(self.job_manager, id="jobs-indicator")
        yield Footer()

    def on_mount(self) -> None:
        """Inject the shared JobManager + ContextBus into the chat engine; prime the indicator.

        Phase 2 (design §E.3/§E.4): the engine is handed the bus exactly as
        ``engine.jobs`` is injected today (``engine.context_bus = self.context_bus``);
        the bus mirrors spec/dataset writes back into the engine's legacy
        ``engine.context`` dict (additive, so ``_build_context`` + the DataPane reader
        keep working); and the bus is registered process-locally via
        ``set_active_bus`` so the read-only ``get_tui_context`` agent tool can read the
        live snapshot. Tab-widget subscriptions land in Phase 3 (NOT here).
        """
        try:
            chat_pane = self.query_one(ChatPane)
            engine = getattr(chat_pane, "_engine", None)
            if engine is not None and hasattr(engine, "jobs"):
                engine.jobs = self.job_manager
            # Phase 5 (design §E.4 job seam, GP-1): share the ONE JobManager with the
            # tool-bound agent too, so a long-running tool the AGENT confirms is
            # submitted as a background job on the SAME manager (instead of running
            # inline). That job's transitions fire ``_on_job_event`` → ``set_job``, so
            # an agent-launched job becomes the monitored ``ActiveJob`` and the Phase-4
            # Dashboard auto-monitors it. Before this, only ``engine.jobs`` was wired
            # and agent long-ops bypassed the manager → never reached the bus. Null-safe.
            tool_backend = getattr(engine, "tool_backend", None) if engine is not None else None
            if tool_backend is not None and hasattr(tool_backend, "job_manager"):
                tool_backend.job_manager = self.job_manager
            # : inject the SAME shared JobManager into the cloud-dispatch tool seam so
            # ``dispatch_cloud_experiment`` submits the blocking InvokeAgentRuntime stream as a
            # background job (chat returns instantly; the terminal folds back via _on_job_event).
            # The live Strands path runs tools inline inside ``agent(prompt)`` and bypasses the
            # ``execute_tool_use`` JobManager seam, so the tool itself must background — hence a
            # process-local handle rather than LONG_RUNNING_TOOLS. Null-safe / best-effort.
            try:
                from ..tools.orchestration_tools import set_cloud_job_manager

                set_cloud_job_manager(self.job_manager)
            except Exception:  # nosec B110
                pass
            if engine is not None and hasattr(engine, "context_bus"):
                engine.context_bus = self.context_bus
                # Mirror bus spec/dataset writes into the engine's legacy context dict.
                engine_context = getattr(engine, "context", None)
                if isinstance(engine_context, dict):
                    self.context_bus.attach_engine_context(engine_context)
            # Phase C2 (design §D.6): wire the session ledger. (a) subscribe it to the bus
            # so run/dataset/spec transitions append timeline entries for free (the bus
            # supports N subscribers); (b) inject it into the engine so ``_build_context``
            # folds its history and ``process_message`` records turns/commands in EVERY
            # mode (mirrors ``engine.context_bus = self.context_bus`` above). Null-safe.
            self.context_bus.subscribe(self.session_ledger.on_bus_event)
            if engine is not None and hasattr(engine, "session_ledger"):
                engine.session_ledger = self.session_ledger
            # Phase F BUG-1 (Offline Bedrock-leak fix): sync the engine's Bedrock
            # chat gate to the DERIVED launch mode at mount. ``_derive_default_mode``
            # only LABELS the mode (read-only, §B.1) and ``set_runtime_mode`` — the
            # only code that flips the gate — is user-triggered AND no-ops when the
            # mode is unchanged, so an app LAUNCHED in Offline previously kept the
            # ``ChatEngine`` default ``bedrock_enabled=True``. With real Strands +
            # creds present that let an Offline NL turn reach REAL Bedrock (the
            # tier-3c tool agent), violating "Offline = no Bedrock". Applying the
            # gate here is a pure runtime FLAG (no env mutation, no client rebuild —
            # strangler-safe), reusing the same ``_set_engine_bedrock_gate`` seam
            # ``set_runtime_mode`` uses. Local / Full-AWS derive gate ON (unchanged).
            self._set_engine_bedrock_gate(self.runtime_mode != "offline")
        except Exception:  # nosec B110
            pass
        # Register the active bus for the headless ``get_tui_context`` agent tool.
        set_active_bus(self.context_bus)
        # Phase C2 (design §D.5): register the active ledger process-locally so
        # ``get_tui_context`` can merge the bounded ``"session"`` timeline (mirrors
        # ``set_active_bus`` above). Null-safe: absent it, the tool is byte-identical.
        set_active_ledger(self.session_ledger)
        self._refresh_jobs_indicator()
        # Phase D2 (resume UX): surface a VISIBLE resumed-session indicator as the LAST
        # step of mount (after the ledger is fully wired). No-op / silent when nothing was
        # resumed (fresh launch byte-identical, strangler-safe).
        self._announce_resumed_session()

    def _announce_resumed_session(self) -> None:
        """Tell the user a prior session was resumed (Phase D2 resume UX). Null-safe.

        Fires ONLY when ``AMLAS_TUI_RESUME_SESSION`` produced a NON-empty reload
        (``self._resumed_session_id`` set in ``_build_session_ledger``). Emits a one-line
        ``notify`` toast + (best-effort) a system chat line so the user knows the ledger
        below the fresh input reflects a prior session. Composes SILENTLY with a fresh
        ledger (a no-op when nothing was resumed) so the default launch is byte-identical.
        Never raises.
        """
        sid = getattr(self, "_resumed_session_id", None)
        if not sid:
            return
        try:
            n = len(self.session_ledger.entries())
        except Exception:
            n = 0
        try:
            self.notify(
                f"Resumed session {sid} ({n} prior events)", severity="information"
            )
        except Exception:  # nosec B110
            pass
        # Best-effort system chat line (additive-if-present: the ChatPane has no dedicated
        # "add system message" API today, so this degrades cleanly via the getattr guard —
        # we do NOT invent a ChatPane method; the toast alone satisfies the indicator).
        try:
            chat_pane = self.query_one(ChatPane)
            add_system = getattr(chat_pane, "add_system_message", None)
            if callable(add_system):
                add_system(
                    f"↻ Resumed prior session {sid} — {n} events restored. "
                    "Ask about earlier work and I'll have the context."
                )
        except Exception:  # nosec B110
            pass

    def action_show_tab(self, tab_id: str) -> None:
        """Switch to a specific tab by ID.

        : guarded against disabled tabs. When a tab is feature-flag
        disabled it is NOT composed, so activating it would raise; instead this is
        a silent no-op (the ``check_action`` hook already hides the binding, but the
        action can still be dispatched programmatically / by a stale keybinding).
        """
        if not is_tab_enabled(tab_id):
            return
        tabbed = self.query_one(TabbedContent)
        tabbed.active = tab_id

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Feature-flag gate for the footer bindings.

        Textual calls this for every binding to decide whether it is active/shown.
        For ``show_tab(<id>)`` bindings we return ``None`` (hide + disable) when the
        target tab is feature-flag disabled, so the Footer only advertises the tabs
        that are actually mounted. Every non-``show_tab`` action (and any enabled
        tab) is unaffected — returns ``True``.
        """
        if action == "show_tab" and parameters:
            tab_id = parameters[0]
            if isinstance(tab_id, str) and not is_tab_enabled(tab_id):
                return None
        return True

    # ─── Phase B2.2 — runtime-mode selector (design §A/§B/§C) ─────────────────

    def set_runtime_mode(self, mode: RuntimeMode | str) -> None:
        """Switch the runtime mode: mutate env knobs + chat gate + refresh health.

        Implements design §B.2 ordered behavior with the §A.6 authoritative env
        mapping. Idempotent + null-safe: a no-op when ``mode`` is unchanged or invalid;
        the env mutation (step 2) + ``runtime_mode`` store (step 4) always complete,
        while the chat gate / health refresh / posture / notify / select-reflect steps
        are each individually guarded so a failure in one never aborts the switch.

        The §A.6 mapping (the authoritative implementation) collapses to three clean
        env states — Offline & Local both set ``AMLAS_BACKENDS_OFFLINE=1`` (registry
        ``auto``→local, name-safe) and differ ONLY in the chat gate + remote-fetch +
        force-aws flag; Full AWS clears the offline flag (``auto``→AWS defaults). It
        never pins a per-port ``*_BACKEND`` (avoids the fail-fast footgun, §A.5) and
        never touches ``AWS_BEARER_TOKEN_BEDROCK`` (launch-time neutralization governs
        it, §A.4).
        """
        # 1. Validation + idempotency (design §B.2.1 — also breaks the Select.Changed
        #    feedback loop, §C.4). Invalid mode → ignore (null-safe, no raise).
        if mode not in _RUNTIME_MODE_ORDER:
            return
        if mode == self.runtime_mode:
            return

        # 2. Mutate in-process env per §A.6 (string values only). This is the source
        #    of truth and must always complete.
        offline = mode in ("offline", "local")
        os.environ["AMLAS_BACKENDS_OFFLINE"] = "1" if offline else "0"
        os.environ["AMLAS_OFFLINE"] = "1" if offline else "0"
        # Offline skips the AWS-first affirmation; Local + Full AWS affirm it.
        os.environ["AMLAS_TUI_FORCE_AWS"] = "0" if mode == "offline" else "1"
        # Offline disables remote (S3/HTTP) reads; Local + Full AWS enable them.
        os.environ["AMLAS_AGENT_REMOTE_FETCH"] = "0" if mode == "offline" else "1"

        # 3. Toggle the chat gate on the engine (design §A.3) — a runtime flag, NOT a
        #    client rebuild. Reached via the ChatPane, null-safe/headless-safe.
        self._set_engine_bedrock_gate(mode != "offline")

        # 4. Store the new mode (source of truth alongside the env).
        self.runtime_mode = mode

        # 5. Re-run health on every mounted SystemHealthWidget (Dashboard-only, so it
        #    may be absent). ``self.query`` returns an empty set when unmounted —
        #    NEVER ``query_one`` (which raises). Guarded per §B.2.
        self._refresh_health_widgets()

        # 5.5. Phase D1 (design §B.3): clear pane-level cached boto3 clients so the
        #    NEXT cloud op rebuilds under the new creds/profile/offline posture. Only
        #    the DataPane caches an S3 client; the helper is ``self.query``-based
        #    (empty-set-safe when the pane is unmounted / on another tab) and
        #    individually guarded so a failure never aborts the switch.
        self._reset_pane_client_caches()

        # 6. Re-affirm the AWS-first posture AFTER the env mutation (best-effort).
        try:
            _affirm_aws_first_posture()
        except Exception:  # nosec B110
            pass

        # 6.5. Phase C2 (design §D.6): note the mode switch on the session ledger — the
        # ONE additive line C2 adds to this method. The HANDOFF itself needs NO other
        # code: because the ledger is mode-independent and accumulates in every mode, the
        # gate flip in step 3 means the NEXT ``_build_context`` already folds the entire
        # prior (offline) ledger into the first Bedrock prompt. This just records the
        # transition in the timeline. Null-safe/try-guarded — never abort a switch on it.
        try:
            self.session_ledger.record(
                "mode",
                summary=f"mode → {_RUNTIME_MODE_LABELS.get(mode, mode)}",
                detail={"mode": mode},
                source="mode_switch",
            )
        except Exception:  # nosec B110
            pass

        # 7. Notify the change with a human label.
        try:
            self.notify(
                f"Runtime mode → {_RUNTIME_MODE_LABELS.get(mode, mode)}",
                severity="information",
            )
        except Exception:  # nosec B110
            pass

        # 8. Reflect the mode in the Select (programmatic switches, e.g. the f2 cycle).
        #    B.2.1 idempotency breaks any Select.Changed re-entrancy this may cause.
        self._reflect_mode_in_select(mode)

    def _set_engine_bedrock_gate(self, enabled: bool) -> None:
        """Apply the runtime chat gate to the engine (null-safe, §A.3/§B.2.3; D1 §B.1).

        Phase D1: routes through the engine's ``reconfigure_mode`` seam when present
        (the single documented mode-reapply entry point — flips ``bedrock_enabled``
        which gates BOTH the tier-3c tool agent and the tier-4 raw converse, without
        rebuilding the cached client). Falls back to setting ``bedrock_enabled``
        directly for a bare/older engine — backward-compatible + null-safe.
        """
        try:
            chat_pane = self.query_one(ChatPane)
            engine = getattr(chat_pane, "_engine", None)
            if engine is None:
                return
            reconfigure = getattr(engine, "reconfigure_mode", None)
            if callable(reconfigure):
                reconfigure(bedrock_enabled=enabled)
            elif hasattr(engine, "bedrock_enabled"):
                engine.bedrock_enabled = enabled
        except Exception:  # nosec B110
            pass

    def _reset_pane_client_caches(self) -> None:
        """Clear pane-level cached boto3 clients on a mode switch (Phase D1, design §B.3).

        Only the DataPane caches an S3 client (``data.py``). ``self.query`` returns an
        empty set when the pane is not mounted (unmounted / another tab) → iterate,
        NEVER ``query_one`` (which raises). Best-effort / never raises.
        """
        try:
            from .screens.data import DataPane

            for pane in self.query(DataPane):
                try:
                    pane.reset_client_cache()
                except Exception:  # nosec B110
                    pass
        except Exception:  # nosec B110
            pass

    def _refresh_health_widgets(self) -> None:
        """Re-run health on every mounted ``SystemHealthWidget`` (design §B.2.5).

        CRITICAL null-safety: the widget lives ONLY inside the Dashboard pane, so it
        may not be mounted when the user switches mode from another tab. ``self.query``
        returns a possibly-empty set — iterate it, never ``query_one`` (which raises).
        """
        try:
            from .widgets.status_bar import SystemHealthWidget

            for widget in self.query(SystemHealthWidget):
                try:
                    widget.refresh_health()
                except Exception:  # nosec B110
                    pass
        except Exception:  # nosec B110
            pass

    def _reflect_mode_in_select(self, mode: RuntimeMode | str) -> None:
        """Mirror the current mode into the ``#mode-select`` control (design §B.2.8)."""
        try:
            from textual.widgets import Select

            select = self.query_one("#mode-select", Select)
            if select.value != mode:
                select.value = mode
        except Exception:  # nosec B110
            pass

    @on(Select.Changed, "#mode-select")
    def _on_mode_select_changed(self, event: Select.Changed) -> None:
        """Route a ``#mode-select`` change to ``set_runtime_mode`` (design §C.4)."""
        value = event.value
        if isinstance(value, str):
            self.set_runtime_mode(value)

    def action_cycle_runtime_mode(self) -> None:
        """Cycle Offline → Local → Full AWS → Offline (the ``f2`` binding, §C.5)."""
        try:
            idx = _RUNTIME_MODE_ORDER.index(self.runtime_mode)
        except ValueError:
            idx = 0
        next_mode = _RUNTIME_MODE_ORDER[(idx + 1) % len(_RUNTIME_MODE_ORDER)]
        self.set_runtime_mode(next_mode)

    # ─── DP-2 job-event surfacing (§3.2) ──────────────────────────────────────

    def _on_job_event(self, job: Job) -> None:
        """JobManager event sink — refresh the indicator on the UI thread.

        Job events fire from background daemon threads (ThreadExecutor); widget
        mutation must happen on the UI thread. ``call_from_thread`` marshals it
        there. When the event originates on the UI thread (the synchronous
        ``submitted`` emit) ``call_from_thread`` raises — we then refresh directly.

        Phase 2 (design §E.4, job seam): additionally mirror the job onto the shared
        ``ContextBus`` (``set_job``) so agent- AND user-started jobs — which all share
        this one ``JobManager`` — become the monitored ``ActiveJob``. The bus mutation
        is thread-safe (its own ``RLock``); it publishes a ``"job"`` event whose UI
        refresh is marshalled by ``_on_context_event`` below (the same discipline as the
        indicator refresh). This is plumbing only — no tab widget consumes it yet
        (Dashboard/Pipeline wiring is Phase 3/4).
        """
        try:
            self.context_bus.set_job(
                job_id=job.id,
                kind=job.kind,
                status=job.status,
                progress=job.progress,
                stage=job.stage,
                run_id=str(job.args.get("run_id")) if job.args.get("run_id") else None,
                started_by=str(job.args.get("started_by", "")) or "app",
            )
        except Exception:  # nosec B110
            pass
        # Phase 5 §2d (D-3): also surface the job as a LOCAL PROCESS. The app's shared
        # JobManager runs local work on OS threads via ``ThreadExecutor`` — so every job
        # it drives IS a local process/thread. Publishing a ``ProcessActivity`` keyed by
        # the job id makes the local training subprocess / HPO thread / submit-poll
        # thread appear in the Dashboard's Local Processes region without any new
        # producer plumbing. Null-safe (its own ``RLock``; never blocks the job event).
        try:
            self.context_bus.set_process_activity(
                ProcessActivity(
                    process_id=str(job.id),
                    kind=str(job.kind or "thread"),
                    status=str(job.status or "running"),
                    started_by=str(job.args.get("started_by", "")) or "app",
                    progress=float(job.progress or 0.0),
                    detail=str(job.stage or ""),
                )
            )
        except Exception:  # nosec B110
            pass
        try:
            self.call_from_thread(self._refresh_jobs_indicator)
        except Exception:
            try:
                self._refresh_jobs_indicator()
            except Exception:  # nosec B110
                pass

    def _refresh_jobs_indicator(self) -> None:
        try:
            indicator = self.query_one("#jobs-indicator", JobsIndicator)
            indicator.refresh_counts()
        except Exception:  # nosec B110
            pass

    # ─── Phase 2 context-event surfacing (design §E.4, mirrors _on_job_event) ──

    def _on_context_event(self, event: ContextEvent, state: AppContextState) -> None:
        """ContextBus event sink — marshal any UI refresh onto the UI thread.

        The identical ``call_from_thread`` discipline as ``_on_job_event``: a
        ``ContextBus`` mutation may originate on a background job thread (via
        ``set_job`` in ``_on_job_event``), so any pane re-render triggered by a
        context event must land on the UI thread. Phase 2 is plumbing only — no pane
        subscribes yet — so this is a null-safe marshalling shim that Phase 3+ pane
        subscribers hang off. It never raises (a bad refresh must not break a mutation).
        """
        try:
            self.call_from_thread(self._refresh_context_subscribers, event, state)
        except Exception:
            try:
                self._refresh_context_subscribers(event, state)
            except Exception:  # nosec B110
                pass

    def _refresh_context_subscribers(
        self, event: ContextEvent, state: AppContextState
    ) -> None:
        """UI-thread hook for context-driven pane re-renders (populated in Phase 3+).

        Intentionally a no-op in Phase 2 (foundation only): tab panes subscribe to
        ``self.context_bus`` directly in their own ``on_mount`` in later phases; this
        method exists so the app's marshalling path is in place and directly verifiable now.
        """
        return


def run() -> None:
    """Launch the Agentic-ML TUI application.

    Layered-config: the shared-base root ``.env`` was applied at package import
    (``amlas._load_dotenv``); here we overlay the TUI-scoped
    ``.env.tui`` (chat model / UX toggles) BEFORE any ``Settings.load()`` or
    Bedrock client is constructed inside the app, so TUI-specific keys (e.g.
    ``AMLAS_CHAT_MODEL``) take effect. Shell exports still win over both files.
    """
    from ..config.env_loader import load_layered_env

    load_layered_env("tui")

    # Universal error handling: give the process-wide ERROR_LOG a real JSONL
    # file sink for this TUI session + install the process-level uncaught-exception
    # funnel, so every captured/escaped error across the TUI lands in the one error file
    # (+ the CloudWatch mirror in managed modes). Idempotent + best-effort — a telemetry
    # init failure must never block the TUI from launching.
    try:
        from .._runtime_guard import install_error_excepthook
        from ..telemetry.error_log import init_error_log

        init_error_log()
        install_error_excepthook()
    except Exception:  # nosec B110
        pass

    # CREDS-01: neutralize a stale/expired (or profile-superseded) inherited
    # AWS_BEARER_TOKEN_BEDROCK ONCE at the TUI entrypoint — BEFORE any Bedrock
    # client (chat backend, tool-agent swarm, or _build real-model path) is
    # constructed. boto3 prefers that bearer token over the AWS profile's SigV4
    # chain, so a dead token makes the TUI fail with AccessDenied even though the
    # operator's AWS_PROFILE is valid. Neutralizing here (they all share one
    # os.environ) means the whole TUI uses the configured account. Opt out with
    # AMLAS_KEEP_BEDROCK_BEARER_TOKEN=1. (Mirrors the runtime deploy helpers' FIX-10 unset.)
    try:
        from ..config.bedrock_creds import neutralize_stale_bearer_token
        from ..config.settings import Settings

        # BUGFIX (CREDS-01b): use Settings.LOAD() — bare ``Settings()`` does NOT read
        # ``AWS_PROFILE`` from the env, so ``effective_profile`` came back None and the
        # neutralizer KEPT a still-valid inherited bearer token (e.g. Claude Code's
        # ``AWS_BEARER_TOKEN_BEDROCK``). That token then overrode the profile's SigV4
        # chain and the Strands tool-agent failed every turn with
        # ``IncompleteSignatureException`` (user report: "prompting does nothing").
        # ``Settings.load()`` resolves AWS_PROFILE/AMLAS_AWS_PROFILE, so a configured
        # profile now correctly supersedes + drops the inherited token.
        neutralize_stale_bearer_token(Settings.load().aws.effective_profile)
    except Exception:  # nosec B110
        # Never let a diagnostics/creds helper block the TUI from launching.
        pass

    # AWS-first affirmation (user directive 2026-07-12): make the AWS-default backend
    # posture an explicit, logged invariant here, right after the bearer-token
    # fix and before any backend resolves. No-op/local under offline mode; verification-only (never
    # mutates env or pins a selector). Best-effort: it must never block the launch.
    _affirm_aws_first_posture()

    app = AgenticMLApp()
    app.run()
