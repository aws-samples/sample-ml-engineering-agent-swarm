"""Pipeline pane — start, stop, and monitor ML pipeline runs.

Phase 3 (TUI context integration, design §E.3 / §F): the Pipeline pane is the
design's **Config pane** — its "Configuration" section (``#spec-input`` /
``#data-input``) is now driven by the shared ``ContextBus`` instead of being a pair
of isolated local widgets (fixes GT-1 / GA-3). It SUBSCRIBES to the ``spec`` /
``dataset`` (and ``problem``) events and prefills the inputs from the active spec /
dataset — so loading a spec on the Specs tab auto-configures the pipeline here (the
user's headline example). Conversely, submitting a path in either input MUTATES the
bus (``set_spec`` / ``set_dataset`` with ``source="pipeline_tab"``) so a path typed
here becomes the cross-tab active entity. All additive + null-safe: with no bus
(headless / pre-mount) the pane behaves exactly as before.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Log, ProgressBar, Static

from ..context import AppContextState, ContextEvent
from ..widgets.chart_view import ChartViewPanel
from ..widgets.progress import PipelineStagesWidget


class PipelinePane(Static):
    """Pipeline control screen — launch and monitor runs."""

    # DP-2: the id of the most recently submitted background job (set when /run
    # returns ``job_submitted``), so the "Stop" button can cancel the real job.
    _active_job_id: str | None = None

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Vertical(classes="pipeline-content"):
            with Vertical(classes="pipeline-inputs"):
                yield Static("Configuration", classes="section-title")
                yield Input(placeholder="Spec path (e.g., specs/ai4i_binary_failure.md)", id="spec-input")
                yield Input(placeholder="Data path (e.g., data/ai4i/ai4i2020.csv)", id="data-input")
                with Horizontal(classes="pipeline-buttons"):
                    yield Button("▶ Run Pipeline", id="run-btn", variant="success")
                    yield Button("⏹ Stop", id="stop-btn", variant="error")
            with Vertical(classes="pipeline-monitor"):
                yield Static("Stages", classes="section-title")
                yield PipelineStagesWidget()
                yield Static("Progress", classes="section-title")
                yield ProgressBar(total=100, show_eta=True, id="run-progress")
                yield Static("Logs", classes="section-title")
                yield Log(id="pipeline-log", highlight=True, max_lines=200)
                # Agent-addressable chart surface: "chart the run progress on pipeline"
                # lands HERE rather than only on the Dashboard (goal item #3).
                yield ChartViewPanel(tab="pipeline", classes="chart-view-section")

    # ─── ContextBus wiring (design §E.3) ─────────────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prefill the Config inputs from active state.

        Null-safe: with no ``app.context_bus`` (bare-pane / headless) this is a
        no-op so the pane still mounts. Priming on mount means switching TO the
        Pipeline tab after a spec is already active shows it immediately.
        """
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_context(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Bus subscriber — marshal the re-render onto the UI thread.

        Same discipline as the App's ``_on_context_event`` / SpecsPane: the
        ``spec`` / ``dataset`` / ``problem`` events change the Config inputs, and
        (Phase 2 §3b, fixes X-2/X-4) the ``job`` event drives the live pipeline stage
        indicators + progress bar from the REAL ``JobManager`` job instead of the
        cosmetic sleep loop. Never raises (a bad refresh must not break the mutation
        that triggered it).
        """
        if event not in ("spec", "dataset", "problem", "job"):
            return
        try:
            app = self.app
        except Exception:
            return
        target = self._apply_job_stages if event == "job" else self._apply_context
        try:
            app.call_from_thread(target, state)
        except Exception:
            try:
                target(state)
            except Exception:  # nosec B110
                pass

    def _apply_context(self, state: AppContextState) -> None:
        """Prefill ``#spec-input`` / ``#data-input`` from the active spec/dataset.

        Only writes when an active entity exists (never clobbers a user-typed path
        with an empty value), and only when the value actually differs (so the
        subscriber can't fight the user mid-edit). Guarded for pre-mount widgets.
        """
        spec_path = state.spec.path
        ds_path = state.dataset.path
        if spec_path:
            try:
                spec_input = self.query_one("#spec-input", Input)
                if spec_input.value != spec_path:
                    spec_input.value = spec_path
            except Exception:  # nosec B110
                pass
        if ds_path:
            try:
                data_input = self.query_one("#data-input", Input)
                if data_input.value != ds_path:
                    data_input.value = ds_path
            except Exception:  # nosec B110
                pass

    # ─── X-2/X-4 (spec §3b): live stage/progress from the bus ``job`` event ──────

    def _stage_index_for(self, stage_text: str) -> int | None:
        """Map a ``JobManager`` job.stage string → a ``PipelineStagesWidget`` index.

        ``JobManager`` stage strings and the widget's ``STAGES`` labels differ slightly
        (e.g. "Experiment Plan" ↔ "Plan", "Evaluation" ↔ "Evaluate"), so the match is
        case-insensitive **contains** in either direction. Returns ``None`` when no
        stage label corresponds (so an unknown stage never crashes / mis-advances).
        """
        s = (stage_text or "").strip().lower()
        if not s:
            return None
        for i, label in enumerate(PipelineStagesWidget.STAGES):
            low = label.lower()
            if low in s or s in low:
                return i
        return None

    def _apply_job_stages(self, state: AppContextState) -> None:
        """Drive the stage indicators + ``#run-progress`` from the active ``job`` (X-2/X-4).

        Retires the cosmetic ``asyncio.sleep(0.8)`` animation: the Pipeline tab's own
        progress is now the LIVE ``JobManager`` job. Only reacts to the pane's own job:
        when ``self._active_job_id`` is set, a ``job`` event for a different id is
        ignored (so a concurrent unrelated job can't repaint our stages); when it is
        unset we follow the focused job. Maps ``job.stage`` → a stage index and marks
        earlier stages complete / the current running / all complete on ``complete`` /
        the current failed on ``failed``. Every widget access is guarded pre-mount.
        """
        job = state.job
        # Only our own job drives the stages when we have one pinned.
        if self._active_job_id is not None and job.job_id != self._active_job_id:
            return
        try:
            stages = self.query_one(PipelineStagesWidget)
        except Exception:
            return

        status = (job.status or "").lower()
        idx = self._stage_index_for(job.stage)

        try:
            if status == "complete":
                for i in range(len(PipelineStagesWidget.STAGES)):
                    stages.set_stage_complete(i)
            elif status == "failed":
                if idx is not None:
                    stages.set_stage_failed(idx)
            else:  # running / submitted / idle-with-stage
                if idx is not None:
                    for i in range(idx):
                        stages.set_stage_complete(i)
                    stages.set_stage_running(idx)
        except Exception:  # nosec B110
            pass

        # Drive the progress bar from job.progress (100 on complete).
        try:
            progress = self.query_one("#run-progress", ProgressBar)
            pct = 100.0 if status == "complete" else float(job.progress or 0.0)
            progress.update(progress=pct)
        except Exception:  # nosec B110
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submitting a Config input MUTATES the shared bus (fixes GA-3).

        Pressing Enter in ``#spec-input`` publishes ``set_spec`` and in
        ``#data-input`` publishes ``set_dataset`` (both ``source="pipeline_tab"``),
        so a path typed here becomes the cross-tab active entity. Submit-based (not
        on-change) so a programmatic prefill from ``_apply_context`` never triggers a
        publish loop. Null-safe.
        """
        bus = self._bus()
        if bus is None:
            return
        value = (event.value or "").strip()
        if not value:
            return
        try:
            if event.input.id == "spec-input":
                bus.set_spec(path=value, source="pipeline_tab")
            elif event.input.id == "data-input":
                bus.set_dataset(path=value, source="pipeline_tab")
        except Exception:  # nosec B110
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle pipeline button presses."""
        if event.button.id == "run-btn":
            self._start_pipeline_run()
        elif event.button.id == "stop-btn":
            self._stop_pipeline_run()

    def _start_pipeline_run(self) -> None:
        """Start a pipeline run — validates inputs and sends /run to ChatEngine."""
        log = self.query_one("#pipeline-log", Log)
        progress = self.query_one("#run-progress", ProgressBar)
        stages_widget = self.query_one(PipelineStagesWidget)

        spec_input = self.query_one("#spec-input", Input)
        data_input = self.query_one("#data-input", Input)

        # B1: honest example fallbacks pointing at REAL shipped files (the old
        # adsa-classifier.yaml / repos/adsa/train.csv defaults never existed and
        # always tripped the "not found" guard below). A blank input now falls back
        # to a real spec/dataset the repo ships.
        spec_path = spec_input.value.strip() or "specs/ai4i_binary_failure.md"
        data_path = data_input.value.strip() or "data/ai4i/ai4i2020.csv"

        # Validate paths exist
        if not Path(spec_path).exists():
            log.write_line(f"❌ Spec file not found: {spec_path}")
            self.notify(f"Spec file not found: {spec_path}", severity="error")
            return

        if not Path(data_path).exists():
            log.write_line(f"❌ Data file not found: {data_path}")
            self.notify(f"Data file not found: {data_path}", severity="error")
            return

        log.write_line("━━━ Pipeline Run Started ━━━")
        log.write_line(f"  Spec: {spec_path}")
        log.write_line(f"  Data: {data_path}")
        log.write_line("")

        # Run pipeline stages with real path validation
        self.run_worker(self._run_stages(log, progress, stages_widget, spec_path, data_path))

    async def _run_stages(
        self,
        log: Log,
        progress: ProgressBar,
        stages: PipelineStagesWidget,
        spec_path: str,
        data_path: str,
    ) -> None:
        """Execute pipeline stages — sends /run to ChatEngine for real processing.

        Phase 2 (spec §3b, fixes X-2/X-4): the pane's stage indicators + progress bar are
        now driven by the LIVE ``JobManager`` ``job`` event (see :meth:`_apply_job_stages`)
        instead of a cosmetic timer. When the ``/run`` dispatch succeeds we RETURN and let
        the real job events advance the widget. The visual ``asyncio.sleep`` loop is kept
        ONLY as an engine-failure fallback (no engine / headless), so that path behaves
        exactly as before.
        """
        # Try to get the ChatEngine from the Chat pane to invoke real /run
        try:
            from ..screens.chat import ChatPane

            chat_pane = self.app.query_one(ChatPane)
            engine = chat_pane._engine

            # Invoke the real /run handler
            response = await engine.process_message(f"/run {spec_path} {data_path}")
            log.write_line(f"  Engine: {response.action}")

            # DP-2: when /run returns ``job_submitted`` capture the job id so the
            # "Stop" button can cancel the REAL background job (no longer cosmetic).
            if response.action == "job_submitted":
                self._active_job_id = response.data.get("job_id")
                log.write_line(f"  ⟳ Job submitted: {self._active_job_id}")

            if response.action == "error":
                log.write_line(f"  ❌ {response.message}")
                self.notify("Pipeline failed to start", severity="error")
                return

            # X-2/X-4: the engine drove the real run — the pane's stages/progress are now
            # driven by the bus ``job`` event (``_apply_job_stages``). Do NOT run the
            # cosmetic sleep loop; the live job is the source of truth.
            log.write_line("  ⟳ Tracking live job stages…")
            return
        except Exception as e:
            log.write_line(f"  ⚠️ Engine connection failed: {e}")
            log.write_line("  Falling back to visual simulation...")

        # Fallback ONLY (no engine / headless): cosmetic stage progression so the tab
        # still shows movement when no real job events will arrive.
        await self._run_stages_fallback(log, progress, stages)

    async def _run_stages_fallback(
        self,
        log: Log,
        progress: ProgressBar,
        stages: PipelineStagesWidget,
    ) -> None:
        """Cosmetic stage animation — engine-failure fallback ONLY (X-2/X-4).

        Preserved verbatim from the pre-Phase-2 behavior so headless / no-engine paths
        are unchanged. NOT used when a real ``/run`` job is dispatched (that path is
        driven by the bus ``job`` event via :meth:`_apply_job_stages`).
        """
        stage_names = ["Research", "Experiment Plan", "Training", "Evaluation", "Leaderboard", "Decision"]
        for i, stage_name in enumerate(stage_names):
            stages.set_stage_running(i)
            log.write_line(f"⟳ {stage_name}...")
            progress.update(progress=int((i / len(stage_names)) * 100))

            # Simulated work per stage (real pipeline would be async callbacks)
            await asyncio.sleep(0.8)

            stages.set_stage_complete(i)
            log.write_line(f"  ✓ {stage_name} complete")

        progress.update(progress=100)
        log.write_line("")
        log.write_line("━━━ Pipeline Run Complete ━━━")
        log.write_line("  Use /status or /leaderboard in Chat for results.")
        self.notify("Pipeline run complete!", severity="information")

    def _stop_pipeline_run(self) -> None:
        """Stop the current pipeline run — DP-2: cancel the REAL background job.

        Retires the "cosmetic Stop" gap (FINAL §4): if a background job was
        submitted by ``/run``, request its cancellation through the app's shared
        ``JobManager`` (cooperative — transitions a running job to ``cancelled``).
        """
        log = self.query_one("#pipeline-log", Log)

        cancelled = False
        job_manager = getattr(self.app, "job_manager", None)
        if job_manager is not None and self._active_job_id:
            cancelled = job_manager.cancel(self._active_job_id)
            if cancelled:
                log.write_line(f"⏹ Cancellation requested for job `{self._active_job_id}`")
                self._active_job_id = None

        if cancelled:
            log.write_line("⏹ Pipeline stopped by user")
            self.notify("Pipeline cancelled", severity="warning")
        else:
            log.write_line("⏹ Pipeline stopped by user")
            self.notify("Pipeline stopped", severity="warning")
