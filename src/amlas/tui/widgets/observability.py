"""Phase 4 — ``ObservabilityPanel``: CloudWatch logs / metrics / spend on the Dashboard.

This is the ``src/amlas/tui/widgets/observability.py`` widget named in the design
(``docs/tui-context-integration-design.md`` §E.7 / §F Phase 4). It closes GO-1 /
GO-2 — the Dashboard surfaces **no** CloudWatch info today even though the tools
exist and are unused by the UI.

**Sources (design §E.7 — the already-existing, unused monitoring tools):**

* ``get_recent_logs`` (T-MON-04, CloudWatch Logs),
* ``get_metrics`` (T-MON-03), and
* ``track_inference_spend`` (T-MON-09, CostAdvisor)

from :mod:`amlas.tools.monitor_neutral`. Each resolves the ``MonitoringBackend`` PORT
via the registry: **live** CloudWatch when creds are present, the local in-process
``OTELTraceBackend`` **offline** — so the panel shows the local observability store and
**never crashes** with no creds (fixes GO-1/GO-2).

**Refresh discipline (design §E.7):** identical to
:class:`~amlas.tui.widgets.status_bar.SystemHealthWidget` — a ``@work(thread=True)``
refresh resolves the sources off the UI thread and repaints via ``call_from_thread``.
The scope re-points to ``state.model.endpoint`` (or a default namespace when no
endpoint) when the bus publishes a new ``model``.

**Dependency-injection seam (design §G.3):** the three sources are instance callables
(``_log_reader`` / ``_metric_reader`` / ``_spend_reader``) so offline validation inject
deterministic, offline fakes — **no real AWS calls in tests**. A raising source is
swallowed (defensive fetch), so the panel degrades to a placeholder rather than
crashing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.widgets import Static

logger = logging.getLogger(__name__)

__all__ = ["ObservabilityPanel"]

# Default namespace/metric surfaced when no endpoint is active (design §E.7).
_DEFAULT_NAMESPACE = "AWS/SageMaker"
_DEFAULT_METRIC = "Invocations"
_DEFAULT_LOG_GROUP = "/aws/amlas/agent-traces"
_MAX_LOG_LINES = 5
_MAX_METRIC_POINTS = 3


# ─── Default sources: the real monitor_neutral tools, resolved + defensive ──────
#
# Lazy import (the tool modules import ``strands``) + resolve the underlying callable
# (real Strands wraps the fn in a DecoratedFunctionTool; the confcompatibility shim leaves it as
# the plain function). Any failure returns the tool's own empty shape so a missing-creds
# / offline call never raises out of the widget.


def _resolve_tool(tool: Any) -> Callable[..., Any]:
    for attr in ("_tool_func", "func", "__wrapped__", "_func", "original"):
        candidate = getattr(tool, attr, None)
        if callable(candidate):
            return candidate  # type: ignore[no-any-return]
    return tool  # type: ignore[no-any-return]


def _default_log_reader(**kwargs: Any) -> dict[str, Any]:
    try:
        from ...tools.monitor_neutral import get_recent_logs

        return _resolve_tool(get_recent_logs)(**kwargs)  # type: ignore[no-any-return]
    except Exception as exc:  # pragma: no cover — offline/degraded
        logger.debug("default log reader degraded: %s", exc)
        return {"events": []}


def _default_metric_reader(**kwargs: Any) -> dict[str, Any]:
    try:
        from ...tools.monitor_neutral import get_metrics

        return _resolve_tool(get_metrics)(**kwargs)  # type: ignore[no-any-return]
    except Exception as exc:  # pragma: no cover — offline/degraded
        logger.debug("default metric reader degraded: %s", exc)
        return {"datapoints": []}


def _default_spend_reader(**kwargs: Any) -> dict[str, Any]:
    try:
        from ...tools.monitor_neutral import track_inference_spend

        return _resolve_tool(track_inference_spend)(**kwargs)  # type: ignore[no-any-return]
    except Exception as exc:  # pragma: no cover — offline/degraded
        logger.debug("default spend reader degraded: %s", exc)
        return {"run_rate": 0.0}


class ObservabilityPanel(Static):
    """CloudWatch logs + metrics + inference-spend for the active endpoint (§E.7)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # DI seam (design §G.3): overridable per-instance so offline validation inject
        # deterministic offline fakes — no real AWS.
        self._log_reader: Callable[..., dict[str, Any]] = _default_log_reader
        self._metric_reader: Callable[..., dict[str, Any]] = _default_metric_reader
        self._spend_reader: Callable[..., dict[str, Any]] = _default_spend_reader

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder(), id="observability-content")

    def on_mount(self) -> None:
        """Kick off the initial (threaded) observability refresh — never blocks the UI."""
        try:
            self._run_refresh_worker()
        except Exception:
            # No running app (bare widget) — degrade to a direct, guarded refresh.
            try:
                self._render_observability(self._fetch_observability())
            except Exception:  # nosec B110
                pass

    # ─── Public refresh (the manual / reset hook) ─────────────────────────────────

    def refresh_observability(self) -> None:
        """Re-fetch + repaint the panel. Synchronous + guarded (never raises).

        The integration/offline validation call this directly after injecting deterministic
        sources; the real app also calls it on a ``model`` bus event to re-point to a
        new endpoint. It fetches (defensively) and renders on the calling thread — the
        initial mount uses the ``@work(thread=True)`` worker for real-AWS latency.
        """
        try:
            data = self._fetch_observability()
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("refresh_observability fetch degraded: %s", exc)
            data = {"logs": {}, "metrics": {}, "spend": {}}
        try:
            self._render_observability(data)
        except Exception:  # nosec B110
            pass

    @work(thread=True, exclusive=True)
    def _run_refresh_worker(self) -> None:
        """Threaded refresh (SystemHealthWidget discipline): fetch off-thread, paint via call_from_thread."""
        data = self._fetch_observability()
        try:
            self.app.call_from_thread(self._render_observability, data)
        except Exception:
            try:
                self._render_observability(data)
            except Exception:  # nosec B110
                pass

    # ─── Fetch (defensive; each source swallowed) ────────────────────────────────

    def _active_endpoint(self) -> str | None:
        """Return the active model endpoint from the bus (or ``None``), null-safe."""
        try:
            bus = getattr(self.app, "context_bus", None)
            if bus is None:
                return None
            return bus.state().model.endpoint  # type: ignore[no-any-return]
        except Exception:
            return None

    @staticmethod
    def _safe_call(fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Call a source callable, swallowing any exception into an empty dict."""
        try:
            out = fn(**kwargs)
            return out if isinstance(out, dict) else {}
        except Exception as exc:
            logger.debug("observability source raised (swallowed): %s", exc)
            return {}

    def _fetch_observability(self) -> dict[str, Any]:
        """Resolve logs/metrics/spend for the active endpoint (or the default scope)."""
        endpoint = self._active_endpoint()
        namespace = _DEFAULT_NAMESPACE
        dims = {"EndpointName": endpoint} if endpoint else None
        logs = self._safe_call(self._log_reader, log_group=_DEFAULT_LOG_GROUP, filter="", since="1h")
        metrics = self._safe_call(
            self._metric_reader, namespace=namespace, metric=_DEFAULT_METRIC, dims=dims, period=300
        )
        spend = self._safe_call(self._spend_reader, endpoint=endpoint or "", window="7d")
        return {"logs": logs, "metrics": metrics, "spend": spend, "endpoint": endpoint}

    # ─── Render (guarded pre-mount; offline-safe placeholder) ────────────────────

    @staticmethod
    def _placeholder() -> str:
        return "No observability data yet (offline) —\n(logs/metrics appear when an endpoint is active)"

    @staticmethod
    def _events_of(logs: Any) -> list[dict[str, Any]]:
        if isinstance(logs, dict):
            return logs.get("events") or []
        if isinstance(logs, list):
            return logs
        return []

    @staticmethod
    def _datapoints_of(metrics: Any) -> list[dict[str, Any]]:
        if isinstance(metrics, dict):
            return metrics.get("datapoints") or []
        if isinstance(metrics, list):
            return metrics
        return []

    def _render_observability(self, data: dict[str, Any]) -> None:
        """Repaint ``#observability-content`` from the fetched data (guarded pre-mount)."""
        logs = data.get("logs", {})
        metrics = data.get("metrics", {})
        spend = data.get("spend", {})
        endpoint = data.get("endpoint")

        events = self._events_of(logs)
        datapoints = self._datapoints_of(metrics)
        run_rate = spend.get("run_rate") if isinstance(spend, dict) else None
        spend_source = spend.get("source") if isinstance(spend, dict) else None

        lines: list[str] = []
        scope = endpoint or "default namespace"
        lines.append(f"☁️ Observability — {scope}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Recent CloudWatch log events (ERROR/WARN emphasized).
        if events:
            lines.append("Recent logs:")
            for ev in events[:_MAX_LOG_LINES]:
                msg = ev.get("message") if isinstance(ev, dict) else str(ev)
                if not msg:
                    continue
                marker = "  🔴" if ("ERROR" in str(msg) or "WARN" in str(msg)) else "  •"
                lines.append(f"{marker} {str(msg)[:120]}")
        else:
            lines.append("  No recent logs (offline / no endpoint) —")

        # Metric datapoints.
        metric_name = metrics.get("metric") if isinstance(metrics, dict) else None
        if datapoints:
            latest = datapoints[-1]
            value = latest.get("value") if isinstance(latest, dict) else latest
            lines.append("")
            lines.append(f"Metric {metric_name or _DEFAULT_METRIC}: {value}")
            if len(datapoints) > 1:
                recent = ", ".join(
                    str(dp.get("value") if isinstance(dp, dict) else dp)
                    for dp in datapoints[-_MAX_METRIC_POINTS:]
                )
                lines.append(f"  last {min(len(datapoints), _MAX_METRIC_POINTS)}: {recent}")
        else:
            lines.append("")
            lines.append(f"Metric {metric_name or _DEFAULT_METRIC}: — (no datapoints)")

        # Inference spend run-rate — labeled ACTUAL vs ESTIMATE (be-cautious hardening).
        # A Cost-Explorer figure is a REAL measured spend: shown precisely, labeled "actual".
        # Anything else (the offline synthetic / local price-book run-rate) is a GUESS: rendered
        # as a caveated, upward-biased range via cost_confidence — never a bare confident figure.
        if isinstance(run_rate, (int, float)) and run_rate:
            lines.append("")
            if spend_source == "cost_explorer":
                lines.append(f"Inference spend (7d): ${run_rate:.2f}/run-rate (actual, Cost Explorer)")
            else:
                try:
                    from ...control_plane.cost_confidence import band_for

                    band = band_for(float(run_rate), source="estimate")
                    lines.append(f"Inference spend (7d): {band.human()}")
                except Exception:  # pragma: no cover — defensive; never crash the panel
                    lines.append(f"Inference spend (7d): ~${run_rate:.2f}/run-rate (rough estimate)")

        text = "\n".join(lines)
        try:
            content = self.query_one("#observability-content", Static)
        except Exception:
            return
        try:
            content.update(text)
        except Exception:  # nosec B110
            pass
