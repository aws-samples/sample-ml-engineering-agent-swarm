"""Centralized error log for all three runtime modes.

This module provides a single, consistent, inspectable error/log capture across the three
runtime modes the TUI exposes (``offline`` / ``local`` / ``full_aws``). Before this module the
codebase had a rich per-run OTEL/JSONL *span* trace (``telemetry/trace.py``) and a queryable
capability *lineage* ledger (``control_plane/traceability.py``), but **no consolidated ERROR
surface** — an error in any mode either surfaced as a bare log line or a toast and was never
funnelled into one queryable place. This closes that gap.

Design (mirrors the proven ``telemetry/trace.py`` sink pattern — offline JSONL default + a
best-effort durable CloudWatch mirror, both PII-scrubbed):

* :class:`ErrorRecord` — one captured error, TAGGED with the runtime ``mode`` so offline / local /
  full_aws errors live in the same surface but stay attributable per-mode.
* :class:`ErrorLog` — the process-wide capture + inspection surface: ``capture`` (from an
  ``Exception`` or a plain string), ``recent`` / ``all`` / ``query`` / ``summary`` inspection, an
  optional JSONL sink (``read_all`` reads it back), and a ``clear`` for run isolation.
* :class:`CloudWatchErrorLog` — the durable managed variant (local/full_aws): mirrors every record
  to a CloudWatch Logs stream while STILL writing the JSONL fallback, so errors survive the process
  without losing the offline path. Any CloudWatch failure is swallowed — telemetry NEVER crashes
  the agent in any mode.
* :func:`resolve_runtime_mode` — derives the mode from the SAME env signals the TUI uses (an
  explicit ``AMLAS_RUNTIME_MODE`` wins; else the offline flag → ``offline``; else ``local``).
* :func:`resolve_error_log` — selects the durable CloudWatch sink when configured, else the
  offline JSONL surface (byte-for-byte unchanged offline default).
* :data:`ERROR_LOG` — the process-wide singleton + :func:`capture_error` convenience the surfaces,
  flows, and tools emit into (the single funnel point).

Every captured message is PII-scrubbed on capture via the shared
:class:`~amlas.backends.adapters.governance.PIIRedactor` (secure-by-default — raw PII/secrets never
land in the error surface), reusing the exact redactor the guardrail + telemetry layers use.
Pure-Python, Textual-free, boto3-only inside the managed mirror (lazy) — so it imports cheaply and
is directly verifiable headless.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config.aws_region import resolve_aws_region

__all__ = [
    "ERROR_LOG",
    "RUNTIME_MODES",
    "CloudWatchErrorLog",
    "ErrorLog",
    "ErrorRecord",
    "RuntimeMode",
    "capture_error",
    "default_error_log_path",
    "init_error_log",
    "resolve_error_log",
    "resolve_runtime_mode",
]

# The 3 runtime modes the TUI exposes (mirrors ``tui/app.py`` — re-declared here so the telemetry
# layer stays import-light / Textual-free). ``offline`` = no AWS; ``local`` = local execution +
# Bedrock inference; ``full_aws`` = managed backends + Bedrock.
RuntimeMode = Literal["offline", "local", "full_aws"]
RUNTIME_MODES: tuple[RuntimeMode, ...] = ("offline", "local", "full_aws")

# Truthy tokens for the offline env switch (mirrors ``tui/app.py`` + ``registry.py`` semantics).
_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on"})
# The env signals — an explicit runtime-mode pin, then the two offline flags the TUI honors.
_ENV_RUNTIME_MODE = "AMLAS_RUNTIME_MODE"
_ENV_OFFLINE_FLAGS = ("AMLAS_BACKENDS_OFFLINE", "AMLAS_OFFLINE")
# Sink selection (mirrors ``resolve_trace_sink``).
_ENV_ERROR_SINK = "AMLAS_ERROR_SINK"
_ENV_ERROR_LOG_GROUP = "AMLAS_ERROR_LOG_GROUP"
_DEFAULT_ERROR_LOG_GROUP = "/aws/amlas/agent-errors"
# JSONL sink path resolution (mirrors ``telemetry/trace.py``'s per-run JSONL convention:
# an explicit path env wins; else ``<output_dir>/errors-<session>.jsonl``).
_ENV_ERROR_LOG_PATH = "AMLAS_ERROR_LOG_PATH"
_ENV_OUTPUT_DIR = "AMLAS_OUTPUT_DIR"
_ENV_SESSION_ID = "AMLAS_SESSION_ID"
_DEFAULT_OUTPUT_DIR = "./output"
# Local-file pruning: the single local error file is append-only and long-lived
# (esp. for the TUI/tui-agent, which reuse it across sessions), so cap it to the most recent
# N lines to keep it bounded. Tunable; a rewrite runs at most once per ``_PRUNE_EVERY`` writes
# so pruning is cheap. ``AMLAS_ERROR_LOG_MAX_LINES=0`` disables pruning.
_ENV_ERROR_LOG_MAX_LINES = "AMLAS_ERROR_LOG_MAX_LINES"
_DEFAULT_ERROR_LOG_MAX_LINES = 5000
_PRUNE_EVERY = 200


def _env_truthy(name: str) -> bool:
    """True when env var ``name`` holds a truthy token (``1``/``true``/``yes``/``on``)."""
    val = os.environ.get(name)
    return val is not None and val.strip().lower() in _TRUTHY_TOKENS


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int env var; garbage/negative → ``default``."""
    try:
        v = int((os.environ.get(name) or "").strip() or default)
        return v if v >= 0 else default
    except ValueError:
        return default


def resolve_runtime_mode() -> RuntimeMode:
    """Derive the current runtime mode from env (the "all 3 modes" tag source).

    Resolution precedence (mirrors the TUI's own mode derivation, ``tui/app.py``
    ``_derive_default_mode`` — but env-only, so it is Textual-free and usable from any layer):

    1. an EXPLICIT, valid ``AMLAS_RUNTIME_MODE`` (``offline`` / ``local`` / ``full_aws``) WINS;
    2. else, if an offline flag (``AMLAS_OFFLINE`` / ``AMLAS_BACKENDS_OFFLINE``) is truthy → ``offline``;
    3. else the honest default ``local`` (local execution + Bedrock inference).

    An invalid/garbage ``AMLAS_RUNTIME_MODE`` is ignored (falls through to steps 2/3) so a typo can
    never mislabel the surface.
    """
    explicit = (os.environ.get(_ENV_RUNTIME_MODE) or "").strip().lower()
    if explicit in RUNTIME_MODES:
        # ``explicit`` is narrowed to a valid RuntimeMode by the membership check.
        return explicit
    if any(_env_truthy(flag) for flag in _ENV_OFFLINE_FLAGS):
        return "offline"
    return "local"


def _sanitize(text: str | None) -> str:
    """PII-scrub a message before it enters the error surface (secure-by-default).

    Reuses the shared :class:`~amlas.backends.adapters.governance.PIIRedactor` (the exact redactor
    the guardrail Safety gate + the telemetry content-scrub use), so emails / SSNs / cards / phones
    / IPs are replaced with ``[REDACTED:<type>]`` before the message is stored or persisted. A
    lazy import keeps this module free of a backend-adapter import at load (avoids a heavy/circular
    import) and offline-safe. If the redactor is somehow unavailable, the raw text is returned
    rather than dropping the error (an error surface must never silently lose an error).
    """
    if not text:
        return ""
    try:
        from ..backends.adapters.governance import PIIRedactor

        return PIIRedactor().redact(text)
    except Exception:
        return text


@dataclass
class ErrorRecord:
    """One captured error — tagged with the runtime ``mode`` so all 3 modes share one surface.

    ``mode`` is the runtime mode the error occurred in (``offline`` / ``local`` / ``full_aws``);
    ``source`` the logical origin (tool/flow/subsystem id); ``error_type`` the exception class
    name (or a caller-supplied label); ``message`` the PII-scrubbed error text; ``run_id`` the
    optional run correlation id; ``context`` any extra structured detail; ``traceback`` the
    (scrubbed) formatted traceback when captured from a live exception.
    """

    error_id: str
    ts: float
    mode: RuntimeMode
    source: str
    error_type: str
    message: str
    run_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-able dict (for the JSONL sink + inspection tools)."""
        return {
            "error_id": self.error_id,
            "ts": self.ts,
            "mode": self.mode,
            "source": self.source,
            "error_type": self.error_type,
            "message": self.message,
            "run_id": self.run_id,
            "context": dict(self.context),
            "traceback": self.traceback,
        }


class ErrorLog:
    """Process-wide centralized error surface — capture + inspect across all 3 runtime modes.

    ``capture`` records an error (from an ``Exception`` or a plain string), tagging it with the
    runtime ``mode`` (derived when not given) and PII-scrubbing the message; ``recent`` / ``all`` /
    ``query`` / ``summary`` inspect the surface; an optional ``path`` mirrors each record to a JSONL
    sink (``read_all`` reads it back) so the surface survives the process. The in-memory list is the
    source of truth for isolated validation. Best-effort persistence — a sink failure never drops the in-memory
    record.
    """

    sink_name = "jsonl"

    def __init__(self, path: str | Path | None = None) -> None:
        self._records: list[ErrorRecord] = []
        self._path = Path(path) if path else None
        self._writes_since_prune = 0

    def capture(
        self,
        error: BaseException | str,
        *,
        source: str,
        mode: RuntimeMode | str | None = None,
        error_type: str | None = None,
        run_id: str = "",
        context: dict[str, Any] | None = None,
        include_traceback: bool = True,
    ) -> ErrorRecord:
        """Capture one error into the centralized surface, tagged with the runtime mode.

        ``error`` may be a live ``Exception`` (its class name + message + traceback are captured)
        or a plain string (paired with an explicit ``error_type``). ``mode`` defaults to
        :func:`resolve_runtime_mode` when not supplied, so any caller that does not know the mode
        still lands in the right bucket. The message (and traceback) are PII-scrubbed before
        storage. Returns the recorded :class:`ErrorRecord`.
        """
        resolved_mode = self._coerce_mode(mode)
        if isinstance(error, BaseException):
            etype = error_type or type(error).__name__
            message = _sanitize(str(error))
            tb = ""
            if include_traceback and error.__traceback__ is not None:
                tb = _sanitize("".join(traceback.format_exception(type(error), error, error.__traceback__)))
        else:
            etype = error_type or "Error"
            message = _sanitize(str(error))
            tb = ""
        rec = ErrorRecord(
            error_id=uuid.uuid4().hex[:16],
            ts=time.time(),
            mode=resolved_mode,
            source=source,
            error_type=etype,
            message=message,
            run_id=run_id,
            context=dict(context or {}),
            traceback=tb,
        )
        self._records.append(rec)
        self._persist(rec)
        return rec

    @staticmethod
    def _coerce_mode(mode: RuntimeMode | str | None) -> RuntimeMode:
        """Resolve an explicit ``mode`` (validated) or derive it from env; garbage → derived."""
        if mode is not None:
            candidate = str(mode).strip().lower()
            if candidate in RUNTIME_MODES:
                return candidate
        return resolve_runtime_mode()

    def _persist(self, rec: ErrorRecord) -> None:
        """Mirror ``rec`` to the JSONL sink when configured (best-effort — never raises)."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
            self._maybe_prune()
        except Exception:  # nosec B110
            pass

    def _maybe_prune(self) -> None:
        """Cap the local JSONL file to the most recent N lines.

        The single local error file is append-only and reused across sessions (esp. the
        TUI/tui-agent), so it would grow unbounded. Every ``_PRUNE_EVERY`` writes we rewrite the
        file keeping the last ``AMLAS_ERROR_LOG_MAX_LINES`` lines. Best-effort — a prune failure
        never drops the just-written record (it stays on disk). ``max_lines=0`` disables pruning.
        """
        self._writes_since_prune += 1
        if self._path is None or self._writes_since_prune < _PRUNE_EVERY:
            return
        self._writes_since_prune = 0
        try:
            max_lines = _env_int(_ENV_ERROR_LOG_MAX_LINES, _DEFAULT_ERROR_LOG_MAX_LINES)
            if max_lines <= 0:
                return
            lines = self._path.read_text(encoding="utf-8").splitlines()
            if len(lines) <= max_lines:
                return
            kept = lines[-max_lines:]
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
            tmp.replace(self._path)  # atomic swap
        except Exception:  # nosec B110
            pass

    # ── inspection ────────────────────────────────────────────────────────────
    def all(self) -> list[ErrorRecord]:
        """Return every captured error (insertion order — oldest first)."""
        return list(self._records)

    def recent(self, n: int = 10) -> list[ErrorRecord]:
        """Return the ``n`` most recent errors, newest first."""
        if n <= 0:
            return []
        return list(reversed(self._records[-n:]))

    def query(
        self,
        *,
        mode: RuntimeMode | str | None = None,
        source: str | None = None,
        error_type: str | None = None,
        run_id: str | None = None,
    ) -> list[ErrorRecord]:
        """Return all records matching the given ``mode`` / ``source`` / ``error_type`` / ``run_id`` (AND)."""
        out = self._records
        if mode is not None:
            want = str(mode).strip().lower()
            out = [r for r in out if r.mode == want]
        if source is not None:
            out = [r for r in out if r.source == source]
        if error_type is not None:
            out = [r for r in out if r.error_type == error_type]
        if run_id is not None:
            out = [r for r in out if r.run_id == run_id]
        return list(out)

    def summary(self) -> dict[str, Any]:
        """Roll the surface up: total count + per-mode + per-source + per-type breakdowns."""
        by_mode: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in self._records:
            by_mode[r.mode] = by_mode.get(r.mode, 0) + 1
            by_source[r.source] = by_source.get(r.source, 0) + 1
            by_type[r.error_type] = by_type.get(r.error_type, 0) + 1
        return {
            "total": len(self._records),
            "by_mode": by_mode,
            "by_source": by_source,
            "by_type": by_type,
        }

    def read_all(self) -> list[dict[str, Any]]:
        """Read back all persisted records from the JSONL sink (for isolated validation + post-run analysis)."""
        if self._path is None or not self._path.exists():
            return []
        return [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def clear(self) -> None:
        """Reset the in-memory surface (used by tests for isolation)."""
        self._records.clear()


class CloudWatchErrorLog(ErrorLog):
    """Durable CloudWatch Logs error sink with an offline JSONL fallback (local/full_aws).

    Mirrors every captured :class:`ErrorRecord` to a CloudWatch Logs stream (``put_log_events``)
    AND keeps writing the JSONL file, so errors survive the process without losing the offline
    path. The Logs client is injectable for isolated validation; the log group/stream are created best-effort.
    Any CloudWatch failure is swallowed (the JSONL fallback is canonical) — telemetry NEVER crashes
    the agent, matching :class:`~amlas.telemetry.trace.CloudWatchTraceLogger`.
    """

    sink_name = "cloudwatch"

    def __init__(
        self,
        path: str | Path,
        *,
        log_group: str,
        log_stream: str | None = None,
        logs_client: Any = None,
        region: str | None = None,
    ) -> None:
        super().__init__(path)
        self._log_group = log_group
        self._log_stream = log_stream or f"errors-{uuid.uuid4().hex[:8]}"
        self._logs = logs_client
        self._region = region or resolve_aws_region(required=False)
        self._seq_token: str | None = None
        self._ensure_log_targets()

    def _client(self) -> Any:
        if self._logs is None:
            import boto3

            self._logs = boto3.client("logs", region_name=resolve_aws_region(self._region))
        return self._logs

    def _ensure_log_targets(self) -> None:
        try:
            c = self._client()
            try:
                c.create_log_group(logGroupName=self._log_group)
            except Exception:  # nosec B110
                pass
            try:
                c.create_log_stream(logGroupName=self._log_group, logStreamName=self._log_stream)
            except Exception:  # nosec B110
                pass
        except Exception:  # nosec B110
            pass

    def _persist(self, rec: ErrorRecord) -> None:
        # Always write the JSONL fallback first (canonical, offline-survivable).
        super()._persist(rec)
        # Then mirror to CloudWatch (durable/queryable). Best-effort — never raise.
        try:
            c = self._client()
            kwargs: dict[str, Any] = {
                "logGroupName": self._log_group,
                "logStreamName": self._log_stream,
                "logEvents": [{
                    "timestamp": int(rec.ts * 1000) or int(time.time() * 1000),
                    "message": json.dumps(rec.to_dict(), default=str),
                }],
            }
            if self._seq_token:
                kwargs["sequenceToken"] = self._seq_token
            resp = c.put_log_events(**kwargs)
            self._seq_token = resp.get("nextSequenceToken")
        except Exception:  # nosec B110
            pass


def resolve_error_log(
    *,
    path: str | Path,
    log_group: str | None = None,
    log_stream: str | None = None,
    logs_client: Any = None,
    region: str | None = None,
    mode: RuntimeMode | str | None = None,
) -> ErrorLog:
    """Select the error sink by RUNTIME MODE:

    * **offline** → local JSONL file ONLY (no AWS reachable — the single, offline-survivable
      error file for the TUI + tui-agent);
    * **local** / **full_aws** → durable **CloudWatch** mirror **+** the local JSONL file
      (so a cloud run is queryable, and the TUI/tui-agent still always have their local file
      even if the box goes offline mid-run).

    An explicit ``AMLAS_ERROR_SINK`` still wins as an operator escape hatch: ``=jsonl`` forces
    local-only even in a cloud mode; ``=cloudwatch`` forces the mirror. Absent that override,
    the mode drives it. CloudWatch is only selected when a log group is resolvable; otherwise
    it safely degrades to the local file. Mirrors :func:`resolve_trace_sink`.
    """
    resolved_mode = mode if mode is not None else resolve_runtime_mode()
    resolved_mode = str(resolved_mode).strip().lower()
    group = log_group or os.environ.get(_ENV_ERROR_LOG_GROUP, _DEFAULT_ERROR_LOG_GROUP)

    explicit_sink = (os.environ.get(_ENV_ERROR_SINK) or "").strip().lower()
    if explicit_sink == "jsonl":
        want_cloud = False
    elif explicit_sink == "cloudwatch":
        want_cloud = True
    else:
        # Mode-driven default: mirror to CloudWatch in every mode EXCEPT offline.
        want_cloud = resolved_mode != "offline"

    if want_cloud and group:
        return CloudWatchErrorLog(path, log_group=group, log_stream=log_stream,
                                  logs_client=logs_client, region=region)
    return ErrorLog(path)


def default_error_log_path() -> Path:
    """Resolve the default per-session JSONL error-file path (the single error file).

    Mirrors ``telemetry/trace.py``'s per-run JSONL convention so the error surface has a
    real, inspectable file per process (the #1 gap: the ``ERROR_LOG`` singleton is
    memory-only until initialized). Resolution:

    1. an explicit ``AMLAS_ERROR_LOG_PATH`` wins (an operator/CI pin);
    2. else ``<output_dir>/errors-<session>.jsonl`` where ``output_dir`` is
       ``AMLAS_OUTPUT_DIR`` (default ``./output``) and ``<session>`` is
       ``AMLAS_SESSION_ID`` when set, else a short random token.

    Pure path arithmetic — creates no directory/file (``init_error_log``/the sink's
    ``_persist`` create the parent lazily on the first captured error).
    """
    explicit = (os.environ.get(_ENV_ERROR_LOG_PATH) or "").strip()
    if explicit:
        return Path(explicit)
    out_dir = (os.environ.get(_ENV_OUTPUT_DIR) or _DEFAULT_OUTPUT_DIR).strip() or _DEFAULT_OUTPUT_DIR
    session = (os.environ.get(_ENV_SESSION_ID) or "").strip() or uuid.uuid4().hex[:8]
    return Path(out_dir) / f"errors-{session}.jsonl"


def init_error_log(
    *,
    path: str | Path | None = None,
    force: bool = False,
    logs_client: Any = None,
    region: str | None = None,
) -> ErrorLog:
    """Point the process-wide :data:`ERROR_LOG` at a REAL sink at process start (the #1 fix).

    By default the module-level singleton is a memory-only ``ErrorLog()`` (no path), so
    every captured error is lost on exit. Every entrypoint (CLI / TUI / AgentCore runtime)
    calls this ONCE at startup so all funneled errors land in one JSONL file per session
    (``<output>/errors-<session>.jsonl`` by default) plus the durable CloudWatch mirror in
    managed modes (via :func:`resolve_error_log` when ``AMLAS_ERROR_SINK=cloudwatch``).

    Idempotent: once the singleton has a real path, a later call is a no-op unless
    ``force`` is set — so a CLI that also launches the TUI does not clobber the sink. The
    resolved log REPLACES the module global (any records already captured into the old
    memory-only singleton are best-effort re-mirrored to the new sink). Best-effort +
    never raises — telemetry init must never block an entrypoint.

    Returns the active :data:`ERROR_LOG` (re-pointed on init, unchanged on a no-op).
    """
    global ERROR_LOG
    try:
        if not force and getattr(ERROR_LOG, "_path", None) is not None:
            return ERROR_LOG
        target = Path(path) if path is not None else default_error_log_path()
        # Mode-driven sink: offline → local file only; local/full_aws → CloudWatch
        # mirror + local file. resolve_error_log derives the mode from env when not passed.
        new_log = resolve_error_log(path=target, logs_client=logs_client, region=region)
        # Best-effort: replay any errors captured before init into the new sink so the
        # single error file is complete even for an early failure.
        try:
            for rec in ERROR_LOG.all():
                new_log._records.append(rec)
                new_log._persist(rec)
        except Exception:  # nosec B110
            pass
        ERROR_LOG = new_log
        return ERROR_LOG
    except Exception:
        # Never let telemetry init crash an entrypoint — keep the current singleton.
        return ERROR_LOG


# The process-wide singleton the surfaces / flows / tools funnel errors into. Memory-only
# until an entrypoint calls :func:`init_error_log` (which gives it a real JSONL file sink).
ERROR_LOG = ErrorLog()


def capture_error(
    error: BaseException | str,
    *,
    source: str,
    mode: RuntimeMode | str | None = None,
    error_type: str | None = None,
    run_id: str = "",
    context: dict[str, Any] | None = None,
    include_traceback: bool = True,
    log: ErrorLog | None = None,
) -> ErrorRecord:
    """Capture an error into the process-wide :data:`ERROR_LOG` (the single funnel point).

    The convenience any layer calls without threading the singleton — mirrors
    :func:`~amlas.control_plane.traceability.record_capability_trace`. Pass ``log`` to target a
    specific :class:`ErrorLog` (tests). Returns the recorded :class:`ErrorRecord`.
    """
    return (log or ERROR_LOG).capture(
        error,
        source=source,
        mode=mode,
        error_type=error_type,
        run_id=run_id,
        context=context,
        include_traceback=include_traceback,
    )
