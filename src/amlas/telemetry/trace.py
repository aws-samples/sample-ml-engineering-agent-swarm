"""JSONL trace logger — open-data-scientist pattern.

Every agent emits spans for each tool call, model call, and phase transition.
File at `runs/<run_id>/trace.jsonl`. Each line is an independent JSON object
for `tail -f`-friendly streaming.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amlas.config._env_compat import env_with_compat

from ..config.aws_region import resolve_aws_region

# SEC-A24 (OWASP A09 / LLM02): the OTEL standard env that toggles GenAI message-content
# capture. The infra layer already defaults it to "false" (CDK; WA-04 §8 #3); these helpers
# enforce the SAME secure default at the application telemetry layer AND add the redact-on-
# capture step the audit flagged as missing (so even when an operator turns capture ON, raw
# PII/secrets never reach a span/log).
_OTEL_CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})


def content_capture_enabled() -> bool:
    """True only if OTEL GenAI message-content capture is EXPLICITLY enabled (SEC-A24).

    Reads ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` and defaults **OFF**
    (redact-by-default in prod). Recognizes the standard truthy tokens
    (``1``/``true``/``yes``/``on``, case-insensitive); anything else (incl. unset/empty)
    is OFF.
    """
    return os.environ.get(_OTEL_CAPTURE_ENV, "").strip().lower() in _TRUE_TOKENS


def redact_for_telemetry(text: str | None) -> str | None:
    """Gate + redact message content for telemetry emission (SEC-A24).

    * Capture OFF (default): returns ``None`` — content is DROPPED, never emitted.
    * Capture ON (explicit opt-in): returns the text with all detected PII redacted via
      :class:`~amlas.backends.adapters.governance.PIIRedactor`, so raw PII/
      secrets never reach a span attribute or log event even when capture is enabled.

    Callers attach the RESULT (not the raw text) to a span attribute / log message; a
    ``None`` result means "do not set the content attribute".
    """
    if text is None or not content_capture_enabled():
        return None
    # Lazy import keeps the telemetry module free of a backend-adapter import at module load
    # (avoids a circular import) and matches the offline-safe, dependency-free redactor.
    from ..backends.adapters.governance import PIIRedactor

    return PIIRedactor().redact(text)


@dataclass
class TraceSpan:
    span_id: str
    parent_id: str | None
    name: str
    kind: str  # "tool" | "model" | "phase" | "agent"
    attributes: dict[str, Any] = field(default_factory=dict)
    start_ts: float = 0.0
    end_ts: float = 0.0
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "attributes": self.attributes,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
        }


class TraceLogger:
    """Append-only JSONL trace emitter. Thread-safe via file-append semantics."""

    sink_name = "jsonl"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stack: list[str] = []

    @contextmanager
    def span(self, name: str, kind: str, **attributes: Any) -> Iterator[TraceSpan]:
        span = TraceSpan(
            span_id=uuid.uuid4().hex[:16],
            parent_id=self._stack[-1] if self._stack else None,
            name=name,
            kind=kind,
            attributes=dict(attributes),
            start_ts=time.time(),
        )
        self._stack.append(span.span_id)
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            span.end_ts = time.time()
            self._stack.pop()
            self._write(span)

    def _write(self, span: TraceSpan) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(span.to_dict(), default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read back all spans from disk — for isolated validation and post-run analysis."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


class CloudWatchTraceLogger(TraceLogger):
    """WA-10 — durable CloudWatch Logs trace sink with an offline JSONL fallback.

    The audit found the offline default sink was on-disk JSONL only (not durable/queryable). This
    sink mirrors every span to a CloudWatch Logs stream (``put_log_events``) AND keeps writing the
    JSONL file, so traces survive the process without losing the offline path. The Logs client is
    injectable for isolated validation; the log group/stream are created best-effort. Any CloudWatch failure is
    swallowed (the JSONL fallback is canonical) so telemetry never crashes the agent.
    """

    sink_name = "cloudwatch"

    def __init__(self, path: str | Path, *, log_group: str, log_stream: str | None = None,
                 logs_client: Any = None, region: str | None = None) -> None:
        super().__init__(path)
        self._log_group = log_group
        self._log_stream = log_stream or f"traces-{uuid.uuid4().hex[:8]}"
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
                # CloudWatch Logs encrypts at rest with AWS-managed keys by
                # default; associate a CMK (logs:AssociateKmsKey) if your
                # compliance baseline requires customer-managed encryption.
                c.create_log_group(logGroupName=self._log_group)
            except Exception:  # nosec B110
                pass
            try:
                c.create_log_stream(logGroupName=self._log_group, logStreamName=self._log_stream)
            except Exception:  # nosec B110
                pass
        except Exception:  # nosec B110
            pass

    def _write(self, span: TraceSpan) -> None:
        # Always write the JSONL fallback first (canonical, offline-survivable).
        super()._write(span)
        # Then mirror to CloudWatch (durable/queryable). Best-effort — never raise.
        try:
            c = self._client()
            kwargs: dict[str, Any] = {
                "logGroupName": self._log_group,
                "logStreamName": self._log_stream,
                "logEvents": [{
                    "timestamp": int(span.end_ts * 1000) or int(time.time() * 1000),
                    "message": json.dumps(span.to_dict(), default=str),
                }],
            }
            if self._seq_token:
                kwargs["sequenceToken"] = self._seq_token
            resp = c.put_log_events(**kwargs)
            self._seq_token = resp.get("nextSequenceToken")
        except Exception:  # nosec B110
            pass


def resolve_trace_sink(*, path: str | Path, log_group: str | None = None,
                       log_stream: str | None = None, logs_client: Any = None,
                       region: str | None = None) -> TraceLogger:
    """Select the trace sink (WA-10): durable CloudWatch when configured, else offline JSONL.

    ``AMLAS_TRACE_SINK=cloudwatch`` (with a ``log_group`` / ``AMLAS_TRACE_LOG_GROUP``) selects the
    :class:`CloudWatchTraceLogger` durable sink (which still writes the JSONL fallback); anything
    else returns the offline :class:`TraceLogger` — so the offline default is byte-for-byte
    unchanged and a deployed runtime gets durable, queryable traces.
    """
    sink = env_with_compat("AMLAS_TRACE_SINK", "jsonl").strip().lower()
    group = log_group or env_with_compat("AMLAS_TRACE_LOG_GROUP", "/aws/amlas/agent-traces")
    if sink == "cloudwatch" and group:
        return CloudWatchTraceLogger(path, log_group=group, log_stream=log_stream,
                                     logs_client=logs_client, region=region)
    return TraceLogger(path)
