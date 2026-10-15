"""B1 + B4 — client-side cloud dispatch: NL → spec → stage → invoke → stream.

This is the operator command that Phase 13 found missing (B1): the ONLY
`invoke_agent_runtime` call outside the async-resume Lambda. It targets the
deployed `app.py` agentic orchestrator (B5), not the offline `MLPipeline`.

Flow (`run_cloud_experiment`):
    1. NL → schema-valid spec YAML (`cloud.spec_builder.build_spec_yaml`).
    2. Upload the chosen dataset to the SESSION bucket; rewrite the spec's
       golden_dataset to that `s3://` URI so the runtime stages it (N1).
    3. Build the app.py payload (`spec_text` + `stage_data` directive + caps).
    4. `bedrock-agentcore InvokeAgentRuntime` against the deployed runtime.
    5. Parse the streamed newline-/SSE-delimited JSON events to completion.

Boto3-only (no strands / bedrock_agentcore) so it runs on any operator machine
and is directly verifiable with real AWS + an injected fake runtime client.
SEC-01 (Phase 3d): the runtime ARN, session bucket, and region are NO LONGER
hardcoded to a live account. They are resolved from the environment at call time
(``AGENTCORE_RUNTIME_ARN``, ``AMLAS_SESSION_BUCKET``/``SESSION_BUCKET``) — and
the region comes from the explicit setup/environment/profile selection.
When a required value is unset, ``run_cloud_experiment`` raises a clear
``ValueError`` (fail fast — never a live default with a real account/bucket).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amlas.config._env_compat import env_with_compat

from ..config.aws_region import resolve_aws_region

# SEC-01: region is sourced from the single canonical region constant
# (config.runtime.DEFAULT_REGION) instead of a re-declared "us-west-2" literal.
from .spec_builder import build_spec_yaml, sanitize_project_id

# SEC-01: NO hardcoded account ID / runtime ARN / session bucket. These are
# resolved from the environment; empty ⇒ the operator MUST supply a value via
# env or the CLI `--runtime-arn` / `--session-bucket` overrides. Kept as
# module-level names for backward-compatible imports (e.g. the `run cloud` CLI
# command), but they NEVER default to a live identifier.
DEFAULT_RUNTIME_ARN = os.environ.get("AGENTCORE_RUNTIME_ARN", "")
DEFAULT_SESSION_BUCKET = env_with_compat("AMLAS_SESSION_BUCKET", os.environ.get("SESSION_BUCKET", ""))

_GOLDEN_RE = re.compile(r"^(\s*golden_dataset:\s*).*$", re.MULTILINE)


def _resolve_runtime_arn() -> str:
    """Resolve the AgentCore runtime ARN from env at instantiation time (SEC-01)."""
    return os.environ.get("AGENTCORE_RUNTIME_ARN", "")


def _resolve_session_bucket() -> str:
    """Resolve the session S3 bucket from env at instantiation time (SEC-01)."""
    return env_with_compat("AMLAS_SESSION_BUCKET", os.environ.get("SESSION_BUCKET", ""))


def resolve_runtime_arn(
    explicit: str = "", *, region: str | None = None, ssm_getter: Any | None = None
) -> str:
    """Entry-point resolver: explicit → env → SSM ``/amlas/agentcore-runtime-arn``.

    FIX-13 (H8 family): the deploy step publishes the LIVE runtime ARN to SSM
    (matrix §5 step 4: "so dispatch + async callback resolve it"). The operator
    surfaces — the ``amlas run cloud`` CLI, the MO-046 story handler, and the
    exploration_loop dispatch — call this so they resolve the current runtime
    WITHOUT a manual ``AGENTCORE_RUNTIME_ARN`` export (and without ever binding a
    stale/dead id). ``DispatchConfig``'s env-only ``default_factory`` is unchanged
    (the SEC-01 pure-logic contract) — this SSM fallback sits ABOVE it, at the
    entry point, and passes the resolved value in explicitly. Empty ⇒ the caller's
    fail-fast ValueError still fires (never a live default).
    """
    if explicit and explicit.strip():
        return explicit.strip()
    from ..config import externalized

    return externalized.resolve("agentcore_runtime_arn", region=region, ssm_getter=ssm_getter)


def resolve_session_bucket(
    explicit: str = "", *, region: str | None = None, ssm_getter: Any | None = None
) -> str:
    """Entry-point resolver: explicit → env → SSM ``/amlas/session-bucket`` (FIX-13)."""
    if explicit and explicit.strip():
        return explicit.strip()
    from ..config import externalized

    return externalized.resolve("session_bucket", region=region, ssm_getter=ssm_getter)


@dataclass(frozen=True)
class DispatchConfig:
    # SEC-01: env-sourced at instantiation (default_factory) so callers/callers can
    # set AGENTCORE_RUNTIME_ARN / AMLAS_SESSION_BUCKET and have a fresh
    # DispatchConfig() pick them up. Empty ⇒ fail fast in run_cloud_experiment.
    runtime_arn: str = field(default_factory=_resolve_runtime_arn)
    session_bucket: str = field(default_factory=_resolve_session_bucket)
    region: str | None = None
    qualifier: str = "DEFAULT"
    # Cheap-smoke compute caps (threaded into the spec budget).
    max_rows: int | None = None
    max_retreats: int = 0
    trials: int = 1
    wall_clock_minutes: int = 5
    # Phase 18 — explicit acceptance lever. When set (e.g. ">= 0.15") it OVERRIDES
    # the inferred default delta rule in spec_builder, raising the acceptance bar
    # to FORCE the retreat/iterate loop. None ⇒ legacy inferred default (">= 0.02").
    acceptance_delta: str | None = None
    no_subgroup_regression: bool = False
    # Regression-path correctness (weather E2E): for a time-series regression the
    # split MUST be time-based (never stratified/random — that leaks future→past),
    # and the baseline recorded in the spec is an ERROR metric (e.g. RMSE ~30), not
    # the classification default 0.55. Threaded into build_spec_yaml so the Arbiter
    # grades the delta against a meaningful baseline. Defaults preserve legacy
    # classification behavior.
    splits_strategy: str = "stratified"
    baseline_metric_value: float = 0.55
    # A real agentic run (multiple Opus Leads + Code Interpreter) can take many
    # minutes; the default botocore read_timeout (~60s) would abort the client
    # mid-stream. Give the socket room to receive the streamed events.
    read_timeout: int = 900
    connect_timeout: int = 60
    # PERF-02: urllib3 connection-pool size for the bedrock-agentcore client.
    # Raised above the default of 10 so the streamed invoke + any parallel
    # session ops are not pool-capped. Pure pool sizing — bytes unchanged.
    max_pool_connections: int = 50


@dataclass
class DispatchResult:
    events: list[dict[str, Any]]
    session_id: str
    spec_text: str
    dataset_uri: str
    payload: dict[str, Any]

    @property
    def status(self) -> str | None:
        """The terminal run status, if a run_completed event was streamed."""
        for e in reversed(self.events):
            if e.get("event") == "run_completed":
                return e.get("status")
        return None

    @property
    def reached_training(self) -> bool:
        """True if the stream shows the run getting past data staging into D2/ML."""
        markers = {"data_staged", "run_started"}
        return any(e.get("event") in markers for e in self.events)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def _s3(region: str, client: Any | None) -> Any:
    if client is not None:
        return client
    import boto3

    return boto3.client("s3", region_name=resolve_aws_region(region))


def _dispatch_boto_config(
    *,
    read_timeout: int = 900,
    connect_timeout: int = 60,
    max_pool_connections: int = 50,
) -> Any:
    """Build the botocore ``Config`` for the bedrock-agentcore dispatch client (PERF-02).

    Behavior-preserving: keeps the long stream ``read_timeout``/``connect_timeout``
    and the single-attempt standard retry policy EXACTLY as before, and ADDS
    ``max_pool_connections`` (default 50) so the streamed invoke is not capped at
    urllib3's default pool of 10. Extracted as a pure helper so it is directly verifiable
    offline (no client construction).
    """
    from botocore.config import Config

    return Config(
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        max_pool_connections=max_pool_connections,
        retries={"max_attempts": 1, "mode": "standard"},
    )


def _agentcore(
    region: str,
    client: Any | None,
    *,
    read_timeout: int = 900,
    connect_timeout: int = 60,
    max_pool_connections: int = 50,
) -> Any:
    if client is not None:
        return client
    import boto3

    # Long read_timeout: the runtime streams events only as the agentic run
    # progresses (minutes), so a short default would abort mid-run client-side.
    cfg = _dispatch_boto_config(
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        max_pool_connections=max_pool_connections,
    )
    return boto3.client("bedrock-agentcore", region_name=resolve_aws_region(region), config=cfg)


# ---------------------------------------------------------------------------
# Steps (each independently testable)
# ---------------------------------------------------------------------------


def upload_dataset_to_session(
    dataset_path: str,
    *,
    bucket: str,
    project_id: str,
    region: str | None = None,
    s3_client: Any | None = None,
    prefix: str = "data",
) -> str:
    """Upload a local dataset to the session bucket; return its ``s3://`` URI."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    client = _s3(region, s3_client)
    key = f"{prefix}/{sanitize_project_id(project_id)}/{p.name}"
    client.upload_file(str(p), bucket, key)
    return f"s3://{bucket}/{key}"


def build_payload(
    spec_text: str,
    *,
    stage_data: bool = True,
    max_rows: int | None = None,
    y_true_path: str | None = None,
) -> dict[str, Any]:
    """Build the app.py InvokeAgentRuntime payload.

    Contract (extended in Phase 14b — see app.py):
        spec_text   — the task_spec.md (required).
        stage_data  — directive telling the runtime to stage golden_dataset →
                      concrete train/test + y_true on the container (N1/N2).
        max_rows    — optional cap for a cheap verification slice.
        y_true_path — optional explicit s3:// y_true override (when not staging).
    """
    payload: dict[str, Any] = {"spec_text": spec_text, "stage_data": bool(stage_data)}
    if max_rows is not None:
        payload["max_rows"] = int(max_rows)
    if y_true_path:
        payload["y_true_path"] = y_true_path
    return payload


def _rewrite_golden_dataset(spec_text: str, uri: str) -> str:
    """Point the spec's golden_dataset at the uploaded session-bucket object."""
    if _GOLDEN_RE.search(spec_text):
        return _GOLDEN_RE.sub(lambda m: f"{m.group(1)}{uri}", spec_text, count=1)
    return spec_text


def make_session_id(project_id: str) -> str:
    """AgentCore runtimeSessionId must be >=33 chars. Pad with project + uuid."""
    pid = sanitize_project_id(project_id)
    sid = f"p14b-{pid}-{uuid.uuid4().hex}"
    return sid[:120] if len(sid) >= 33 else (sid + "0" * (33 - len(sid)))


# FIX-14: the app.py entrypoint's LAST yield is always one of these terminal
# events (``run_completed`` on any run end incl. budget_exhausted/failed/accepted,
# or ``error``). Returning as soon as one arrives is what unblocks the client.
_TERMINAL_EVENTS = frozenset({"run_completed", "error", "budget_exhausted"})


def _parse_event_line(line: str) -> dict[str, Any] | None:
    """Decode one streamed line (tolerating SSE ``data:`` framing + blanks)."""
    line = line.strip()
    if line.startswith("data:"):
        line = line[len("data:") :].strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_invoke_response(
    response: dict[str, Any],
    *,
    stop_events: frozenset[str] = _TERMINAL_EVENTS,
    event_sink: Any | None = None,
) -> list[dict[str, Any]]:
    """Parse the streamed InvokeAgentRuntime body into a list of event dicts.

    FIX-14 (root cause of the recurring ``ReadTimeoutError``): the deployed
    AgentCore runtime streams each yielded event as an SSE line as the agentic
    run progresses, but keeps the HTTP connection OPEN after the terminal event
    instead of sending EOF promptly. A ``StreamingBody.read()``-to-EOF therefore
    blocks for the FULL client ``read_timeout`` even though every event — incl.
    the terminal ``run_completed`` — has already been delivered (verified by
    the streaming probe: ``iter_lines()`` returned the terminal event
    in ~6s while ``read()`` hung to the timeout). We now consume the body
    INCREMENTALLY via ``iter_lines()`` and return the moment a terminal event
    (``run_completed``/``error``/``budget_exhausted``) arrives — never waiting on
    the trailing EOF. Falls back to the legacy read-to-EOF / chunk-iteration path
    for non-streaming fakes (offline validation) or if incremental reading is unavailable.

    Tolerates: raw newline-delimited JSON, SSE ``data: {...}`` framing, blank
    lines, and unparseable lines (skipped). Mirrors what app.py yields.

    ``event_sink`` (FIX-39, durable streaming): an optional ``callable(event_dict)`` invoked for
    EVERY parsed event AS IT ARRIVES — before the terminal, before this function returns. This makes
    the run RECOVERABLE: a caller that persists each event to disk (see ``run_cloud_experiment``'s
    ``events_path``) keeps the full streamed trace + terminal ``run_completed`` (with the
    arbiter_report) even if the calling process is killed / times out mid-run. The sink is
    best-effort — a sink error never breaks the stream parse.
    """
    body = response.get("response", [])
    events: list[dict[str, Any]] = []

    def _emit(obj: dict[str, Any]) -> None:
        events.append(obj)
        if event_sink is not None:
            try:
                event_sink(obj)
            except Exception:  # nosec B110
                pass  # durable-sink failures never break the live stream parse

    # Preferred path: a real botocore StreamingBody — read line-by-line and stop
    # at the terminal event so the trailing keep-alive never times us out.
    if hasattr(body, "iter_lines"):
        try:
            for raw_line in body.iter_lines():
                line = (
                    raw_line.decode("utf-8", errors="replace")
                    if isinstance(raw_line, (bytes, bytearray))
                    else str(raw_line)
                )
                obj = _parse_event_line(line)
                if obj is None:
                    continue
                _emit(obj)
                if obj.get("event") in stop_events:
                    break
            return events
        except Exception:
            # Any streaming error: surface whatever we already collected (the
            # terminal event may already be in-hand); do NOT re-read the body
            # (it is partially consumed). Only fall back to read-to-EOF when we
            # got nothing at all AND a re-readable buffer is available.
            if events:
                return events

    # Fallback path (test fakes / list-of-chunks / no iter_lines): read to EOF.
    raw = bytearray()
    if hasattr(body, "read"):  # botocore StreamingBody
        raw.extend(body.read())
    else:
        for chunk in body or []:
            if isinstance(chunk, (bytes, bytearray)):
                raw.extend(chunk)
            elif isinstance(chunk, str):
                raw.extend(chunk.encode("utf-8"))
            elif isinstance(chunk, dict) and "chunk" in chunk:
                data = chunk["chunk"].get("bytes", b"")
                raw.extend(data if isinstance(data, (bytes, bytearray)) else str(data).encode())

    for line in raw.decode("utf-8", errors="replace").splitlines():
        obj = _parse_event_line(line)
        if obj is not None:
            _emit(obj)
    return events


# ---------------------------------------------------------------------------
# B1 — end-to-end operator command
# ---------------------------------------------------------------------------


def run_cloud_experiment(
    description: str,
    *,
    dataset_path: str,
    project_id: str,
    target: str,
    config: DispatchConfig | None = None,
    primary_metric: str | None = None,
    session_id: str | None = None,
    s3_client: Any | None = None,
    agentcore_client: Any | None = None,
    events_path: str | None = None,
) -> DispatchResult:
    """Drive a full NL → cloud experiment against the deployed AgentCore runtime.

    All AWS calls go through injectable clients so the whole flow is unit-tested
    against real AWS: real S3 + the REAL deployed AgentCore runtime (Phase 6 de-mock).
    SEC-01: fails fast with a clear ``ValueError`` when the runtime ARN or
    session bucket is unset (no live default). Operators supply them via env
    (``AGENTCORE_RUNTIME_ARN`` / ``AMLAS_SESSION_BUCKET``) or the CLI overrides.

    ``events_path`` (FIX-39, durable streaming): when set, EVERY streamed event is appended to that
    JSONL file AS IT ARRIVES, so the full trace + terminal ``run_completed`` (with arbiter_report)
    survives even if THIS process is killed/times out before the synchronous stream finishes. The
    session_id is written first as a ``{"event":"session_started",...}`` line so a killed run is
    correlatable to its CloudWatch stream. Defaults to ``AMLAS_DISPATCH_EVENTS_PATH`` env if unset.
    """
    cfg = config or DispatchConfig()

    # SEC-01 fail-fast: never invoke against an empty/unconfigured target.
    if not cfg.runtime_arn:
        raise ValueError("AGENTCORE_RUNTIME_ARN not configured — set the env var or pass --runtime-arn")
    if not cfg.session_bucket:
        raise ValueError(
            "session bucket not configured — set AMLAS_SESSION_BUCKET (or SESSION_BUCKET) "
            "or pass --session-bucket"
        )

    # 1. Upload dataset → session bucket, get s3:// URI (N1 client side).
    dataset_uri = upload_dataset_to_session(
        dataset_path,
        bucket=cfg.session_bucket,
        project_id=project_id,
        region=cfg.region,
        s3_client=s3_client,
    )

    # 2. NL → schema-valid spec, golden_dataset = the uploaded object.
    spec_text = build_spec_yaml(
        description,
        project_id=project_id,
        golden_dataset=dataset_uri,
        target=target,
        primary_metric=primary_metric,
        baseline_metric_value=cfg.baseline_metric_value,
        max_retreats=cfg.max_retreats,
        trials=cfg.trials,
        wall_clock_minutes=cfg.wall_clock_minutes,
        acceptance_delta=cfg.acceptance_delta,
        no_subgroup_regression=cfg.no_subgroup_regression,
        splits_strategy=cfg.splits_strategy,
    )
    # (defensive) ensure the golden_dataset line truly points at the upload.
    spec_text = _rewrite_golden_dataset(spec_text, dataset_uri)

    # 3. Build the app.py payload (stage_data → runtime stages + materializes y_true).
    payload = build_payload(spec_text, stage_data=True, max_rows=cfg.max_rows)

    # 4. InvokeAgentRuntime against the deployed runtime (B1/B5). An explicit
    #    session_id (when provided) lets the caller correlate the run to its
    #    CloudWatch log stream even if the client read-times-out before the
    #    streamed events are delivered; otherwise we mint one.
    session_id = session_id or make_session_id(project_id)
    client = _agentcore(
        cfg.region,
        agentcore_client,
        read_timeout=cfg.read_timeout,
        connect_timeout=cfg.connect_timeout,
        max_pool_connections=cfg.max_pool_connections,
    )
    # : the invoke_agent_runtime CONNECT (establishing the SSE stream) is retried with
    # shared full-jitter backoff on TRANSIENT failures (throttle/capacity/timeout) — previously
    # the botocore config used max_attempts=1 (no retry), so a transient connect blip failed the
    # whole dispatch. Only the connect is retried; the mid-stream read below is NOT (a partial
    # stream can't be safely re-driven). Best-effort import so an env without the module still runs.
    def _connect() -> Any:
        return client.invoke_agent_runtime(
            agentRuntimeArn=cfg.runtime_arn,
            runtimeSessionId=session_id,
            qualifier=cfg.qualifier,
            payload=json.dumps(payload).encode("utf-8"),
        )

    try:
        from ..errors import retry_with_backoff

        response = retry_with_backoff(_connect, max_attempts=4)
    except ImportError:
        response = _connect()

    # FIX-39: durable event sink — persist each streamed event to a JSONL file as it arrives, so a
    # killed/timed-out caller can still recover the full trace + terminal gate (with arbiter_report).
    _events_path = events_path or os.environ.get("AMLAS_DISPATCH_EVENTS_PATH") or ""
    _sink: Callable[[dict[str, Any]], None] | None = None
    _fh = None
    if _events_path:
        try:
            from pathlib import Path as PathT

            PathT(_events_path).parent.mkdir(parents=True, exist_ok=True)
            _fh = open(_events_path, "w", encoding="utf-8")  # nosemgrep: open-never-closed
            _fh.write(json.dumps({"event": "session_started", "session_id": session_id,
                                  "dataset_uri": dataset_uri}) + "\n")
            _fh.flush()

            def _sink(ev: dict[str, Any]) -> None:
                _fh.write(json.dumps(ev, default=str) + "\n")
                _fh.flush()  # flush per-event so a kill loses at most the in-flight line
        except Exception:
            _sink = None  # durable sink is best-effort; never block the dispatch

    # 5. Parse the streamed events to completion (persisting each via the sink).
    try:
        events = parse_invoke_response(response, event_sink=_sink)
    finally:
        if _fh is not None:
            try:
                _fh.close()
            except Exception:  # nosec B110
                pass
    return DispatchResult(
        events=events,
        session_id=session_id,
        spec_text=spec_text,
        dataset_uri=dataset_uri,
        payload=payload,
    )
