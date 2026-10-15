"""C1 — shared dataset-URI resolution + smart tool-call defaults (Phase C).

The single seam that makes a sparse NL-derived tool call VALID:
  * a logical ``s3://amlas/...`` URI is rewritten to the physical lake bucket
    (reusing the lake chokepoint :func:`amlas.tools.lake.resolve_lake_bucket`), so
    ETL / feature-store / processing tools hit the real ``amlas-lake-<acct>-<region>``
    bucket EXACTLY like the lake tools already do;
  * a bare relative path (``ai4i2020.csv``) is resolved against the ``data/`` tree
    (generalizing B1's :func:`amlas.tui.chat.engine.resolve_run_data_path` search);
  * smart defaults fill an omitted ``output`` / ``transform_spec`` so a sparse
    prompt still yields a runnable tool call.

Offline-safe + additive: unresolvable inputs pass through UNCHANGED (never raises
for resolution; a caller that needs a local file still gets its own not-found
error). ``resolve_lake_bucket`` already returns the logical name unchanged when the
physical bucket can't be resolved (no env/SSM), so in offline mode a
logical ``s3://amlas/...`` stays ``s3://amlas/...`` — no regression.

STRANDS-FREE + BOTO3-LAZY at import: the lake chokepoint (which imports ``strands``)
and the S3 URI parser are imported LAZILY inside the functions, so importing this
module is cheap and headless-safe (mirrors :mod:`amlas.tools.pathref` discipline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "resolve_dataset_uri",
    "resolve_feature_store_ingest",
    "resolve_processing_io",
]


def _looks_like_s3(uri: str) -> bool:
    """True iff ``uri`` is an ``s3://`` URI (same test as engine/pathref classify)."""
    return isinstance(uri, str) and uri.strip().lower().startswith("s3://")


def _resolve_local_data_path(path: str) -> str | None:
    """Resolve a bare/relative local filename against the ``data/`` tree.

    Generalizes B1's ``resolve_run_data_path`` one-level subdir search
    (``engine.py:233-251``): if ``path`` does not exist, look for the same filename
    directly under ``data/`` and one level of its subdirectories
    (``ai4i2020.csv`` → ``data/ai4i/ai4i2020.csv``). Returns the resolved path
    string on a hit, else ``None``. Filesystem-only (``Path.exists``/``iterdir``) —
    never touches the network, safe under ``_running_offline``.
    """
    p = Path(path)
    if p.exists():
        return str(p)

    filename = p.name
    if not filename:
        return None

    search_dirs: list[Path] = []
    data_root = Path("data")
    try:
        if data_root.is_dir():
            search_dirs.append(data_root)
            search_dirs.extend(d for d in sorted(data_root.iterdir()) if d.is_dir())
    except OSError:
        return None

    for candidate_dir in search_dirs:
        candidate = candidate_dir / filename
        if candidate.exists():
            return str(candidate)
    return None


def resolve_dataset_uri(uri: str) -> str:
    """Normalize a dataset reference to a concrete, backend-usable location.

    - ``s3://<alias>/...`` (alias in the lake alias set) → ``s3://<physical>/...``
      via :func:`resolve_lake_bucket` (key/prefix preserved EXACTLY).
    - ``s3://<physical-or-other>/...`` → unchanged (regression-critical).
    - a local path that exists → unchanged.
    - a bare/relative local path that does NOT exist → the ``data/`` one-level
      subdir match if found (``ai4i2020.csv`` → ``data/ai4i/ai4i2020.csv``), else
      the input unchanged (the tool surfaces its own not-found).

    Never raises: a best-effort normalizer. Empty/None → returned as-is.
    """
    if not uri or not isinstance(uri, str):
        return uri

    if _looks_like_s3(uri):
        # Lazy imports: the lake module imports ``strands`` and staging imports
        # pandas/numpy — keep this module import cheap + headless.
        try:
            from ..cloud.staging import parse_s3_uri
            from .lake import resolve_lake_bucket

            bucket, key = parse_s3_uri(uri)
        except Exception:
            # Malformed s3 uri (no key) or import failure → pass through unchanged.
            return uri
        physical = resolve_lake_bucket(bucket)
        return f"s3://{physical}/{key}"

    # Local reference: explicit-local-first, then the data/ subdir search.
    resolved = _resolve_local_data_path(uri)
    return resolved if resolved is not None else uri


def _sibling_output(resolved_input: str) -> str:
    """Derive a deterministic sibling ``output`` next to a RESOLVED input (§C.3).

    - ``s3://<bucket>/<key>`` → ``s3://<bucket>/silver/processed/<stem>/`` (the
      writable-tier ``silver`` per ``WRITABLE_TIERS``; never bronze/raw).
    - local ``<stem><suffix>`` → ``<stem>.processed<suffix>`` (the InProcess
      adapter's ``_out_path`` sibling convention, ``etl.py:56-62``).
    An unparseable input falls back to the input unchanged.

    Phase D3 (P1-6): a malformed input — ``None`` / a non-str / ``""`` — is a
    best-effort pass-through (returned as-is), NEVER a raise. Pre-D3 the local
    branch ``Path(resolved_input).with_name(...)`` raised ``TypeError`` for
    ``None``/non-str and ``ValueError`` for ``""`` (empty name), which broke the
    ``resolve_processing_io`` contract that a sparse/garbage tool arg must degrade
    without raising. This guard closes that gap while leaving every valid-string
    result byte-identical.
    """
    if _looks_like_s3(resolved_input):
        try:
            from ..cloud.staging import parse_s3_uri

            bucket, key = parse_s3_uri(resolved_input)
        except Exception:
            return resolved_input
        stem = Path(key).stem or "dataset"
        return f"s3://{bucket}/silver/processed/{stem}/"

    # D3 P1-6: only a non-empty string can be turned into a sibling path; a
    # ``None`` / non-str / ``""`` input passes through unchanged (no raise).
    if not resolved_input or not isinstance(resolved_input, str):
        return resolved_input

    p = Path(resolved_input)
    return str(p.with_name(f"{p.stem}.processed{p.suffix or '.csv'}"))


def resolve_processing_io(
    input_uri: str, output: str = "", transform_spec: dict[str, Any] | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Resolve ``run_processing_job`` / ``run_glue_job`` args to a valid triple.

    - ``input_uri`` → :func:`resolve_dataset_uri`.
    - ``output`` supplied → normalized via :func:`resolve_dataset_uri` (a logical
      ``s3://amlas/...`` output is rewritten too) but otherwise KEPT (explicit
      wins). ``output`` empty → a deterministic sibling next to the resolved input
      (:func:`_sibling_output`, §C.3).
    - ``transform_spec`` falsy → ``{"op": "auto_preprocess"}`` (the InProcess
      adapter's own default op, ``etl.py:92``) so a bare "featurize" call runs.

    Returns ``(resolved_input, resolved_output, transform_spec)``.
    """
    resolved_input = resolve_dataset_uri(input_uri)

    if output:
        resolved_output = resolve_dataset_uri(output)
    else:
        resolved_output = _sibling_output(resolved_input)

    spec = dict(transform_spec) if transform_spec else {"op": "auto_preprocess"}

    return resolved_input, resolved_output, spec


def resolve_feature_store_ingest(
    data_path: str, record_id: str | None, event_time: str | None
) -> tuple[str, str | None, str | None]:
    """Resolve ``feature_store_ingest`` args.

    Only NORMALIZES ``data_path`` via :func:`resolve_dataset_uri` (so a bare
    filename is found under ``data/`` before the tool's ``Path.exists`` gate).
    ``record_id`` / ``event_time`` defaults ALREADY exist in the tool (first column
    / synthetic ``event_time``, ``feature_store.py:163-164``), so they pass through
    UNTOUCHED — C1 does not duplicate that inference (§C.4).
    """
    return resolve_dataset_uri(data_path), record_id, event_time
