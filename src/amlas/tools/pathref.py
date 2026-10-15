"""Provider-neutral path-resolution seam for the TUI agent's file/dataset tools.

Feature 1 (``docs/tui-agent-fs-remote-and-thought-trace-design.md`` §2). One shared
seam that classifies a reference (``classify``), validates + normalizes it
(``resolve_ref``), reads its bytes (``open_bytes``), or loads it as a DataFrame
(``read_dataframe``) — transparently across **local paths**, **``s3://`` URIs**, and
**``http(s)://`` URLs**, with a pragmatic security posture.

Design invariants honored here:
  * **Local** paths keep the EXISTING sandbox containment — this module OWNS
    :func:`_resolve_within_sandbox` / :func:`_allowed_roots` (moved here from
    ``fs_tools`` so both the fs tools and the data tools share ONE containment
    implementation, no duplication). A sandbox escape raises :class:`PathRefError`.
  * **S3** rides the resolved AWS profile (``Settings.load().aws.effective_profile``)
    via a boto3 ``Session`` → ``get_object`` → ``io.BytesIO`` (so ``s3fs`` stays
    OPTIONAL — only core ``boto3`` is required). Never embeds credentials (SEC-01).
  * **HTTP(S)** is ANONYMOUS-only (no ``Authorization`` header ever attached) with a
    MANDATORY SSRF guard (reject link-local/loopback/private/metadata IPs incl.
    ``169.254.169.254``) plus an optional host allow-list.
  * **Remote is gated** behind ``AMLAS_AGENT_REMOTE_FETCH`` — default **ON** to honor
    the AWS-first directive (S3 IS the AWS account). Set ``0`` to force local-only.
  * A **remote size ceiling** (:data:`REMOTE_MAX_BYTES`) bounds unbounded downloads;
    ``read_file``'s 256 KB cap flows through ``max_bytes`` (S3 ``Range`` / HTTP read).

STRANDS-FREE + REMOTE-CLIENT-LAZY: importing this module needs only stdlib +
``pandas`` + :class:`Settings`. ``boto3`` and ``httpx`` are imported only inside
remote branches, so the module imports cheaply and the offline path stays lean.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..config.aws_region import resolve_aws_region

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


# ─── Tunables / env seams ────────────────────────────────────────────────────

#: Env flag gating remote (S3/HTTP) reads. Default ON (AWS-first): unset/empty →
#: enabled; only the standard falsey tokens disable it. Mirrors the
#: ``AMLAS_SANDBOX_EGRESS`` opt-out precedent in ``code_interpreter.py``.
_REMOTE_FETCH_ENV = "AMLAS_AGENT_REMOTE_FETCH"
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

#: Comma-separated host allow-list for HTTP(S). Empty = allow any PUBLIC host, still
#: SSRF-guarded (private/loopback/link-local/metadata always rejected).
_HTTP_ALLOW_HOSTS_ENV = "AMLAS_AGENT_HTTP_ALLOW_HOSTS"

#: Hard ceiling on a single remote read (bytes) when no explicit ``max_bytes`` is
#: given (e.g. dataset loads). Generous for CSVs; rejects pathological objects
#: rather than silently truncating (which would corrupt a DataFrame).
REMOTE_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB

#: HTTP GET timeout (seconds) — bounds a hung remote host.
HTTP_TIMEOUT = 30.0


class PathScheme(StrEnum):
    """The three reference families the seam understands (``str`` mixin → JSON-friendly)."""

    LOCAL = "local"
    S3 = "s3"
    HTTP = "http"  # covers http + https


@dataclass(frozen=True)
class ResolvedRef:
    """A classified, validated reference to a byte source (local or remote)."""

    raw: str  # the original user string
    scheme: PathScheme
    # LOCAL: the sandbox-resolved absolute path (str). Remote: the normalized URI.
    location: str
    bucket: str | None = None  # S3 only
    key: str | None = None  # S3 only
    is_remote: bool = False


class PathRefError(Exception):
    """Unsupported scheme, sandbox escape, disabled remote scheme, or SSRF block."""


# ─── Sandbox containment (moved here from fs_tools; shared by ALL file tools) ──


def _allowed_roots() -> list[Path]:
    """Resolved allowed roots: workspace + data/output/specs.

    Roots come from :class:`LocalConfig` (``Settings().local``). All are fully
    resolved so containment compares real paths. ``Settings`` is imported lazily to
    keep this module's import cheap and side-effect-free.
    """
    from ..config.settings import Settings  # lazy: avoid import-time cost

    local = Settings.load().local
    roots = [
        local.resolved_workspace,
        local.resolved_data,
        local.resolved_output,
        local.resolved_specs,
    ]
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _resolve_within_sandbox(path: str) -> Path:
    """Resolve ``path`` to a real absolute path confined to an allowed root.

    Expands ``~``, resolves ``..`` and symlinks (``Path.resolve()``), then requires
    the result to live inside one of :func:`_allowed_roots`. Raises ``PermissionError``
    on any escape so local fs-tool callers translate it into a fail-soft dict exactly
    as before (behavior byte-identical to the prior ``fs_tools`` implementation).
    """
    candidate = Path(os.path.expanduser(path)).resolve()
    for root in _allowed_roots():
        if candidate == root or candidate.is_relative_to(root):
            return candidate
    raise PermissionError(f"path outside sandbox: {path}")


def resolve_dataset_path(path: str) -> Path:
    """Resolve an agent-supplied dataset path with traversal containment.

    Path-traversal guard (Security review, CWE-22): agent-facing data tools
    (``detect_pii`` / ``diagnose_missingness`` / ``submit_training_trial`` …)
    receive ``data_path`` from LLM tool calls, so a hostile/hallucinated
    ``../../etc/passwd`` must not escape the working area. The resolved real
    path must live inside one of: the sandbox roots (:func:`_allowed_roots`),
    the process CWD (project tree), or the system temp dir (flows stage
    intermediate artifacts under ``tempfile.mkdtemp``). Anything else raises
    ``PermissionError`` — callers translate it into their fail-soft error dict.
    """
    import tempfile

    candidate = Path(os.path.expanduser(path)).resolve()
    roots = list(_allowed_roots())
    roots.append(Path.cwd().resolve())
    roots.append(Path(tempfile.gettempdir()).resolve())
    for root in roots:
        if candidate == root or candidate.is_relative_to(root):
            return candidate
    raise PermissionError(f"dataset path outside allowed roots: {path}")


def _resolve_local(path: str) -> Path:
    """Resolve an EXPLICIT, user-provided local path WITHOUT sandbox containment.

    Regression fix (2026-07-12): the data tools (``load_dataset`` /
    ``describe_dataset`` / ``split_dataset``) load a path the USER typed at a
    tier-1 ``/data`` prompt — pre-Feature-1 they did ``pd.read_csv(Path(path))``
    with NO containment, so an absolute dataset path outside the workspace roots
    (or, under the isolated harness's ``chdir(tmp)`` isolation, an absolute repo path)
    loaded fine. This helper restores that exact contract: it only expands ``~``
    and resolves the real path — it does NOT consult :func:`_allowed_roots`. The
    AGENT-facing fs tools keep :func:`_resolve_within_sandbox` (containment) so the
    LLM's autonomous directory exploration stays sandboxed. See ``resolve_ref``'s
    ``sandbox_local`` flag.
    """
    return Path(os.path.expanduser(path)).resolve()


# ─── Public: classify / flag ───────────────────────────────────────────────────


def classify(path: str) -> PathScheme:
    """Map a reference to its :class:`PathScheme`.

    ``s3://…`` → S3 ; ``http://`` / ``https://`` → HTTP ; anything else → LOCAL.
    """
    low = path.strip().lower()
    if low.startswith("s3://"):
        return PathScheme.S3
    if low.startswith("http://") or low.startswith("https://"):
        return PathScheme.HTTP
    return PathScheme.LOCAL


def remote_fetch_enabled() -> bool:
    """True unless ``AMLAS_AGENT_REMOTE_FETCH`` is a falsey token (default ON).

    AWS-first: an unset or empty flag means ENABLED (S3 == the AWS account). Only the
    explicit ``0``/``false``/``no``/``off`` tokens (case-insensitive) disable remote.
    """
    raw = os.environ.get(_REMOTE_FETCH_ENV)
    if raw is None:
        return True
    token = raw.strip().lower()
    if token == "":  # nosec B105 — feature-flag env token emptiness check, not a password comparison
        return True
    return token not in _FALSE_TOKENS


# ─── S3 URI parsing ──────────────────────────────────────────────────────────


def _parse_s3(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/parts`` → ``("bucket", "key/parts")`` (key may be "")."""
    rest = uri[len("s3://"):] if uri.lower().startswith("s3://") else uri
    if "/" in rest:
        bucket, key = rest.split("/", 1)
    else:
        bucket, key = rest, ""
    if not bucket:
        raise PathRefError(f"invalid S3 URI (no bucket): {uri}")
    return bucket, key


# ─── SSRF guard ────────────────────────────────────────────────────────────────


def _resolve_host_ips(host: str) -> list[str]:
    """Resolve ``host`` to IP strings. A literal IP resolves to itself (no DNS).

    Monkeypatched in tests to avoid real DNS; in production it uses
    ``socket.getaddrinfo`` for hostnames and returns the literal for IP hosts.
    """
    try:
        ipaddress.ip_address(host)
        return [host]
    except ValueError:
        pass
    infos = socket.getaddrinfo(host, None)
    return list({str(info[4][0]) for info in infos})


def _is_blocked_ip(ip_str: str) -> bool:
    """True if ``ip_str`` is loopback/private/link-local/metadata/reserved/etc.

    Fail-CLOSED: an unparseable address is treated as blocked. ``169.254.169.254``
    (cloud metadata) is caught by ``is_link_local`` (169.254/16); the RFC1918 ranges
    (10/8, 172.16/12, 192.168/16), IPv6 ``::1`` and ``fd00::/8`` by
    ``is_private``/``is_loopback``.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return bool(
        ip.is_private  # nosemgrep
        or ip.is_loopback  # nosemgrep
        or ip.is_link_local  # nosemgrep
        or ip.is_reserved  # nosemgrep
        or ip.is_multicast  # nosemgrep
        or ip.is_unspecified  # nosemgrep
    )


def _guard_http(url: str) -> None:
    """Enforce the host allow-list (if set) + the MANDATORY SSRF guard for ``url``."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise PathRefError(f"invalid URL (no host): {url}")
    if parsed.username is not None or parsed.password is not None:
        raise PathRefError("URL userinfo credentials are not allowed for anonymous HTTP fetches")

    allow = [h.strip() for h in os.environ.get(_HTTP_ALLOW_HOSTS_ENV, "").split(",") if h.strip()]
    if allow and host not in allow:
        raise PathRefError(f"host not in AMLAS_AGENT_HTTP_ALLOW_HOSTS allow-list: {host}")

    try:
        ips = _resolve_host_ips(host)
    except Exception as exc:
        raise PathRefError(f"cannot resolve host {host!r}: {exc}") from exc
    if not ips:
        raise PathRefError(f"cannot resolve host {host!r}")
    for ip in ips:
        if _is_blocked_ip(ip):
            raise PathRefError(f"blocked by SSRF guard: {host} -> {ip}")


# ─── Public: resolve_ref ─────────────────────────────────────────────────────


def resolve_ref(
    path: str, *, allow_remote: bool = True, sandbox_local: bool = True
) -> ResolvedRef:
    """Classify + validate ``path`` into a :class:`ResolvedRef` (or raise).

    LOCAL → sandbox containment (raises :class:`PathRefError` on escape).
    S3    → parse bucket/key; requires remote enabled.
    HTTP  → host allow-list + SSRF guard; requires remote enabled.

    ``sandbox_local`` (default ``True``) governs ONLY the LOCAL branch. The
    AGENT-facing fs tools keep the default so the LLM's autonomous directory
    exploration stays confined to :func:`_allowed_roots`. The data tools pass
    ``sandbox_local=False`` for an EXPLICIT, user-provided dataset path (a tier-1
    ``/data`` action) — restoring the pre-Feature-1 ``pd.read_csv(Path(path))``
    contract that imposed no containment on a path the user chose. It does NOT
    affect remote gating: S3/HTTP still ride ``allow_remote`` + the SSRF guard.
    """
    scheme = classify(path)

    if scheme is PathScheme.LOCAL:
        if not sandbox_local:
            # Explicit user-provided local path: resolve WITHOUT containment.
            real = _resolve_local(path)
            return ResolvedRef(
                raw=path, scheme=PathScheme.LOCAL, location=str(real), is_remote=False
            )
        try:
            real = _resolve_within_sandbox(path)
        except PermissionError as exc:
            raise PathRefError(str(exc)) from exc
        return ResolvedRef(raw=path, scheme=PathScheme.LOCAL, location=str(real), is_remote=False)

    # Remote (S3/HTTP) — gated by the flag AND the caller's allow_remote.
    if not allow_remote or not remote_fetch_enabled():
        raise PathRefError(
            f"remote fetch disabled (set {_REMOTE_FETCH_ENV}=1 to enable): {path}"
        )

    if scheme is PathScheme.S3:
        bucket, key = _parse_s3(path)
        return ResolvedRef(
            raw=path, scheme=PathScheme.S3, location=path, bucket=bucket, key=key, is_remote=True
        )

    # HTTP(S)
    _guard_http(path)
    return ResolvedRef(raw=path, scheme=PathScheme.HTTP, location=path, is_remote=True)


# ─── S3 client (uses the resolved AWS profile; boto3 lazy) ─────────────────────


def _s3_client() -> Any:
    """Build a boto3 S3 client on the RESOLVED AWS profile (SEC-01: no hardcoded creds).

    ``Settings.load().aws.effective_profile`` (``None`` → boto3 default chain) makes S3
    reads ride the SAME AWS account the rest of the TUI uses (AWS-first). ``boto3`` is
    imported lazily so the module stays importable without it.
    """
    import boto3  # lazy

    profile: str | None = None
    region: str | None = None
    try:
        from ..config.settings import Settings

        aws = Settings.load().aws
        profile = aws.effective_profile
        region = aws.region or None
    except Exception:
        profile = None
        region = None

    session = boto3.Session(profile_name=profile)
    return session.client("s3", region_name=resolve_aws_region(region, profile_name=profile))


# ─── Public: open_bytes ──────────────────────────────────────────────────────


def open_bytes(ref: ResolvedRef, *, max_bytes: int | None = None) -> bytes:
    """Read bytes from a resolved reference (local file / S3 object / HTTP URL).

    When ``max_bytes`` is given, at most that many bytes are returned (S3 uses a
    ``Range`` header; HTTP uses a bounded ``read``; local uses ``read(max_bytes)``).
    When ``max_bytes`` is ``None`` for a REMOTE ref, the whole object is read subject
    to the :data:`REMOTE_MAX_BYTES` ceiling (exceeding it raises :class:`PathRefError`
    rather than silently truncating). May raise on I/O errors — tool wrappers catch.
    """
    if ref.scheme is PathScheme.LOCAL:
        with open(ref.location, "rb") as fh:
            if max_bytes is not None and max_bytes >= 0:
                return fh.read(max_bytes)
            return fh.read()

    if ref.scheme is PathScheme.S3:
        return _s3_get_bytes(ref, max_bytes)

    if ref.scheme is PathScheme.HTTP:
        return _http_get_bytes(ref, max_bytes)

    raise PathRefError(f"unsupported scheme for open_bytes: {ref.scheme}")


def _s3_get_bytes(ref: ResolvedRef, max_bytes: int | None) -> bytes:
    client = _s3_client()
    if max_bytes is not None and max_bytes >= 0:
        if max_bytes == 0:
            return b""
        resp = client.get_object(
            Bucket=ref.bucket, Key=ref.key, Range=f"bytes=0-{max_bytes - 1}"
        )
        # boto3's StreamingBody.read() is typed ``Any``; pin to ``bytes`` so the
        # declared return contract holds without an Any-return.
        data: bytes = resp["Body"].read()
        return data[:max_bytes]
    resp = client.get_object(Bucket=ref.bucket, Key=ref.key)
    data = resp["Body"].read()
    if len(data) > REMOTE_MAX_BYTES:
        raise PathRefError(
            f"remote object exceeds size cap ({REMOTE_MAX_BYTES} bytes): {ref.raw}"
        )
    return data


def _http_get_bytes(ref: ResolvedRef, max_bytes: int | None) -> bytes:
    """Fetch an anonymous HTTP resource without following redirects.

    Disabling redirects closes the second-hop SSRF path where a public, validated
    origin redirects to a private or metadata address. Streaming enforces the byte
    ceiling on decoded content without first materializing an unbounded response.
    """
    import httpx  # lazy

    if max_bytes is not None and max_bytes >= 0:
        explicitly_bounded = True
        limit = max_bytes
    else:
        explicitly_bounded = False
        limit = REMOTE_MAX_BYTES
    if limit == 0:
        return b""

    content = bytearray()
    try:
        with httpx.Client(
            timeout=HTTP_TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": "amlas-pathref/1.0"},
        ) as client:
            with client.stream("GET", ref.location) as response:
                if response.is_redirect:  # nosemgrep
                    raise PathRefError(
                        f"HTTP redirect responses are not followed: {ref.raw}"
                    )
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    if explicitly_bounded:
                        remaining = limit - len(content)
                        content.extend(chunk[:remaining])
                        if len(content) >= limit:
                            break
                    else:
                        if len(content) + len(chunk) > REMOTE_MAX_BYTES:
                            raise PathRefError(
                                "remote object exceeds size cap "
                                f"({REMOTE_MAX_BYTES} bytes): {ref.raw}"
                            )
                        content.extend(chunk)
    except PathRefError:
        raise
    except httpx.HTTPError as error:
        raise PathRefError(f"HTTP fetch failed for {ref.raw}: {error}") from error
    return bytes(content)


# ─── Public: read_dataframe ────────────────────────────────────────────────────


def read_dataframe(ref: ResolvedRef, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV reference into a pandas DataFrame across local/S3/HTTP.

    LOCAL → ``pd.read_csv(location)``. Remote → ``open_bytes`` → ``io.BytesIO`` →
    ``pd.read_csv`` (the boto3 ``get_object`` path, so ``s3fs`` is NOT required).
    """
    import io

    import pandas as pd

    if ref.scheme is PathScheme.LOCAL:
        return pd.read_csv(ref.location, **read_csv_kwargs)

    data = open_bytes(ref)
    return pd.read_csv(io.BytesIO(data), **read_csv_kwargs)


# ─── S3 prefix listing (for fs_tools list_directory / find_files) ──────────────


def list_prefix(ref: ResolvedRef, *, max_results: int = 1000) -> list[dict[str, Any]]:
    """Enumerate S3 objects under ``ref`` via paginated ``list_objects_v2``.

    Mirrors ``tui/screens/data.py::_list_all_objects`` (PERF-13s): follows
    ``NextContinuationToken`` until exhausted or ``max_results`` reached. Returns the
    raw ``Contents`` dicts (``Key``/``Size``/``LastModified``).
    """
    if ref.scheme is not PathScheme.S3:
        raise PathRefError(f"list_prefix only supports S3, got {ref.scheme}")
    client = _s3_client()
    contents: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {"Bucket": ref.bucket}
    if ref.key:
        kwargs["Prefix"] = ref.key
    while True:
        resp = client.list_objects_v2(**kwargs)
        if not isinstance(resp, dict):
            break
        contents.extend(resp.get("Contents", []) or [])
        if len(contents) >= max_results:
            break
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
        kwargs["ContinuationToken"] = token
    return contents
