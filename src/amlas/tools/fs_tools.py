"""Sandboxed, read-only filesystem tools for the TUI chat agent (design doc 06).

The ML lifecycle chat agent had 25 ML tools but ZERO filesystem tools, so when a
user pasted a *directory* (``data/ai4i/bogie/archive``) and asked it to "read this
data", the agent dead-ended ("I'm unable to auto-discover the filename …, run
``ls`` yourself"). This module gives it the "claude-code-like" directory verbs it
was missing — list, glob/find, content-search, read, and stat — so it can explore
a directory autonomously, discover the CSVs, then hand them to
``describe_dataset``/``load_dataset`` for analysis.

Five ``@tool``s, all **read-only**, all **sandboxed**, all **fail-soft**:
  - :func:`list_directory`  — ``ls`` (immediate or recursive), type+size per entry.
  - :func:`find_files`      — glob/``find`` by ``fnmatch`` pattern (e.g. ``*.csv``).
  - :func:`search_in_files` — ripgrep-style content search → matching file+line.
  - :func:`read_file`       — safe text read; 256 KB cap; head/tail; refuses binary.
  - :func:`file_stat`       — ``stat`` metadata (size/mtime/type/exists), no read.

Safety / sandbox (design §6.1, MANDATORY):
  * Access confined to allowed roots — workspace + ``data``/``output``/``specs``/
    ``papers`` — via :func:`_resolve_within_sandbox`, which resolves the REAL path
    (expanding ``~`` and following ``..``/symlinks) and rejects anything escaping
    containment (so ``../../etc/passwd``, ``/etc/passwd`` and symlink-escapes fail).
  * No writes, no ``subprocess``/shell, no ``os.system``/``eval`` — pure stdlib
    (``pathlib``/``os``/``fnmatch``/``glob``).
  * ``read_file`` caps at 256 KB and refuses binary (NUL-byte sniff).
  * List/find/search cap result counts and set ``"truncated": True`` when capped.
  * Every tool returns a JSON-serializable ``dict`` — on ANY error it returns
    ``{"success": False, "ok": False, "error": "...", "path": "..."}`` rather than
    raising, so the LLM always gets a usable message, never a crash.

Return-shape note (reconciliation): the authoritative design doc §6 documents
``"success": True`` (matching the project convention and the existing tools that
``test_tool_binding`` reads via ``result.get("success")``); the task summary's
illustrative ``{"ok": False, ...}`` is honoured too — every return carries BOTH a
primary ``"success"`` key and an ``"ok"`` mirror with the same value.

LAZY/STDLIB-ONLY: like ``data_tools``, this module begins with
``from strands import tool``; ``Settings`` is imported lazily inside the sandbox
helper so importing the module stays cheap and side-effect-free.
"""

from __future__ import annotations

import fnmatch
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strands import tool

from ..errors import tool_error_boundary

# Feature 1 (remote reach): the sandbox containment (``_allowed_roots`` /
# ``_resolve_within_sandbox``) now lives in the shared ``pathref`` seam so the fs
# tools and the data tools share ONE implementation (no duplication). They are
# re-imported here so ``fs_tools._resolve_within_sandbox`` keeps resolving for any
# caller. ``classify``/``resolve_ref``/``open_bytes``/``list_prefix`` add the
# transparent ``s3://`` + ``http(s)://`` branches.
from .pathref import (
    PathRefError,
    PathScheme,
    _allowed_roots,
    _resolve_within_sandbox,
    _s3_client,
    classify,
    list_prefix,
    open_bytes,
    resolve_ref,
)

__all__ = [
    "_allowed_roots",
    "_resolve_within_sandbox",
    "file_stat",
    "find_files",
    "list_directory",
    "read_file",
    "search_in_files",
]

# ─── Tunable caps (design §6.1) ────────────────────────────────────────────────

MAX_READ_BYTES = 262_144          # 256 KB hard cap for read_file
_BINARY_SNIFF_BYTES = 8_192       # bytes inspected for NUL / non-UTF-8
DEFAULT_MAX_RESULTS = 500         # list_directory / find_files entry cap
DEFAULT_MAX_HITS = 200            # search_in_files match cap
_LINE_TRIM = 300                  # max chars per reported search line
_PER_FILE_SEARCH_BYTES = MAX_READ_BYTES  # don't scan more than this per file

# Hidden/junk skipped by default (design §6.1 #8).
_SKIP_NAMES = {".DS_Store", ".git", "__pycache__"}
_SKIP_SUFFIXES = {".pyc"}


# ─── Sandbox containment (shared by every tool, design §6.1) ───────────────────
#
# ``_allowed_roots`` / ``_resolve_within_sandbox`` now live in ``pathref`` (imported
# at the top of this module) so the fs tools and the data tools share ONE containment
# implementation. Behavior is byte-identical to the prior in-module version — the code
# was MOVED, not changed — and both names are re-exported here for any caller that
# imports ``fs_tools._resolve_within_sandbox`` / ``fs_tools._allowed_roots``.


def _err(path: str, message: str, **extra: Any) -> dict[str, Any]:
    """Structured fail-soft error dict (carries both ``success`` and ``ok``)."""
    out: dict[str, Any] = {"success": False, "ok": False, "error": message, "path": path}
    out.update(extra)
    return out


def _ok(**fields: Any) -> dict[str, Any]:
    """Structured success dict (``ok`` mirrors ``success``)."""
    out: dict[str, Any] = {"success": True, "ok": True}
    out.update(fields)
    return out


def _mtime_iso(p: Path) -> str:
    """File mtime as an ISO-8601 UTC string (JSON-serializable)."""
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return ""


def _is_hidden_or_junk(name: str) -> bool:
    if name in _SKIP_NAMES or name.startswith("."):
        return True
    return any(name.endswith(suf) for suf in _SKIP_SUFFIXES)


def _entry(p: Path) -> dict[str, Any]:
    """Build a single directory-entry dict (name/type/size/mtime/ext)."""
    try:
        is_dir = p.is_dir()
        size = p.stat().st_size if not is_dir else 0
    except OSError:
        is_dir = False
        size = 0
    return {
        "name": p.name,
        "type": "dir" if is_dir else "file",
        "size": size,
        "mtime": _mtime_iso(p),
        "ext": "" if is_dir else p.suffix,
    }


def _looks_binary(raw: bytes) -> bool:
    """Heuristic binary sniff: NUL byte present, or not valid UTF-8 (design §6.1 #6)."""
    if b"\x00" in raw:
        return True
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


# ─── Remote helpers (Feature 1) ────────────────────────────────────────────────


def _s3_entry(obj: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Build a directory-entry dict from an S3 ``Contents`` object (name relative to prefix)."""
    key = str(obj.get("Key", ""))
    name = key[len(prefix):] if prefix and key.startswith(prefix) else key
    name = name.lstrip("/") or key
    mtime = obj.get("LastModified")
    if mtime is None:
        mtime_str = ""
    elif hasattr(mtime, "isoformat"):
        mtime_str = mtime.isoformat()
    else:
        mtime_str = str(mtime)
    ext = ""
    if "." in name:
        ext = name[name.rindex("."):]
    return {
        "name": name,
        "type": "file",
        "size": int(obj.get("Size", 0) or 0),
        "mtime": mtime_str,
        "ext": ext,
        "key": key,
    }


def _list_directory_s3(path: str, max_results: int) -> dict[str, Any]:
    """S3 prefix listing for ``list_directory`` (mirrors screens/data.py pagination)."""
    try:
        ref = resolve_ref(path)
        objects = list_prefix(ref, max_results=max_results)
    except PathRefError as e:
        return _err(path, str(e))
    except Exception as e:
        return _err(path, f"failed to list S3 prefix: {e}")

    prefix = ref.key or ""
    truncated = len(objects) >= max_results
    entries = [_s3_entry(o, prefix) for o in objects[:max_results]]
    entries.sort(key=lambda e: e["name"])
    return _ok(
        path=path,
        is_dir=True,
        entries=entries,
        n_entries=len(entries),
        truncated=truncated,
    )


def _find_files_s3(path: str, pattern: str, max_results: int) -> dict[str, Any]:
    """S3 prefix listing filtered by an ``fnmatch`` NAME pattern (find_files shape)."""
    try:
        ref = resolve_ref(path)
        objects = list_prefix(ref, max_results=max_results * 4 if max_results else 1000)
    except PathRefError as e:
        return _err(path, str(e))
    except Exception as e:
        return _err(path, f"failed to list S3 prefix: {e}")

    matches: list[str] = []
    truncated = False
    for obj in objects:
        key = str(obj.get("Key", ""))
        name = key.rsplit("/", 1)[-1]
        if not name or _is_hidden_or_junk(name):
            continue
        if not fnmatch.fnmatch(name, pattern):
            continue
        if len(matches) >= max_results:
            truncated = True
            break
        matches.append(f"s3://{ref.bucket}/{key}")
    matches.sort()
    return _ok(
        path=path,
        pattern=pattern,
        matches=matches,
        n_matches=len(matches),
        truncated=truncated,
    )


# ─── 1. list_directory ─────────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.fs_tools")
def list_directory(
    path: str,
    recursive: bool = False,
    include_hidden: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    """List the files and sub-directories in a directory (like ``ls``).

    Use this FIRST when the user gives you a directory path (or you are unsure of an
    exact filename): it turns a folder into a concrete file list with each entry's
    type and size, so you can pick the dataset to analyze — never ask the user to
    run ``ls`` themselves.

    Also accepts an ``s3://bucket/prefix/`` URI → an S3 prefix listing (paginated).
    ``http(s)://`` URLs are unsupported (you cannot glob a URL) and fail soft.

    Args:
        path: Directory path (workspace sandbox) or ``s3://bucket/prefix/`` URI.
        recursive: If True, walk sub-directories too; otherwise immediate children.
        include_hidden: If True, include dotfiles / junk (``.DS_Store`` etc.).
        max_results: Cap on entries returned; sets ``truncated`` when exceeded.
    """
    scheme = classify(path)
    if scheme is PathScheme.S3:
        return _list_directory_s3(path, max_results)
    if scheme is PathScheme.HTTP:
        return _err(path, "listing is not supported for a URL (cannot glob an http(s):// URL)")

    try:
        real = _resolve_within_sandbox(path)
    except PermissionError as e:
        return _err(path, str(e))
    except OSError as e:
        return _err(path, f"cannot resolve path: {e}")

    if not real.exists():
        return _err(str(real), f"path not found: {path}")
    if not real.is_dir():
        return _err(str(real), f"not a directory: {path}", is_dir=False)

    entries: list[dict[str, Any]] = []
    truncated = False
    try:
        iterator = real.rglob("*") if recursive else real.iterdir()
        for child in iterator:
            if not include_hidden and _is_hidden_or_junk(child.name):
                continue
            if len(entries) >= max_results:
                truncated = True
                break
            entries.append(_entry(child))
    except OSError as e:
        return _err(str(real), f"failed to list directory: {e}")

    entries.sort(key=lambda e: (e["type"] != "dir", e["name"]))
    return _ok(
        path=str(real),
        is_dir=True,
        entries=entries,
        n_entries=len(entries),
        truncated=truncated,
    )


# ─── 2. find_files ──────────────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.fs_tools")
def find_files(
    path: str,
    pattern: str = "*",
    recursive: bool = True,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    """Find files matching a glob pattern under a directory (like ``find`` / glob).

    Example: ``find_files("data/ai4i/bogie/archive", "*.csv")`` returns every CSV in
    that folder. Use this to locate datasets by extension/name when you do not know
    the exact filename. Also accepts an ``s3://bucket/prefix/`` URI → a name-filtered
    S3 prefix listing; ``http(s)://`` URLs fail soft (you cannot glob a URL).

    Args:
        path: Directory (workspace sandbox) or ``s3://bucket/prefix/`` URI.
        pattern: ``fnmatch`` glob applied to the file NAME (e.g. ``*.csv``,
            ``dataset*.csv``).
        recursive: If True, descend into sub-directories.
        max_results: Cap on matches returned; sets ``truncated`` when exceeded.
    """
    scheme = classify(path)
    if scheme is PathScheme.S3:
        return _find_files_s3(path, pattern, max_results)
    if scheme is PathScheme.HTTP:
        return _err(path, "find is not supported for a URL (cannot glob an http(s):// URL)")

    try:
        real = _resolve_within_sandbox(path)
    except PermissionError as e:
        return _err(path, str(e))
    except OSError as e:
        return _err(path, f"cannot resolve path: {e}")

    if not real.exists():
        return _err(str(real), f"path not found: {path}")
    if not real.is_dir():
        return _err(str(real), f"not a directory: {path}")

    matches: list[str] = []
    truncated = False
    try:
        iterator = real.rglob("*") if recursive else real.iterdir()
        for child in iterator:
            try:
                if not child.is_file():
                    continue
            except OSError:
                continue
            if _is_hidden_or_junk(child.name):
                continue
            if not fnmatch.fnmatch(child.name, pattern):
                continue
            if len(matches) >= max_results:
                truncated = True
                break
            matches.append(str(child))
    except OSError as e:
        return _err(str(real), f"failed to search directory: {e}")

    matches.sort()
    return _ok(
        path=str(real),
        pattern=pattern,
        matches=matches,
        n_matches=len(matches),
        truncated=truncated,
    )


# ─── 3. search_in_files ──────────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.fs_tools")
def search_in_files(
    path: str,
    query: str,
    pattern: str = "*",
    max_results: int = DEFAULT_MAX_HITS,
    ignore_case: bool = True,
) -> dict[str, Any]:
    """Search text-file contents for a substring under a directory (ripgrep-style).

    Returns each matching file with the 1-based line number and the matched line
    (trimmed). Binary and oversized files are skipped automatically. Content search
    is LOCAL-only — ``s3://`` and ``http(s)://`` references fail soft (use
    ``list_directory``/``find_files`` to discover remote objects, then read one).

    Args:
        path: Directory to search within (confined to the workspace sandbox).
        query: Substring to look for in each text line.
        pattern: ``fnmatch`` glob restricting which files are scanned (e.g. ``*.csv``).
        max_results: Cap on hits returned; sets ``truncated`` when exceeded.
        ignore_case: Case-insensitive matching when True.
    """
    scheme = classify(path)
    if scheme is PathScheme.S3:
        return _err(path, "content search is not supported for an s3:// URI (list the prefix, then read a file)")
    if scheme is PathScheme.HTTP:
        return _err(path, "content search is not supported for a URL (cannot glob an http(s):// URL)")

    try:
        real = _resolve_within_sandbox(path)
    except PermissionError as e:
        return _err(path, str(e))
    except OSError as e:
        return _err(path, f"cannot resolve path: {e}")

    if not real.exists():
        return _err(str(real), f"path not found: {path}")
    if not real.is_dir():
        return _err(str(real), f"not a directory: {path}")

    needle = query.lower() if ignore_case else query
    hits: list[dict[str, Any]] = []
    files_scanned = 0
    truncated = False

    try:
        for child in sorted(real.rglob("*")):
            if truncated:
                break
            try:
                if not child.is_file():
                    continue
            except OSError:
                continue
            if _is_hidden_or_junk(child.name):
                continue
            if not fnmatch.fnmatch(child.name, pattern):
                continue
            try:
                raw = child.read_bytes()[:_PER_FILE_SEARCH_BYTES]
            except OSError:
                continue
            if _looks_binary(raw):
                continue  # skip binary, never crash
            files_scanned += 1
            text = raw.decode("utf-8", errors="ignore")
            for line_no, line in enumerate(text.splitlines(), start=1):
                hay = line.lower() if ignore_case else line
                if needle in hay:
                    hits.append(
                        {
                            "file": str(child),
                            "line_no": line_no,
                            "line": line[:_LINE_TRIM],
                        }
                    )
                    if len(hits) >= max_results:
                        truncated = True
                        break
    except OSError as e:
        return _err(str(real), f"failed to search files: {e}")

    return _ok(
        path=str(real),
        query=query,
        pattern=pattern,
        hits=hits,
        n_hits=len(hits),
        files_scanned=files_scanned,
        truncated=truncated,
    )


# ─── 4. read_file ────────────────────────────────────────────────────────────────


def _read_file_remote(path: str, cap: int, head: int | None, tail: int | None) -> dict[str, Any]:
    """Remote (S3/HTTP) ``read_file``: capped byte read via ``pathref.open_bytes``.

    Applies the SAME 256 KB byte cap as the local path (via ``open_bytes(max_bytes=cap)``),
    the same binary sniff, and the same head/tail line slicing on the bytes actually
    read. ``truncated`` is set when the capped read may have stopped short (we cannot
    cheaply know the full remote size, so a full-cap read conservatively flags truncated).
    """
    try:
        ref = resolve_ref(path)
    except PathRefError as e:
        return _err(path, str(e))

    # Read cap+1 so we can DETECT truncation (more bytes exist beyond the cap).
    try:
        raw = open_bytes(ref, max_bytes=cap + 1)
    except PathRefError as e:
        return _err(path, str(e))
    except Exception as e:
        return _err(path, f"failed to read remote file: {e}")

    truncated = len(raw) > cap
    raw = raw[:cap]

    if _looks_binary(raw[:_BINARY_SNIFF_BYTES]):
        return _err(path, "binary or non-text file", is_binary=True)

    bytes_read = len(raw)
    content = raw.decode("utf-8", errors="ignore")

    if head is not None:
        content = "\n".join(content.splitlines()[: max(head, 0)])
        if content:
            content += "\n"
    elif tail is not None:
        lines = content.splitlines()
        content = "\n".join(lines[-max(tail, 0):]) if tail and tail > 0 else ""
        if content:
            content += "\n"

    return _ok(
        path=path,
        size=bytes_read,
        bytes_read=bytes_read,
        truncated=truncated,
        is_binary=False,
        content=content,
    )


@tool
@tool_error_boundary(source="tools.fs_tools")
def read_file(
    path: str,
    max_bytes: int = MAX_READ_BYTES,
    head: int | None = None,
    tail: int | None = None,
) -> dict[str, Any]:
    """Safely read a small TEXT file (headers, README, spec, config) — capped.

    Reads at most ``max_bytes`` (256 KB default) and refuses binary files. With
    ``head=N``/``tail=N`` returns the first/last N lines. This is for SMALL text
    peeks only — for multi-MB CSV data, discover the file then use
    ``describe_dataset``/``load_dataset`` instead of reading it here. Also accepts
    ``s3://`` URIs (resolved AWS profile) and ``http(s)://`` URLs (anonymous,
    SSRF-guarded) — the same 256 KB cap applies to remote reads.

    Args:
        path: File path (workspace sandbox) or ``s3://`` / ``http(s)://`` reference.
        max_bytes: Hard cap on bytes read (never exceeds 256 KB even if larger).
        head: If set, return only the first N lines.
        tail: If set, return only the last N lines.
    """
    cap = min(max_bytes, MAX_READ_BYTES)

    if classify(path) is not PathScheme.LOCAL:
        return _read_file_remote(path, cap, head, tail)

    try:
        real = _resolve_within_sandbox(path)
    except PermissionError as e:
        return _err(path, str(e))
    except OSError as e:
        return _err(path, f"cannot resolve path: {e}")

    if not real.exists():
        return _err(str(real), f"file not found: {path}")
    if real.is_dir():
        return _err(str(real), f"path is a directory, not a file: {path}", is_dir=True)

    try:
        size = real.stat().st_size
        with open(real, "rb") as fh:
            # Always sniff the first chunk for a binary refusal BEFORE reading more,
            # so we never slurp a multi-MB binary.
            sniff = fh.read(min(_BINARY_SNIFF_BYTES, cap))
            if _looks_binary(sniff):
                return _err(
                    str(real),
                    "binary or non-text file",
                    is_binary=True,
                    size=size,
                )

            if tail is not None and tail > 0:
                # Tail: seek to read only the trailing window (bounded by cap), so a
                # tail peek of a huge file stays cheap. Re-sniff that window too.
                window = min(cap, size)
                fh.seek(max(size - window, 0))
                raw = fh.read(window)
                if _looks_binary(raw[:_BINARY_SNIFF_BYTES]):
                    return _err(
                        str(real), "binary or non-text file", is_binary=True, size=size
                    )
            elif head is not None and head >= 0:
                # Head: read incrementally and STOP as soon as we have N+1 newlines
                # (or hit the cap) — a 1-line header peek of a 66 MB CSV reads only a
                # few KB, never the whole file (design §4 size note).
                raw = sniff
                while raw.count(b"\n") <= head and len(raw) < cap:
                    chunk = fh.read(min(_BINARY_SNIFF_BYTES, cap - len(raw)))
                    if not chunk:
                        break
                    raw += chunk
            else:
                # No head/tail: read up to the byte cap.
                rest = fh.read(cap - len(sniff)) if cap > len(sniff) else b""
                raw = sniff + rest
    except OSError as e:
        return _err(str(real), f"failed to read file: {e}")

    bytes_read = len(raw)
    truncated = bytes_read < size
    content = raw.decode("utf-8", errors="ignore")

    # head/tail line slicing operates on the bytes actually read.
    if head is not None:
        content = "\n".join(content.splitlines()[: max(head, 0)])
        if content:
            content += "\n"
        # A clean head selection is the requested slice, not a byte-cap truncation.
        if head >= 0 and bytes_read < size and raw.count(b"\n") > head:
            truncated = False
    elif tail is not None:
        lines = content.splitlines()
        content = "\n".join(lines[-max(tail, 0):]) if tail and tail > 0 else ""
        if content:
            content += "\n"

    return _ok(
        path=str(real),
        size=size,
        bytes_read=bytes_read,
        truncated=truncated,
        is_binary=False,
        content=content,
    )


# ─── 5. file_stat ────────────────────────────────────────────────────────────────


def _file_stat_s3(path: str) -> dict[str, Any]:
    """S3 ``file_stat`` via ``head_object`` — size/mtime, no body read."""
    try:
        ref = resolve_ref(path)
    except PathRefError as e:
        return _err(path, str(e), exists=False)
    try:
        client = _s3_client()
        head = client.head_object(Bucket=ref.bucket, Key=ref.key)
    except PathRefError as e:
        return _err(path, str(e), exists=False)
    except Exception as e:
        return _err(path, f"failed to stat S3 object: {e}", exists=False)

    mtime = head.get("LastModified")
    if mtime is None:
        mtime_str = ""
    elif hasattr(mtime, "isoformat"):
        mtime_str = mtime.isoformat()
    else:
        mtime_str = str(mtime)
    key = ref.key or ""
    tail_name = key.rsplit("/", 1)[-1]
    ext = tail_name[tail_name.rindex("."):] if "." in tail_name else ""
    return _ok(
        path=path,
        type="file",
        size=int(head.get("ContentLength", 0) or 0),
        mtime=mtime_str,
        ext=ext,
        is_dir=False,
        exists=True,
    )


@tool
@tool_error_boundary(source="tools.fs_tools")
def file_stat(path: str) -> dict[str, Any]:
    """Return ``stat`` metadata for a path — size, mtime, type — without reading it.

    Cheap and safe even on huge files (no content read). Use it to check a file's
    size/type before deciding whether to ``read_file`` it or hand it to a data tool.
    Also accepts ``s3://`` URIs (via ``head_object``); ``http(s)://`` URLs are
    unsupported for a metadata-only stat and fail soft (fetch with ``read_file`` instead).

    Args:
        path: File/dir path (workspace sandbox) or ``s3://bucket/key`` URI.
    """
    scheme = classify(path)
    if scheme is PathScheme.S3:
        return _file_stat_s3(path)
    if scheme is PathScheme.HTTP:
        return _err(path, "stat is not supported for a URL (use read_file to fetch it)", exists=False)

    try:
        real = _resolve_within_sandbox(path)
    except PermissionError as e:
        return _err(path, str(e), exists=False)
    except OSError as e:
        return _err(path, f"cannot resolve path: {e}", exists=False)

    if not real.exists():
        return _err(str(real), f"path not found: {path}", exists=False)

    try:
        st = real.stat()
        is_dir = real.is_dir()
        is_symlink = real.is_symlink()
    except OSError as e:
        return _err(str(real), f"failed to stat path: {e}", exists=False)

    if is_symlink:
        ftype = "symlink"
    elif is_dir:
        ftype = "dir"
    else:
        ftype = "file"

    return _ok(
        path=str(real),
        type=ftype,
        size=st.st_size if not is_dir else 0,
        mtime=_mtime_iso(real),
        ext="" if is_dir else real.suffix,
        is_dir=is_dir,
        exists=True,
    )
