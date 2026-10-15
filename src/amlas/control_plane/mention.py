"""`@`-mention context injection — T-LAKE-07 (roadmap M7; doc 05 §8.6; US-CC.06).

The Claude-Code `@`-mention analog for ML: pull a precise **dataset / spec / run /
artifact** into the conversation context by reference, so the user can say
*"profile @dataset:silver/sensors"* or *"compare against @run:exp-042"* and the control
plane injects the resolved context (schema slice / spec summary / run metrics / artifact
metadata) WITHOUT the user pasting it.

A mention ref is ``@<kind>:<ref>`` (kind ∈ ``dataset``/``spec``/``run``/``artifact``;
a bare ``@<ref>`` defaults to ``dataset``). Resolution routes by kind to the EXISTING
seam, all READ-ONLY:

  * ``dataset`` → the ``LakeBackend`` port (``read_slice`` schema/rows; the S3 medallion
    / local-FS adapter the registry resolves) — the catalog/slice view of a tier dataset;
  * ``spec``    → the project ``specs/`` dir (the spec card the M6 framing flow writes);
  * ``run``     → the experiment-tracking leaderboard / run store (run metrics + verdict);
  * ``artifact``→ a project artifact path (model card / pipeline yaml / report) — metadata.

This is the read side of the control-plane facet (doc 06 §1.1e); the funnel
(``capability.invoke_capability``) handles flow dispatch, and the gate
(``approvals``) handles spend — a mention is a CHEAP_READ and never gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["MentionContext", "mention_context", "parse_mention"]

# ``@kind:ref`` or bare ``@ref`` (kind defaults to dataset). kind is a known noun;
# ref is the rest (a uri / name / id), allowed to contain ``/``, ``.``, ``-``, ``_``, ``:``.
_MENTION_RE = re.compile(r"^@?(?:(?P<kind>dataset|spec|run|artifact):)?(?P<ref>[\w./:\-]+)$")
_KNOWN_KINDS = ("dataset", "spec", "run", "artifact")


@dataclass
class MentionContext:
    """The resolved context of one ``@`` mention (the injected_context contract).

    Args:
        ref: The raw mention string as given (``"@dataset:silver/x"``).
        kind: The resolved kind (``dataset``/``spec``/``run``/``artifact``).
        target: The bare reference (uri / name / id) sans the ``@kind:`` prefix.
        resolved: Whether resolution found the referenced entity.
        summary: A one-line human summary the control plane injects into context.
        context: The structured payload (schema / metrics / metadata) to inject.
        adapter: The backend/source that resolved it (for the trace).
        error: An error message when ``resolved`` is False.
    """

    ref: str
    kind: str = "dataset"
    target: str = ""
    resolved: bool = False
    summary: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    adapter: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref, "kind": self.kind, "target": self.target,
            "resolved": self.resolved, "summary": self.summary,
            "context": self.context, "adapter": self.adapter, "error": self.error,
        }


def parse_mention(ref: str) -> tuple[str, str]:
    """Parse ``@<kind>:<target>`` (or bare ``@<target>``) → ``(kind, target)``.

    A missing/unknown kind defaults to ``dataset`` (the most common ML mention). Raises
    ``ValueError`` for an empty/malformed ref.
    """
    raw = (ref or "").strip()
    m = _MENTION_RE.match(raw)
    if not m or not m.group("ref"):
        raise ValueError(f"malformed mention {ref!r}; expected @<kind>:<ref> or @<ref>")
    kind = (m.group("kind") or "dataset").lower()
    if kind not in _KNOWN_KINDS:
        kind = "dataset"
    return kind, m.group("ref")


def _resolve_dataset(target: str, *, steering: Any | None, scale: Any | None) -> MentionContext:
    """Resolve a ``@dataset`` mention via the LakeBackend port (read-only slice/schema)."""
    from ..backends.ports import LakeBackend
    from ..backends.registry import REGISTRY

    try:
        resolved = REGISTRY.resolve(LakeBackend, steering=steering, scale=scale)
    except Exception as e:
        return MentionContext(ref=target, kind="dataset", target=target, resolved=False,
                              error=f"lake resolve failed: {e}")
    try:
        # A tiny schema-only slice: 1 row is enough to inject schema + tier + lineage.
        ds = resolved.adapter.read_slice(name_or_uri=target, max_rows=1)
    except Exception as e:
        return MentionContext(ref=target, kind="dataset", target=target,
                              adapter=resolved.adapter_name, resolved=False,
                              error=f"read_slice failed: {e}")
    schema = dict(getattr(ds, "schema", {}))
    tier = str(getattr(ds, "tier", ""))
    rows = int(getattr(ds, "rows", 0))
    return MentionContext(
        ref=target, kind="dataset", target=target, resolved=True,
        adapter=resolved.adapter_name,
        summary=f"dataset {target} (tier={tier or '?'}, {len(schema)} cols, ~{rows} rows sampled)",
        context={"uri": getattr(ds, "uri", target), "tier": tier, "rows": rows,
                 "schema": schema, "lineage": dict(getattr(ds, "lineage", {}))},
    )


def _resolve_file(target: str, *, kind: str, project_dir: str | Path,
                  subdir: str, exts: tuple[str, ...]) -> MentionContext:
    """Resolve a file-backed mention (spec/artifact) from a project subdir (read-only).

    Looks for ``<project>/<subdir>/<target>`` then ``<project>/<target>`` then, when no
    extension given, the first matching ``<target>.<ext>``. Returns a metadata context
    (size, lines, a short head) — never reads a huge file fully.

    Privacy note: the 40-line head preview is injected into conversation context —
    do not ``@``-mention files containing PII/secrets (spec/artifact files are
    expected to be project metadata, not raw data).
    """
    base = Path(project_dir).expanduser()
    candidates: list[Path] = []
    direct = Path(target)
    if direct.is_absolute():
        candidates.append(direct)
    candidates.append(base / subdir / target)
    candidates.append(base / target)
    if "." not in Path(target).name:
        for ext in exts:
            candidates.append(base / subdir / f"{target}{ext}")
            candidates.append(base / f"{target}{ext}")

    # Path-traversal guard (Security review, CWE-22): EVERY candidate —
    # including a user-supplied absolute path — must resolve inside the project
    # dir. `@spec:/etc/passwd` therefore cannot read outside the project.
    def _contained(p: Path) -> bool:
        try:
            return p.resolve().is_relative_to(base.resolve())
        except OSError:
            return False

    found = next((p for p in candidates if p.is_file() and _contained(p)), None)
    if found is None:
        return MentionContext(ref=target, kind=kind, target=target, resolved=False,
                              adapter="project_fs",
                              error=f"no {kind} file found for {target!r} under {base / subdir}")
    try:
        text = found.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return MentionContext(ref=target, kind=kind, target=target, resolved=False,
                              adapter="project_fs", error=str(e))
    lines = text.splitlines()
    head = "\n".join(lines[:40])
    return MentionContext(
        ref=target, kind=kind, target=target, resolved=True, adapter="project_fs",
        summary=f"{kind} {found.name} ({len(lines)} lines, {len(text)} chars)",
        context={"path": str(found), "lines": len(lines), "chars": len(text), "head": head},
    )


def _resolve_run(target: str, *, project_dir: str | Path,
                 leaderboard: Any | None) -> MentionContext:
    """Resolve a ``@run`` mention from the experiment leaderboard / run store (read-only).

    Accepts an injected ``leaderboard`` (a mapping ``run_id -> metrics`` or an object with
    a ``get_run``/``__getitem__``) for hermetic tests; falls back to the EXISTING local
    leaderboard store when present. Returns the run's metrics + verdict context.
    """
    record: Any = None
    src = "leaderboard"
    if leaderboard is not None:
        if hasattr(leaderboard, "get_run"):
            record = leaderboard.get_run(target)
        elif isinstance(leaderboard, dict):
            record = leaderboard.get(target)
        else:
            try:
                record = leaderboard[target]
            except Exception:
                record = None
    else:
        # best-effort: the EXISTING local leaderboard store (no hard dependency).
        try:
            from ..evaluation.leaderboard import load_leaderboard  # type: ignore

            board = load_leaderboard(project_dir=str(project_dir))
            record = board.get(target) if isinstance(board, dict) else None
            src = "local_leaderboard"
        except Exception:
            record = None

    if record is None:
        return MentionContext(ref=target, kind="run", target=target, resolved=False,
                              adapter=src, error=f"no run {target!r} in {src}")
    metrics = record if isinstance(record, dict) else _as_dict(record)
    return MentionContext(
        ref=target, kind="run", target=target, resolved=True, adapter=src,
        summary=f"run {target} ({len(metrics)} metric(s))",
        context={"run_id": target, "metrics": metrics},
    )


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"value": str(obj)}


def mention_context(
    ref: str,
    *,
    project_dir: str | Path = ".",
    steering: Any | None = None,
    scale: Any | None = None,
    leaderboard: Any | None = None,
) -> dict[str, Any]:
    """``T-LAKE-07`` — resolve an ``@`` mention into injectable context (US-CC.06).

    Parses ``ref`` (``@<kind>:<target>`` or bare ``@<target>``) and routes by kind to the
    EXISTING read seam (LakeBackend port for datasets; project ``specs/`` for specs;
    experiment leaderboard for runs; project artifacts for artifacts). All resolution is
    READ-ONLY — a mention is a CHEAP_READ that never spends or gates.

    Args:
        ref: The mention string (e.g. ``"@dataset:silver/sensors"``, ``"@run:exp-042"``).
        project_dir: Project root for spec/artifact/run resolution.
        steering: Optional steering profile forwarded to the lake registry resolution.
        scale: Optional data-scale hint forwarded to the lake registry resolution.
        leaderboard: Optional injected run store (mapping/object) for hermetic run resolution.

    Returns:
        The :class:`MentionContext` as a dict (``ref/kind/target/resolved/summary/
        context/adapter/error``) — the ``injected_context`` the control plane adds to the turn.
    """
    try:
        kind, target = parse_mention(ref)
    except ValueError as e:
        return MentionContext(ref=ref, resolved=False, error=str(e)).to_dict()

    if kind == "dataset":
        return _resolve_dataset(target, steering=steering, scale=scale).to_dict()
    if kind == "spec":
        return _resolve_file(target, kind="spec", project_dir=project_dir,
                             subdir="specs", exts=(".json", ".md", ".yaml", ".yml")).to_dict()
    if kind == "artifact":
        return _resolve_file(target, kind="artifact", project_dir=project_dir,
                             subdir="output", exts=(".json", ".md", ".yaml", ".yml", ".txt")).to_dict()
    if kind == "run":
        return _resolve_run(target, project_dir=project_dir, leaderboard=leaderboard).to_dict()
    # parse_mention only returns known kinds, so this is unreachable; safe fallback.
    return MentionContext(ref=ref, kind=kind, target=target, resolved=False,
                          error=f"unsupported mention kind {kind!r}").to_dict()
