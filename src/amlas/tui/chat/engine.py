"""Chat engine — processes messages and routes to appropriate system actions.

The engine works in two modes:
1. Local mode (no LLM): Slash commands + pattern matching for common requests
2. Agentic mode (Bedrock configured): Full conversational agent via Converse API

Local mode always works — slash commands are deterministic.
Bedrock mode adds real LLM responses for natural-language queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import os.path
import re
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from amlas.config._env_compat import env_with_compat

from ...config.aws_region import resolve_aws_region
from ...config.chat_model import CHAT_HISTORY_MESSAGES, chat_model_id
from ...config.settings import Settings

# Phase 11 — Phase 6–10 tools the agentic slash-commands drive (module scope so
# callers can override the dispatch layer; heavy deps import lazily in the tools).
from ...tools.data_quality import diagnose_missingness, scan_dq_patterns
from ...tools.imputation import apply_imputation_plan, recommend_imputation_plan
from ...tools.lake import read_lake_slice, write_lake_dataset
from ...tools.mlops import compute_mlops_kpis
from ...tools.synthetic import augment_imbalance, build_synthesis_spec, synthesize_tabular, validate_synthetic
from ..jobs import JOB_COMPLETE, JobManager
from .commands import (
    COMMANDS,
    format_help_text,
    parse_slash_command,
    validate_command_args,
)
from .intent_agent import IntentAgent
from .models import ChatMessage, ChatResponse
from .tool_binding_backend import ToolBindingBackend, render_agent_turn

logger = logging.getLogger(__name__)


# ── announced graceful degradation (issue #35, 2.1) ───────────────────────────
#
# Tier 4 (the raw-Converse fallback brain) is deleted, so when a real ``strands``
# package is not importable, free-form NL has no LLM and is answered by the
# deterministic tier-5 keyword responder. Announce that state at WARNING.
#
# Log once per process rather than once per engine. Compatibility validation may
# construct many engines under an inert identity shim; repeating the warning would
# turn a useful signal into noise. A normal TUI process constructs one engine and
# therefore sees the warning once at full volume.
_WARNED_CHAT_LLM_ABSENT = False


def reset_degradation_warning() -> None:
    """Re-arm the one-shot no-LLM-chat warning. Validation seam; safe anytime."""
    global _WARNED_CHAT_LLM_ABSENT
    _WARNED_CHAT_LLM_ABSENT = False


def _warn_chat_llm_absent() -> None:
    """Log the no-LLM-chat degradation warning, at most once per process."""
    global _WARNED_CHAT_LLM_ABSENT
    if _WARNED_CHAT_LLM_ABSENT:
        return
    _WARNED_CHAT_LLM_ABSENT = True
    logger.warning(
        "Chat tool-binding DISABLED — no real 'strands' package is importable, so "
        "free-form chat will be answered by the DETERMINISTIC local keyword fallback, "
        "NOT an LLM. Slash commands and the offline handlers still work normally. "
        "'strands-agents>=1.1' is a BASE dependency (pyproject.toml [project.dependencies]), "
        "so a real install should never reach this — treat it as a broken environment and "
        "reinstall. Logged once per process."
    )


# ── injected-context budget ───────────────────────────────────────
#
# Section-priority order for truncation, HIGHEST value first. When the assembled
# ``_build_context`` block exceeds the budget, sections are dropped from the BOTTOM of
# this order (least decision-relevant first) so the most useful state always survives.
#
# Rationale for the ranking: the intent hint and the awaiting-confirmation marker change
# how the CURRENT turn must be interpreted (dropping them can make a bare "yes"
# unresolvable); the live bus snapshot is the authoritative current state; session history
# and the active spec/data summary ground the answer; the directory LISTINGS ("here are 10
# csv files") are the most verbose and least often decision-relevant, so they go first.
_CONTEXT_SECTION_PRIORITY: tuple[str, ...] = (
    "## Awaiting confirmation",
    "## Intent hint",
    "## Active TUI Context",
    "## Data Summary",
    "## Active Spec",
    "## Pipeline Run",
    "## Running Jobs",
    "## Best Model",
    "## Session history",
    "## Experiment History",
    "## Attached Files",
    "## Projects",
    "## Available Specs",
    "## Available Data Files",
)
_CONTEXT_BUDGET_ENV = "AMLAS_CONTEXT_BUDGET_CHARS"
_CONTEXT_TRUNCATION_NOTE = (
    "\n## Context truncated\n"
    "Some lower-priority state sections were omitted to stay within the context budget. "
    "Ask for a specific item (e.g. `/data`, `/leaderboard`) if you need it."
)


def _context_section_rank(part: str) -> int:
    """Rank a ``_build_context`` part by priority (lower = keep for longer).

    Parts are the strings ``_build_context`` appends. A part whose first line starts with
    a known ``## `` header takes that header's rank; a continuation line (e.g. one
    ``- spec.yaml`` bullet under a header) inherits the rank of the header it follows,
    which the caller tracks. An unrecognized header sorts mid-table rather than last, so a
    NEW section added later is not silently dropped first.
    """
    head = (part.lstrip("\n").splitlines() or [""])[0]
    for rank, prefix in enumerate(_CONTEXT_SECTION_PRIORITY):
        if head.startswith(prefix):
            return rank
    return len(_CONTEXT_SECTION_PRIORITY) // 2


def resolve_context_budget_chars(default: int) -> int:
    """Resolve the aggregate context budget from env (``0``/negative disables it)."""
    raw = (os.environ.get(_CONTEXT_BUDGET_ENV) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        logger.debug("invalid %s=%r — using default %d", _CONTEXT_BUDGET_ENV, raw, default)
        return default
    return val if val > 0 else 0


def _apply_context_budget(parts: list[str], *, budget: int | None = None) -> str:
    """Join ``parts`` under an aggregate char budget, dropping lowest-priority first.

    Groups the flat ``parts`` list into sections (a ``## `` header plus the lines that
    follow it), then drops whole sections from the least-decision-relevant end until the
    joined result fits. Dropping is section-atomic — a header is never kept without its
    body, and a body never orphaned from its header. Appends a short note when anything
    was omitted so the agent knows the view is partial rather than the state being absent.

    Returns the full unbudgeted join when the budget is ``0`` (disabled) or already met.
    """
    limit = resolve_context_budget_chars(ChatEngine.CONTEXT_BUDGET_CHARS if budget is None else budget)
    full = "\n".join(parts)
    if limit <= 0 or len(full) <= limit:
        return full

    # Group into (rank, [lines]) sections; a part without a recognized header continues
    # the section above it (keeps bullet lists attached to their heading).
    sections: list[tuple[int, list[str]]] = []
    for part in parts:
        head = (part.lstrip("\n").splitlines() or [""])[0]
        is_header = head.startswith("## ")
        if is_header or not sections:
            sections.append((_context_section_rank(part), [part]))
        else:
            sections[-1][1].append(part)

    # Drop the worst-ranked sections until it fits (stable: preserves original order
    # among kept sections, so the prompt's shape is unchanged apart from omissions).
    order = sorted(range(len(sections)), key=lambda i: sections[i][0])
    dropped: set[int] = set()
    for idx in reversed(order):
        kept = [s for i, s in enumerate(sections) if i not in dropped]
        body = "\n".join("\n".join(lines) for _rank, lines in kept)
        if len(body) + len(_CONTEXT_TRUNCATION_NOTE) <= limit:
            break
        dropped.add(idx)
        if len(dropped) == len(sections):  # pragma: no cover — degenerate, keep something
            break

    kept = [s for i, s in enumerate(sections) if i not in dropped]
    body = "\n".join("\n".join(lines) for _rank, lines in kept)
    if not dropped:
        return body
    out = body + _CONTEXT_TRUNCATION_NOTE
    # Last resort: even the highest-priority section alone can exceed the budget (e.g. a
    # pathological Data Summary). Hard-clip rather than emit an oversized prompt.
    if len(out) > limit:
        out = out[: max(0, limit - len(_CONTEXT_TRUNCATION_NOTE))] + _CONTEXT_TRUNCATION_NOTE
    logger.debug(
        "context budget: %d chars → %d (dropped %d/%d sections)",
        len(full),
        len(out),
        len(dropped),
        len(sections),
    )
    return out


# Regex for extracting file paths embedded in natural language
_EMBEDDED_PATH_RE = re.compile(
    r"""((?:[~.]?/|\.\./)[\w./~\-]+\.(?:csv|parquet|json|yaml|yml|xlsx|md|txt))"""
    r"""|"""
    r"""([\w][\w./\-]*\.(?:csv|parquet|json|yaml|yml|xlsx))""",
    re.VERBOSE,
)

# Regexes for extracting a BARE DIRECTORY path (no file extension) embedded in a
# sentence — the second failure shape from the filesystem-tools investigation
# (design doc 06 §3.3): a folder pasted inside a sentence was missed by the
# extension-only regex above, so the turn never reached the tool-capable agent and
# the assistant could only ask the user to run ``ls``. These match absolute,
# home-relative, dot-relative and known-data-root directory tokens; a match is only
# ACCEPTED as a directory when it is an existing dir on disk OR ends with ``/`` (see
# :func:`_extract_embedded_directory`), so plain prose never false-matches.
_EMBEDDED_DIR_RES = (
    re.compile(r"(?:^|\s)(/(?:[\w.\-]+/)*[\w.\-]+/?)"),  # /abs/dir/path
    re.compile(r"(~/(?:[\w.\-]+/)*[\w.\-]+/?)"),  # ~/home/dir
    re.compile(r"(\.\.?/(?:[\w.\-]+/)*[\w.\-]+/?)"),  # ./ or ../ dir
    re.compile(r"((?:data|repos|specs|output|papers|results|models)/[\w./\-]+)"),  # data/foo/bar
)


def _is_directory_path(token: str) -> bool:
    """True when ``token`` denotes a directory (existing dir on disk, or ends ``/``).

    Used by the tier-3 router: a directory embedded in a sentence must go to the
    tool-capable agent (which can ``list_directory`` it), NOT to ``_cmd_data``
    (which expects a single CSV file for ``pd.read_csv``).
    """
    if not token:
        return False
    if token.endswith("/"):
        return True
    try:
        return os.path.isdir(os.path.expanduser(token))
    except OSError:
        return False


def _extract_embedded_directory(text: str) -> str | None:
    """Extract a bare DIRECTORY path embedded in a sentence (no file extension).

    Returns the first directory-like token that is an existing directory OR ends
    with ``/``; otherwise ``None``. The existence/trailing-slash gate keeps plain
    prose (and not-yet-created relative tokens) from being misread as a directory.
    """
    for pat in _EMBEDDED_DIR_RES:
        m = pat.search(text)
        if not m:
            continue
        token = m.group(1)
        # Drop a trailing sentence period that is not part of the path.
        if token.endswith(".") and not token.endswith("/."):
            token = token.rstrip(".")
        if _is_directory_path(token):
            return token
    return None


def _normalize_fs_path(raw: str) -> str:
    """Normalize a user-supplied filesystem path argument.

    Offline commands receive a raw ``args[0]`` token that a user typed or pasted.
    Python's ``pathlib.Path`` does **not** expand ``~`` and does not strip matched
    surrounding quotes or whitespace, so ``/dqscan ~/data/x.csv`` and a pasted
    ``"/some/quoted path.csv"`` both fail a bare ``Path(arg).exists()`` even when
    the file is right there. This helper makes the local filesystem contract
    correct BEFORE the existence check and before the path reaches the local tool:

    * ``None`` / non-``str`` → ``""`` (never raises);
    * strip surrounding whitespace;
    * strip a SINGLE matched pair of surrounding quotes (``"..."`` or ``'...'``);
    * ``os.path.expanduser`` so ``~`` / ``~user`` resolve to the real home dir.

    Left untouched otherwise (plain relative/absolute paths, ``s3://`` and
    ``http(s)://`` URIs — ``expanduser`` is a no-op for those), so it is safe to
    apply uniformly at every filesystem command entrypoint.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return os.path.expanduser(text)


def _looks_like_path(text: str) -> bool:
    """Detect if user input looks like a single file path (not a sentence).

    Matches:
      - Paths starting with ./ ../ / ~/
      - Relative paths with slashes and file extensions (data/foo/bar.csv)
      - Bare filenames with known data extensions (train.csv)

    Returns False for sentences (text containing spaces) unless the
    entire text is a path with escaped spaces.
    """
    text = text.strip()
    # Sentences with spaces are NOT bare paths — they go through NL extraction
    if " " in text:
        return False
    # Starts with common path prefixes
    if text.startswith(("./", "../", "/", "~/")):
        return True
    # Contains path separators and file extension
    if "/" in text and "." in text.split("/")[-1]:
        return True
    # Known data extensions (bare filename like train.csv)
    if re.match(r"^[\w./~\-]+\.(csv|parquet|json|yaml|yml|xlsx|md)$", text):
        return True
    return False


def _extract_embedded_path(text: str) -> str | None:
    """Extract a file path embedded in a natural-language sentence.

    Returns the first path-like token with a data file extension, or None.
    Examples:
      "analyze data/ai4i/bogie/archive/dataset1.csv" -> "data/ai4i/bogie/archive/dataset1.csv"
      "use ./train.csv for training" -> "./train.csv"
    """
    m = _EMBEDDED_PATH_RE.search(text)
    if m:
        return m.group(1) or m.group(2)
    return None


# Task verbs/nouns that mark an embedded-path sentence as REAL WORK (build a plan,
# diagnose, impute, train, …) rather than a bare "load/show me this file". When the
# tool-capable agent is present, a task-bearing prompt must reach it (it can
# ``load_dataset`` itself + then act) instead of being intercepted by the load-only
# ``_cmd_data`` fast-path — the TUI NL-path routing bug where a rich prompt like
# "build a data-cleansing + imputation plan for data/x.csv" only printed
# "Dataset loaded" and the agent never ran. (project_tui_nl_path_routing_bug)
_TASK_INTENT_RE = re.compile(
    r"\b("
    r"plan|diagnos\w*|imput\w*|clean\w*|cleans\w*|preprocess\w*|impute|"
    r"train\w*|tune\w*|tuning|build\w*|fit|model\w*|predict\w*|forecast\w*|"
    r"analy\w*|assess\w*|evaluat\w*|report\w*|summar\w*|profile|profil\w*|"
    r"feature\w*|engineer\w*|select\w*|"
    r"quality|missing\w*|mechanism|strateg\w*|justif\w*|"
    r"detect\w*|anomal\w*|outlier\w*|drift|leakage|balance\w*|"
    r"synth\w*|augment\w*|split|encode\w*|scale\w*|normal\w*|"
    r"recommend\w*|research|compare|explain\w*|eda"
    r")\b",
    re.IGNORECASE,
)


def _is_bare_load_request(text: str, path: str) -> bool:
    """True when an embedded-path sentence is JUST "load/show me this file" (no task).

    We strip the path token from the sentence and ask whether any analytical task
    verb/noun (:data:`_TASK_INTENT_RE`) survives. If none does — e.g. "load
    data/x.csv", "open ./train.csv", "here is my data foo.csv" — it is a bare load
    request that the deterministic ``_cmd_data`` fast-path should handle. If a task
    verb IS present ("build an imputation plan for data/x.csv"), it is real work
    that (when the agent is tool-capable) must reach the tool-bound agent instead.
    """
    remainder = text.replace(path, " ")
    return _TASK_INTENT_RE.search(remainder) is None


# Offline verb → LOCAL no-LLM handler. Offline there is no agent, so a
# task-bearing embedded-path prompt used to fall through to the load-only fast-path and
# every request — "summarize", "how does the data look", "run a quality check" — returned
# the same "Dataset loaded" preview. These patterns map the verb onto the deterministic
# slash handler that already does the work, so plain English reaches real work offline.
# Order matters: the FIRST match wins, so the more specific intent is listed first.
_OFFLINE_TASK_HANDLERS: tuple[tuple[str, str], ...] = (
    # missingness / mechanism diagnosis → /diagnose
    (r"\b(missing\w*|mcar|mar|mnar|mechanism|diagnos\w*)\b", "diagnose"),
    # quality / EDA / anomaly-and-outlier scanning → /quality
    (r"\b(quality|eda|dq|outlier\w*|anomal\w*|leakage|drift)\b", "quality"),
    # summarize / describe / profile / "how does the data look" → /describe
    (
        r"\b(summar\w*|describ\w*|profil\w*|overview|look|shape|columns?|stats?|"
        r"statistics|distribution\w*)\b",
        "describe",
    ),
)

_OFFLINE_TASK_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), handler) for pat, handler in _OFFLINE_TASK_HANDLERS
)


def _offline_task_handler(text: str) -> str | None:
    """Map an offline task-bearing prompt to a LOCAL no-LLM handler name.

    Returns ``"describe"`` / ``"quality"`` / ``"diagnose"`` when the sentence carries a
    task verb this engine can satisfy offline, else ``None`` — meaning either a bare load
    ("load data/x.csv") or a verb with no local equivalent ("train a classifier on
    data/x.csv"), both of which keep the existing ``_cmd_data`` load preview. Returning
    ``None`` rather than guessing is deliberate: the load preview is the honest answer
    when there is no offline capability behind the verb.

    Every mapped handler is pure local Python (no Bedrock, no credentials), so the
    credential-free offline contract is preserved.

    The patterns below are the discriminator, deliberately NOT
    :data:`_TASK_INTENT_RE`: that regex is the online load-vs-task gate and does not
    know "describe" or "look", both of which this must route ("how does the data look
    for data/x.csv" is the reported prompt). A bare load matches none of these patterns
    and so still returns ``None``.
    """
    path = _extract_embedded_path(text)
    remainder = text.replace(path, " ") if path else text
    for pattern, handler in _OFFLINE_TASK_RES:
        if pattern.search(remainder):
            return handler
    return None


# ─── Phase A — local /run data-path resolution (A2/A3) ────────────────────────


class RunDataPathError(Exception):
    """Raised when a ``/run`` data path can't be resolved to a local file (A2/A3).

    The message is user-actionable: it lists the attempted paths and any nearby
    candidate files (A2), or explains that a spec's ``s3://`` golden_dataset needs
    a local override in local mode (A3).
    """


def _looks_like_s3(uri: str) -> bool:
    """True iff ``uri`` is an ``s3://`` URI (cannot be read locally)."""
    return isinstance(uri, str) and uri.strip().lower().startswith("s3://")


def _nearby_candidates(requested: str, limit: int = 10) -> list[str]:
    """List candidate data files near a requested (unresolvable) path (A2).

    Searches the ``data/`` tree (and the requested path's parent, if it exists)
    for CSV/parquet files so the error can point the user at real files.
    """
    seen: list[str] = []
    roots: list[Path] = [Path("data"), Path.cwd() / "data"]
    parent = Path(requested).parent
    if parent and str(parent) not in (".", ""):
        roots.append(parent)
    for root in roots:
        try:
            if not root.is_dir():
                continue
            for pattern in ("*.csv", "*.parquet", "*/*.csv", "*/*.parquet"):
                for p in sorted(root.glob(pattern)):
                    s = str(p)
                    if s not in seen:
                        seen.append(s)
                    if len(seen) >= limit:
                        return seen
        except OSError:
            continue
    return seen


def resolve_run_data_path(data_path: str, spec_dict: dict[str, Any]) -> str:
    """Resolve the effective LOCAL data file for a ``/run`` (A2 + A3).

    Precedence (local-first):
      1. An explicit local ``data_path`` that exists → used verbatim (A3: this
         overrides the spec's ``golden_dataset`` even when that is an ``s3://`` URI).
      2. If the given path doesn't exist, attempt sensible resolution (A2):
         resolve relative to the project root, or search one level of
         subdirectories under ``data/`` for a matching filename
         (``data/ai4i2020.csv`` → ``data/ai4i/ai4i2020.csv``).
      3. If still unresolved, fall back to the spec's ``golden_dataset`` when it is
         a LOCAL path that exists.
      4. Otherwise raise :class:`RunDataPathError` with a clear, actionable message
         (listing attempted paths + nearby candidates, or explaining the s3://
         golden_dataset needs a local override).

    Returns:
        The resolved local filesystem path (string).

    Raises:
        RunDataPathError: when no local file can be resolved.
    """
    attempted: list[str] = []
    data_path = (data_path or "").strip()
    golden = spec_dict.get("golden_dataset", "") if spec_dict else ""

    # 1. Explicit local path that exists — highest precedence (A3 override).
    if data_path and not _looks_like_s3(data_path):
        p = Path(data_path)
        attempted.append(str(p))
        if p.exists():
            return str(p)

        # 2a. Resolve relative to the project root (parent of the cwd data dir).
        # 2b. Search one level of subdirs under data/ for the same filename.
        filename = p.name
        search_dirs: list[Path] = []
        data_root = Path("data")
        if data_root.is_dir():
            search_dirs.append(data_root)
            search_dirs.extend(d for d in sorted(data_root.iterdir()) if d.is_dir())
        # Also try the given path's own parent tree one level down.
        for candidate_dir in search_dirs:
            candidate = candidate_dir / filename
            attempted.append(str(candidate))
            if candidate.exists():
                logger.info(
                    "Data path '%s' resolved to '%s' (searched data/ subdirs).",
                    data_path,
                    candidate,
                )
                return str(candidate)

    # 3. Fall back to a LOCAL golden_dataset from the spec.
    if golden and not _looks_like_s3(golden):
        gp = Path(golden)
        attempted.append(str(gp))
        if gp.exists():
            return str(gp)

    # 4. Nothing resolved → clear, actionable error.
    if _looks_like_s3(golden) and (not data_path or _looks_like_s3(data_path)):
        raise RunDataPathError(
            f"golden_dataset is an s3:// URI ({golden!r}) and cannot be read in "
            f"local mode. Provide a local data path: `/run <spec> <local.csv>`, "
            f"or switch to AWS mode."
        )

    candidates = _nearby_candidates(data_path or golden)
    cand_str = ("\n  Nearby candidates:\n    " + "\n    ".join(candidates)) if candidates else ""
    raise RunDataPathError(
        f"Data file not found: `{data_path or golden}`. Attempted paths:\n    "
        + "\n    ".join(attempted or [str(data_path)])
        + cand_str
    )


# ─── Phase 9 — agent-turn → bus fold tool-family maps (extend the narrow fold) ──
#
# The Phase-5/6 fold mapped only a handful of tools (spec/dataset/run/model/problem)
# onto the ContextBus, so every OTHER real @tool the agent ran published nothing but
# the always-on ``agents`` activity record — its registry-declared DOMAIN event/panel
# never fired (the 91-case Phase-8 gap). These NAME-keyed families extend the fold so
# each tool family reaches the correct ContextBus mutation → the correct tab reacts,
# exactly as the Phase-2 §4 reaction table + the 100-use-case registry require. Each
# set is grounded in the tool's REAL purpose (analysis/transform tools mutate the
# active dataset; monitoring tools re-point the active model; flow/ETL tools launch a
# job; lake tools land a slice), NOT invented behavior.

# Analysis / DQ / validation / missingness / dedup / outlier / profiling / PII /
# transform / feature tools that operate ON a dataset → publish the active dataset
# (``set_dataset`` → the Data tab repaints). Carries ``overall_score`` → ``dq_score``
# when the tool returns one.
_DATASET_FOLD_TOOLS: frozenset[str] = frozenset(
    {
        "scan_dq_patterns",
        "check_expectations",
        "recommend_dq_rules",
        "evaluate_dq",
        "diagnose_missingness",
        "recommend_imputation_plan",
        "impute_numeric",
        "impute_categorical",
        "impute_timeseries",
        "evaluate_imputation",
        "dedupe_dataset",
        "detect_anomalies",
        "score_data_quality",
        "detect_ocr_artifacts",
        "query_lineage",
        "leakage_check_tool",
        "detect_pii",
        "redact_pii",
        "pii_compliance_gate",
        "auto_preprocess",
        "apply_transform_flow",
        "split_dataset",
        "feature_pit_dataset",
        "save_feature_manifest",
        "build_synthesis_spec",
        "llm_synthesize_rows",
        "validate_synthetic",
        "generate_eda_report",
        "run_profile",
        "score_data_quality_6dim",
    }
)
# Lake read/write tools land a slice → active dataset (source="lake_tab") + selection.
_LAKE_DATASET_FOLD_TOOLS: frozenset[str] = frozenset({"read_lake_slice", "write_lake_dataset"})
# Lake register/query/partition tools → active dataset (source="lake_tab").
_LAKE_REGISTER_FOLD_TOOLS: frozenset[str] = frozenset(
    {"register_dataset_version", "run_sql", "partition_dataset", "recommend_partition_strategy"}
)
# MLOps / monitoring / drift tools re-point the active model → MLOps + Observability.
_MODEL_FOLD_TOOLS: frozenset[str] = frozenset(
    {
        "analyze_drift",
        "configure_alert",
        "evaluate_retrain_trigger",
        "setup_model_monitor",
        "run_clarify",
        "mlops_kpi_scorecard",
        "run_guardrail_check",
        "feature_store_get",
        # MLOps registry / governance / deploy-strategy / monitoring-report tools all act
        # ON the active model, so each must re-point it (else the MLOps tab and the
        # Observability panel never react). Added when the 60 ML-* use cases were first
        # EXECUTED — the registry declared these reactions and the fold did not produce
        # them (ML-REG-002/003/006, ML-DEP-003, ML-MON-008, ML-LIFE-003/004).
        "generate_model_card_tool",
        "create_model_card",
        "query_lineage",
        "check_data_access",
        "select_deployment_strategy",
        "compile_monitoring_report",
        "analyze_production_failures",
        "carry_forward_learnings",
        "get_metrics",
        "get_recent_logs",
        "track_inference_spend",
        "update_endpoint_canary",
        "invoke_endpoint",
        "rollback_deployment",
        "retire_model",
        "deploy_endpoint",
        "package_model",
    }
)
# Evaluation / selection tools publish the active COMPLETED run → the Results tab.
# ``get_best_model``/``get_recommendation`` are handled by their own richer branch; these
# are the remaining evaluation verbs whose result names a run/score (ML-TRAIN-003/004,
# ML-HPO-004, ML-EVAL-001/002/007/008).
_RUN_FOLD_TOOLS: frozenset[str] = frozenset(
    {
        "log_experiment_run",
        "train_ensemble",
        "get_leaderboard",
        "compare_experiments",
        "build_model_comparison_table",
        "evaluate_offline_metrics",
        "search_runs",
    }
)
# Capability FLOWS whose lifecycle stage produces a completed RUN (→ Results tab) or acts
# on a MODEL (→ MLOps + Observability). ``_fold_flow_dispatch`` always publishes the job;
# these decide which additional surface must react, since a data flow, an evaluation flow,
# and a deploy flow are not observable in the same place.
_RUN_FLOW_IDS: frozenset[str] = frozenset(
    {"evaluation_flow", "baseline_train_flow", "hpo_flow", "exploration_loop", "ml_pipeline"}
)
_MODEL_FLOW_IDS: frozenset[str] = frozenset(
    {"deploy_flow", "monitor_flow", "readiness_flow", "improve_retrain_flow"}
)
# Job-producing flow/ETL tools → active job (Pipeline stage/progress + Active-Jobs).
_JOB_FOLD_TOOLS: frozenset[str] = frozenset(
    {
        "run_glue_job",
        "run_batch_transform",
        "run_processing_job",
        "run_crawler",
        "apply_imputation_plan",
        # Training / tuning dispatch also creates a tracked job (ML-TRAIN-002/003,
        # ML-HPO-001/003): the Pipeline + Active-Jobs panels must show it running.
        "train_model",
        "tune_model",
        "run_hpo",
        "run_automl",
        "submit_training_run",
        "create_labeling_job",
    }
)
# Job tools that ALSO register a local process/thread (Dashboard Local-Processes).
_PROCESS_FOLD_TOOLS: frozenset[str] = frozenset({"run_processing_job"})
# Human labels for the job ``kind`` column (else the tool name is used).
_JOB_KIND_LABELS: dict[str, str] = {
    "run_glue_job": "glue",
    "run_batch_transform": "batch_transform",
    "run_processing_job": "processing",
    "run_crawler": "crawler",
    "apply_imputation_plan": "imputation",
    # Training/tuning kinds matter beyond cosmetics: ``choose_dynamic_view`` keys the
    # Dashboard's default chart off the job kind, so "train"/"tune"/"hpo" here is what
    # makes a training job show a training curve rather than the neutral overview.
    "train_model": "train",
    "tune_model": "tune",
    "run_hpo": "hpo",
    "run_automl": "automl",
    "submit_training_run": "train",
    "create_labeling_job": "labeling",
}


class ChatEngine:
    """Processes chat messages and routes to system actions.

    Maintains conversation history, file context, and pipeline state.
    Routes slash commands to direct tool invocation and natural language
    to Bedrock Sonnet (if available) or pattern-matched handlers.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        job_manager: JobManager | None = None,
        *,
        approval_policy: Any | None = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.history: list[ChatMessage] = []
        self.context: dict[str, Any] = {
            "attachments": [],  # List of attached file paths
            "current_spec": None,  # Path to active spec
            "running_jobs": [],  # Active pipeline run IDs
            "data_summary": None,  # Summary of last attached dataset
        }
        self._pipeline_state: dict[str, Any] = {}

        # Phase 2 (TUI context integration, design §E.3/§E.4): the shared
        # ``ContextBus`` — the App owns it and injects it here on mount (exactly like
        # ``engine.jobs`` today). ``None`` by default so a headless ``ChatEngine()``
        # (every existing caller) is unchanged and null-safe: bus writes/reads are all
        # guarded by ``if self.context_bus is not None``. When set, chat handlers ALSO
        # mutate the bus and ``_build_context`` prefers its snapshot — additive, so the
        # 500+ existing callers stay green.
        self.context_bus: Any | None = None

        # Phase C2 (session-wide context store, design §D.2/§D.5): the append-only
        # ``SessionLedger`` timeline that LAYERS ON TOP of ``context_bus``. The App owns
        # it and injects it here on mount (mirrors ``context_bus`` exactly). ``None`` by
        # default so a headless ``ChatEngine()`` (every existing caller) is unchanged and
        # null-safe: every ledger write/read is guarded by ``if ... is not None``. When
        # set, the engine records turns/tools/commands here (all modes, offline included)
        # and ``_build_context`` folds its bounded history AFTER the authoritative bus
        # snapshot (precedence: bus = current, ledger = history) — additive/null-safe.
        self.session_ledger: Any | None = None

        # Phase 11 — the session domain-steering "dial" (master plan §3.1). Set
        # by `/steer set ...`, honored (HARD) by `/impute`, `/synth`, `/dqscan`.
        self.steering_profile: Any | None = None

        # DP-2 (Phase 11): background-execution manager. The DEFAULT uses the
        # inline executor, so a headless ``ChatEngine()`` (every existing caller)
        # runs long ops synchronously inside ``submit`` — the job is already
        # ``complete`` when ``submit`` returns and the long-op handlers return
        # their real ``*_complete`` response exactly as before. The Textual app
        # injects a thread-backed JobManager so ``submit`` returns immediately and
        # the handlers return ``job_submitted`` (non-blocking input). See design §3
        # and ``docs/audit/phase11-background-and-routing.md``.
        self.jobs: JobManager = job_manager or JobManager()

        # Pending input state: tracks when engine asked user for a specific input.
        # Values: None, "data_path", "spec_path"
        self._awaiting: str | None = None

        # Hop-1 C-2 (assessment §D.2): a cross-turn pending y/n confirmation marker.
        # When a turn's tool-bound agent returned ``needs_confirmation`` for a
        # state-changing tool, this holds ``{"tool": name, "tools": [...]}`` so the NEXT
        # turn's injected context explicitly states the awaiting y/n (a bare "yes" then
        # resolves deterministically). Cleared once a later turn resolves it (executes
        # or produces no new pending). ``None`` by default → no awaiting line (null-safe).
        self._awaiting_confirmation: dict[str, Any] | None = None

        # Intent agent for natural language → action execution
        self._intent_agent = IntentAgent()

        # Phase B2.2 (design §A.3): the runtime Bedrock-chat gate. Issue #35 deleted
        # the second chat brain (the raw-Converse ``BedrockChatBackend``, tier 4), so
        # this flag now gates exactly one tier — the tier-3c tool agent, whose Strands
        # agent is built per turn but whose credentials/mode are NOT re-read on a mode
        # switch. Offline mode flips this runtime flag rather than mutating an env var,
        # which the NL routing guard honors to BYPASS the LLM tier (degrade to the
        # offline/regex path) WITHOUT rebuilding anything. Default ``True`` preserves
        # the exact behavior of every existing ``ChatEngine()`` (additive/null-safe):
        # ``AgenticMLApp.set_runtime_mode`` sets it ``False`` only for Offline.
        self.bedrock_enabled: bool = True

        # DP-1 (Phase 10): embedded tool-bound agent — the ONLY LLM chat brain
        # (issue #35). Under the dev/test shim this reports is_tool_capable == False,
        # so free-form NL degrades to the deterministic local fallback (tier 5) and
        # the prior tests are unaffected. When real Strands + creds are present,
        # free-form NL reaches this agent for real function-calling (design §2 / §4).
        self.tool_backend = (
            ToolBindingBackend(self.settings, approval_policy=approval_policy)
            if approval_policy is not None else ToolBindingBackend(self.settings)
        )
        if self.tool_backend.is_tool_capable:  # nosemgrep
            logger.info("Chat engine tool-binding ENABLED (embedded Strands agent)")
        else:
            # Announce the degradation instead of letting it be silent (task 6.3):
            # with tier 4 gone there is no second brain behind this, only tier 5.
            _warn_chat_llm_absent()

    def reconfigure_mode(self, *, bedrock_enabled: bool) -> None:
        """Re-apply runtime-mode knobs to the chat brain (Phase D1, design §B.1).

        A mode switch is LIVE for the backend registry (it re-reads env per
        ``resolve`` — ``registry.py:232``) but the tool-agent tier is gated by a
        runtime FLAG, not rebuilt (§A.3 of the B2 design; §6 invariant of the D
        design). This is the single seam the App calls so the NEXT turn's routing
        honors the new mode:

        * sets ``self.bedrock_enabled`` — since issue #35 deleted the raw-Converse
          fallback brain (tier 4), this now gates exactly ONE tier: the tier-3c
          tool agent at ``_route_free_form_nl`` (see that guard);
        * never rebuilds anything (null-safe / no cred re-read — the tier is simply
          bypassed when the flag is ``False``, degrading to tier 5).

        Idempotent; never raises. Additive: a bare ``ChatEngine()`` that never
        calls this stays byte-identical (``bedrock_enabled`` defaults ``True``).
        """
        self.bedrock_enabled = bool(bedrock_enabled)

    def _aws_gate(self, action_label: str) -> ChatResponse | None:
        """Phase E2 FIX-2 — mode-aware AWS gate for cloud side-effect commands.

        The AWS-gated slash commands (``/upload``, ``/download``, ``/lake
        read|write``) previously keyed their gate ONLY on bucket/cred presence
        (``AMLAS_S3_BUCKET`` autoloaded from ``.env``), never on the runtime
        MODE. So in **Offline** mode — where ``.env`` still resolves a real
        bucket + ``~/.aws/credentials`` — they executed REAL S3/Glue calls (and
        ``/lake read`` leaked a raw ``EntityNotFoundException``). That violates
        the contract "Offline means NO AWS" (audit §2 P-2, §3.3, FIX-2).

        This shared guard consults the SAME predicate the backend registry uses
        to pick local-vs-AWS adapters — :func:`registry._running_offline` (True
        when ``AMLAS_OFFLINE``/``AMLAS_BACKENDS_OFFLINE`` is set). When offline it
        returns an HONEST gate ``ChatResponse``
        that the caller returns BEFORE any boto3/tool call; otherwise it returns
        ``None`` and the caller proceeds exactly as before (Local / Full-AWS are
        NOT over-gated — proven by the stubbed non-offline test).

        The import is deliberately deferred to CALL time (not module scope) so a
        callers can replace ``amlas.backends.registry._running_offline`` and have
        it honored here (a module-level ``from ... import`` would bind the original
        function object).

        Args:
            action_label: The user-facing command name for the copy, e.g. ``/upload``.

        Returns:
            An honest-gate :class:`ChatResponse` when Offline; ``None`` otherwise.
        """
        from amlas.backends.registry import _running_offline

        if not _running_offline():
            return None
        return ChatResponse(
            message=(
                f"☁️ `{action_label}` requires AWS. You're in **Offline mode** — "
                "switch to **Local** or **Full-AWS** mode (top-left selector), or unset "
                "`AMLAS_OFFLINE`/`AMLAS_BACKENDS_OFFLINE`, to enable cloud actions."
            ),
            action="aws_gated",
            data={"command": action_label, "offline": True},
        )

    async def process_message(self, message: str, thought_trace: Any | None = None) -> ChatResponse:
        """Top-level chat boundary — capture any escaped error, then re-raise.

        The single chat engine boundary the ChatPane worker calls. It delegates to
        :meth:`_process_message_impl` (the real routing) and, on any UNEXPECTED escaped
        exception, funnels it into the centralized ``ERROR_LOG`` with
        ``source="tui.chat.engine"`` BEFORE re-raising it unchanged — so the pane's
        existing ``_process_and_respond`` handling (its ``finally`` cleanup + toast) is
        byte-for-byte preserved. The capture is guarded so telemetry can never mask the
        original error. Nearly every turn returns a ``ChatResponse`` normally (the per-
        handler try/excepts below already convert expected failures into a response);
        this net catches only a truly unexpected escape.
        """
        try:
            return await self._process_message_impl(message, thought_trace=thought_trace)
        except Exception as exc:
            try:
                from ...telemetry.error_log import capture_error

                capture_error(exc, source="tui.chat.engine")
            except Exception:  # nosec B110
                pass
            raise

    async def _process_message_impl(self, message: str, thought_trace: Any | None = None) -> ChatResponse:
        """Process a user message and route it to the right handler (DP-3, design §4).

        The FINALIZED routing decision flow — deterministic tiers first, the
        tool-bound LLM agent as the free-form NL brain, the deterministic local
        keyword responder as the graceful fallback:

        0.   Awaiting input → if the engine previously asked for a path, accept it.
        0.5. **Escape hatch** (``!…`` or ``/ask …``) → FORCE the tool-bound agent,
             bypassing the regex intent (power users composing multiple tools, §4).
        1.   Slash commands → deterministic ``_cmd_*`` handler (never reach the LLM).
        2.   Bare file paths → auto-route to ``/data`` (never reach the LLM).
        3.   NL with an embedded file path → extract & load.
        3b.  Regex ``IntentAgent`` (HIGH-confidence fast-path) → deterministic
             handler. KEPT (design §4) — a clear "show leaderboard" stays a free,
             offline, instant regex hit and never costs an LLM round-trip.
        3c.  Free-form NL → tool-bound Strands agent (real function-calling), ONLY
             when ``is_tool_capable``. Long-op tools enqueue DP-2 jobs. This is the
             ONLY LLM chat brain (issue #35 deleted the tier-4 raw-Converse one).
        5.   Local keyword fallback (deterministic, always works, offline). NOT an
             LLM — the tier numbering keeps its gap where tier 4 used to sit.

        Under the dev/test shim ``is_tool_capable`` is False, so tier 3c is skipped
        and tier 5 answers exactly as before — the prior tests are unaffected.
        """
        text = message.strip()

        # Record user message
        user_msg = ChatMessage(role="user", content=message)
        self.history.append(user_msg)

        # FIX-24: publish a RUNNING chat-agent activity at turn START so the Dashboard
        # lights up the instant a prompt is sent — in EVERY mode (offline/local
        # fallback or tool agent), unlike the post-turn fold which only fires after a
        # completed tool turn under bedrock+tool-capable. The completing fold
        # (_publish_agent_activity, status="complete") then flips this to done. Null-safe.
        self._publish_agent_turn_start(text)

        # 0. Check if we're awaiting specific input from a prior exchange
        if self._awaiting == "data_path" and (_looks_like_path(text) or os.path.exists(text)):
            self._awaiting = None
            response = await self._cmd_data([text])
        elif self._awaiting == "spec_path" and (_looks_like_path(text) or os.path.exists(text)):
            self._awaiting = None
            response = await self._cmd_attach([text])

        # 0.5. Escape hatch — force the tool-bound agent (skip regex intent).
        # A leading ``!`` or ``/ask `` routes straight to the LLM tool agent so a
        # power user can compose multiple tools even when a regex intent would
        # otherwise match (design §4). When NOT tool-capable, the stripped query
        # degrades to the deterministic local fallback (tier 5).
        elif (forced := self._extract_forced_agent_query(text)) is not None:
            response = await self._route_free_form_nl(forced, forced, thought_trace=thought_trace)

        # 1. Slash commands always handled locally
        elif (parsed := parse_slash_command(message)) is not None:
            cmd_name, args = parsed
            # Distinguish real commands from absolute paths (e.g., /home/user/data.csv)
            if cmd_name in COMMANDS:
                response = await self._handle_slash_command(cmd_name, args)
            elif _looks_like_path(text):
                # Absolute path misidentified as slash command
                response = await self._cmd_data([text])
            else:
                response = await self._handle_slash_command(cmd_name, args)

        # 2. Bare file path detected — auto-route to /data handler
        elif _looks_like_path(text):
            response = await self._cmd_data([text])

        # 3. Natural language with an embedded file path.
        # A BARE load request ("load data/x.csv", "open ./train.csv") → the
        # deterministic load-only ``_cmd_data`` fast-path (free, offline, instant).
        # But a TASK-BEARING prompt ("build an imputation plan for data/x.csv",
        # "diagnose missingness in foo.csv and apply a strategy") is real work: when
        # the tool-bound agent is available it must reach tier 3c — the agent can
        # ``load_dataset`` itself and then actually DO the task — instead of being
        # swallowed by the loader (the fix for the TUI NL-path routing bug, where a
        # rich prompt only printed "Dataset loaded" and the agent never ran). Under
        # the shim (``tool_backend`` absent / not tool-capable) we keep the load-only
        # behavior so the deterministic offline suite is unchanged.
        elif (embedded := _extract_embedded_path(text)) is not None:
            if (
                self.bedrock_enabled
                and self.tool_backend
                and self.tool_backend.is_tool_capable  # nosemgrep
                and not _is_bare_load_request(text, embedded)
            ):
                response = await self._route_free_form_nl(message, text, thought_trace=thought_trace)
            elif (local := _offline_task_handler(text)) is not None:
                # No agent available (offline / shim / Bedrock off) but the sentence DOES
                # carry a task verb we can satisfy locally. Dispatch to the
                # matching deterministic no-LLM handler instead of returning the generic
                # load preview, which used to swallow every offline request that happened
                # to name a dataset. Still credential-free: these handlers are pure Python.

                # Load FIRST, then run the task: ``_cmd_data`` is the only writer of
                # ``context["data_summary"]``, which later turns read to resolve "now check
                # quality" against the active dataset (see ``_execute_intent``). Dispatching
                # straight to the task handler would leave the dataset unregistered and
                # silently break the multi-turn follow-up. The load reply is discarded — the
                # user asked for the task, not the preview — but its side effect is kept.
                load_resp = await self._cmd_data([embedded])
                if load_resp.action == "error":
                    # The path did not resolve; the loader's message is the useful one.
                    response = load_resp
                else:
                    handler = {
                        "describe": self._cmd_describe,
                        "quality": self._cmd_quality,
                        "diagnose": self._cmd_diagnose,
                    }[local]
                    response = await handler([embedded])
            else:
                # Load the data file found in the sentence. This is the path for a
                # bare load, for the offline/shim config (no agent), AND for Offline
                # mode (``bedrock_enabled=False``) — where the agent CANNOT run, so we
                # must still load the referenced dataset rather than dropping the turn
                # to the local keyword fallback (which would never load the file). Also
                # the fallback for a task verb with no local equivalent ("train ...").
                response = await self._cmd_data([embedded])

        # 3a. Natural language with an embedded DIRECTORY path → tool agent.
        # The extension-only embedded-path extractor (tier 3) misses a bare folder
        # pasted inside a sentence (design doc 06 §3.3), so the broken user story
        # ("read this data here … /Volumes/.../bogie/archive") never reached a tool.
        # A directory can't be ``pd.read_csv``'d by ``_cmd_data``; route it to the
        # tool-capable agent, which now has ``list_directory``/``find_files`` to
        # discover the CSVs itself — exactly like claude-code (never asks for ``ls``).
        elif _extract_embedded_directory(text) is not None:
            response = await self._route_free_form_nl(message, text, thought_trace=thought_trace)

        # 3b. Regex IntentAgent — kept as a fast, offline, deterministic path ONLY
        # when the tool-bound agent is NOT available (the shim / no-Strands runtime).
        # When the real agent IS available it must ALWAYS be engaged for free-form NL
        # (the agent uses judgment; a regex must never answer INSTEAD of it), so the
        # classification is demoted to a soft HINT passed into the agent's context
        # rather than executed directly. This keeps the offline suite's deterministic
        # intent tests green (tool_backend disabled) while making the live surface
        # agent-first (user directive: intelligence first, classifiers only hint).
        elif (
            not (self.tool_backend and self.tool_backend.is_tool_capable)  # nosemgrep
            and (intent_result := self._intent_agent.classify(text)) is not None
        ):
            intent, params = intent_result
            response = await self._execute_intent(intent, params, text)

        # 3c/5. Free-form NL → tool agent → deterministic local fallback. When the
        # tool agent is engaged, pass any regex-intent classification as a non-binding
        # hint so the agent has a weighted signal without being overridden by it.
        else:
            intent_hint = self._intent_agent.classify(text)
            response = await self._route_free_form_nl(
                message, text, intent_hint=intent_hint, thought_trace=thought_trace
            )

        # Record assistant response. Hop-1 C-1: persist the compact per-turn
        # tool-result summary (set by ``_route_free_form_nl`` from the structured tool
        # trace) so ``_build_agent_history`` can replay what was computed/looked up to
        # the next turn. Empty for non-tool turns → history replay unchanged.
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.message,
            action_taken=response.action,
            tool_summary=str(response.data.get("tool_summary", "") or ""),
        )
        self.history.append(assistant_msg)

        # Hop-1 C-2: maintain the cross-turn pending-confirmation marker. A tool-agent
        # turn always reports ``pending_confirmation`` in its data (a list of tools that
        # returned ``needs_confirmation``). Set the marker when non-empty so the NEXT
        # turn's context states the awaiting y/n; clear it once a later tool-agent turn
        # resolves it (executes / no new pending). Turns that don't touch the tool agent
        # (slash commands, fallbacks) omit the key and leave the marker untouched.
        if "pending_confirmation" in response.data:
            pending = response.data.get("pending_confirmation") or []
            if pending:
                self._awaiting_confirmation = {"tool": pending[0], "tools": list(pending)}
            else:
                self._awaiting_confirmation = None

        # Phase C2 (design §D.3/§D.6): fold this completed turn into the session-wide
        # ``SessionLedger`` timeline. This is the DETERMINISTIC accumulation seam that
        # runs in EVERY mode — slash commands, offline local-fallback NL, and tool-agent
        # turns all pass through here — so the ledger fills even when ``bedrock_enabled``
        # is False (the crux offline requirement). Null-safe: a no-op when no ledger is
        # attached (headless / every existing caller) and never raises.
        self._record_turn_to_ledger(message, response)

        return response

    def _find_state_file(self, filename: str) -> Path | None:
        """Resolve a state file (e.g. evaluation_store.json) without unbounded rglob.

        CTX-01: the previous implementation called ``Path(".").rglob(...)`` on
        every chat turn, recursively walking output/, repos/, and nested
        project trees, then picking an arbitrary
        ``[0]``. This is slow and order-dependent. Instead we check a small,
        deterministic set of candidate directories derived from Settings, plus a
        shallow (depth-limited) scan of the output dir as a fallback.
        """
        candidates = [
            Path(self.settings.local.output_dir) / filename,
            Path(self.settings.local.output_dir).resolve() / filename,
            Path(self.settings.local.workspace_dir) / filename,
            Path(filename),
        ]
        for c in candidates:
            if c.exists():
                return c

        # Shallow fallback: scan only the immediate children of the output dir
        # (one level), never the whole tree.
        out = Path(self.settings.local.output_dir)
        if out.is_dir():
            for child in out.iterdir():
                if child.is_dir():
                    candidate = child / filename
                    if candidate.exists():
                        return candidate
        return None

    @staticmethod
    def _extract_forced_agent_query(text: str) -> str | None:
        """Return the query when the user forces the tool agent (``!…`` / ``/ask …``).

        The DP-3 escape hatch (design §4): a leading ``!`` or the ``/ask`` command
        forces free-form NL to the tool-bound agent, bypassing the regex intent.
        Returns the stripped query (may be empty) or ``None`` if not a forced turn.
        """
        if text.startswith("!"):
            return text[1:].strip()
        # ``/ask`` (and ``/ask <query>``) — case-insensitive, exact command token.
        lowered = text.lower()
        if lowered == "/ask" or lowered.startswith("/ask "):
            return text[4:].strip()
        return None

    async def _route_free_form_nl(
        self,
        message: str,
        text: str,
        intent_hint: tuple[str, dict[str, Any]] | None = None,
        thought_trace: Any | None = None,
    ) -> ChatResponse:
        """Tiers 3c→5: the tool-bound agent, then the deterministic local fallback.

        Shared by the normal free-form NL branch and the ``!``/``/ask`` escape
        hatch. Returns the first non-None response in priority order. The tool-bound
        agent tier is taken ONLY when ``is_tool_capable`` (real Strands present),
        so under the shim this collapses to the deterministic offline behavior.

        Issue #35 removed the intervening tier 4 (the raw-Converse
        ``BedrockChatBackend``): there is now exactly ONE LLM brain here. That tier
        was already unreachable — ``strands-agents`` is a BASE dependency, so
        ``is_tool_capable`` is True in any real install and ``chat_with_tools``
        returns ``None`` only when it is False, so tier 4's ``response is None``
        guard could never open while the tier-3c gate did.

        ``intent_hint`` is the (non-binding) regex ``IntentAgent`` classification for
        this turn, when one matched. It is injected into the agent's context as a soft
        signal — the agent uses its own judgment and is never forced to follow it
        (user directive: intelligence first, classifiers only hint/weight).

        ``thought_trace`` (Feature 2, additive) is the per-turn ephemeral
        :class:`~amlas.tui.chat.thought_trace.ThoughtTrace` for the "stock ticker".
        When present it is threaded into ``chat_with_tools`` so live tool/step events
        feed the transient ticker. It is NEVER persisted (design §3.3). Default
        ``None`` keeps every existing caller/test byte-identical.
        """
        response: ChatResponse | None = None

        # 3c. DP-1 tool-bound agent (the ONE NL brain) — only when tool-capable AND
        # the runtime chat gate is ON. Phase D1 (design §B.2): the tool agent drives
        # the real Strands agent, which uses Bedrock — so in Offline mode
        # (``bedrock_enabled=False``) it must be bypassed, degrading to the
        # deterministic local fallback (tier 5). Default ``bedrock_enabled=True`` +
        # shim ``is_tool_capable=False`` keeps every existing engine/test
        # byte-identical (the branch was already skipped under pytest).
        if self.bedrock_enabled and self.tool_backend and self.tool_backend.is_tool_capable:  # nosemgrep
            context = self._build_context(intent_hint=intent_hint)
            # STATEFUL CHAT FIX: replay the PRIOR conversation turns to the agent so
            # this turn is not stateless. Without this, a fresh Strands ``Agent`` was
            # built every turn with an empty ``.messages`` and only the latest prompt
            # was sent — so an affirmative like "yes to both" could not resolve against
            # the two questions the assistant asked last turn. ``_build_agent_history``
            # returns the bounded user+assistant history (the current user message is
            # the prompt, never duplicated into it); the ephemeral ticker is NOT in it.
            history = self._build_agent_history()
            turn = await asyncio.to_thread(
                self.tool_backend.chat_with_tools,
                message,
                context,
                thought_trace=thought_trace,
                history=history,
            )
            if turn is not None:
                response = ChatResponse(
                    message=render_agent_turn(turn),
                    action="tool_agent_turn",
                    data={
                        "tools_used": [e.tool_name for e in turn.executions if e.status == "executed"],
                        "pending_confirmation": [
                            e.tool_name for e in turn.executions if e.status == "needs_confirmation"
                        ],
                        # Hop-1 C-1: a compact per-tool result summary (one short
                        # "tool: k=v" line per executed tool, from the structured
                        # ``ToolExecution.result_summary`` — NOT a raw dump, NOT the
                        # ephemeral ticker) that ``process_message`` persists onto the
                        # assistant ChatMessage for bounded history replay next turn.
                        "tool_summary": self._summarize_tool_results(turn.executions),
                    },
                )
                # Phase 5 (design §E.4/§E.8, GP-2/GO-4): fold this turn's local-agent
                # tool trace onto the shared bus — one ``AgentActivity`` record for the
                # ``tui_chat_agent`` plus any spec/dataset the agent produced — so the
                # local chat agent's work appears in the Dashboard's AgentActivityPanel
                # (alongside cloud swarm agents) and its generated spec/dataset show up
                # on the Specs/Data/Pipeline tabs for review. Null-safe when no bus.
                self._fold_agent_turn(turn, message)

        # 5. Local keyword fallback (always works, offline).
        if response is None:
            response = await self._handle_natural_language(message)

        return response

    # Bound on how many PRIOR messages are replayed to the tool agent each turn, so
    # the replayed prompt never grows unbounded.
    #
    # Issue #35 criterion 4.1: an ALIAS of the single bound in
    # [`config.chat_model.CHAT_HISTORY_MESSAGES`](../../config/chat_model.py:50). It used
    # to be its own ``40`` carrying the comment "Matches
    # ``BedrockChatBackend.MAX_HISTORY_ENTRIES``" — i.e. a duplicate the reader was asked
    # to keep in sync by hand. Both names now read the one definition. The attribute name
    # is kept: it is this class's public window bound and is asserted on.
    MAX_AGENT_HISTORY_MESSAGES = CHAT_HISTORY_MESSAGES

    def _build_agent_history(self) -> list[dict[str, Any]]:
        """Build the bounded prior-turn history to replay to the tool agent.

        Returns the accumulated conversation as Strands ``Message`` dicts
        (``{"role": "user"|"assistant", "content": [{"text": ...}]}``) — the exact
        shape ``Agent(messages=[...])`` accepts — so a fresh per-turn Agent still
        sees the earlier turns (the stateful-chat fix).

        Invariants:
          * EXCLUDES the current user message. ``process_message`` appends it to
            ``self.history`` BEFORE routing here ([`engine.py:253`](engine.py:253)),
            and it is sent separately as the agent PROMPT — replaying it too would
            double-send the current turn.
          * Only ``user``/``assistant`` roles (what Strands accepts); any transient
            ``system`` bubble is skipped. The ephemeral thought-ticker is NEVER in
            ``self.history`` (it only mounts UI widgets, design §3.3 / T-TT-05), so
            it cannot leak here by construction.
          * Bounded to the last ``MAX_AGENT_HISTORY_MESSAGES`` and trimmed so the
            replay begins on a ``user`` turn (a valid Bedrock/Strands transcript
            starts with the user).
        """
        prior = self.history[:-1] if self.history else []
        messages: list[dict[str, Any]] = []
        for m in prior:
            if m.role not in ("user", "assistant") or not (m.content or "").strip():
                continue
            content: list[dict[str, Any]] = [{"text": m.content}]
            # Hop-1 C-1 (assessment §D.2): replay the prior turn's COMPACT tool-result
            # summary so a follow-up ("what was that F1 again?") can ground on the
            # earlier tool output. It is an EXTRA content block on the SAME assistant
            # message (message count unchanged → the ``MAX_AGENT_HISTORY_MESSAGES``
            # window bound is unaffected). Sourced from the structured tool trace, so
            # the ephemeral ticker can never leak here (T-TT-05 holds by construction).
            summary = getattr(m, "tool_summary", "") or ""
            if m.role == "assistant" and summary.strip():
                content.append({"text": f"[tool results] {summary}"})
            messages.append({"role": m.role, "content": content})
        # Sliding window: keep only the most recent N messages.
        if len(messages) > self.MAX_AGENT_HISTORY_MESSAGES:
            messages = messages[-self.MAX_AGENT_HISTORY_MESSAGES :]
        # A valid transcript starts with a user turn; drop a leading assistant that
        # a window boundary may have exposed.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages

    # ─── the injected-context budget ──────────────────────────────

    # ``_build_context`` assembles a live-state block that is prepended to EVERY chat
    # turn. Each section was individually capped (``spec_files[:10]``, ``columns[:15]``,
    # ledger ``max_entries=12``) but nothing bounded the AGGREGATE, so a wide dataset plus
    # a rich session could push a single turn over the model's window — on turn ONE, where
    # no history compaction can help, because this block arrives fresh each turn and is
    # immune to conversation-manager trimming.

    # ``AMLAS_CONTEXT_BUDGET_CHARS`` tunes it; ``0`` disables the budget entirely.
    CONTEXT_BUDGET_CHARS = 24_000

    def _build_context(self, intent_hint: tuple[str, dict[str, Any]] | None = None) -> str:
        """Build rich context from REAL system state for Bedrock.

        ``intent_hint`` (optional) is the regex ``IntentAgent`` classification for the
        current turn. It is surfaced to the agent as an explicit NON-BINDING hint — a
        weighted signal the agent may use or ignore — never an instruction (user
        directive: intelligence first, classifiers only hint).
        """
        parts: list[str] = []

        # 0. Non-binding intent hint (a weighted signal, not a command).
        if intent_hint is not None:
            hint_intent, hint_params = intent_hint
            param_str = ", ".join(f"{k}={v}" for k, v in hint_params.items()) or "none"
            parts.append(
                "## Intent hint (NON-BINDING — use your own judgment)\n"
                f"A lightweight classifier guessed the user's intent may be "
                f"`{hint_intent}` (params: {param_str}). Treat this only as a weak "
                f"signal; decide for yourself whether it fits before acting."
            )

        # 0.5 Active TUI context (Phase 2, design §E.4) — rendered from the shared
        # ``ContextBus`` snapshot when the App has injected one. This is the
        # cross-tab source of truth (the spec loaded on the Specs tab, the dataset
        # selected on the Data tab, the running job) — NOT the per-turn disk re-scan
        # below. Null-safe/additive: absent a bus this whole block is skipped and the
        # legacy disk-scan sections run exactly as before.
        bus_section = self._render_bus_context()
        if bus_section:
            parts.append(bus_section)

        # 0.55 Session history (Phase C2, design §D.5) — the append-only ``SessionLedger``
        # timeline rendered AFTER the authoritative bus snapshot. PRECEDENCE: the live bus
        # above is "current/authoritative"; this ledger is the HISTORY layer (clearly
        # labeled so a stale ledger entry never overrides the live snapshot). It
        # accumulates in ALL modes (offline included), so on an offline→local mode flip the
        # very first Bedrock prompt already folds the full prior offline history. Null-safe:
        # absent a ledger (headless / every existing caller) this block is skipped entirely.
        ledger_section = self._render_session_ledger()
        if ledger_section:
            parts.append(ledger_section)

        # 0.6 Pending confirmation (Hop-1 C-2, assessment §D.2) — when the PRIOR turn
        # returned ``needs_confirmation`` for a state-changing tool, state the awaiting
        # y/n EXPLICITLY so a bare "yes" this turn resolves deterministically against a
        # concrete tool (rather than relying only on history recall of the prose). This
        # is bus-independent (works headless) and cleared once resolved. Null-safe:
        # ``_awaiting_confirmation is None`` → this whole block is skipped (unchanged).
        pending_section = self._render_pending_confirmation()
        if pending_section:
            parts.append(pending_section)

        # 1. Load evaluation store (bounded resolution — CTX-01)
        try:
            from ...evaluation.store import EvaluationStore

            store_path = self._find_state_file("evaluation_store.json")

            if store_path is not None and store_path.exists():
                store = EvaluationStore(store_path=store_path)
                results = store.get_all()
                if results:
                    parts.append(f"## Experiment History ({len(results)} results)")
                    # Include last 10 results
                    for r in results[-10:]:
                        parts.append(
                            f"- {r.algorithm}: {r.primary_metric}={r.primary_score:.4f} "
                            f"({r.status}) [{r.timestamp}]"
                        )

                    # Best result
                    successful = [r for r in results if r.status == "success"]
                    if successful:
                        best = max(successful, key=lambda x: x.primary_score)
                        parts.append(
                            f"\n## Best Model: {best.algorithm} "
                            f"({best.primary_metric}={best.primary_score:.4f})"
                        )
        except Exception:  # nosec B110
            pass

        # 2. Load project state (bounded resolution — CTX-01)
        try:
            scheduler_path = self._find_state_file("scheduler_state.json")
            if scheduler_path is not None:
                from ...scheduler.project_manager import ProjectManager

                pm = ProjectManager(str(scheduler_path.parent))
                pm.load_state()
                projects = pm.list_projects()
                if projects:
                    parts.append(f"\n## Projects ({len(projects)} total)")
                    for p in projects:
                        title = (
                            p.spec.get("title", p.project_id) if isinstance(p.spec, dict) else p.project_id
                        )
                        parts.append(f"- {title}: {p.status}")
        except Exception:  # nosec B110
            pass

        # 3. List available specs
        specs_dirs = [Path(self.settings.local.specs_dir), Path("specs"), Path("output")]
        spec_files: list[Path] = []
        for d in specs_dirs:
            if d.exists():
                spec_files.extend(d.glob("*.yaml"))
                spec_files.extend(d.glob("*.yml"))
        # Deduplicate
        spec_files = list({sf.resolve(): sf for sf in spec_files}.values())
        if spec_files:
            parts.append(f"\n## Available Specs ({len(spec_files)})")
            for sf in spec_files[:10]:
                parts.append(f"- {sf.name}")

        # 4. List available data files
        data_dirs = [
            Path(self.settings.local.data_dir),
            Path("data"),
            Path("../repos/adsa"),
            Path("."),
        ]
        data_files: list[Path] = []
        for d in data_dirs:
            if d.exists():
                data_files.extend(d.glob("*.csv"))
        # Deduplicate
        data_files = list({df.resolve(): df for df in data_files}.values())
        if data_files:
            parts.append(f"\n## Available Data Files ({len(data_files)})")
            for df in data_files[:10]:
                parts.append(f"- {df.name}")

        # 5. Attached files context
        if self.context.get("attachments"):
            parts.append("\n## Attached Files")
            for att in self.context["attachments"]:
                parts.append(f"- {att}")

        # 6. Active spec
        if self.context.get("current_spec"):
            parts.append(f"\n## Active Spec: {self.context['current_spec']}")

        # 7. Data summary (if we described data)
        if self.context.get("data_summary"):
            summary = self.context["data_summary"]
            parts.append(
                f"\n## Data Summary: {summary.get('path', '?')} "
                f"({summary.get('rows', 0)} rows × {summary.get('cols', 0)} cols)"
            )
            if summary.get("columns"):
                parts.append(f"  Columns: {', '.join(summary['columns'][:15])}")

        # 8. Running jobs / pipeline state
        if self.context.get("running_jobs"):
            parts.append(f"\n## Running Jobs: {', '.join(self.context['running_jobs'])}")

        if self._pipeline_state:
            state = self._pipeline_state
            parts.append(f"\n## Pipeline Run: {state.get('run_id', '?')} status={state.get('status', '?')}")

        if not parts:
            return "No system state available yet. Run /status or /leaderboard to load state."
        return _apply_context_budget(parts)

    # ─── Phase 2 — shared ContextBus helpers (null-safe / additive) ───────────

    def _render_bus_context(self) -> str:
        """Render the shared ``ContextBus`` snapshot as a context section (design §E.4).

        Returns an empty string when no bus is attached (headless / offline) so
        ``_build_context`` is unchanged in that case. When present, surfaces the
        cross-tab active spec/dataset/job/run/model so the agent grounds answers in
        what the user loaded/selected in ANY tab — not just chat-set state.
        """
        bus = self.context_bus
        if bus is None:
            return ""
        try:
            state = bus.state()
        except Exception:
            return ""

        lines: list[str] = []
        spec = state.spec
        if spec.path or spec.project_id:
            extra = []
            if spec.task_type:
                extra.append(f"task={spec.task_type}")
            if spec.target:
                extra.append(f"target={spec.target}")
            if spec.primary_metric:
                extra.append(f"metric={spec.primary_metric}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"- Active spec: {spec.path or spec.project_id}{suffix}")

        ds = state.dataset
        if ds.path:
            shape = ""
            if ds.rows is not None and ds.cols is not None:
                shape = f" ({ds.rows} rows × {ds.cols} cols)"
            lines.append(f"- Active dataset: {ds.path}{shape}")

        job = state.job
        if job.job_id:
            lines.append(
                f"- Active job: {job.job_id} (kind={job.kind}, status={job.status}"
                + (f", stage={job.stage}" if job.stage else "")
                + ")"
            )

        run = state.run
        if run.run_id:
            lines.append(
                f"- Active run: {run.run_id}" + (f" best={run.best_algorithm}" if run.best_algorithm else "")
            )

        model = state.model
        if model.model_ref or model.endpoint:
            lines.append(
                f"- Active model: {model.model_ref or '?'}"
                + (f" endpoint={model.endpoint}" if model.endpoint else "")
            )

        problem = state.problem
        if problem.problem_id:
            lines.append(
                f"- Active problem: {problem.problem_id}" + (f" (kpi={problem.kpi})" if problem.kpi else "")
            )

        # Hop-1 C-3 (assessment §D.2): the cross-tab ``selection`` — WHAT the user
        # currently has selected on a tab (Specs/Data/MLOps/Lake row) — is the implicit
        # referent for "impute THIS dataset" / "deploy THAT run". The bus HOLDS it and
        # ``get_tui_context`` returns it, but it was omitted here, so those phrases had
        # no injected referent unless the agent proactively called the tool. Surface it.
        selection = state.selection or {}
        if selection:
            tab = selection.get("tab", "?")
            row_id = selection.get("row_id") or selection.get("id") or "?"
            lines.append(f"- Current selection: {tab} → {row_id}")

        # Hop-1 C-4 (assessment §D.2): the session steering "dial" (forbid/pins/notes)
        # is dual-written to the bus by the Steering tab (screens/steering.py:133) and
        # exposed by ``get_tui_context``, but it was NOT rendered here — so the agent
        # only "saw" a user's HARD constraints if it proactively called the tool. Render
        # a compact one-line summary so it reasons with the dial by DEFAULT. Coerce
        # defensively: a ``SteeringProfile`` exposes ``forbid_methods``/``pin_methods``/
        # ``domain_notes``; anything else falls back to ``str(...)`` (never raises).
        steering_line = self._steering_summary_line(state.steering)
        if steering_line:
            lines.append(steering_line)

        if not lines:
            return ""
        return "## Active TUI Context (shared across tabs)\n" + "\n".join(lines)

    def _render_session_ledger(self) -> str:
        """Render the session-wide ``SessionLedger`` timeline (Phase C2, design §D.5).

        Returns an empty string when no ledger is attached (headless / every existing
        test) so ``_build_context`` is byte-identical in that case — additive/null-safe,
        the identical pattern to :meth:`_render_bus_context`. When present, surfaces the
        bounded most-recent history (``render_context(max_entries=12)``) under a clearly
        labeled "Session history" header so the agent treats it as *what happened this
        session*, NOT *what is current* (the live bus snapshot above remains authoritative
        for current state — the explicit precedence rule, §D.5). This is what makes an
        offline→local mode flip lossless: the accumulated offline ledger folds into the
        first Bedrock prompt with no handoff code.
        """
        led = getattr(self, "session_ledger", None)
        if led is None:
            return ""
        try:
            body = led.render_context(max_entries=12)
        except Exception:
            return ""
        if not body:
            return ""
        return f"## Session history (this session, all modes)\n{body}"

    def _record_turn_to_ledger(self, message: str, response: Any) -> None:
        """Fold a completed ``process_message`` turn into the ``SessionLedger`` (Phase C2).

        This is the DETERMINISTIC accumulation seam — it runs on EVERY turn in EVERY mode
        (slash commands, offline local-fallback NL, and tool-agent turns all reach here),
        so the ledger fills even with ``bedrock_enabled=False`` (§D.6, the crux offline
        requirement). It classifies the turn:

        * a slash command (``/help``, ``/run`` …) → ``record_command`` (turn boundary);
        * otherwise a chat turn → ``record_turn`` (turn boundary), plus one ``record_tool``
          per executed tool the agent fired (from ``response.data['tools_used']``).

        Null-safe/additive: a no-op when no ledger is attached (headless / every existing
        test) and never raises — a ledger fold must never break a chat response.
        """
        led = getattr(self, "session_ledger", None)
        if led is None:
            return
        try:
            text = (message or "").strip()
            reply = getattr(response, "message", "") or ""
            data = getattr(response, "data", None) or {}

            # Slash command? (the deterministic tier-1 path — runs in every mode).
            # Mirror ``process_message`` routing: a real command is identified by
            # ``cmd_name in COMMANDS`` REGARDLESS of ``_looks_like_path`` (a leading "/"
            # trivially matches the path heuristic, but ``/help`` is still a command).
            # Only a parsed token that is NOT a registered command falls through to the
            # data-load path below (recorded as a turn).
            parsed = parse_slash_command(message)
            if parsed is not None and parsed[0] in COMMANDS:
                cmd_name = parsed[0]
                led.record_command(cmd_name, reply)
                return

            # Otherwise a chat/NL turn. Record each executed tool first (agent turns),
            # then the turn itself (with the tool names for the summary).
            tools_used = data.get("tools_used") or []
            if isinstance(tools_used, list) and tools_used:
                tool_summary = str(data.get("tool_summary", "") or "")
                for tool_name in tools_used:
                    led.record_tool(str(tool_name), tool_summary, source="agent")
                led.record_turn(text, reply, tools=[str(t) for t in tools_used])
            else:
                led.record_turn(text, reply)
        except Exception:
            logger.debug("SessionLedger turn fold failed (best-effort)", exc_info=True)

    @staticmethod
    def _steering_summary_line(steering: Any) -> str:
        """Render the active steering dial as ONE compact prose line (Hop-1 C-4).

        Returns ``""`` when no dial is set (null-safe). Surfaces the HARD constraints
        (forbid/pins) plus a short notes snippet so the agent sees the user's dial in
        prose without calling ``get_tui_context``. Never raises on a non-standard
        steering object — it degrades to ``str(steering)``.
        """
        if steering is None:
            return ""
        forbid = getattr(steering, "forbid_methods", None)
        pins = getattr(steering, "pin_methods", None)
        notes = getattr(steering, "domain_notes", None)
        if forbid is None and pins is None and notes is None:
            # Unknown shape — degrade to a bounded string repr rather than omit it.
            text = str(steering).strip()
            return f"- Active steering: {text[:160]}" if text else ""
        segments: list[str] = []
        if forbid:
            segments.append(f"forbid={list(forbid)}")
        if pins:
            segments.append(f"pins={dict(pins)}")
        if notes:
            segments.append(f"notes={str(notes)[:120]}")
        if not segments:
            return ""
        return "- Active steering: " + ", ".join(segments)

    def _render_pending_confirmation(self) -> str:
        """Render the cross-turn awaiting-confirmation marker as a context section (C-2).

        Returns ``""`` when nothing is pending (null-safe). When a prior turn returned
        ``needs_confirmation`` for a state-changing tool, this states the awaiting y/n
        EXPLICITLY so a bare "yes" resolves against a concrete tool this turn.
        """
        pending = self._awaiting_confirmation
        if not pending:
            return ""
        tool = pending.get("tool") or "the pending action"
        tools = pending.get("tools") or [tool]
        joined = ", ".join(str(t) for t in tools)
        return (
            "## Awaiting confirmation\n"
            f"You are awaiting the user's yes/no confirmation to run `{joined}`. "
            f'If the user affirms (e.g. "yes", "go ahead", "do it"), proceed to run '
            f"`{tool}`; if they decline, do not run it and clear this."
        )

    @staticmethod
    def _summarize_tool_results(executions: list[Any]) -> str:
        """Build the COMPACT per-turn tool-result summary for history replay (C-1).

        One short ``tool: <result_summary>`` clause per EXECUTED tool, joined with
        ``"; "`` — deliberately concise (uses the already-bounded
        ``ToolExecution.result_summary``, which ``_summarize`` capped at 200 chars),
        NOT a raw dump of the full result dict. Only ``executed`` executions contribute
        (pending/error turns add no summary). Returns ``""`` for a no-tool turn so the
        assistant message carries no summary and history replay is unchanged.
        """
        clauses: list[str] = []
        for ex in executions or []:
            if getattr(ex, "status", "") != "executed":
                continue
            name = getattr(ex, "tool_name", "") or "tool"
            summary = (getattr(ex, "result_summary", "") or "").strip()
            clauses.append(f"{name}: {summary}" if summary else name)
        return "; ".join(clauses)

    def _sync_bus(self, method: str, **kwargs: Any) -> None:
        """Call a ``ContextBus`` mutator when a bus is attached; no-op otherwise.

        Every chat/agent write-point (``_cmd_data``/``_cmd_spec``/``_attach_yaml``/
        ``_cmd_run``) routes through here so the shared context stays in sync WITHOUT
        coupling the headless engine to a bus (null-safe/additive, design §E.3). Never
        raises — a bus mutation must never break a slash command.
        """
        bus = self.context_bus
        if bus is None:
            return
        try:
            getattr(bus, method)(**kwargs)
        except Exception:
            logger.debug("ContextBus %s sync failed", method, exc_info=True)

    # ─── Phase 5 — agent-turn result seam (design §E.4 / §E.8) ────────────────

    def _fold_agent_turn(self, turn: Any, message: str = "") -> None:
        """Fold a completed tool-bound agent turn onto the shared bus (design §E.4/§E.8).

        This is the **result seam** for the LOCAL chat agent (``tui_chat_agent``): after
        an agent turn returns, its ``AgentTurn.executions`` (the real tools it fired) are
        mapped to bus mutations so agent work appears across the TUI:

        * **AgentActivity (§E.8 / GO-4):** one folded ``AgentActivity`` record — agent
          name, ``last_tool`` (latest executed/backgrounded tool), model tier,
          ``aws_services`` (via :func:`aws_services_for_tool` + Bedrock for the LLM
          itself), status — published via ``set_agent_activity`` so the local agent shows
          up in the Dashboard's ``AgentActivityPanel`` beside the cloud swarm.
        * **Spec (§E.4 / GP-2):** a ``save_spec`` / ``generate_spec_from_description``
          execution publishes the active spec (``set_spec`` source=``"agent"``) so an
          agent-generated spec appears on the Specs tab and prefills Pipeline/Config.
        * **Dataset (§E.4 / GP-2):** a ``load_dataset`` / ``describe_dataset`` execution
          publishes the active dataset (``set_dataset`` source=``"agent"``).

        Null-safe/additive: a no-op when no bus is attached (headless) or the turn has no
        tool executions; never raises (a fold failure must not break the chat response).
        """
        if self.context_bus is None:
            return
        executions = getattr(turn, "executions", None) or []
        # Only tools that actually did work count as agent activity / effects.
        effective = [e for e in executions if getattr(e, "status", "") in ("executed", "job_submitted")]
        if not effective:
            return
        try:
            self._publish_agent_activity(effective)
        except Exception:
            logger.debug("agent-activity fold failed", exc_info=True)
        for execution in effective:
            try:
                self._fold_execution_effect(execution, message)
            except Exception:
                logger.debug("execution-effect fold failed", exc_info=True)

    def _publish_agent_turn_start(self, text: str) -> None:
        """Publish a RUNNING ``tui_chat_agent`` activity at turn START (FIX-24, null-safe).

        The Dashboard's ``AgentActivityPanel`` / ``AgentLogsPane`` / ``RunningLogPane`` all
        repaint on the bus ``"agents"`` event. Before FIX-24 the only chat→bus publish was
        the POST-turn fold (``_publish_agent_activity``, status="complete"), gated on
        bedrock+tool-capability — so a prompt in offline/local-fallback mode published nothing
        and the operator saw NO Dashboard movement after pressing Send. Publishing a running
        record here (via ``_sync_bus``, which is null-safe and mode-agnostic) means the chat
        agent appears the INSTANT the turn begins; the completing fold flips it to done.
        """
        from ..context.models import AgentActivity

        self._sync_bus(
            "set_agent_activity",
            activity=AgentActivity(
                agent_name="tui_chat_agent",
                phase="thinking",
                last_model=self._chat_model_label(),
                aws_services=["Bedrock"],
                status="running",
                updated_at=datetime.now(UTC).isoformat(),
            ),
        )

    def _publish_agent_activity(self, executions: list[Any]) -> None:
        """Publish ONE ``AgentActivity`` for the local chat agent from its tool trace."""
        from ..context.models import AgentActivity
        from ..context.trace_reader import aws_services_for_tool

        last_tool = executions[-1].tool_name
        aws_services: list[str] = []
        for execution in executions:
            for svc in aws_services_for_tool(getattr(execution, "tool_name", "")):
                if svc not in aws_services:
                    aws_services.append(svc)
        # The local chat agent always runs on a Bedrock model tier.
        if "Bedrock" not in aws_services:
            aws_services.append("Bedrock")

        self._sync_bus(
            "set_agent_activity",
            activity=AgentActivity(
                agent_name="tui_chat_agent",
                phase=None,
                current_span=last_tool,
                last_tool=last_tool,
                last_model=self._chat_model_label(),
                aws_services=aws_services,
                status="complete",
                updated_at=datetime.now(UTC).isoformat(),
            ),
        )

    # : map each local-pipeline stage to the roster lead that owns it, so the
    # Dashboard's Agent Logs pane (which lists the seven ML_AUTOMATION_AGENTS leads) lights
    # up on a local ``/run``. The local ``MLPipeline`` writes no ``trace.jsonl`` for the
    # Dashboard trace poller, and plain chat never starts the poller — so without this the
    # seven leads sat permanently ``idle`` and the pane read as "nothing".
    _STAGE_TO_LEAD: ClassVar[dict[str, str]] = {
        "research": "research_lead",
        "plan": "ml_execution_lead",
        "train": "ml_execution_lead",
        "evaluate": "deploy_serving_lead",
        "decision": "deploy_serving_lead",
        "complete": "deploy_serving_lead",
    }

    def _publish_pipeline_stage(self, stage_key: str, label: str) -> None:
        """Publish an ``AgentActivity`` for the lead owning ``stage_key``.

        Called by ``MLPipeline.run(on_stage=...)`` as each stage begins. Marks the owning
        lead ``running`` (``complete`` for the terminal ``complete`` stage) with the stage
        label as its phase, so the Dashboard Agent Logs pane shows real per-lead progress on
        a local run in EVERY mode (no trace file / poller dependency). Null-safe via
        ``_sync_bus``; best-effort — never breaks the run.
        """
        from ..context.models import AgentActivity

        lead = self._STAGE_TO_LEAD.get(stage_key)
        if lead is None:
            return
        status = "complete" if stage_key == "complete" else "running"
        try:
            self._sync_bus(
                "set_agent_activity",
                activity=AgentActivity(
                    agent_name=lead,
                    phase=label,
                    current_span=stage_key,
                    last_model=self._chat_model_label(),
                    aws_services=["Bedrock"],
                    status=status,
                    updated_at=datetime.now(UTC).isoformat(),
                ),
            )
        except Exception:
            logger.debug("pipeline-stage publish failed for %s", stage_key, exc_info=True)

    def _fold_execution_effect(self, execution: Any, message: str = "") -> None:
        """Map a single known tool execution to a bus write (spec/dataset/run/model/problem).

        Phase 6 (design §E.4, fixes GT-4 / GT-7 / use case #1): extends the Phase-5 spec/
        dataset fold so agent-run evaluation/decision tools reach the Results tab
        (``set_run``), MLOps registry tools reach the MLOps tab (``set_model``), and
        business-framing tools reach the active problem (``set_problem``) — so an agent
        that evaluates a run, approves a model, or frames a problem is observed across the
        TUI, exactly as the design's Section-C matrix requires. Null-safe/additive.

        Phase 9 (100-use-case fold breadth): extends the fold so EVERY analysis /
        validation / missingness / dedup / outlier / profiling / PII / transform /
        feature / lake / mlops-monitoring tool family reaches its registry-declared
        ContextBus mutation (``dataset`` / ``selection`` / ``model`` / ``job`` /
        ``processes`` / ``problem`` / ``spec``) → the correct tab reacts, per the
        Phase-2 §4 reaction table. Each mapping is driven by the tool NAME + its real
        result shape; ``message`` (the chat prompt) supplies the dataset/model referent
        the tool operates on when the tool result itself does not echo it.
        """
        name = getattr(execution, "tool_name", "")
        args = getattr(execution, "args", None) or {}
        result = getattr(execution, "result", None)
        result = result if isinstance(result, dict) else {}

        if name in ("save_spec", "generate_spec_from_description"):
            path = result.get("path") or args.get("path")
            self._sync_bus(
                "set_spec",
                path=path,
                spec_dict=result or None,
                task_type=result.get("task_type"),
                target=result.get("target"),
                primary_metric=result.get("primary_metric"),
                source="agent",
            )
        elif name in ("load_dataset", "describe_dataset"):
            path = args.get("path") or result.get("path")
            if not path:
                return
            self._sync_bus(
                "set_dataset",
                path=str(path),
                rows=result.get("n_rows"),
                cols=result.get("n_cols"),
                columns=result.get("columns"),
                dtypes=result.get("dtypes"),
                source="agent",
            )
            # DQ-PROF-005: a describe FOCUSED on the target column also frames a
            # target-profile problem (the registry expects a ``problem`` event when the
            # request is about the target/class balance, e.g. "profile the target column
            # machine_failure … class balance"). Gated on the prompt mentioning "target"
            # so pure schema/stats describes (PROF-001/002/003) do NOT frame a problem.
            if name == "describe_dataset" and "target" in (message or "").lower():
                problem_id = self._slugify_problem(message)
                if problem_id:
                    self._sync_bus(
                        "set_problem",
                        problem_id=problem_id,
                        kpi="class_balance",
                    )
        elif name in _DATASET_FOLD_TOOLS:
            # Analysis / DQ / validation / missingness / dedup / outlier / profiling /
            # PII / transform / feature / synthesis tools operate ON a dataset → publish
            # the active dataset so the Data tab repaints (Phase-2 §4: dataset → Data).
            # A tool whose prompt names a LAKEHOUSE table (e.g. run_profile on a lakehouse
            # table, DQ-LAKE-008) is sourced "lake_tab" so the Lake tab ALSO reacts; the
            # Data tab reacts either way (it keys on the path, not the source).
            src = "lake_tab" if "lake" in (message or "").lower() else "agent"
            self._fold_dataset_effect(name, args, result, message, source=src)
            # A tool listed in BOTH families is genuinely dual-surface: lineage answers
            # "which data produced this MODEL", so it must publish the model too, not just
            # the dataset (this ``elif`` chain would otherwise stop at the first match —
            # ML-REG-003). Kept explicit rather than reordering the chain, so the existing
            # DQ expectations for these tools are untouched.
            if name in _MODEL_FOLD_TOOLS:
                self._fold_model_effect(name, args, result, message)
        elif name == "list_lake_catalog":
            # DQ-LAKE-003: listing the catalog makes it the active Lake selection so the
            # Lake tab reacts (it is a read-only browse → only ``selection`` + ``agents``).
            self._sync_bus("set_selection", selection={"tab": "lake", "row_id": "catalog"})
        elif name == "validate_spec":
            # DQ-VAL-011: validating the open spec re-publishes it as the active spec so
            # the Specs viewer reflects the validated contract (spec → Specs tab).
            self._sync_bus(
                "set_spec",
                path=args.get("path") or result.get("path"),
                spec_dict=result or {"validated": True},
                task_type=result.get("task_type"),
                target=result.get("target"),
                primary_metric=result.get("primary_metric"),
                valid=result.get("valid"),
                source="agent",
            )
        elif name in _LAKE_DATASET_FOLD_TOOLS:
            # Lake read/write lands a slice → active dataset (source="lake_tab") + the
            # cross-tab selection marker (Phase-2 §4: Lake produces dataset + selection).
            self._fold_dataset_effect(name, args, result, message, source="lake_tab")
            path = self._dataset_path_from(args, result, message)
            if path:
                self._sync_bus("set_selection", selection={"tab": "lake", "row_id": str(path)})
        elif name in _LAKE_REGISTER_FOLD_TOOLS:
            # Lake register/ingest/profile/query/partition → active dataset (lake_tab)
            # so the Lake + Data tabs reflect the landed/registered slice.
            self._fold_dataset_effect(name, args, result, message, source="lake_tab")
        elif name in _MODEL_FOLD_TOOLS:
            # MLOps monitoring / drift / guardrail / feature-store-read tools re-point the
            # active model → MLOps + Observability panels react (Phase-2 §4: model → MLOps).
            self._fold_model_effect(name, args, result, message)
        elif name == "feature_store_ingest":
            # DQ-FEAT-004: ingesting features touches BOTH the dataset (source of the
            # features) AND the feature store on the MLOps surface — fire dataset + model.
            self._fold_dataset_effect(name, args, result, message, source="agent")
            self._fold_model_effect(name, args, result, message)
        elif name in _JOB_FOLD_TOOLS:
            # Flow/ETL job tools launch a server-side job → active job (Pipeline stage +
            # Active-Jobs). A local (Processing) job ALSO registers a process row.
            self._fold_job_effect(name, args, result, message)
        elif name == "run_ml_capability":
            # Capability flow dispatch (data_quality_flow / imputation_flow /
            # feature_engineering_flow) → a job (Pipeline + Active-Jobs) AND the dataset
            # the flow operates on (Data tab), per the registry's (job, dataset) events.
            self._fold_flow_dispatch(args, result, message)
        elif name in ("get_best_model", "get_recommendation"):
            # Evaluation/decision tools (use case #11 evaluation_flow, fixes GT-4) →
            # publish the active COMPLETED run so the Results tab scopes to it. Routed
            # through the same merging helper as the rest of the run family: the REAL
            # ``get_recommendation`` returns a decision with NO run identity, and a
            # wholesale ``set_run`` would blank the Results card it is commenting on.
            if not (result.get("success") or result.get("best_algorithm") or result.get("recommendation")):
                return
            self._fold_run_effect(name, args, result, message)
        elif name in _RUN_FOLD_TOOLS:
            # Evaluation / experiment-tracking tools name a RUN and its score, so they
            # publish the active completed run → the Results tab scopes to it. Added when
            # the 60 ML-* use cases were first executed: the registry declared a Results
            # reaction for these and the fold produced only ``agents``
            # (ML-TRAIN-004, ML-HPO-004, ML-EVAL-002/008).
            self._fold_run_effect(name, args, result, message)
        elif name == "render_dashboard_view":
            # ADR-0006: the agent's chart request. The tool itself publishes the ``view``
            # event to the ACTIVE bus (``set_active_bus``), which the app registers at
            # mount. Folding here as well makes the request land on the engine's OWN bus
            # too — the seam a headless engine / a isolated harness holds — so the Dashboard
            # reacts regardless of which bus handle the caller wired up.
            self._fold_view_effect(name, args, result, message)
        elif name in ("build_problem_card", "capture_business_problem"):
            # Business-framing tools (use case #1 business_framing_flow) → publish the
            # active problem so the framed problem is observable across the TUI + agent.
            self._fold_problem_effect(name, args, result)
        elif name in ("register_model", "approve_model", "recommend_inference_config"):
            # MLOps registry/deploy tools (use cases #12/#13, fixes GT-7) → active model.
            model_ref = (
                result.get("model_ref")
                or result.get("model_package_arn")
                or result.get("model_name")
                or args.get("model_ref")
                or args.get("model_name")
            )
            if not (model_ref or result.get("endpoint")):
                return
            self._sync_bus(
                "set_model",
                model_ref=str(model_ref) if model_ref else None,
                version=result.get("version") or result.get("model_version"),
                approval=result.get("approval") or result.get("approval_status"),
                endpoint=result.get("endpoint") or result.get("endpoint_name"),
            )

    def _fold_run_effect(self, name: str, args: dict[str, Any], result: dict[str, Any], message: str) -> None:
        """Publish the active COMPLETED run from an evaluation / tracking tool result.

        The Results tab is a ``run`` consumer, so an evaluation verb that names a run and
        a score must publish one or the tab never reacts. Fields are read from the tool's
        REAL result shape under the several key spellings the tool family uses
        (``best_algorithm``/``algorithm``, ``best_score``/``score``/``primary_score``);
        nothing is invented — a result with no identifying field publishes the run with
        those fields ``None`` rather than a fabricated algorithm/score.

        A leaderboard/comparison tool may return ``rows``/``results``; the FIRST entry is
        the winner (these tools return ranked output), so it supplies the run identity.
        """
        rows = result.get("rows") or result.get("results") or result.get("leaderboard")
        top: dict[str, Any] = {}
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            top = rows[0]

        # ``ContextBus.set_run`` REPLACES the record wholesale, so a tool that reports only
        # part of a run (e.g. the real ``get_recommendation`` returns a decision + rationale
        # and NO run identity) would blank the Results card — a recommendation ABOUT the
        # active run must not erase it. So each field falls back to the currently-active
        # run. This is a merge, not an invention: the fallback value is a real prior
        # observation, never a synthesized one.
        current = None
        try:
            bus = self.context_bus
            current = bus.state().run if bus is not None else None
        except Exception:
            current = None

        def _pick(*keys: str, prior: str | None = None) -> Any:
            for key in keys:
                for source in (result, top, args):
                    value = source.get(key) if isinstance(source, dict) else None
                    if value not in (None, ""):
                        return value
            if prior and current is not None:
                return getattr(current, prior, None)
            return None

        score = _pick("best_score", "primary_score", "score", prior="best_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        self._sync_bus(
            "set_run",
            run_id=_pick("best_run_id", "run_id", "id", prior="run_id"),
            spec_id=_pick("spec_id", "project_id", prior="spec_id"),
            best_algorithm=_pick("best_algorithm", "algorithm", "model", "estimator", prior="best_algorithm"),
            primary_metric=_pick("primary_metric", "metric", prior="primary_metric"),
            best_score=score,
            decision=_pick("recommendation", "decision", prior="decision"),
            artifacts_dir=getattr(current, "artifacts_dir", None) if current else None,
            status=getattr(current, "status", None) if current else None,
        )

    def _fold_view_effect(
        self, name: str, args: dict[str, Any], result: dict[str, Any], message: str
    ) -> None:
        """Publish the agent's dashboard chart request onto THIS engine's bus (ADR-0006).

        ``render_dashboard_view`` already renders the chart and publishes to the process
        ACTIVE bus; this mirrors the request onto the engine's own bus handle so the
        Dashboard reacts even when the two differ (headless engine, isolated harness). The
        payload is reconstructed from the tool's ``args`` — the chart data the agent
        actually passed — so nothing is invented. A tool result that FAILED (the
        never-fabricate-a-chart path) publishes nothing.
        """
        if result.get("success") is False:
            return
        chart = str(args.get("chart") or result.get("chart") or "").strip().lower()
        if not chart:
            return
        try:
            from ..context.models import ActiveView

            self._sync_bus(
                "set_view",
                view=ActiveView(
                    view_id=str(args.get("view_id") or chart),
                    chart=chart,
                    title=str(result.get("title") or args.get("title") or chart.title()),
                    series={
                        str(k): [float(v) for v in (vals or [])]
                        for k, vals in (args.get("series") or {}).items()
                    },
                    values={str(k): float(v) for k, v in (args.get("values") or {}).items()},
                    matrix=[list(row) for row in (args.get("matrix") or [])],
                    x=list(args.get("x") or []),
                    y=list(args.get("y") or []),
                    labels=[str(v) for v in (args.get("labels") or [])],
                    source="agent",
                    detail=str(result.get("detail") or ""),
                    # Which tab the agent addressed (empty = Dashboard, ADR-0006 default).
                    target=str(args.get("target") or result.get("target") or ""),
                ),
            )
        except Exception:
            logger.debug("view fold failed for %s", name, exc_info=True)

    def _fold_problem_effect(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        """Publish ``set_problem`` from a business-framing tool result (use case #1).

        Handles both ``build_problem_card`` (nested ``problem_card`` dict) and
        ``capture_business_problem`` (flat ``statement``). Derives a stable
        ``problem_id`` slug from the business problem statement so the Specs/Dashboard
        and the agent's ``get_tui_context`` can reference it. Never raises.
        """
        raw_card = result.get("problem_card")
        card: dict[str, Any] = raw_card if isinstance(raw_card, dict) else {}
        statement = (
            card.get("business_problem")
            or result.get("statement")
            or args.get("business_problem")
            or args.get("statement")
            or ""
        )
        problem_id = self._slugify_problem(statement)
        if not problem_id:
            return
        raw_kpi = card.get("kpi")
        kpi: dict[str, Any] = raw_kpi if isinstance(raw_kpi, dict) else {}
        kpi_label = None
        if kpi:
            kpi_label = next(iter(kpi.keys()))
        kpi_label = kpi_label or card.get("success_metric") or result.get("kpi")
        self._sync_bus(
            "set_problem",
            problem_id=problem_id,
            kpi=str(kpi_label) if kpi_label else None,
            problem_card_path=result.get("path") or result.get("problem_card_path"),
        )

    # ─── Phase 9 — tool-family fold helpers (extend the fold to all 100 cases) ──

    @staticmethod
    def _dataset_path_from(args: dict[str, Any], result: dict[str, Any], message: str) -> str | None:
        """Resolve the dataset path/URI a tool operated on (args → result → prompt).

        Reads the common arg/result keys the real tools use (``data_path`` / ``path`` /
        ``dataset_uri`` / ``local_path`` / ``uri`` / ``dest_uri`` / ``input_uri`` /
        ``written_uri`` / ``imputed_path`` / ``synthetic_path`` / ``output`` …). When the
        tool result/args don't echo a path (many read-only analysis tools take the path
        positionally or return only findings), falls back to the ``*.csv``/``*.parquet``/
        ``s3://`` reference embedded in the chat prompt — the dataset the user named.
        """
        for src in (result, args):
            for key in (
                "path",
                "data_path",
                "dataset_uri",
                "local_path",
                "uri",
                "dest_uri",
                "input_uri",
                "written_uri",
                "imputed_path",
                "imputed_uri",
                "synthetic_path",
                "augmented_path",
                "output",
                "output_uri",
                "output_dir",
            ):
                val = src.get(key)
                if isinstance(val, str) and val.strip():
                    return val
        m = re.search(r"([\w./\-]+\.(?:csv|parquet))", message or "")
        if m:
            return m.group(1)
        m = re.search(r"(s3://[\w./\-]+)", message or "")
        if m:
            return m.group(1)
        m = re.search(r"((?:silver|gold|bronze)://[\w./\-]+)", message or "")
        if m:
            return m.group(1)
        return None

    def _fold_dataset_effect(
        self,
        name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        message: str,
        *,
        source: str,
    ) -> None:
        """Publish the active dataset from an analysis/transform/lake tool (Data tab).

        The dataset the tool operated on becomes the active dataset so the Data tab
        repaints (Phase-2 §4: ``dataset`` → Data). Carries ``dq_score`` when the tool
        returned an ``overall_score`` / ``score`` (the scoring tools), and rows/cols
        when the tool echoes them. Null-safe: no resolvable path → no-op.
        """
        path = self._dataset_path_from(args, result, message)
        if not path:
            return
        dq_score = result.get("overall_score")
        if dq_score is None:
            dq_score = result.get("score")
        rows = result.get("rows") or result.get("n_rows")
        cols = result.get("columns_count") or result.get("n_cols")
        columns = result.get("columns") if isinstance(result.get("columns"), list) else None
        self._sync_bus(
            "set_dataset",
            path=str(path),
            rows=int(rows) if isinstance(rows, int) else None,
            cols=int(cols) if isinstance(cols, int) else None,
            columns=columns,
            dq_score=float(dq_score) if isinstance(dq_score, (int, float)) else None,
            source=source,
        )

    @staticmethod
    def _model_ref_from_message(message: str) -> str:
        """Derive a stable model/endpoint referent from the chat prompt (mlops fold).

        Prefers a ``<name> model`` / ``<name> endpoint`` phrase (e.g. "the bogie model"
        → ``bogie``); else a ``model/endpoint <name>`` phrase; else the safe default
        ``amlas``. Used by monitoring tools whose result carries no model name so the
        active model still re-points the MLOps/Observability panels to a concrete ref.
        """
        text = (message or "").lower()
        m = re.search(r"([a-z0-9_\-]+)\s+(?:model|endpoint)", text)
        if m:
            return m.group(1)
        m = re.search(r"(?:model|endpoint)\s+([a-z0-9_\-]+)", text)
        if m:
            return m.group(1)
        return "amlas"

    def _fold_model_effect(
        self, name: str, args: dict[str, Any], result: dict[str, Any], message: str
    ) -> None:
        """Publish the active model from an MLOps/monitoring tool (MLOps + Observability).

        Monitoring / drift / guardrail / feature-store tools re-point the active model so
        the MLOps active-model line AND the Observability CloudWatch panel react
        (Phase-2 §4: ``model`` → MLOps + Observability). The model_ref/endpoint are read
        from the tool result where present (``model_ref`` / ``schedule_name`` / ``target``
        / ``group`` …) and otherwise derived from the model the prompt names — so the
        Observability panel always has a concrete endpoint scope to render.
        """
        model_ref = (
            result.get("model_ref")
            or result.get("model_name")
            or result.get("model")
            or result.get("schedule_name")
            or result.get("feature_group")
            or args.get("model_ref")
            or args.get("model")
            or args.get("group")
            or self._model_ref_from_message(message)
        )
        endpoint = (
            result.get("endpoint")
            or result.get("endpoint_name")
            or args.get("target")
            or args.get("endpoint")
            or f"{model_ref}-endpoint"
        )
        self._sync_bus(
            "set_model",
            model_ref=str(model_ref) if model_ref else None,
            version=result.get("version") or result.get("model_version"),
            approval=result.get("approval") or result.get("approval_status"),
            endpoint=str(endpoint) if endpoint else None,
        )

    def _fold_job_effect(self, name: str, args: dict[str, Any], result: dict[str, Any], message: str) -> None:
        """Publish an active job (+ local process) from a flow/ETL job tool.

        Server-side job tools (``run_glue_job`` / ``run_batch_transform`` /
        ``run_processing_job`` / ``run_crawler`` / ``apply_imputation_plan``) launch a
        job → ``set_job`` (Pipeline stage/progress + Active-Jobs, Phase-2 §4: ``job`` →
        Pipeline + Active-Jobs). A local Processing job ALSO registers a
        ``ProcessActivity`` row (Dashboard Local-Processes). The job id is the tool's own
        ``job_id`` / ``job_name`` when returned, else a stable generated id.
        """
        job_id = (
            result.get("job_id")
            or result.get("job_name")
            or result.get("run_id")
            or result.get("crawler_name")
            or f"{name}-{uuid.uuid4().hex[:8]}"
        )
        kind = _JOB_KIND_LABELS.get(name, name)
        run_id = result.get("run_id") or (
            job_id if kind in ("glue", "processing", "batch_transform") else None
        )
        self._sync_bus(
            "set_job",
            job_id=str(job_id),
            kind=kind,
            status="running",
            progress=50.0,
            stage=str(result.get("stage") or "Training"),
            run_id=str(run_id) if run_id else None,
            started_by="agent",
        )
        if name in _PROCESS_FOLD_TOOLS:
            self._publish_process(str(job_id), kind, message)
        # DQ-LAKE-004: a Glue crawler is a LAKE job — the crawled target becomes the
        # active Lake selection so the Lake tab reacts (in addition to Active-Jobs).
        if name == "run_crawler":
            target = self._dataset_path_from(args, result, message) or "crawl"
            self._sync_bus("set_selection", selection={"tab": "lake", "row_id": str(target)})

    def _fold_flow_dispatch(self, args: dict[str, Any], result: dict[str, Any], message: str) -> None:
        """Publish job + dataset for a ``run_ml_capability`` capability-flow dispatch.

        The registry's flow-dispatch cases (data_quality_flow / imputation_flow /
        feature_engineering_flow) expect BOTH a ``job`` (Pipeline + Active-Jobs) AND the
        ``dataset`` the flow operates on (Data tab). Emits ``set_job`` keyed by the flow
        capability id + ``set_dataset`` for the prompt's dataset (source="agent"). For a
        Processing-heavy flow this also registers a local process row.
        """
        capability_id = str(args.get("capability_id") or result.get("capability_id") or "flow")
        job_id = str(
            result.get("job_id") or result.get("run_id") or f"{capability_id}-{uuid.uuid4().hex[:8]}"
        )
        self._sync_bus(
            "set_job",
            job_id=job_id,
            kind=capability_id,
            status="running",
            progress=50.0,
            stage="Plan",
            run_id=job_id,
            started_by="agent",
        )
        path = self._dataset_path_from(args, result, message)
        if path:
            self._sync_bus("set_dataset", path=str(path), source="agent")

        # A flow's OWN lifecycle stage decides which additional surface must react — a
        # data flow lands a dataset (above), but an evaluation flow produces a RUN and a
        # deploy/monitor/readiness flow acts on a MODEL. Without this, dispatching
        # ``evaluation_flow`` or ``deploy_flow`` from chat left the Results / MLOps tabs
        # inert (ML-EVAL-001, ML-REG-004, ML-DEP-001, ML-MON-001).
        if capability_id in _RUN_FLOW_IDS:
            self._fold_run_effect(capability_id, args, result, message)
        elif capability_id in _MODEL_FLOW_IDS:
            self._fold_model_effect(capability_id, args, result, message)

    def _publish_process(self, process_id: str, kind: str, message: str) -> None:
        """Register a local ProcessActivity row for a local (Processing) job (D-3)."""
        from ..context.models import ProcessActivity

        self._sync_bus(
            "set_process_activity",
            activity=ProcessActivity(
                process_id=process_id,
                kind=kind,
                status="running",
                started_by="agent",
                progress=50.0,
                detail=(message or "")[:80],
                updated_at=datetime.now(UTC).isoformat(),
            ),
        )

    @staticmethod
    def _slugify_problem(statement: str) -> str | None:
        """Turn a free-text business problem into a short stable id slug (or ``None``)."""
        words = [w for w in (statement or "").lower().split() if w.isalnum()]
        if not words:
            return None
        return "-".join(words[:5])

    @staticmethod
    def _chat_model_label() -> str:
        """A friendly label for the TUI chat agent's model tier (design §E.8 column).

        Reads the SAME source the tool backend builds the agent from —
        [`config.chat_model.chat_model_id`](../../config/chat_model.py:53) (issue #35
        criterion 4.1) — so the AgentActivityPanel's ``Model`` column cannot report a
        different tier than the one actually driving the local chat agent. This used to
        re-resolve ``AMLAS_CHAT_MODEL`` itself, making it a THIRD independent copy of the
        lookup and a place the label could silently drift from the agent.

        Never raises. ``chat_model_id()`` cannot raise (pure env read with a default), so
        the guard is belt-and-braces for the label path, kept because a widget column must
        never take a turn down.
        """
        try:
            return chat_model_id()
        except Exception:  # pragma: no cover — chat_model_id() cannot raise
            return "chat"

    # ─── Slash Command Routing ────────────────────────────────────────────────

    def _slash_handlers(self) -> dict[str, Any]:
        """Return the slash-command → handler dispatch table.

        BUG-09 / TUI-01: EVERY registered command maps to a handler that performs
        a real action (invokes an agent/tool or sets session state) — no handler
        is a cosmetic "record-only" no-op. The Phase-11 agentic commands
        (``dqscan``/``diagnose``/``impute``/``synth``/``synth-from``/``steer``/
        ``lake``/``mlops``) join the Phase-3 real-invocation contract that
        ``run``/``train`` already satisfy.
        """
        return {
            "help": self._cmd_help,
            # Aliases advertised by the chat welcome text + input placeholder.
            "commands": self._cmd_help,
            "cmds": self._cmd_help,
            "spec": self._cmd_spec,
            "run": self._cmd_run,
            "status": self._cmd_status,
            "leaderboard": self._cmd_leaderboard,
            "research": self._cmd_research,
            "quality": self._cmd_quality,
            "train": self._cmd_train,
            "attach": self._cmd_attach,
            # Phase 11 — agentic Data-Engineering / MLOps commands (§10).
            "dqscan": self._cmd_dqscan,
            "diagnose": self._cmd_diagnose,
            "impute": self._cmd_impute,
            "synth": self._cmd_synth,
            "synth-from": self._cmd_synth_from,
            "steer": self._cmd_steer,
            "lake": self._cmd_lake,
            "mlops": self._cmd_mlops,
            "data": self._cmd_data,
            # Phase 5 — newly wired commands (G4, G5, G10, G12)
            "export": self._cmd_export,
            "dashboard": self._cmd_dashboard,
            "tune": self._cmd_tune,
            "experiment": self._cmd_experiment,
            "modelcard": self._cmd_modelcard,
            # Phase 9 — EXPOSE-set parity commands (G21–G37 + #20)
            "split": self._cmd_split,
            "describe": self._cmd_describe,
            "anomalies": self._cmd_anomalies,
            "leakage": self._cmd_leakage,
            "preprocess": self._cmd_preprocess,
            "ensemble": self._cmd_ensemble,
            "best": self._cmd_best,
            "recommend": self._cmd_recommend,
            "nextsteps": self._cmd_nextsteps,
            "project": self._cmd_project,
            "queue": self._cmd_queue,
            "watch": self._cmd_watch,
            "upload": self._cmd_upload,
            "download": self._cmd_download,
            "health": self._cmd_health,
            "plan": self._cmd_plan,
            # M7 — one-capability across surfaces (ADR-010): chat → the shared funnel.
            "flows": self._cmd_flows,
            "flow": self._cmd_flow,
            "mention": self._cmd_mention,
        }

    async def _handle_slash_command(self, cmd_name: str, args: list[str]) -> ChatResponse:
        """Route a parsed slash command to its handler."""
        # Validate arguments
        error = validate_command_args(cmd_name, args)
        if error:
            return ChatResponse(message=f"❌ {error}")

        handler = self._slash_handlers().get(cmd_name)
        if handler is None:
            return ChatResponse(message=f"❌ Unknown command: /{cmd_name}")

        result: ChatResponse = await handler(args)
        return result

    # ─── M7 — one-capability across surfaces (ADR-010): chat → the shared funnel ──

    async def _cmd_flows(self, args: list[str]) -> ChatResponse:
        """``/flows`` — list every registered capability via the shared funnel (ADR-010).

        Delegates to ``tui.chat.flow_dispatch.format_capabilities`` (a thin renderer over
        ``control_plane.capability.list_capabilities``) so chat discovery is identical to the
        CLI ``capability list`` + the TUI pane — NO flow logic in the chat layer.
        """
        from .flow_dispatch import format_capabilities

        return format_capabilities()

    async def _cmd_flow(self, args: list[str]) -> ChatResponse:
        """``/flow <id> [k=v ...] [--plan]`` — run (or preview) a capability via the funnel.

        The first arg is the capability id; remaining ``k=v`` tokens become flow params; a
        trailing ``--plan`` returns the Plan-Mode preview. Gated capabilities route through
        the approval gate exactly as on every other surface (ADR-010 parity).
        """
        from .flow_dispatch import dispatch_flow

        flow_id, rest = args[0], args[1:]
        return dispatch_flow(flow_id, rest)

    async def _cmd_mention(self, args: list[str]) -> ChatResponse:
        """``/mention @<kind>:<ref>`` — resolve an ``@`` mention into context (T-LAKE-07)."""
        from .flow_dispatch import dispatch_mention

        return dispatch_mention(args[0])

    # ─── Phase 11 — steering "dial" helpers ──────────────────────────────────

    def _steering_profile_dict(self) -> dict[str, Any] | None:
        """Return the session ``SteeringProfile`` as a tool ``profile=`` dict.

        ``None`` when no dial is set, so the no-steering path stays clean. This is
        what threads the HARD constraints (forbid/pin/thresholds) into the tools.
        """
        if self.steering_profile is None:
            return None
        out: dict[str, Any] = self.steering_profile.as_profile_dict()
        return out

    async def _cmd_help(self, args: list[str]) -> ChatResponse:
        """Show available commands."""
        return ChatResponse(
            message=format_help_text(),
            action="help_shown",
        )

    async def _cmd_spec(self, args: list[str]) -> ChatResponse:
        """Generate a spec from a description and PERSIST it to disk.

        Phase-5 (G3): replaces the old text-only template with the REAL
        ``generate_spec_from_description`` tool, builds a schema-valid spec
        YAML, and persists it via the REAL ``save_spec`` tool. Returns the
        saved path so callers can ``/run`` it.
        """
        # G33: /spec list → list_example_specs; /spec show <path> → render file.
        if args and args[0].lower() == "list":
            return await self._cmd_spec_list()
        if args and args[0].lower() == "show":
            if len(args) < 2:
                return ChatResponse(message="❌ Usage: /spec show <path>", action="error")
            return await self._cmd_spec_show(args[1])

        description = " ".join(args)

        try:
            from ...tools.spec_generation_tools import generate_spec_from_description

            generated = generate_spec_from_description(description)
        except Exception as e:  # pragma: no cover — defensive
            return ChatResponse(
                message=f"❌ Spec generation failed: {e}",
                action="error",
            )

        # Build a schema-valid SpecModel dict from the generated fields.
        base = generated.get("title") or description or "ml_project"
        project_id = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")[:48] or "ml_project"
        task_type = generated.get("task_type", "classification")
        if task_type not in ("classification", "regression", "segmentation", "detection"):
            task_type = "classification"
        target = generated.get("target", "target")
        primary_metric = generated.get("primary_metric", "f1_macro")

        spec_dict = {
            "project_id": project_id,
            "task_type": task_type,
            "target": target,
            "primary_metric": primary_metric,
            "golden_dataset": "./data/train.csv",
            "baseline": {"type": "majority_class"},
            "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.0"},
        }

        import yaml

        front = yaml.safe_dump(spec_dict, sort_keys=False, default_flow_style=False)
        yaml_text = f"---\n{front}---\n\n{description.strip()}\n"

        # Persist via the REAL save_spec tool (validates against SpecModel first).
        specs_dir = Path(self.settings.local.specs_dir)
        spec_path = str(specs_dir / f"{project_id}.md")
        try:
            from ...tools.spec_tools import save_spec

            result = save_spec(yaml_text, spec_path)
        except Exception as e:  # pragma: no cover — defensive
            return ChatResponse(
                message=f"❌ Spec save failed: {e}",
                action="error",
            )

        if not result.get("saved"):
            errs = "; ".join(result.get("errors", []) or ["unknown validation error"])
            return ChatResponse(
                message=f"❌ Spec validation failed: {errs}",
                action="error",
                data={"errors": result.get("errors", [])},
            )

        saved_path = result.get("path", spec_path)
        self.context["current_spec"] = saved_path
        # Phase 2: mirror the active spec onto the shared bus (design §E.3).
        self._sync_bus(
            "set_spec",
            path=saved_path,
            spec_dict=spec_dict,
            project_id=spec_dict.get("project_id"),
            task_type=task_type,
            target=target,
            primary_metric=primary_metric,
            source="chat",
        )

        return ChatResponse(
            message=(
                f"📝 **Spec generated & saved**\n\n"
                f"  Path: `{saved_path}`\n"
                f"  Task type: `{task_type}`\n"
                f"  Target: `{target}`\n"
                f"  Metric: `{primary_metric}`\n\n"
                f"💡 `/run {saved_path} <data-path>` to execute the pipeline."
            ),
            action="spec_generated",
            data={"spec_path": saved_path, "task_type": task_type, "spec": spec_dict},
            suggestions=[f"/run {saved_path} <data-path>"],
        )

    def _load_spec_dict(self, spec_path: str) -> dict[str, Any]:
        """Parse a spec file's YAML front-matter into a plain dict.

        ``MLPipeline.run`` accepts a plain spec dict (not a strict ``SpecModel``),
        so we parse leniently and don't enforce schema validation here.

        Security note (CWE-22 review 2026-07): ``spec_path``/``data_path`` handled
        by these TUI slash-commands are typed by the LOCAL operator at their own
        terminal (same trust model as ``cat <path>``); they cross no privilege
        boundary. AGENT/LLM-supplied paths go through the sandboxed
        ``amlas.tools.pathref`` containment instead.
        """
        try:
            import frontmatter

            post = frontmatter.loads(Path(spec_path).read_text(encoding="utf-8"))
            d: dict[str, Any] = dict(post.metadata)
            body = (post.content or "").strip()
            if body:
                d.setdefault("domain_notes", body)
            return d
        except Exception:
            return {}

    # ─── DP-2 background-execution helper (Phase 11) ──────────────────────────

    def _run_as_job(
        self,
        kind: str,
        blocking_fn: Callable[[], ChatResponse],
        *,
        submitted_message: str,
        job_args: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Submit a long op as a background job and shape the immediate response.

        ``blocking_fn`` is the long op's REAL body — it captures its own errors and
        returns a finished ``ChatResponse`` (so the job result IS that response).

        With the default inline ``JobManager`` (headless engine / every existing
        test) the job is already ``complete`` when ``submit`` returns, so we return
        the real completion ``ChatResponse`` — synchronous behaviour is preserved
        EXACTLY. With a thread/deferred ``JobManager`` (the Textual app) ``submit``
        returns immediately with the job not yet terminal, so we return a
        ``job_submitted`` response — input is never blocked (design §3.2 acceptance).
        """
        job = self.jobs.submit(
            kind,
            lambda progress_cb, cancel_token: blocking_fn(),
            args=job_args or {},
        )

        if job.status == JOB_COMPLETE and isinstance(job.result, ChatResponse):
            return job.result
        if job.is_terminal:  # nosemgrep
            # A blocking_fn that raised (should be rare — they capture internally)
            # or returned a non-ChatResponse: surface a clean error.
            if isinstance(job.result, ChatResponse):  # pragma: no cover
                return job.result
            return ChatResponse(
                message=f"❌ {kind} failed: {job.error or 'unknown error'}",
                action="error",
                data={"job_id": job.id, "error": job.error},
            )

        # Non-terminal → backgrounded. Return immediately without blocking input.
        return ChatResponse(
            message=submitted_message + f"\n\n  Job: `{job.id}` (status: {job.status})",
            action="job_submitted",
            data={"job_id": job.id, "kind": kind, **(job_args or {})},
        )

    async def _cmd_run(self, args: list[str]) -> ChatResponse:
        """Run the REAL deterministic ML pipeline as a background job (G1, DP-2 / TUI-01).

        Wires ``/run <spec> <data>`` to ``MLPipeline.run`` — research → plan →
        train → evaluate → leaderboard → decision → model card. Path validation is
        synchronous (errors return immediately); the pipeline body runs through the
        ``JobManager`` (inline for isolated validation → synchronous ``pipeline_complete``,
        threaded in the app — non-blocking).
        """
        spec_path = args[0]
        data_path = args[1]

        # Validate spec path exists — synchronous, BEFORE submitting any job.
        if not Path(spec_path).exists():
            return ChatResponse(
                message=f"❌ Spec file not found: `{spec_path}`\nAvailable specs in specs/ directory.",
                action="error",
            )

        spec_dict = self._load_spec_dict(spec_path)

        # A2/A3: resolve the effective LOCAL data file. An explicit local path
        # overrides the spec's (possibly s3://) golden_dataset; a mistyped path is
        # resolved against data/ subdirs; an s3://-only spec in local mode errors.
        try:
            resolved_data_path = resolve_run_data_path(data_path, spec_dict)
        except RunDataPathError as e:
            return ChatResponse(
                message=f"❌ {e}",
                action="error",
            )

        # A7: validate the spec's declared target against the resolved dataset
        # BEFORE launching a doomed run. Uses the A1 normalization so
        # ``machine_failure`` matches ``Machine failure``.
        validation_error = self._validate_target_column(resolved_data_path, spec_dict)
        if validation_error is not None:
            return validation_error

        run_id = f"run-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        output_dir = str(Path(self.settings.local.output_dir) / run_id)
        # A6: capture a real start time at submission so /status never shows
        # "Started: unknown".
        started_at = datetime.now(UTC).isoformat()

        def _run_pipeline_blocking() -> ChatResponse:
            try:
                from ...pipeline.orchestrator import MLPipeline

                pipeline = MLPipeline(output_dir=output_dir)
                result = pipeline.run(
                    spec_dict=spec_dict,
                    data_path=str(resolved_data_path),
                    # : publish per-lead activity to the Dashboard Agent Logs pane
                    # as each stage runs, so the 7 roster leads light up on a local /run
                    # (the local pipeline writes no trace file for the poller to read).
                    on_stage=self._publish_pipeline_stage,
                )
            except Exception as e:
                self._pipeline_state = {
                    "run_id": run_id,
                    "spec_path": spec_path,
                    "data_path": str(resolved_data_path),
                    "status": "failed",
                    "started_at": started_at,  # A6
                    "error": str(e),
                    "n_successful": 0,
                    # TUI-01: record per-stage status so /status + the run-executed
                    # invariant see real progress (not all-'pending').
                    "stages": {
                        "research": "failed",
                        "feature_engineering": "failed",
                        "training": "failed",
                        "evaluation": "failed",
                        "decision": "failed",
                    },
                }
                # B1.1: publish a FAILED run + the attempted dataset so the
                # Results/Dashboard/Data tabs show the failure (not a stale
                # placeholder). Null-safe via _sync_bus.
                self._sync_bus(
                    "set_job",
                    job_id=run_id,
                    kind="run",
                    status="failed",
                    progress=100.0,
                    run_id=run_id,
                    started_by="chat",
                )
                self._sync_bus(
                    "set_run",
                    run_id=run_id,
                    spec_id=spec_dict.get("project_id"),
                    primary_metric=spec_dict.get("primary_metric"),
                    status="failed",
                )
                self._sync_bus(
                    "set_dataset",
                    path=str(resolved_data_path),
                    source="chat",
                )
                return ChatResponse(
                    message=f"❌ Pipeline run failed: {e}",
                    action="error",
                    data={"run_id": run_id, "error": str(e)},
                )

            decision = getattr(getattr(result, "decision", None), "recommendation", "unknown")
            artifacts_dir = getattr(result, "artifacts_dir", "")
            leaderboard_md = getattr(result, "leaderboard", "")
            duration = getattr(result, "total_duration_seconds", 0.0)
            try:
                duration_str = f"{float(duration):.2f}s"
            except (TypeError, ValueError):
                duration_str = "n/a"

            # A5: inspect per-experiment outcomes. Only branch on real result lists
            # (a stubbed MLPipeline hands back a MagicMock — we leave those alone so
            # existing mock-based tests keep their synchronous-complete contract).
            results_obj = getattr(result, "results", None)
            n_total: int | None
            n_ok: int | None
            n_fail: int | None
            if isinstance(results_obj, list):
                successful = [r for r in results_obj if getattr(r, "status", None) == "success"]
                failed = [r for r in results_obj if getattr(r, "status", None) != "success"]
                n_total, n_ok, n_fail = len(results_obj), len(successful), len(failed)
            else:
                successful, failed = [], []
                n_total = n_ok = n_fail = None  # unknown (stubbed) — treat as complete

            # ── A5: 0 successful models out of a REAL, non-empty result set → FAIL.
            if n_total is not None and n_total > 0 and n_ok == 0:
                first_err = next(
                    (getattr(r, "notes", "") for r in failed if getattr(r, "notes", "")),
                    "all experiments failed",
                )
                fail_summary = self._summarize_failed_experiments(failed)
                self._pipeline_state = {
                    "run_id": run_id,
                    "spec_path": spec_path,
                    "data_path": str(resolved_data_path),
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "error": first_err,
                    "n_successful": 0,
                    "n_failed": n_fail,
                    "decision": decision,
                    # Research + feature engineering ran; training/eval had 0 wins.
                    "stages": {
                        "research": "complete",
                        "feature_engineering": "complete",
                        "training": "failed",
                        "evaluation": "failed",
                        "decision": "complete",
                    },
                }
                self._sync_bus(
                    "set_job",
                    job_id=run_id,
                    kind="run",
                    status="failed",
                    progress=100.0,
                    run_id=run_id,
                    started_by="chat",
                )
                # B1.1: publish a FAILED run + the attempted dataset so the
                # Results/Dashboard/Data tabs reflect the failure (0 models trained)
                # rather than a stale placeholder. Null-safe via _sync_bus.
                self._sync_bus(
                    "set_run",
                    run_id=run_id,
                    spec_id=spec_dict.get("project_id"),
                    primary_metric=spec_dict.get("primary_metric"),
                    decision=decision,
                    status="failed",
                )
                self._sync_bus(
                    "set_dataset",
                    path=str(resolved_data_path),
                    source="chat",
                )
                return ChatResponse(
                    message=(
                        f"❌ **Pipeline failed** (Run ID: `{run_id}`)\n\n"
                        f"  0 of {n_total} models trained successfully.\n"
                        f"  First error: {first_err}\n\n"
                        f"{fail_summary}"
                    ),
                    action="error",
                    data={"run_id": run_id, "error": first_err, "n_failed": n_fail},
                )

            # ── A5: some succeeded AND some failed → PARTIAL success.
            partial = (
                n_total is not None and n_ok is not None and n_fail is not None and n_ok > 0 and n_fail > 0
            )
            status = "partial" if partial else "complete"
            best_score = None
            best_algorithm = None
            if successful:
                best = max(successful, key=lambda r: getattr(r, "primary_score", 0.0))
                best_score = getattr(best, "primary_score", None)
                # B1.1: keep the winning algorithm so the Results tab shows a REAL
                # best model (previously computed then discarded).
                best_algorithm = getattr(best, "algorithm", None)

            self._pipeline_state = {
                "run_id": run_id,
                "spec_path": spec_path,
                "data_path": str(resolved_data_path),
                "status": status,
                "artifacts_dir": artifacts_dir,
                "decision": decision,
                "started_at": started_at,  # A6
                "completed_at": datetime.now(UTC).isoformat(),
                "n_successful": n_ok,
                "n_failed": n_fail,
                "best_score": best_score,
                # The run executed all stages; when partial, training still produced
                # ≥1 winner so the stages are complete (the partial count is surfaced).
                "stages": {
                    "research": "complete",
                    "feature_engineering": "complete",
                    "training": "complete",
                    "evaluation": "complete",
                    "decision": "complete",
                },
            }
            self.context["running_jobs"].append(run_id)
            # Phase 2: surface the completed run as the active job on the bus so the
            # Dashboard/Pipeline (later phases) can monitor it (design §E.3/§E.4).
            self._sync_bus(
                "set_job",
                job_id=run_id,
                kind="run",
                status="complete",
                progress=100.0,
                run_id=run_id,
                started_by="chat",
            )
            # B1.1: publish the COMPLETED run (real best algorithm / metric / score /
            # decision) so the Results tab's BestModelCard repaints from a real run,
            # and publish the active dataset so the Data tab reflects it. rows=None
            # (not fabricated — a cheap count isn't computed here). Null-safe via
            # _sync_bus. ``best_algorithm`` is None for stubbed-pipeline validation paths
            # (n_total unknown → successful==[]), which is harmless.
            self._sync_bus(
                "set_run",
                run_id=run_id,
                spec_id=spec_dict.get("project_id"),
                best_algorithm=best_algorithm,
                primary_metric=spec_dict.get("primary_metric"),
                best_score=best_score,
                decision=decision,
                artifacts_dir=artifacts_dir,
                status=status,
            )
            self._sync_bus(
                "set_dataset",
                path=str(resolved_data_path),
                source="chat",
            )

            header = "✅ **Pipeline complete**"
            counts_line = ""
            if partial:
                header = "⚠️ **Pipeline complete (partial)**"
                counts_line = f"  Models: {n_ok} succeeded, {n_fail} failed\n"
                fail_summary = self._summarize_failed_experiments(failed)
            else:
                fail_summary = ""

            return ChatResponse(
                message=(
                    f"{header} (Run ID: `{run_id}`)\n\n"
                    f"{counts_line}"
                    f"  Decision: **{decision}**\n"
                    f"  Duration: {duration_str}\n"
                    f"  Artifacts: `{artifacts_dir}`\n\n"
                    f"{leaderboard_md}" + (f"\n\n{fail_summary}" if fail_summary else "")
                ),
                action="pipeline_complete",
                data={
                    "run_id": run_id,
                    "artifacts_dir": artifacts_dir,
                    "decision": decision,
                    "leaderboard": leaderboard_md,
                    "duration_seconds": duration,
                    "n_successful": n_ok,
                    "n_failed": n_fail,
                },
                suggestions=["/leaderboard", f"/export {run_id} <dest>"],
            )

        # A6: record the run as started BEFORE the (possibly threaded) job body runs
        # so /status shows a real start time immediately on the app path.
        self._pipeline_state = {
            "run_id": run_id,
            "spec_path": spec_path,
            "data_path": str(resolved_data_path),
            "status": "running",
            "started_at": started_at,
            "stages": {
                "research": "running",
                "feature_engineering": "pending",
                "training": "pending",
                "evaluation": "pending",
                "decision": "pending",
            },
        }

        return self._run_as_job(
            "run",
            _run_pipeline_blocking,
            submitted_message=(
                f"🚀 **Pipeline submitted** (Run ID: `{run_id}`)\n\n"
                f"  Spec: `{spec_path}`\n"
                f"  Data: `{resolved_data_path}`\n"
                f"  Running in the background — `/status` to monitor."
            ),
            job_args={"run_id": run_id, "spec_path": spec_path, "data_path": str(resolved_data_path)},
        )

    def _validate_target_column(self, data_path: str, spec_dict: dict[str, Any]) -> ChatResponse | None:
        """A7: validate the spec's declared target against the dataset columns.

        Returns an error :class:`ChatResponse` when the declared target can't be
        resolved to a dataset column (using the A1 normalization), naming the
        target and listing the dataset's actual columns. Returns ``None`` when the
        target validates, when no target is declared, or when the header can't be
        read locally (a non-CSV / remote path — the pipeline surfaces those).
        """
        target = (spec_dict or {}).get("target")
        if not target:
            return None
        if _looks_like_s3(data_path) or not str(data_path).lower().endswith(".csv"):
            return None  # can't cheaply read a header → defer to the pipeline
        try:
            import pandas as pd

            from ...tools.training_tools import resolve_target_column

            header = pd.read_csv(data_path, nrows=0)
            columns = list(header.columns)
        except Exception:
            return None  # unreadable header → let the pipeline surface the real error
        try:
            resolve_target_column(columns, target)
        except ValueError:
            return ChatResponse(
                message=(
                    f"❌ Spec target `{target}` is not a column in the dataset.\n"
                    f"  Dataset: `{data_path}`\n"
                    f"  Available columns: {columns}\n\n"
                    f"Fix the spec's `target:` to match one of the columns above."
                ),
                action="error",
                data={"target": target, "columns": columns},
            )
        return None

    @staticmethod
    def _summarize_failed_experiments(failed: list[Any]) -> str:
        """Render a short 'Failed Experiments' summary from ExperimentResults (A5)."""
        if not failed:
            return ""
        lines = ["**Failed experiments:**"]
        seen: set[str] = set()
        for r in failed:
            algo = getattr(r, "algorithm", "?")
            note = getattr(r, "notes", "") or "(no detail)"
            key = f"{algo}:{note}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  • {algo}: {note}")
            if len(lines) > 6:  # cap the list
                lines.append("  • …")
                break
        return "\n".join(lines)

    async def _cmd_status(self, args: list[str]) -> ChatResponse:
        """Show current pipeline/system status with real state."""
        if self._pipeline_state:
            # Active pipeline — show its progress
            state = self._pipeline_state
            stage_icons = {
                "pending": "○",
                "running": "⟳",
                "complete": "✓",
                "failed": "✗",
            }

            stage_lines = []
            for stage_name, stage_status in state.get("stages", {}).items():
                icon = stage_icons.get(stage_status, "?")
                stage_lines.append(f"  {icon} {stage_name.replace('_', ' ').title()}")

            # A5: surface a real model-outcome count when we have one.
            counts_line = ""
            n_ok = state.get("n_successful")
            n_fail = state.get("n_failed")
            if n_ok is not None:
                total = n_ok + (n_fail or 0)
                counts_line = f"  Models: {n_ok}/{total} succeeded\n"

            # A4: render the captured error so the user can SEE why it failed —
            # previously the error was stored but never displayed.
            error_line = ""
            err = state.get("error")
            if err:
                error_line = f"  ⚠️ Error: {err}\n"

            return ChatResponse(
                message=(
                    f"🔄 **Pipeline Status** (Run: `{state.get('run_id', 'unknown')}`)\n\n"
                    f"  Status: {state.get('status', 'unknown')}\n"
                    f"  Started: {state.get('started_at', 'unknown')}\n"
                    + counts_line
                    + error_line
                    + "\n"
                    + "\n".join(stage_lines)
                ),
                action="status_shown",
                data=state,
            )

        # No active pipeline — show overall system state
        lines = ["ℹ️ **System Status**\n"]
        lines.append("  No active pipeline runs.\n")

        # Query evaluation store for last run info
        try:
            from ...evaluation.store import EvaluationStore

            store_path = self._find_state_file("evaluation_store.json")

            if store_path is not None and store_path.exists():
                store = EvaluationStore(store_path=store_path)
                results = store.get_all()
                if results:
                    last = results[-1]
                    lines.append(
                        f"  **Last experiment:** {last.algorithm} — {last.primary_metric}={last.primary_score:.4f} ({last.status})"
                    )
                    lines.append(f"  **Total experiments:** {len(results)}")
                    successful = [r for r in results if r.status == "success"]
                    if successful:
                        best = max(successful, key=lambda x: x.primary_score)
                        lines.append(
                            f"  **Best model:** {best.algorithm} ({best.primary_metric}={best.primary_score:.4f})"
                        )
                    lines.append("")
        except Exception:  # nosec B110
            pass

        # Query scheduler for projects (bounded resolution — CTX-01)
        try:
            scheduler_path = self._find_state_file("scheduler_state.json")
            if scheduler_path is not None:
                from ...scheduler.project_manager import ProjectManager

                pm = ProjectManager(str(scheduler_path.parent))
                pm.load_state()
                projects = pm.list_projects()
                if projects:
                    by_status: dict[str, int] = {}
                    for p in projects:
                        by_status[p.status] = by_status.get(p.status, 0) + 1
                    status_str = ", ".join(f"{k}: {v}" for k, v in by_status.items())
                    lines.append(f"  **Projects:** {len(projects)} ({status_str})")
                    lines.append("")
        except Exception:  # nosec B110
            pass

        lines.append(f"  Attached files: {len(self.context['attachments'])}")
        lines.append(f"  Messages in session: {len(self.history)}")
        lines.append("\nUse `/run <spec> <data>` to start a pipeline.")

        return ChatResponse(
            message="\n".join(lines),
            action="status_shown",
        )

    async def _cmd_leaderboard(self, args: list[str]) -> ChatResponse:
        """Show the current model leaderboard via the REAL system formatter (G11).

        Replaces the bespoke top-10 medal text with the system's
        ``Leaderboard.format_leaderboard_markdown`` so the TUI and CLI share a
        single source of truth for ranking output (markdown table).
        """
        empty_msg = (
            "📊 **Leaderboard**\n\n  No experiments recorded yet.\n  Run a pipeline to see results here."
        )
        try:
            from ...evaluation.leaderboard import Leaderboard
            from ...evaluation.store import EvaluationStore

            # CTX-01: bounded resolution (no per-turn unbounded Path('.').rglob).
            store_path = self._find_state_file("evaluation_store.json")
            if store_path is None or not store_path.exists():
                return ChatResponse(message=empty_msg, action="leaderboard_shown")

            store = EvaluationStore(store_path=store_path)
            results = store.get_all()
            if not results:
                return ChatResponse(message=empty_msg, action="leaderboard_shown")

            # spec_id: explicit arg wins, else the most recent result's spec.
            spec_id = args[0] if args else results[-1].spec_id
            metric = results[-1].primary_metric or "f1_macro"

            leaderboard = Leaderboard(store=store)
            md = leaderboard.format_leaderboard_markdown(spec_id=spec_id, metric=metric)

            return ChatResponse(
                message=f"📊 **Leaderboard**\n\n{md}",
                action="leaderboard_shown",
                data={"spec_id": spec_id, "metric": metric},
            )

        except Exception:
            logger.debug("Leaderboard load failed", exc_info=True)
            return ChatResponse(message=empty_msg, action="leaderboard_shown")

    async def _cmd_research(self, args: list[str]) -> ChatResponse:
        """Search the local algorithm catalog or the external research aggregator.

        G32: a leading ``--web`` flag routes to the REAL external aggregator
        ``research_search`` (arxiv + Semantic Scholar + Papers-with-Code +
        Tavily). Network-gated: degrades gracefully when offline / no keys.
        """
        if args and args[0] == "--web":
            return await self._cmd_research_web(args[1:])

        query = " ".join(args)

        try:
            from ...tools.research_tools import search_algorithms

            # Search algorithm KB
            algo_result = search_algorithms(
                task_type="classification",
                data_type="tabular",
                data_size="medium",
            )

            # Format response
            lines = [f"🔬 **Research Results** for: *{query}*\n"]

            # Algorithms
            algos = algo_result.get("algorithms", []) if isinstance(algo_result, dict) else []
            if algos:
                lines.append("**Recommended Algorithms:**")
                for algo in algos[:5]:
                    name = algo.get("display_name", algo.get("name", "unknown"))
                    strengths = ", ".join(algo.get("strengths", [])[:2])
                    lines.append(f"  • {name} — {strengths}")
                lines.append("")

            if not algos:
                lines.append("  No results found. Try a broader search query.")

            return ChatResponse(
                message="\n".join(lines),
                action="research_complete",
                data={"algorithms": algos[:5]},
            )

        except Exception as e:
            logger.debug("Research command failed", exc_info=True)
            return ChatResponse(
                message=(
                    f"🔬 **Research** for: *{query}*\n\n"
                    f"  ⚠️ Research tools unavailable: {e}\n"
                    f"  Ensure dependencies are installed."
                ),
                action="error",
            )

    async def _cmd_quality(self, args: list[str]) -> ChatResponse:
        """Run data quality check."""
        data_path = _normalize_fs_path(args[0])  # : expand ~, strip quotes

        if not Path(data_path).exists():
            return ChatResponse(
                message=f"❌ File not found: `{data_path}`",
                action="error",
            )

        try:
            from ...tools.data_quality_tools import score_data_quality

            # QUAL-01: infer the real target column (no hardcoded "target"). When
            # it can't be determined, fall back to the last column rather than a
            # literal "target" that exists in no real dataset.
            target_column = self._infer_target_column(data_path) or ""
            result = score_data_quality(data_path=data_path, target_column=target_column)

            if isinstance(result, dict):
                overall = result.get("overall_score", 0)
                summary = result.get("summary", "Assessment complete.")
                issues = result.get("issues", [])

                lines = [
                    f"✅ **Data Quality Report** for `{data_path}`\n",
                    f"  Overall Score: **{overall:.1f}/100**\n",
                    f"  {summary}\n",
                ]

                if issues:
                    lines.append(f"  Issues found: {len(issues)}")
                    for issue in issues[:5]:
                        col = issue.get("column", "?")
                        desc = issue.get("issue", "unknown")
                        sev = issue.get("severity", "?")
                        lines.append(f"    • [{sev}] {col}: {desc}")

                return ChatResponse(
                    message="\n".join(lines),
                    action="quality_checked",
                    data=result,
                )
            else:
                return ChatResponse(
                    message=f"✅ Data quality check completed for `{data_path}`.",
                    action="quality_checked",
                )

        except Exception as e:
            logger.debug("Quality command failed", exc_info=True)
            return ChatResponse(
                message=f"❌ Data quality check failed: {e}",
                action="error",
            )

    def _infer_target_column(self, data_path: str, default: str = "target") -> str:
        """Best-effort target column inference for train/tune commands.

        Prefers the active dataset summary, falls back to a known target name in
        the CSV header, else the last column, else ``default``.
        """
        summary = self.context.get("data_summary") or {}
        cols = summary.get("columns") if isinstance(summary, dict) else None
        if not cols:
            try:
                import pandas as pd

                cols = list(pd.read_csv(data_path, nrows=0).columns)
            except Exception:
                cols = None
        if cols:
            for cand in ("target", "label", "y", "class"):
                if cand in cols:
                    return cand
            return str(cols[-1])
        return default

    async def _cmd_train(self, args: list[str]) -> ChatResponse:
        """Train a specific model via the REAL ``train_model`` tool (G2, DP-2 / TUI-01).

        Long-running → backgrounded via the ``JobManager`` (inline for isolated validation; the
        app threads it). Validation is synchronous; ``train_model`` runs in the job.
        """
        algorithm = args[0]
        data_path = args[1]

        if not Path(data_path).exists():
            return ChatResponse(
                message=f"❌ Data file not found: `{data_path}`",
                action="error",
            )

        target_column = self._infer_target_column(str(data_path))

        def _train_blocking() -> ChatResponse:
            try:
                from ...tools.training_tools import train_model

                result = train_model(
                    data_path=str(data_path),
                    target_column=target_column,
                    algorithm=algorithm,
                )
            except Exception as e:
                return ChatResponse(
                    message=f"❌ Training failed: {e}",
                    action="error",
                    data={"algorithm": algorithm, "data_path": str(data_path)},
                )

            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            model_path = result.get("model_path", "") if isinstance(result, dict) else ""
            metric_lines = (
                "\n".join(
                    f"  • {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  • {k}: {v}"
                    for k, v in metrics.items()
                )
                or "  (no metrics returned)"
            )

            return ChatResponse(
                message=(
                    f"🏋️ **Training complete**\n\n"
                    f"  Algorithm: `{algorithm}`\n"
                    f"  Target: `{target_column}`\n"
                    f"  Model: `{model_path}`\n\n"
                    f"**Metrics:**\n{metric_lines}"
                ),
                action="training_complete",
                data={"algorithm": algorithm, "data_path": str(data_path), "result": result},
                suggestions=["/leaderboard", "/status"],
            )

        return self._run_as_job(
            "train",
            _train_blocking,
            submitted_message=(
                f"🏋️ **Training submitted**\n\n"
                f"  Algorithm: `{algorithm}`\n"
                f"  Target: `{target_column}`\n"
                f"  Running in the background — `/status` to monitor."
            ),
            job_args={"algorithm": algorithm, "data_path": str(data_path)},
        )

    # ─── Phase 5 — newly wired commands (G4, G5, G10, G12) ───────────────────

    def _resolve_store_path(self) -> Path:
        """Locate the evaluation store JSON (bounded resolution — CTX-01).

        Uses the deterministic, depth-limited ``_find_state_file`` helper instead
        of a per-call unbounded ``Path('.').rglob`` walk. Falls back to the
        configured output-dir path (which may not yet exist) so callers that
        create-on-write keep working.
        """
        found = self._find_state_file("evaluation_store.json")
        if found is not None:
            return found
        return Path(self.settings.local.output_dir) / "evaluation_store.json"

    async def _cmd_export(self, args: list[str]) -> ChatResponse:
        """Export a run's model artifact to a destination path (G4).

        Mirrors the CLI ``results export`` artifact-copy: looks up the run in
        the evaluation store and copies its ``artifact_path`` to ``dest``.
        """
        run_id = args[0]
        dest = args[1]

        try:
            from ...evaluation.store import EvaluationStore

            store_path = self._resolve_store_path()
            if not store_path.exists():
                return ChatResponse(
                    message="❌ No evaluation store found — nothing to export.",
                    action="error",
                )
            store = EvaluationStore(store_path=store_path)
            matches = store.get_by_run_ids([run_id])
            if not matches:
                return ChatResponse(
                    message=f"❌ Run not found in store: `{run_id}`",
                    action="error",
                )
            artifact_path = matches[0].artifact_path
            if not artifact_path or not Path(artifact_path).exists():
                return ChatResponse(
                    message=f"❌ Artifact for run `{run_id}` not found on disk: `{artifact_path}`",
                    action="error",
                )

            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact_path, dest_path)

            return ChatResponse(
                message=(
                    f"📥 **Export complete**\n\n"
                    f"  Run: `{run_id}`\n"
                    f"  From: `{artifact_path}`\n"
                    f"  To: `{dest_path}`"
                ),
                action="export_complete",
                data={"run_id": run_id, "source": artifact_path, "dest": str(dest_path)},
            )
        except Exception as e:
            return ChatResponse(message=f"❌ Export failed: {e}", action="error")

    async def _cmd_dashboard(self, args: list[str]) -> ChatResponse:
        """Show the project dashboard via the REAL ``get_dashboard`` tool (G5)."""
        try:
            from ...tools.scheduler_tools import get_dashboard

            md = get_dashboard()
            return ChatResponse(
                message=f"🏠 **Dashboard**\n\n{md}",
                action="dashboard_shown",
                data={"dashboard": md},
            )
        except Exception as e:
            return ChatResponse(message=f"❌ Dashboard unavailable: {e}", action="error")

    async def _cmd_tune(self, args: list[str]) -> ChatResponse:
        """Run Optuna HPO via the REAL ``tune_model`` tool (G10, DP-2 background)."""
        algorithm = args[0]
        data_path = args[1]

        if not Path(data_path).exists():
            return ChatResponse(
                message=f"❌ Data file not found: `{data_path}`",
                action="error",
            )

        target_column = self._infer_target_column(str(data_path))

        def _tune_blocking() -> ChatResponse:
            try:
                from ...tools.hpo_tools import tune_model

                result = tune_model(
                    algorithm=algorithm,
                    data_path=str(data_path),
                    target_column=target_column,
                    task_type="classification",
                )
            except Exception as e:
                return ChatResponse(message=f"❌ Tuning failed: {e}", action="error")

            best_params = result.get("best_params", {}) if isinstance(result, dict) else {}
            best_score = result.get("best_score", "n/a") if isinstance(result, dict) else "n/a"
            n_trials = result.get("n_trials", "n/a") if isinstance(result, dict) else "n/a"

            return ChatResponse(
                message=(
                    f"🎛️ **Tuning complete**\n\n"
                    f"  Algorithm: `{algorithm}`\n"
                    f"  Best score: {best_score}\n"
                    f"  Trials: {n_trials}\n"
                    f"  Best params: `{best_params}`"
                ),
                action="tune_complete",
                data={"algorithm": algorithm, "result": result},
                suggestions=[f"/train {algorithm} {data_path}", "/leaderboard"],
            )

        return self._run_as_job(
            "tune",
            _tune_blocking,
            submitted_message=(
                f"🎛️ **Tuning submitted**\n\n"
                f"  Algorithm: `{algorithm}`\n"
                f"  Running Optuna HPO in the background — `/status` to monitor."
            ),
            job_args={"algorithm": algorithm, "data_path": str(data_path)},
        )

    async def _cmd_experiment(self, args: list[str]) -> ChatResponse:
        """Run a multi-algorithm experiment plan via the REAL ExperimentRunner (G10, DP-2)."""
        spec_path = args[0]
        data_path = args[1]

        if not Path(spec_path).exists():
            return ChatResponse(message=f"❌ Spec file not found: `{spec_path}`", action="error")
        if not Path(data_path).exists():
            return ChatResponse(message=f"❌ Data file not found: `{data_path}`", action="error")

        spec_dict = self._load_spec_dict(spec_path)
        spec_dict.setdefault("data_path", str(data_path))
        spec_dict.setdefault("spec_id", spec_dict.get("project_id", "experiment"))

        def _experiment_blocking() -> ChatResponse:
            try:
                from ...evaluation.store import EvaluationStore
                from ...experiments.runner import ExperimentRunner
                from ...tools.research_tools import recommend_approaches

                try:
                    recommendations = recommend_approaches(spec_json=json.dumps(spec_dict))
                except Exception:
                    recommendations = {"recommended_algorithms": []}

                store_path = Path(self.settings.local.output_dir) / "evaluation_store.json"
                store = EvaluationStore(store_path=store_path)
                runner = ExperimentRunner(store=store)
                plan = runner.create_plan_from_recommendations(spec_dict, recommendations)
                results = runner.run_plan(plan)
                report = runner.get_report(plan)
            except Exception as e:
                return ChatResponse(message=f"❌ Experiment failed: {e}", action="error")

            n_results = len(results) if hasattr(results, "__len__") else 0
            return ChatResponse(
                message=(f"🧪 **Experiment complete**\n\n  Runs: {n_results}\n\n{report}"),
                action="experiment_complete",
                data={"n_results": n_results},
                suggestions=["/leaderboard"],
            )

        return self._run_as_job(
            "experiment",
            _experiment_blocking,
            submitted_message=(
                f"🧪 **Experiment submitted**\n\n"
                f"  Spec: `{spec_path}`\n"
                f"  Running the experiment plan in the background — `/status` to monitor."
            ),
            job_args={"spec_path": spec_path, "data_path": str(data_path)},
        )

    async def _cmd_modelcard(self, args: list[str]) -> ChatResponse:
        """Generate a model card for a recorded run via the REAL generator (G12)."""
        run_id = args[0]
        try:
            from ...decisions.model_card import generate_model_card
            from ...evaluation.store import EvaluationStore

            store_path = self._resolve_store_path()
            if not store_path.exists():
                return ChatResponse(
                    message="❌ No evaluation store found — nothing to document.",
                    action="error",
                )
            store = EvaluationStore(store_path=store_path)
            matches = store.get_by_run_ids([run_id])
            if not matches:
                return ChatResponse(
                    message=f"❌ Run not found in store: `{run_id}`",
                    action="error",
                )
            result = matches[0]
            spec = {
                "project_id": result.spec_id,
                "task_type": "classification",
                "target": "target",
                "primary_metric": result.primary_metric,
            }
            card_md = generate_model_card(result, spec)
            return ChatResponse(
                message=card_md,
                action="modelcard_generated",
                data={"run_id": run_id},
            )
        except Exception as e:
            return ChatResponse(message=f"❌ Model card generation failed: {e}", action="error")

    # ─── Phase 9 — EXPOSE-set parity handlers (G21–G37 + #20) ────────────────

    def _spec_id_arg(self, args: list[str]) -> str:
        """Resolve a spec_id from an optional arg, active spec, or store."""
        if args:
            return args[0]
        try:
            store_path = self._resolve_store_path()
            if store_path.exists():
                from ...evaluation.store import EvaluationStore

                results = EvaluationStore(store_path=store_path).get_all()
                if results:
                    return results[-1].spec_id
        except Exception:  # nosec B110
            pass
        return "default"

    async def _cmd_split(self, args: list[str]) -> ChatResponse:
        """Split a dataset into train/test via the REAL ``split_dataset`` (G22)."""
        data_path, target = args[0], args[1]
        test_size = 0.2
        if len(args) >= 3:
            try:
                test_size = float(args[2])
            except ValueError:
                return ChatResponse(message=f"❌ test_size must be a float: `{args[2]}`", action="error")
        if not Path(data_path).exists():
            return ChatResponse(message=f"❌ Data file not found: `{data_path}`", action="error")
        try:
            from ...tools.data_tools import split_dataset

            result = split_dataset(path=str(data_path), target_column=target, test_size=test_size)
        except Exception as e:
            return ChatResponse(message=f"❌ Split failed: {e}", action="error")
        if isinstance(result, dict) and not result.get("success", True):
            return ChatResponse(message=f"❌ Split failed: {result.get('error')}", action="error")
        train_n = result.get("train_size", "?") if isinstance(result, dict) else "?"
        test_n = result.get("test_size", "?") if isinstance(result, dict) else "?"
        return ChatResponse(
            message=(
                f"✂️ **Split complete**\n\n"
                f"  Train rows: {train_n}\n"
                f"  Test rows: {test_n}\n"
                f"  Target: `{target}` | test_size: {test_size}"
            ),
            action="split_complete",
            data={"result": result},
        )

    async def _cmd_describe(self, args: list[str]) -> ChatResponse:
        """Full dataset summary stats via the REAL ``describe_dataset`` (G23)."""
        data_path = _normalize_fs_path(args[0])
        target = args[1] if len(args) >= 2 else None
        if not Path(data_path).exists():
            return ChatResponse(message=f"❌ Data file not found: `{data_path}`", action="error")
        try:
            from ...tools.data_tools import describe_dataset

            result = describe_dataset(path=str(data_path), target_column=target)
        except Exception as e:
            return ChatResponse(message=f"❌ Describe failed: {e}", action="error")
        if isinstance(result, dict) and not result.get("success", True):
            return ChatResponse(message=f"❌ Describe failed: {result.get('error')}", action="error")
        n_rows = result.get("n_rows", "?") if isinstance(result, dict) else "?"
        n_cols = result.get("n_cols", "?") if isinstance(result, dict) else "?"
        cols = result.get("columns", []) if isinstance(result, dict) else []
        col_display = ", ".join(map(str, cols[:12]))
        msg = (
            f"📋 **Dataset description:** `{Path(data_path).name}`\n\n"
            f"  Shape: {n_rows} rows × {n_cols} columns\n"
            f"  Columns: {col_display}"
        )
        # Surface the class balance the tool already computes when a target is given
        # Surface class balance so users do not miss an imbalance trap.
        target_dist = result.get("target_distribution") if isinstance(result, dict) else None
        if target_dist:
            total = sum(target_dist.values()) or 1
            parts = ", ".join(f"{k}: {v} ({100 * v / total:.1f}%)" for k, v in target_dist.items())
            msg += f"\n  Target `{target}` balance: {parts}"
            majority = max(target_dist.values()) / total
            if majority >= 0.90:
                msg += (
                    f"\n  ⚠️ Imbalanced — the majority class is {majority * 100:.1f}%. "
                    f"Prefer macro-F1 over accuracy for this target."
                )
        return ChatResponse(
            message=msg,
            action="describe_complete",
            data={"result": result},
        )

    async def _cmd_anomalies(self, args: list[str]) -> ChatResponse:
        """Detect outliers via the REAL ``detect_anomalies`` (G24)."""
        data_path = _normalize_fs_path(args[0])
        method = args[1] if len(args) >= 2 else "iqr"
        if not Path(data_path).exists():
            return ChatResponse(message=f"❌ Data file not found: `{data_path}`", action="error")
        try:
            from ...tools.data_quality_tools import detect_anomalies

            result = detect_anomalies(data_path=str(data_path), method=method)
        except Exception as e:
            return ChatResponse(message=f"❌ Anomaly detection failed: {e}", action="error")
        total = result.get("total_anomalies", 0) if isinstance(result, dict) else 0
        rate = result.get("anomaly_rate", 0.0) if isinstance(result, dict) else 0.0
        rec = result.get("recommendation", "keep") if isinstance(result, dict) else "keep"
        return ChatResponse(
            message=(
                f"🔎 **Anomaly report** (`{method}`)\n\n"
                f"  Total anomalies: {total}\n"
                f"  Anomaly rate: {rate}\n"
                f"  Recommendation: **{rec}**"
            ),
            action="anomalies_checked",
            data={"result": result},
        )

    async def _cmd_leakage(self, args: list[str]) -> ChatResponse:
        """Run train/test leakage checks via the REAL ``leakage_check_tool`` (G25)."""
        train_path, test_path, target = (
            _normalize_fs_path(args[0]),
            _normalize_fs_path(args[1]),
            args[2],
        )
        for pth in (train_path, test_path):
            if not Path(pth).exists():
                return ChatResponse(message=f"❌ File not found: `{pth}`", action="error")
        try:
            from ...tools.agent_tools import leakage_check_tool

            result = leakage_check_tool(train_path=str(train_path), test_path=str(test_path), target=target)
        except Exception as e:
            return ChatResponse(message=f"❌ Leakage check failed: {e}", action="error")
        blocking = result.get("blocking", False) if isinstance(result, dict) else False
        summary = result.get("summary", "") if isinstance(result, dict) else ""
        findings = result.get("findings", []) if isinstance(result, dict) else []
        icon = "🚫" if blocking else "✅"
        return ChatResponse(
            message=(
                f"{icon} **Leakage check** (target=`{target}`)\n\n"
                f"  Blocking: {blocking}\n"
                f"  Findings: {len(findings)}\n"
                f"  {summary}"
            ),
            action="leakage_checked",
            data={"result": result},
        )

    async def _cmd_preprocess(self, args: list[str]) -> ChatResponse:
        """Auto feature-engineer via the REAL ``auto_preprocess`` (G26, DP-2)."""
        data_path = _normalize_fs_path(args[0])
        target = args[1] if len(args) >= 2 else self._infer_target_column(str(data_path))
        if not Path(data_path).exists():
            return ChatResponse(message=f"❌ Data file not found: `{data_path}`", action="error")

        def _preprocess_blocking() -> ChatResponse:
            try:
                from ...tools.fe_tools import auto_preprocess

                result = auto_preprocess(path=str(data_path), target_column=target)
            except Exception as e:
                return ChatResponse(message=f"❌ Preprocess failed: {e}", action="error")
            if isinstance(result, dict) and not result.get("success", True):
                return ChatResponse(message=f"❌ Preprocess failed: {result.get('error')}", action="error")
            n_out = result.get("n_features_out", "?") if isinstance(result, dict) else "?"
            dropped = result.get("dropped_columns", []) if isinstance(result, dict) else []
            return ChatResponse(
                message=(
                    f"🧹 **Preprocessing complete**\n\n"
                    f"  Target: `{target}`\n"
                    f"  Features out: {n_out}\n"
                    f"  Dropped columns: {len(dropped)}"
                ),
                action="preprocess_complete",
                data={"result": result},
            )

        return self._run_as_job(
            "preprocess",
            _preprocess_blocking,
            submitted_message=(
                f"🧹 **Preprocessing submitted**\n\n"
                f"  Target: `{target}`\n"
                f"  Feature-engineering in the background — `/status` to monitor."
            ),
            job_args={"data_path": str(data_path), "target": target},
        )

    async def _cmd_ensemble(self, args: list[str]) -> ChatResponse:
        """Train a stacking/voting ensemble via the REAL ``train_ensemble`` (G27, DP-2)."""
        data_path = args[0]
        algorithms = args[1:]
        if not Path(data_path).exists():
            return ChatResponse(message=f"❌ Data file not found: `{data_path}`", action="error")
        target = self._infer_target_column(str(data_path))

        def _ensemble_blocking() -> ChatResponse:
            try:
                from ...tools.hpo_tools import train_ensemble

                result = train_ensemble(
                    algorithms=list(algorithms),
                    data_path=str(data_path),
                    target_column=target,
                    task_type="classification",
                )
            except Exception as e:
                return ChatResponse(message=f"❌ Ensemble training failed: {e}", action="error")
            if isinstance(result, dict) and not result.get("success", True):
                return ChatResponse(
                    message=f"❌ Ensemble training failed: {result.get('error')}", action="error"
                )
            method = result.get("method", "stacking") if isinstance(result, dict) else "stacking"
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            metric_lines = (
                "\n".join(
                    f"  • {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  • {k}: {v}"
                    for k, v in metrics.items()
                )
                or "  (no metrics returned)"
            )
            return ChatResponse(
                message=(
                    f"🎼 **Ensemble complete** (`{method}`)\n\n"
                    f"  Base algorithms: {', '.join(algorithms)}\n\n"
                    f"**Metrics:**\n{metric_lines}"
                ),
                action="ensemble_complete",
                data={"result": result},
                suggestions=["/leaderboard"],
            )

        return self._run_as_job(
            "ensemble",
            _ensemble_blocking,
            submitted_message=(
                f"🎼 **Ensemble submitted**\n\n"
                f"  Base algorithms: {', '.join(algorithms)}\n"
                f"  Training in the background — `/status` to monitor."
            ),
            job_args={"data_path": str(data_path), "algorithms": list(algorithms)},
        )

    async def _cmd_best(self, args: list[str]) -> ChatResponse:
        """Show the current best model via the REAL ``get_best_model`` (G28)."""
        spec_id = self._spec_id_arg(args)
        store_path = self._resolve_store_path()
        if not store_path.exists():
            return ChatResponse(message="❌ No evaluation store found.", action="error")
        try:
            from ...tools.evaluation_tools import get_best_model

            result = get_best_model(store_path=str(store_path), spec_id=spec_id)
        except Exception as e:
            return ChatResponse(message=f"❌ Best-model lookup failed: {e}", action="error")
        if not (isinstance(result, dict) and result.get("success")):
            err = result.get("error", "no results") if isinstance(result, dict) else "no results"
            return ChatResponse(message=f"❌ {err}", action="error")
        return ChatResponse(
            message=(
                f"🥇 **Best model** for `{spec_id}`\n\n"
                f"  Algorithm: `{result.get('best_algorithm')}`\n"
                f"  {result.get('primary_metric')}: {result.get('best_score')}\n"
                f"  Run: `{result.get('best_run_id')}`"
            ),
            action="best_model_shown",
            data={"result": result},
        )

    async def _cmd_recommend(self, args: list[str]) -> ChatResponse:
        """Deploy/iterate/reject recommendation via REAL ``get_recommendation`` (G29)."""
        spec_id = self._spec_id_arg(args)
        store_path = self._resolve_store_path()
        if not store_path.exists():
            return ChatResponse(message="❌ No evaluation store found.", action="error")
        spec_json = json.dumps({"project_id": spec_id, "spec_id": spec_id})
        try:
            from ...tools.decision_tools import get_recommendation

            result = get_recommendation(store_path=str(store_path), spec_id=spec_id, spec_json=spec_json)
        except Exception as e:
            return ChatResponse(message=f"❌ Recommendation failed: {e}", action="error")
        if not (isinstance(result, dict) and result.get("success")):
            err = result.get("error", "no analysis") if isinstance(result, dict) else "no analysis"
            return ChatResponse(message=f"❌ {err}", action="error")
        steps = result.get("next_steps", []) or []
        step_lines = "\n".join(f"  • {s}" for s in steps) or "  (none)"
        return ChatResponse(
            message=(
                f"🧭 **Recommendation** for `{spec_id}`\n\n"
                f"  Decision: **{result.get('recommendation')}**\n"
                f"  Confidence: {result.get('confidence')}\n"
                f"  Risk: {result.get('risk_level')}\n\n"
                f"  {result.get('rationale', '')}\n\n"
                f"**Next steps:**\n{step_lines}"
            ),
            action="recommendation_shown",
            data={"result": result},
        )

    async def _cmd_nextsteps(self, args: list[str]) -> ChatResponse:
        """Generate a next-steps report via REAL ``generate_next_steps_report`` (G30)."""
        spec_id = self._spec_id_arg(args)
        store_path = self._resolve_store_path()
        if not store_path.exists():
            return ChatResponse(message="❌ No evaluation store found.", action="error")
        spec_json = json.dumps({"project_id": spec_id, "spec_id": spec_id})
        try:
            from ...tools.decision_tools import generate_next_steps_report

            result = generate_next_steps_report(
                store_path=str(store_path), spec_id=spec_id, spec_json=spec_json
            )
        except Exception as e:
            return ChatResponse(message=f"❌ Next-steps report failed: {e}", action="error")
        if not (isinstance(result, dict) and result.get("success")):
            err = result.get("error", "no report") if isinstance(result, dict) else "no report"
            return ChatResponse(message=f"❌ {err}", action="error")
        return ChatResponse(
            message=result.get("report", "(empty report)"),
            action="nextsteps_shown",
            data={"spec_id": spec_id},
        )

    async def _cmd_research_web(self, args: list[str]) -> ChatResponse:
        """External research aggregator via REAL ``research_search`` (G32)."""
        query = " ".join(args)
        if not query:
            return ChatResponse(message="❌ Usage: /research --web <query>", action="error")
        try:
            from ...tools.research.strands_tools import research_search

            result = research_search(query, max_results=5)
        except Exception as e:
            return ChatResponse(
                message=(
                    f"🌐 **Web research** for: *{query}*\n\n"
                    f"  ⚠️ External search unavailable: {e}\n"
                    f"  (Network-gated — needs connectivity / API keys.)"
                ),
                action="error",
            )
        results = result.get("results", []) if isinstance(result, dict) else []
        lines = [f"🌐 **Web research** for: *{query}* ({len(results)} hits)\n"]
        for r in results[:5]:
            title = r.get("title", "Untitled") if isinstance(r, dict) else str(r)
            url = r.get("url", "") if isinstance(r, dict) else ""
            lines.append(f"  • {title}\n    {url}")
        if not results:
            lines.append("  No external results (offline or no API keys).")
        return ChatResponse(
            message="\n".join(lines),
            action="research_web_complete",
            data={"result": result},
        )

    async def _cmd_spec_list(self) -> ChatResponse:
        """List example specs via the REAL ``list_example_specs`` (G33).

        Phase E2 FIX-1: ``list_example_specs`` is a pure filesystem glob and the
        module now imports ``strands`` under a guarded fallback, so a
        partial/minimal install (no ``strands``) no longer breaks this. The
        ``except`` copy names the REAL failing operation ("could not read the
        specs/ directory") so a genuine env/FS problem is actionable rather than
        the old confusing "Spec listing failed: No module named 'strands'".
        """
        try:
            from ...tools.spec_tools import list_example_specs

            result = list_example_specs()
        except Exception as e:  # pragma: no cover - defensive; FIX-1 removed the strands cause
            return ChatResponse(
                message=(
                    f"❌ Could not read the specs/ directory: {e}\n"
                    "  (This is a local filesystem read — check the specs/ path exists.)"
                ),
                action="error",
            )
        specs = result.get("specs", []) if isinstance(result, dict) else []
        if not specs:
            return ChatResponse(
                message="📝 **Specs**\n\n  No example specs found in the specs/ directory.",
                action="spec_listed",
                data={"specs": []},
            )
        lines = [f"📝 **Example Specs** ({len(specs)} found):\n"]
        for s in specs[:20]:
            lines.append(f"  • `{s.get('name', '?')}`")
        lines.append("\n💡 `/spec show <path>` to view a spec's contents.")
        return ChatResponse(
            message="\n".join(lines),
            action="spec_listed",
            data={"specs": [s.get("name") for s in specs]},
        )

    async def _cmd_spec_show(self, path: str) -> ChatResponse:
        """Render a spec file's contents (CLI ``spec show`` equivalent, #47)."""
        p = Path(path)
        if not p.exists():
            return ChatResponse(message=f"❌ Spec file not found: `{path}`", action="error")
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            return ChatResponse(message=f"❌ Could not read spec: {e}", action="error")
        return ChatResponse(
            message=f"📝 **Spec:** `{p.name}`\n\n```\n{content}\n```",
            action="spec_shown",
            data={"path": str(p)},
        )

    async def _cmd_project(self, args: list[str]) -> ChatResponse:
        """Manage queued projects: submit | list | status (G34)."""
        sub = args[0].lower()
        rest = args[1:]
        if sub == "submit":
            if len(rest) < 2:
                return ChatResponse(message="❌ Usage: /project submit <spec> <data>", action="error")
            spec_path, data_path = rest[0], rest[1]
            if not Path(spec_path).exists():
                return ChatResponse(message=f"❌ Spec file not found: `{spec_path}`", action="error")
            spec_dict = self._load_spec_dict(spec_path)
            spec_dict.setdefault("project_id", Path(spec_path).stem)
            try:
                from ...tools.scheduler_tools import submit_project

                result = submit_project(json.dumps(spec_dict), str(data_path))
            except Exception as e:
                return ChatResponse(message=f"❌ Project submit failed: {e}", action="error")
            if isinstance(result, dict) and result.get("error"):
                return ChatResponse(message=f"❌ {result['error']}", action="error")
            return ChatResponse(
                message=(
                    f"📨 **Project submitted**\n\n"
                    f"  ID: `{result.get('project_id')}`\n"
                    f"  Queue position: {result.get('queue_position')}\n"
                    f"  Status: {result.get('status')}"
                ),
                action="project_submitted",
                data={"result": result},
            )
        if sub == "list":
            try:
                from ...tools.scheduler_tools import list_all_projects

                result = list_all_projects()
            except Exception as e:
                return ChatResponse(message=f"❌ Project list failed: {e}", action="error")
            projects = result.get("projects", []) if isinstance(result, dict) else []
            lines = [f"📋 **Projects** ({result.get('total', len(projects))} total)\n"]
            for pr in projects[:15]:
                lines.append(f"  • `{pr.get('project_id')}` — {pr.get('name')} [{pr.get('status')}]")
            if not projects:
                lines.append("  (queue empty)")
            return ChatResponse(
                message="\n".join(lines),
                action="projects_listed",
                data={"result": result},
            )
        if sub == "status":
            if not rest:
                return ChatResponse(message="❌ Usage: /project status <id>", action="error")
            try:
                from ...tools.scheduler_tools import get_project_status

                result = get_project_status(rest[0])
            except Exception as e:
                return ChatResponse(message=f"❌ Project status failed: {e}", action="error")
            if isinstance(result, dict) and result.get("error"):
                return ChatResponse(message=f"❌ {result['error']}", action="error")
            return ChatResponse(
                message=(
                    f"📦 **Project** `{result.get('project_id')}`\n\n"
                    f"  Status: {result.get('status')}\n"
                    f"  Priority: {result.get('priority')}\n"
                    f"  Created: {result.get('created_at')}"
                ),
                action="project_status_shown",
                data={"result": result},
            )
        return ChatResponse(
            message="❌ Usage: /project submit <spec> <data> | /project list | /project status <id>",
            action="error",
        )

    async def _cmd_queue(self, args: list[str]) -> ChatResponse:
        """Run the project queue via REAL ``run_project_queue`` → ProjectScheduler.run_all (G34, DP-2).

        State-changing/long op: exposed as an explicit slash command only (safety
        default), and (Phase 11) executed as a background job — the ``ProjectScheduler``
        is itself a ThreadPoolExecutor, so it composes naturally with the JobManager.
        """
        if args[0].lower() != "run":
            return ChatResponse(message="❌ Usage: /queue run [max_concurrent]", action="error")
        max_concurrent = 3
        if len(args) >= 2:
            try:
                max_concurrent = int(args[1])
            except ValueError:
                return ChatResponse(message=f"❌ max_concurrent must be an int: `{args[1]}`", action="error")

        def _queue_blocking() -> ChatResponse:
            try:
                from ...tools.scheduler_tools import run_project_queue

                result = run_project_queue(max_concurrent=max_concurrent)
            except Exception as e:
                return ChatResponse(message=f"❌ Queue run failed: {e}", action="error")
            return ChatResponse(
                message=(
                    f"⚙️ **Queue run complete**\n\n"
                    f"  Processed: {result.get('processed_count')}\n"
                    f"  Completed: {result.get('completed')}\n"
                    f"  Failed: {result.get('failed')}\n"
                    f"  Remaining queued: {result.get('remaining_queued')}"
                ),
                action="queue_complete",
                data={"result": result},
            )

        return self._run_as_job(
            "queue",
            _queue_blocking,
            submitted_message=(
                f"⚙️ **Queue run submitted** (max_concurrent={max_concurrent})\n\n"
                f"  Running the project queue in the background — `/status` to monitor."
            ),
            job_args={"max_concurrent": max_concurrent},
        )

    async def _cmd_watch(self, args: list[str]) -> ChatResponse:
        """Live run/store summary snapshot via REAL ``MLPipeline.get_summary`` (G35, #101).

        DP-2 (live auto-refresh) is deferred; this returns a single synchronous
        snapshot of the store summary now.
        """
        store_path = self._resolve_store_path()
        try:
            from ...pipeline.orchestrator import MLPipeline

            pipeline = MLPipeline(
                output_dir=str(
                    store_path.parent if store_path.exists() else Path(self.settings.local.output_dir)
                )
            )
            summary = pipeline.get_summary()
        except Exception as e:
            return ChatResponse(message=f"❌ Watch snapshot failed: {e}", action="error")
        running = self.context.get("running_jobs", [])
        header = f"👁️ **Run Watch** ({len(running)} tracked job(s))\n\n"
        return ChatResponse(
            message=header + summary,
            action="watch_shown",
            data={"summary": summary, "running_jobs": running},
        )

    async def _cmd_upload(self, args: list[str]) -> ChatResponse:
        """Upload a local file to S3 via boto3 (G36 — creds-gated slash command).

        Safety default: cloud side-effect exposed as an explicit command only,
        never auto-invoked by NL/LLM. Phase E2 FIX-2: mode-gated FIRST — in
        Offline mode this returns an honest gate and makes NO boto3 call.
        """
        gate = self._aws_gate("/upload")
        if gate is not None:
            return gate
        local_path, s3_key = args[0], args[1]
        if not Path(local_path).exists():
            return ChatResponse(message=f"❌ Local file not found: `{local_path}`", action="error")
        bucket = env_with_compat("AMLAS_S3_BUCKET")
        if not bucket:
            return ChatResponse(
                message=(
                    "☁️ Upload requires a bucket. Set AGENTICML_S3_BUCKET and AWS "
                    "credentials (AWS_PROFILE/.env) to enable."
                ),
                action="error",
            )
        try:
            import boto3

            client = boto3.client("s3", region_name=resolve_aws_region())
            client.upload_file(str(local_path), bucket, s3_key)
        except Exception as e:
            return ChatResponse(message=f"❌ S3 upload failed: {e}", action="error")
        return ChatResponse(
            message=(f"☁️ **Upload complete**\n\n  From: `{local_path}`\n  To: `s3://{bucket}/{s3_key}`"),
            action="upload_complete",
            data={"bucket": bucket, "key": s3_key, "local": str(local_path)},
        )

    async def _cmd_download(self, args: list[str]) -> ChatResponse:
        """Download an S3 object to a local path via boto3 (G36 — creds-gated).

        Phase E2 FIX-2: mode-gated FIRST — Offline returns an honest gate and
        makes NO boto3 call.
        """
        gate = self._aws_gate("/download")
        if gate is not None:
            return gate
        s3_key, local_path = args[0], args[1]
        bucket = env_with_compat("AMLAS_S3_BUCKET")
        if not bucket:
            return ChatResponse(
                message=(
                    "☁️ Download requires a bucket. Set AGENTICML_S3_BUCKET and AWS "
                    "credentials (AWS_PROFILE/.env) to enable."
                ),
                action="error",
            )
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import boto3

            client = boto3.client("s3", region_name=resolve_aws_region())
            client.download_file(bucket, s3_key, str(dest))
        except Exception as e:
            return ChatResponse(message=f"❌ S3 download failed: {e}", action="error")
        return ChatResponse(
            message=(f"☁️ **Download complete**\n\n  From: `s3://{bucket}/{s3_key}`\n  To: `{dest}`"),
            action="download_complete",
            data={"bucket": bucket, "key": s3_key, "local": str(dest)},
        )

    async def _cmd_health(self, args: list[str]) -> ChatResponse:
        """Check AWS connectivity readiness via REAL ``check_aws_health`` (G37)."""
        try:
            from ...config.health import check_aws_health

            status = check_aws_health(self.settings)
        except Exception as e:
            # Sensitive-data handling: surface only the exception TYPE in the
            # user-facing message (a raw ``{e}`` can leak file paths / internal state); the full
            # detail is logged to the internal logger, not the chat pane.
            logger.warning("health check failed", exc_info=True)
            return ChatResponse(
                message=f"❌ Health check failed ({type(e).__name__}). See logs for detail.",
                action="error",
            )
        connected = getattr(status, "aws_connected", False)
        icon = "💚" if connected else "🔴"
        try:
            line = status.summary_line
        except Exception:
            line = "connected" if connected else "not connected"
        return ChatResponse(
            message=(
                f"{icon} **AWS Health**\n\n"
                f"  Connected: {connected}\n"
                f"  {line}\n"
                f"  Bedrock: {getattr(status, 'bedrock_available', False)} | "
                f"S3: {getattr(status, 's3_available', False)}"
            ),
            action="health_shown",
            data={"connected": connected},
        )

    async def _cmd_plan(self, args: list[str]) -> ChatResponse:
        """Preview an experiment plan via REAL ``create_experiment_plan`` (#20)."""
        spec_path = args[0]
        if not Path(spec_path).exists():
            return ChatResponse(message=f"❌ Spec file not found: `{spec_path}`", action="error")
        spec_dict = self._load_spec_dict(spec_path)
        spec_dict.setdefault("spec_id", spec_dict.get("project_id", "experiment"))
        if len(args) >= 2:
            spec_dict.setdefault("data_path", str(args[1]))
        store_path = self._resolve_store_path()
        try:
            from ...tools.experiment_tools import create_experiment_plan
            from ...tools.research_tools import recommend_approaches

            try:
                recs = recommend_approaches(spec_json=json.dumps(spec_dict))
            except Exception:
                recs = {"recommended_algorithms": []}
            result = create_experiment_plan(
                spec_json=json.dumps(spec_dict),
                recommendations_json=json.dumps(recs),
                store_path=str(store_path),
            )
        except Exception as e:
            return ChatResponse(message=f"❌ Plan preview failed: {e}", action="error")
        if isinstance(result, dict) and not result.get("success", True):
            return ChatResponse(message=f"❌ Plan preview failed: {result.get('error')}", action="error")
        n = result.get("n_experiments", result.get("n_configs", 0)) if isinstance(result, dict) else 0
        algos = result.get("algorithms", []) if isinstance(result, dict) else []
        return ChatResponse(
            message=(
                f"🗺️ **Experiment plan preview**\n\n"
                f"  Plan ID: `{result.get('plan_id', '?')}`\n"
                f"  Configs: {n}\n"
                f"  Algorithms: {', '.join(map(str, algos)) or '(none)'}"
            ),
            action="plan_shown",
            data={"result": result},
        )

    # ─── Phase 11 — agentic Data-Engineering / MLOps slash-commands (§10) ─────

    async def _cmd_dqscan(self, args: list[str]) -> ChatResponse:
        """`/dqscan PATH` — run the 25-pattern DQ catalog scan (honors /steer)."""
        path = _normalize_fs_path(args[0])  # : expand ~, strip quotes
        if not Path(path).exists():
            return ChatResponse(message=f"❌ File not found: `{path}`", action="error")
        result = await asyncio.to_thread(
            scan_dq_patterns._tool_func, data_path=path, profile=self._steering_profile_dict()
        )
        if result.get("success") is False:
            return ChatResponse(message=f"❌ DQ scan failed: {result.get('error')}", action="error")
        detected = result.get("patterns_detected", [])
        id_set = sorted({f.get("pattern_id", "") for f in detected})
        ids = ", ".join(id_set) or "none"
        # Build the base summary
        lines = [
            f"🩺 **DQ Pattern Scan** for `{path}`\n",
            f"  Overall score: **{result.get('overall_score', 0):.1f}/100**",
            f"  Patterns detected: {len(detected)} findings across "  # nosemgrep: string-concat-in-list
            f"{len(id_set)} pattern types: {ids}",
        ]
        # Expand content-safety findings (SAFE-*) with actionable detail
        safety = [f for f in detected if f.get("pattern_id", "").startswith("SAFE")]
        if safety:
            lines.append("")
            lines.append("  ⚠️ **Content Safety Issues** (cloud-only, via Amazon Comprehend):")
            for f in safety:
                pid = f.get("pattern_id", "")
                evidence = f.get("evidence", "")
                if pid == "SAFE-TOX":
                    lines.append(f"  🔴 **Toxic content:** {evidence}")
                    lines.append(
                        "     → Recommendation: review and remove/re-label toxic rows before training."
                    )
                elif pid == "SAFE-PII":
                    lines.append(f"  🟠 **PII detected:** {evidence}")
                    lines.append(
                        "     → Recommendation: redact PII (Amazon Comprehend can auto-redact) before training."
                    )
            lines.append("")
            lines.append("  ⛔ This data is NOT safe to train on as-is. Address safety findings first.")
        return ChatResponse(
            message="\n".join(lines),
            action="dqscan_complete",
            data=result,
        )

    async def _cmd_diagnose(self, args: list[str]) -> ChatResponse:
        """`/diagnose PATH` — diagnose MCAR/MAR/MNAR per column (honors /steer)."""
        path = _normalize_fs_path(args[0])
        if not Path(path).exists():
            return ChatResponse(message=f"❌ File not found: `{path}`", action="error")
        result = await asyncio.to_thread(
            diagnose_missingness._tool_func, data_path=path, profile=self._steering_profile_dict()
        )
        if result.get("success") is False:
            return ChatResponse(message=f"❌ Diagnosis failed: {result.get('error')}", action="error")
        per_col = result.get("per_column", {})
        lines = [f"🧪 **Missingness Diagnosis** for `{path}`", ""]
        lines.append(f"  Overall: {result.get('overall_mechanism', 'unknown')}")
        for col, info in list(per_col.items())[:10]:
            lines.append(f"  • {col}: {info.get('mechanism', '?')}")
        return ChatResponse(message="\n".join(lines), action="diagnose_complete", data=result)

    async def _cmd_impute(self, args: list[str]) -> ChatResponse:
        """`/impute PATH [--method auto] [--llm] [--evaluate]` — steered imputation.

        Recommends a per-column plan (``recommend_imputation_plan``) and executes
        it through ``apply_imputation_plan``, which enforces the session /steer
        dial's HARD constraints deterministically.
        """
        path = _normalize_fs_path(args[0])
        if not Path(path).exists():
            return ChatResponse(message=f"❌ File not found: `{path}`", action="error")
        flags = {a for a in args[1:] if a.startswith("--")}
        method = "auto"
        for i, a in enumerate(args[1:]):
            if a == "--method" and i + 2 <= len(args[1:]):
                method = args[1:][i + 1]
        profile = self._steering_profile_dict()

        def _execute() -> dict[str, Any]:
            if method == "auto":
                rec = recommend_imputation_plan._tool_func(data_path=path, profile=profile)
                if rec.get("success") is False:
                    return rec
                plan = {
                    c: {k: v for k, v in spec.items() if k in ("method", "params")}
                    for c, spec in rec.get("plan", {}).items()
                }
            else:
                import pandas as pd

                df = pd.read_csv(path)
                plan = {c: {"method": method} for c in df.columns if df[c].isnull().any()}
            if not plan:
                return {"imputed_path": path, "plan_applied": {}, "steering_violations": []}
            out = apply_imputation_plan._tool_func(data_path=path, plan=plan, profile=profile)
            if "--evaluate" in flags and out.get("imputed_path"):
                out["evaluation"] = "requested"
            return out

        result = await asyncio.to_thread(_execute)
        if result.get("success") is False:
            return ChatResponse(message=f"❌ Imputation failed: {result.get('error')}", action="error")
        applied = result.get("plan_applied", {})
        lines = [f"🩹 **Imputation complete** → `{result.get('imputed_path', '?')}`", ""]
        for col, info in applied.items():
            lines.append(f"  • {col}: {info.get('method')} (filled {info.get('n_filled', 0)})")
        if profile:
            lines.append(f"\n  Steering honored (forbid={profile.get('forbid_methods') or '∅'}).")
        return ChatResponse(message="\n".join(lines), action="impute_complete", data=result)

    async def _cmd_synth(self, args: list[str]) -> ChatResponse:
        """`/synth PATH N [--balance --target Y] [--validate]` — synth / augment."""
        path = _normalize_fs_path(args[0])
        if not Path(path).exists():
            return ChatResponse(message=f"❌ File not found: `{path}`", action="error")
        rest = args[1:]
        rows = 1000
        for a in rest:
            if a.isdigit():
                rows = int(a)
        flags = {a for a in rest if a.startswith("--")}
        target = None
        for i, a in enumerate(rest):
            if a == "--target" and i + 1 < len(rest):
                target = rest[i + 1]
        profile = self._steering_profile_dict()

        def _execute() -> dict[str, Any]:
            if "--balance" in flags:
                if not target:
                    return {"success": False, "error": "--balance requires --target"}
                res = augment_imbalance._tool_func(data_path=path, target=target, strategy="smote")
                synth_path = res.get("augmented_path")
            else:
                res = synthesize_tabular._tool_func(data_path=path, rows=rows, generator="gaussian_copula")
                synth_path = res.get("synthetic_path")
            if res.get("success") is False:
                return res
            if "--validate" in flags and synth_path:
                thresholds = (profile or {}).get("thresholds") or None
                res["validation"] = validate_synthetic._tool_func(
                    real_path=path, synthetic_path=synth_path, target=target, thresholds=thresholds
                )
            return res

        result = await asyncio.to_thread(_execute)
        if result.get("success") is False:
            return ChatResponse(message=f"❌ Synthesis failed: {result.get('error')}", action="error")
        if "--balance" in flags:
            msg = (
                f"🧬 **Augmented** → `{result.get('augmented_path')}`\n\n"
                f"  after: {result.get('class_counts_after')}"
            )
        else:
            msg = (
                f"🧬 **Synthetic data** → `{result.get('synthetic_path')}`\n\n"
                f"  {result.get('rows', '?')} rows via {result.get('generator', '?')}"
            )
        if "validation" in result:
            msg += f"\n  validation accepted: {result['validation'].get('accepted')}"
        return ChatResponse(message=msg, action="synth_complete", data=result)

    async def _cmd_synth_from(self, args: list[str]) -> ChatResponse:
        """`/synth-from "requirement"` — build a SynthesisSpec from NL (C3).

        WIRING FIX (06-tui-e2e-hardening, bug #2): when an active dataset is
        loaded (``/data <path>``), thread its path as ``seed_path`` so the tool's
        DETERMINISTIC schema-inference path produces a real spec OFFLINE — the
        "graceful fallback spec otherwise" the manual doc (Step 13) promises.
        Previously this passed ONLY the requirement, so with no injected LLM the
        tool hard-errored ("cannot build a spec from a bare requirement without an
        LLM or a reference dataset") on every offline turn, even with a dataset
        already loaded in the session — a real dead-end in the chat flow.
        """
        requirement = " ".join(args)
        summary = self.context.get("data_summary") or {}
        seed_path = summary.get("path") if isinstance(summary, dict) else None
        kwargs: dict[str, Any] = {"requirement": requirement}
        if seed_path and Path(str(seed_path)).exists():
            kwargs["seed_path"] = str(seed_path)
        result = await asyncio.to_thread(build_synthesis_spec._tool_func, **kwargs)
        if result.get("success") is False:
            err = str(result.get("error") or "")
            # Phase E2 FIX-3: the "bare requirement without an LLM or a reference
            # dataset" case is a legitimate offline dead-end — but the raw tool
            # copy doesn't tell the user how to proceed. Rewrite it to be
            # actionable (load a dataset for the deterministic schema-inference
            # path, or switch mode to use the LLM). Other errors pass through.
            if "without an llm" in err.lower() or "reference dataset" in err.lower():
                return ChatResponse(
                    message=(
                        "❌ I can't build a synthesis spec from a bare requirement offline "
                        "without a reference dataset.\n"
                        "  • Load a dataset first with `/data <path>` so I can infer a schema "
                        "OFFLINE (the deterministic path), then re-run `/synth-from`.\n"
                        "  • Or switch to **Local** / **Full-AWS** mode to use the LLM spec writer."
                    ),
                    action="error",
                    # Sensitive-data handling: attach ONLY non-sensitive metadata
                    # (status + error class), never the raw result dict which may carry
                    # user file paths / dataset contents / tool error detail.
                    data={"status": result.get("status"), "error": result.get("error")}
                    if isinstance(result, dict)
                    else None,
                )
            return ChatResponse(message=f"❌ Spec build failed: {err}", action="error")
        spec = result.get("synthesis_spec", {})
        return ChatResponse(
            message=(
                f'📐 **SynthesisSpec** from: *"{requirement}"*\n\n'
                f"  generator: {spec.get('generator', '?')}\n"
                f"  volume: {spec.get('volume', '?')}\n"
                f"  conditions: {spec.get('conditions', {})}"
            ),
            action="synth_from_complete",
            data=result,
        )

    async def _cmd_steer(self, args: list[str]) -> ChatResponse:
        """`/steer set ...` | `/steer show` — manage the session steering dial."""
        sub = args[0].lower()
        if sub == "show":
            if self.steering_profile is None:
                return ChatResponse(
                    message="🎛️ No steering profile set for this session.", action="steer_show"
                )
            p = self.steering_profile
            return ChatResponse(
                message=(
                    "🎛️ **Active steering**\n\n"
                    f"  forbid: {p.forbid_methods or '∅'}\n"
                    f"  pins: {p.pin_methods or '∅'}\n"
                    f"  mechanism_hints: {p.mechanism_hints or '∅'}\n"
                    f"  thresholds: {p.thresholds or '∅'}\n"
                    f"  notes: {p.domain_notes or '∅'}"
                ),
                action="steer_show",
                data=p.as_profile_dict(),
            )
        if sub == "set":
            from ...data_eng.steering import SteeringProfile

            kwargs = _parse_steer_tokens(args[1:])
            self.steering_profile = SteeringProfile(**kwargs)
            return ChatResponse(
                message=(
                    "🎛️ **Steering saved** for this session — honored by "
                    "`/impute`, `/synth`, `/dqscan`.\n\n"
                    f"  forbid: {self.steering_profile.forbid_methods or '∅'}, "
                    f"pins: {self.steering_profile.pin_methods or '∅'}"
                ),
                action="steer_set",
                data=self.steering_profile.as_profile_dict(),
            )
        return ChatResponse(message="Usage: /steer set ... | /steer show", action="error")

    async def _cmd_lake(self, args: list[str]) -> ChatResponse:
        """`/lake read URI` | `/lake write LOCAL DEST` — medallion read/write.

        Phase E2 FIX-2: mode-gated FIRST — Offline returns an honest gate and
        makes NO Glue/S3 call. This replaces the previously-leaked raw
        ``EntityNotFoundException`` from ``/lake read`` in Offline mode (audit
        §3.3 B2) with the honest "requires AWS — switch mode" copy.
        """
        gate = self._aws_gate("/lake")
        if gate is not None:
            return gate
        action = args[0].lower()
        profile = self._steering_profile_dict()
        if action == "read":
            if len(args) < 2:
                return ChatResponse(message="Usage: /lake read <uri>", action="error")
            result = await asyncio.to_thread(read_lake_slice._tool_func, args[1])
            if result.get("success") is False:
                return ChatResponse(message=f"❌ Lake read failed: {result.get('error')}", action="error")
            return ChatResponse(
                message=(
                    f"🏞️ **Lake read** → `{result.get('local_path')}`\n\n"
                    f"  {result.get('rows', 0)} rows, tier={result.get('tier', '?')}"
                ),
                action="lake_complete",
                data=result,
            )
        if action == "write":
            if len(args) < 3:
                return ChatResponse(message="Usage: /lake write <local> <dest_uri>", action="error")
            result = await asyncio.to_thread(
                write_lake_dataset._tool_func, local_path=args[1], dest_uri=args[2], profile=profile
            )
            if result.get("success") is False:
                return ChatResponse(message=f"❌ Lake write failed: {result.get('error')}", action="error")
            return ChatResponse(
                message=(
                    f"🏞️ **Lake write** → `{result.get('written_uri')}`\n\n"
                    f"  tier={result.get('tier')}, lineage={result.get('lineage_id')}"
                ),
                action="lake_complete",
                data=result,
            )
        return ChatResponse(message="Usage: /lake read|write ...", action="error")

    async def _cmd_mlops(self, args: list[str]) -> ChatResponse:
        """`/mlops status|pipeline|register|monitor` — MLOps actions (D1-D7)."""
        action = args[0].lower()
        if action == "status":
            scorecard = await asyncio.to_thread(compute_mlops_kpis, {})
            overall = scorecard.get("overall", {})
            return ChatResponse(
                message=(
                    "🛰️ **MLOps status**\n\n"
                    f"  maturity: {overall.get('maturity_level', 'L0')}\n"
                    f"  composite health: {overall.get('composite_health', 0.0):.2f}"
                ),
                action="mlops_complete",
                data=scorecard,
            )
        # pipeline / register / monitor require AWS — surface guidance for the CLI.
        return ChatResponse(
            message=(
                f"🛰️ **MLOps {action}** is an AWS action — run it from the CLI "
                f"(`agentcli mlops {action} ...`) or the MLOps screen. "
                "Use `/mlops status` for the KPI scorecard here."
            ),
            action="mlops_complete",
            data={"action": action},
        )

    async def _cmd_attach(self, args: list[str]) -> ChatResponse:
        """Attach a file to chat context."""
        file_path = args[0]
        return self.attach_file(file_path)

    async def _cmd_data(self, args: list[str]) -> ChatResponse:
        """List available datasets or load a dataset by path.

        With no args: scans data/ directory for CSV/parquet files, lists them.
        With path arg: validates path exists, loads basic info (shape, columns),
        sets as active dataset.
        """
        if not args:
            # No args — scan for available datasets.

            # PERF FIX (F1c): the old scan used ``Path('.').rglob('*.csv')`` which
            # descends into EVERY subdir of the cwd — including ``.venv`` /
            # ``site-packages`` (514 vendored sample CSVs on this tree). That took
            # ~12.5 s warm / ~40 s cold, blocking the chat worker long enough that
            # the headless smoke harness (and a real user) saw NO response, AND it
            # polluted the listing with useless ``.venv/.../*.csv`` paths. We now
            # prune hidden + vendored directories during the walk so the scan is
            # near-instant and the results are the user's ACTUAL datasets. Run it
            # off the event loop via ``asyncio.to_thread`` so even a large real
            # project tree never stalls the UI.
            data_dirs = [Path("data"), Path("repos"), Path(".")]
            found_files = await asyncio.to_thread(_scan_data_files, data_dirs)

            if found_files:
                # Sort and deduplicate
                found_files = sorted(set(found_files))
                lines = [f"📊 **Available Datasets** ({len(found_files)} found):\n"]
                for f in found_files[:15]:
                    lines.append(f"  • `{f}`")
                if len(found_files) > 15:
                    lines.append(f"  ... and {len(found_files) - 15} more")
                lines.append("")
                lines.append("💡 Use `/data <path>` to load a specific dataset.")
                return ChatResponse(
                    message="\n".join(lines),
                    action="data_listed",
                    data={"files": found_files[:15]},
                    suggestions=[f"/data {found_files[0]}"] if found_files else [],
                )
            else:
                return ChatResponse(
                    message=(
                        "📊 **No datasets found**\n\n"
                        "  No CSV or Parquet files found in data/, repos/, or current directory.\n"
                        "  Use `/attach <path>` to load a file by explicit path."
                    ),
                    action="data_listed",
                )
        else:
            # Path provided — load dataset info
            data_path = _normalize_fs_path(args[0])  # : expand ~, strip quotes
            p = Path(data_path)

            if not p.exists():
                return ChatResponse(
                    message=f"❌ File not found: `{data_path}`",
                    action="error",
                )

            suffix = p.suffix.lower()
            if suffix not in {".csv", ".parquet", ".json"}:
                # Not a data file — route to attach
                return self.attach_file(data_path)

            # Try to load basic info.
            try:
                import pandas as pd

                # G21: route CSVs through the REAL load_dataset tool so the TUI
                # and agent system share one loader. Non-CSV formats keep the
                # pandas dispatch (load_dataset is CSV-only).
                columns: list[str]
                dtypes: dict[str, str]
                if suffix == ".csv":
                    from ...tools.data_tools import load_dataset

                    loaded = load_dataset(path=str(p))
                    if not (isinstance(loaded, dict) and loaded.get("success")):
                        # Preserve the prior graceful contract: the file is still
                        # attached for reference and we surface ``data_loaded``
                        # with an error note (parity checks cover the happy path).
                        err = (
                            loaded.get("error", "unknown error")
                            if isinstance(loaded, dict)
                            else "load failed"
                        )
                        self.context["attachments"].append(str(p.resolve()))
                        return ChatResponse(
                            message=(
                                f"📊 **Dataset attached:** `{p.name}`\n\n"
                                f"  ⚠️ Could not parse: {err}\n"
                                f"  File added to context for reference."
                            ),
                            action="data_loaded",
                            data={"path": str(p), "error": err},
                        )
                    n_rows = int(loaded.get("n_rows", 0))
                    n_cols = int(loaded.get("n_cols", 0))
                    columns = list(loaded.get("columns", []))
                    dtypes = dict(loaded.get("dtypes", {}))
                else:
                    if suffix == ".parquet":
                        df = pd.read_parquet(p)
                    else:
                        df = pd.read_json(p)
                    n_rows, n_cols = df.shape
                    columns = list(df.columns)
                    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

                # Store summary in context
                summary = {
                    "path": str(p),
                    "rows": n_rows,
                    "cols": n_cols,
                    "columns": columns,
                    "dtypes": dtypes,
                }
                self.context["data_summary"] = summary
                self.context["attachments"].append(str(p.resolve()))
                # Phase 2: mirror the active dataset onto the shared bus (design §E.3).
                self._sync_bus(
                    "set_dataset",
                    path=str(p),
                    rows=n_rows,
                    cols=n_cols,
                    columns=columns,
                    dtypes=dtypes,
                    source="chat",
                )

                # Format column display
                col_display = ", ".join(columns[:10])
                if len(columns) > 10:
                    col_display += f", ... (+{len(columns) - 10} more)"

                return ChatResponse(
                    message=(
                        f"📊 **Dataset loaded:** `{p.name}`\n\n"
                        f"  Shape: {n_rows:,} rows × {n_cols} columns\n"
                        f"  Columns: {col_display}\n\n"
                        f"💡 Suggestions:\n"
                        f"  • `/quality {data_path}` — Run data quality check\n"
                        f'  • `/spec "describe your task"` — Generate a spec\n'
                        f"  • `/run <spec> {data_path}` — Start a pipeline"
                    ),
                    action="data_loaded",
                    data=summary,
                    suggestions=[
                        f"/quality {data_path}",
                        '/spec "predict <target> from this data"',
                    ],
                )

            except ImportError:
                # pandas not available — provide basic file info
                file_size = p.stat().st_size
                self.context["attachments"].append(str(p.resolve()))
                return ChatResponse(
                    message=(
                        f"📊 **Dataset attached:** `{p.name}`\n\n"
                        f"  Size: {file_size:,} bytes\n"
                        f"  (Install pandas for full dataset analysis)\n\n"
                        f"💡 Use `/quality {data_path}` to run quality checks."
                    ),
                    action="data_loaded",
                    data={"path": str(p), "size": file_size},
                )

            except Exception as e:
                return ChatResponse(
                    message=(
                        f"📊 **Dataset attached:** `{p.name}`\n\n"
                        f"  ⚠️ Could not parse: {e}\n"
                        f"  File added to context for reference."
                    ),
                    action="data_loaded",
                    data={"path": str(p), "error": str(e)},
                )

    # ─── File Attachment ──────────────────────────────────────────────────────

    def attach_file(self, path: str) -> ChatResponse:
        """Attach a file to the chat context.

        Supported: .csv (data), .yaml/.yml (spec), .md (documentation), .txt
        For CSV: auto-runs describe_dataset and adds summary to context.
        For YAML: auto-validates as potential spec.
        """
        path = _normalize_fs_path(path)  # : expand ~, strip surrounding quotes
        p = Path(path)
        if not p.exists():
            return ChatResponse(
                message=f"❌ File not found: `{path}`",
                action="error",
            )

        suffix = p.suffix.lower()
        supported = {".csv", ".yaml", ".yml", ".md", ".txt", ".json"}
        if suffix not in supported:
            return ChatResponse(
                message=(f"⚠️ Unsupported file type: `{suffix}`\nSupported: {', '.join(sorted(supported))}"),
                action="error",
            )

        # Add to attachments
        self.context["attachments"].append(str(p.resolve()))

        # Process based on type
        if suffix == ".csv":
            return self._attach_csv(p)
        elif suffix in {".yaml", ".yml"}:
            return self._attach_yaml(p)
        else:
            file_size = p.stat().st_size
            return ChatResponse(
                message=(
                    f"📎 **Attached:** `{p.name}`\n"
                    f"  Type: {suffix} | Size: {file_size:,} bytes\n"
                    f"  File added to context for this session."
                ),
                action="file_attached",
                data={"path": str(p), "type": suffix, "size": file_size},
            )

    def _attach_csv(self, path: Path) -> ChatResponse:
        """Process and attach a CSV file with auto-description."""
        try:
            import pandas as pd

            df = pd.read_csv(path)
            n_rows, n_cols = df.shape
            columns = list(df.columns)

            # Store summary in context
            summary = {
                "path": str(path),
                "rows": n_rows,
                "cols": n_cols,
                "columns": columns,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            self.context["data_summary"] = summary
            # Phase 2: mirror the active dataset onto the shared bus (design §E.3).
            self._sync_bus(
                "set_dataset",
                path=str(path),
                rows=n_rows,
                cols=n_cols,
                columns=columns,
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                source="chat",
            )

            # Format column list (truncate if too many)
            col_display = ", ".join(columns[:10])
            if len(columns) > 10:
                col_display += f", ... (+{len(columns) - 10} more)"

            return ChatResponse(
                message=(
                    f"📎 **Attached:** `{path.name}` ({n_rows:,} rows × {n_cols} columns)\n\n"
                    f"  Columns: {col_display}\n\n"
                    f"💡 Suggestions:\n"
                    f"  • `/quality {path}` — Run data quality check\n"
                    f'  • `/spec "describe your task"` — Generate a spec\n'
                    f"  • `/research <topic>` — Find suitable algorithms"
                ),
                action="file_attached",
                data=summary,
                suggestions=[
                    f"/quality {path}",
                    '/spec "predict <target> from this data"',
                ],
            )

        except Exception as e:
            return ChatResponse(
                message=f"📎 Attached `{path.name}` (CSV parsing failed: {e})",
                action="file_attached",
                data={"path": str(path), "error": str(e)},
            )

    def _attach_yaml(self, path: Path) -> ChatResponse:
        """Process and attach a YAML file (potential spec)."""
        try:
            content = path.read_text(encoding="utf-8")
            line_count = len(content.splitlines())

            # Check if it looks like a spec (has frontmatter delimiters)
            is_spec = content.strip().startswith("---")

            msg = f"📎 **Attached:** `{path.name}` ({line_count} lines)\n"
            if is_spec:
                self.context["current_spec"] = str(path)
                # Phase 2: mirror the active spec onto the shared bus (design §E.3).
                self._sync_bus("set_spec", path=str(path), source="chat")
                msg += (
                    f"  Detected as ML spec file.\n"
                    f"  Set as active spec for this session.\n\n"
                    f"💡 Use `/run {path} <data-path>` to start a pipeline."
                )
            else:
                msg += "  YAML configuration file added to context."

            return ChatResponse(
                message=msg,
                action="file_attached",
                data={"path": str(path), "is_spec": is_spec, "lines": line_count},
            )

        except Exception as e:
            return ChatResponse(
                message=f"📎 Attached `{path.name}` (parse note: {e})",
                action="file_attached",
                data={"path": str(path)},
            )

    # ─── Intent Execution ─────────────────────────────────────────────────────

    async def _execute_intent(self, intent: str, params: dict[str, Any], original_text: str) -> ChatResponse:
        """Execute a detected intent directly — no user action required.

        This is the core behavioral change: instead of telling users "use /attach",
        the system classifies their intent and EXECUTES the appropriate command.
        """
        if intent == "load_data":
            path = params.get("path")
            if path:
                return await self._cmd_data([path])
            else:
                return await self._cmd_data([])  # list available datasets

        elif intent == "run_pipeline":
            # Try to infer spec + data from context
            path = params.get("path") or params.get("data")
            spec = params.get("spec")
            # Use active dataset if available
            if not path and self.context.get("data_summary"):
                path = self.context["data_summary"].get("path")
            if not spec and self.context.get("current_spec"):
                spec = self.context["current_spec"]
            if spec and path:
                return await self._cmd_run([spec, path])
            return ChatResponse(
                message=(
                    "🚀 I'm ready to run a pipeline! I need:\n\n"
                    f"  • Spec: {spec or '❌ not set — use `/spec` to create one'}\n"
                    f"  • Data: {path or '❌ not set — use `/data <path>` to load'}\n\n"
                    "Once both are set, I'll start automatically."
                ),
                action="intent_needs_params",
                suggestions=['/spec "describe your task"', "/data"],
            )

        elif intent == "check_quality":
            path = params.get("path")
            if not path and self.context.get("data_summary"):
                path = self.context["data_summary"].get("path")
            if path:
                return await self._cmd_quality([path])
            return await self._cmd_data([])

        elif intent == "show_results":
            return await self._cmd_leaderboard([])

        elif intent == "create_spec":
            description = params.get("description", original_text)
            return await self._cmd_spec([description])

        elif intent == "show_status":
            return await self._cmd_status([])

        elif intent == "research":
            query = params.get("query", original_text)
            return await self._cmd_research([query])

        # Fallback — intent recognized but couldn't execute
        return ChatResponse(
            message=(
                f"I understood your intent (`{intent}`) but need more details.\n"
                "Try being more specific or use `/help` to see available commands."
            ),
            action="intent_incomplete",
            suggestions=["/help"],
        )

    # ─── Natural Language Processing ─────────────────────────────────────────

    async def _handle_natural_language(self, message: str) -> ChatResponse:
        """Handle natural language messages using pattern matching.

        Without an LLM, uses keyword detection to route to appropriate responses.
        With Bedrock configured, would use the pipeline agent for full conversation.
        """
        lower = message.lower().strip()

        # Pattern: asking about status / progress
        if any(kw in lower for kw in ["status", "progress", "how is", "running"]):
            return await self._cmd_status([])

        # Pattern: asking about results / leaderboard
        if any(kw in lower for kw in ["leaderboard", "results", "best model", "scores", "ranking"]):
            return await self._cmd_leaderboard([])

        # Pattern: asking for help
        if any(kw in lower for kw in ["help", "commands", "what can you do", "how do i"]):
            return await self._cmd_help([])

        # Pattern: spec generation request
        if any(kw in lower for kw in ["spec", "define", "create a", "i need to predict", "i want to"]):
            return ChatResponse(
                message=(
                    "I can help you define an ML spec! Tell me more about:\n\n"
                    "  1. What are you trying to predict? (target variable)\n"
                    "  2. What type of task? (classification, regression)\n"
                    "  3. What metric matters most? (f1, accuracy, rmse)\n"
                    "  4. What threshold for success?\n\n"
                    'Or use: `/spec "<description>"` for quick generation.\n\n'
                    "💡 Tip: `/attach <data-file>` first so I can analyze your columns."
                ),
                action="spec_guidance",
                suggestions=[
                    '/spec "predict wheel failures from sensor data"',
                    "/attach data/train.csv",
                ],
            )

        # Pattern: asking about data
        if any(kw in lower for kw in ["data", "dataset", "csv", "columns"]):
            if self.context.get("data_summary"):
                summary = self.context["data_summary"]
                return ChatResponse(
                    message=(
                        f"📊 Current dataset: `{summary.get('path', 'unknown')}`\n"
                        f"  {summary.get('rows', 0):,} rows × {summary.get('cols', 0)} columns\n\n"
                        f"  Use `/quality <path>` for a full quality report."
                    ),
                    action="data_info",
                )
            # Set awaiting state so next bare path input auto-loads as data
            self._awaiting = "data_path"
            return ChatResponse(
                message=(
                    "No dataset attached yet. Paste a file path or use `/attach <path>` to load one.\n\n"
                    "Example: `repos/adsa/train.csv`"
                ),
                action="data_guidance",
                suggestions=["/attach data/train.csv"],
            )

        # Pattern: greeting
        if any(kw in lower for kw in ["hello", "hi", "hey", "good morning", "good evening"]):
            return ChatResponse(
                message=(
                    "👋 Hello! I'm your ML engineering assistant.\n\n"
                    "I can help you:\n"
                    "  • Define ML specs from descriptions\n"
                    "  • Run automated ML pipelines\n"
                    "  • Research algorithms and approaches\n"
                    "  • Check data quality\n"
                    "  • View results and leaderboards\n\n"
                    "Type `/help` to see all commands, or just describe what you need!"
                ),
                action="greeting",
            )

        # Default: generic response with suggestions
        return ChatResponse(
            message=(
                f'I understand you\'re asking about: *"{message}"*\n\n'
                "I can help with ML engineering tasks. Try:\n"
                "  • `/help` — see all available commands\n"
                '  • `/spec "description"` — generate an ML spec\n'
                "  • `/attach file.csv` — load data for analysis\n"
                "  • `/research topic` — find relevant algorithms\n\n"
                "Or describe your ML task in more detail!"
            ),
            action="general_response",
            suggestions=["/help"],
        )

    # ─── Spec Generation Helpers ─────────────────────────────────────────────

    def _generate_spec_template(self, description: str, task_type: str) -> str:
        """Generate a YAML spec template from a description."""
        # Extract potential target from description
        target = "target"
        words = description.lower().split()
        if "predict" in words:
            idx = words.index("predict")
            if idx + 1 < len(words):
                target = words[idx + 1]

        # Choose metric based on task type
        metric = "f1_macro" if task_type == "classification" else "rmse"
        threshold = "0.80" if task_type == "classification" else "10.0"

        # Create a project ID from description
        project_id = re.sub(r"[^a-z0-9]+", "-", description.lower())[:30].strip("-")

        return (
            f"---\n"
            f"project_id: {project_id}\n"
            f"task_type: {task_type}\n"
            f"target: {target}\n"
            f"primary_metric: {metric}\n"
            f"threshold: {threshold}\n"
            f'description: "{description}"\n'
            f"---\n"
            f"\n"
            f"# {description.title()}\n"
            f"\n"
            f"Auto-generated spec. Edit fields above as needed.\n"
        )


# ─── Phase 11 — /steer token parser (module scope; pure) ──────────────────────


# Directories that never hold a USER dataset and are catastrophically large to
# walk (virtualenvs, VCS, caches, node deps). Pruning these turns the ``/data``
# no-arg scan from ~12.5 s warm / ~40 s cold (514 vendored site-package CSVs on a
# typical tree) into a near-instant listing of the user's ACTUAL data. F1c.
_DATA_SCAN_PRUNE_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "site-packages",
        ".tox",
        ".eggs",
        "build",
        "dist",
        ".idea",
        ".vscode",
        ".agentcore",
        ".f1_diag",
        ".claude",
    }
)
_DATA_SCAN_SUFFIXES: tuple[str, ...] = (".csv", ".parquet")


def _scan_data_files(roots: list[Path]) -> list[str]:
    """Recursively list ``*.csv``/``*.parquet`` under ``roots``, pruning junk dirs.

    Uses ``os.walk`` (NOT ``Path.rglob``) specifically so we can prune
    ``dirnames`` IN PLACE — ``rglob`` offers no way to skip a subtree, so it
    descended into ``.venv``/``site-packages`` and paid for hundreds of vendored
    sample CSVs. We additionally skip any hidden directory (leading ``.``). The
    walk is bounded and deterministic; callers run it via ``asyncio.to_thread``
    so even a huge real-project tree never blocks the chat event loop.

    Returns a flat list of path strings (deduped + sorted by the caller, to
    preserve the existing ``data_listed`` formatting contract).
    """
    import os

    found: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune in place: drop hidden dirs and known-vendored/cache dirs so
            # os.walk never recurses into them.
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _DATA_SCAN_PRUNE_DIRS]
            for fn in filenames:
                if fn.endswith(_DATA_SCAN_SUFFIXES):
                    found.append(str(Path(dirpath) / fn))
    return found


def _parse_steer_tokens(tokens: list[str]) -> dict[str, Any]:
    """Parse ``/steer set`` ``key=value`` tokens into ``SteeringProfile`` kwargs.

    Recognized token forms (master plan §3.1 — the domain-steering "dial"):

        forbid=mean[,knn]      → ``forbid_methods``   (comma-split list, accumulates)
        preferred=median[,knn] → ``preferred_methods``(comma-split list)
        pin=col:method         → ``pin_methods``      ``{col: method}`` (accumulates)
        mechanism=col:MNAR     → ``mechanism_hints``  ``{col: MCAR|MAR|MNAR}``
        threshold=name:0.9     → ``thresholds``       ``{name: float}``
        notes=free_text        → ``domain_notes``     (last token wins)
        maxllm=0.1             → ``max_llm_fraction`` (float, hard LLM cap)
        novelty=3              → ``novelty_budget``   (int)

    HARD constraints (forbid/pin/mechanism/thresholds/maxllm) flow into the tools
    as ``profile=`` and are enforced deterministically; SOFT signals
    (preferred/notes/novelty) flow into prompts. Unknown keys are ignored
    (fail-soft) so a typo never crashes the session dial.

    Args:
        tokens: The argument tokens AFTER the ``set`` sub-command, e.g.
            ``["forbid=mean", "pin=a:median", "notes=km_after_1980"]``.

    Returns:
        A kwargs dict suitable for ``SteeringProfile(**kwargs)``.
    """
    kwargs: dict[str, Any] = {}
    forbid: list[str] = []
    preferred: list[str] = []
    pins: dict[str, str] = {}
    mechs: dict[str, str] = {}
    thresholds: dict[str, float] = {}

    for tok in tokens:
        if "=" not in tok:
            continue
        key, _, value = tok.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key == "forbid":
            forbid.extend(v.strip() for v in value.split(",") if v.strip())
        elif key == "preferred":
            preferred.extend(v.strip() for v in value.split(",") if v.strip())
        elif key == "pin":
            col, _, method = value.partition(":")
            if col.strip() and method.strip():
                pins[col.strip()] = method.strip()
        elif key == "mechanism":
            col, _, mech = value.partition(":")
            if col.strip() and mech.strip():
                mechs[col.strip()] = mech.strip().upper()
        elif key == "threshold":
            name, _, raw = value.partition(":")
            if name.strip() and raw.strip():
                try:
                    thresholds[name.strip()] = float(raw)
                except ValueError:
                    pass
        elif key in ("notes", "note"):
            kwargs["domain_notes"] = value
        elif key in ("maxllm", "max_llm_fraction"):
            try:
                kwargs["max_llm_fraction"] = float(value)
            except ValueError:
                pass
        elif key in ("novelty", "novelty_budget"):
            try:
                kwargs["novelty_budget"] = int(value)
            except ValueError:
                pass

    if forbid:
        kwargs["forbid_methods"] = forbid
    if preferred:
        kwargs["preferred_methods"] = preferred
    if pins:
        kwargs["pin_methods"] = pins
    if mechs:
        kwargs["mechanism_hints"] = mechs
    if thresholds:
        kwargs["thresholds"] = thresholds
    return kwargs
