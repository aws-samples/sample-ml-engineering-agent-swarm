"""Custom slash-command authoring — T-CTL-01 / US-CC.02 (doc 05 §3.1d; ADR-016).

The Claude-Code custom-command analog (the ``.claude/commands/*.md`` mechanism): let a DS
author a reusable, **markdown-defined ML slash command** with YAML frontmatter + an
argument-hint + the flow/tool sequence it triggers, and have it become invocable through the
control plane. Commands live under a project commands dir (``.agenticml/commands/*.md`` by
default) so they are version-controlled project memory (loaded alongside ``.agenticml.yaml``).

A command file looks like::

    ---
    name: rul-baseline
    description: impute -> train xgboost -> grade vs RMSE 12 -> register
    argument-hint: "<dataset_uri>"
    flow: ml_pipeline
    steps:
      - tool: impute
      - tool: submit_training_run
        args: {algo: xgboost}
      - tool: grade_submission
        args: {threshold: 12}
      - tool: register_model
    ---
    # /rul-baseline
    Run the standard RUL baseline pipeline on $ARGUMENTS.

``author_custom_command`` (T-CTL-01) writes such a file; ``load_custom_commands`` reads the
commands dir into a registry; ``invoke_custom_command`` expands an authored command into its
flow/tool sequence (resolving ``$ARGUMENTS``) ready for the control plane to dispatch. Pure
filesystem + YAML — no AWS, no LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "COMMANDS_DIR",
    "CustomCommand",
    "author_custom_command",
    "get_custom_command",
    "invoke_custom_command",
    "load_custom_commands",
]

COMMANDS_DIR = ".agenticml/commands"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class CustomCommand:
    """A parsed custom slash command (the ``.agenticml/commands/<name>.md`` analog).

    ``name`` is the slash-command id (invoked as ``/<name>``); ``description`` + ``argument_hint``
    document it; ``flow`` is the flow it dispatches (or empty for a pure tool sequence); ``steps``
    is the ordered ``[{tool, args}]`` sequence; ``body`` is the markdown prompt template (with
    ``$ARGUMENTS``); ``path`` is the source file.
    """

    name: str
    description: str = ""
    argument_hint: str = ""
    flow: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    body: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description, "argument_hint": self.argument_hint,
            "flow": self.flow, "steps": self.steps, "body": self.body, "path": self.path,
        }


def _commands_dir(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser() / COMMANDS_DIR


def author_custom_command(
    *,
    name: str,
    description: str = "",
    steps: list[dict[str, Any]] | None = None,
    flow: str = "",
    argument_hint: str = "",
    body: str = "",
    project_dir: str | Path = ".",
    overwrite: bool = False,
) -> dict[str, Any]:
    """``T-CTL-01`` — author a reusable markdown slash command (US-CC.02; ADR-016).

    Writes ``<project>/.agenticml/commands/<name>.md`` with YAML frontmatter (name/description/
    argument-hint/flow/steps) + the markdown body, so the command round-trips through
    :func:`load_custom_commands` and becomes invocable. Validates the command name (slug form) and
    refuses to clobber an existing command unless ``overwrite`` (safe-by-default).

    Args:
        name: The slash-command id (lowercase slug, e.g. ``"rul-baseline"``).
        description: One-line human description.
        steps: Ordered ``[{tool, args}]`` the command expands to.
        flow: Optional flow id the command dispatches (else a pure tool sequence).
        argument_hint: The ``$ARGUMENTS`` hint shown to the user.
        body: The markdown prompt template (``$ARGUMENTS`` placeholder supported).
        project_dir: The project root holding the commands dir.
        overwrite: Allow replacing an existing command file.

    Returns:
        ``{"name","path","created": bool,"command": {...}}``.

    Raises:
        ValueError: an invalid command name.
    """
    slug = (name or "").strip().lower()
    if not _NAME_RE.match(slug):
        raise ValueError(f"invalid command name {name!r}; must be a lowercase slug [a-z0-9_-]{{1,64}}")

    cmd_dir = _commands_dir(project_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    target = cmd_dir / f"{slug}.md"
    if target.exists() and not overwrite:
        return {"name": slug, "path": str(target), "created": False,
                "command": load_one(target).to_dict(), "note": "exists (pass overwrite=True to replace)"}

    front: dict[str, Any] = {"name": slug}
    if description:
        front["description"] = description
    if argument_hint:
        front["argument-hint"] = argument_hint
    if flow:
        front["flow"] = flow
    if steps:
        front["steps"] = steps

    md_body = body or f"# /{slug}\n\n{description or 'Run the ' + slug + ' command'} on $ARGUMENTS."
    content = "---\n" + yaml.safe_dump(front, sort_keys=False, default_flow_style=False) + "---\n\n" + md_body
    target.write_text(content, encoding="utf-8")
    return {"name": slug, "path": str(target), "created": True,
            "command": load_one(target).to_dict()}


def load_one(path: str | Path) -> CustomCommand:
    """Parse a single ``<name>.md`` command file into a :class:`CustomCommand`."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    front: dict[str, Any] = {}
    body = text
    if m:
        try:
            front = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            front = {}
        body = m.group(2).strip()
    name = str(front.get("name", p.stem))
    raw_steps = front.get("steps")
    steps: list[Any] = raw_steps if isinstance(raw_steps, list) else []
    return CustomCommand(
        name=name,
        description=str(front.get("description", "")),
        argument_hint=str(front.get("argument-hint", front.get("argument_hint", ""))),
        flow=str(front.get("flow", "")),
        steps=[dict(s) for s in steps],
        body=body,
        path=str(p),
    )


def load_custom_commands(*, project_dir: str | Path = ".") -> dict[str, CustomCommand]:
    """Load all authored commands from ``<project>/.agenticml/commands/*.md`` into a registry.

    Returns ``{name: CustomCommand}`` (empty when the commands dir is absent). The control plane
    calls this at session start so authored commands are invocable (loaded from project memory).
    """
    cmd_dir = _commands_dir(project_dir)
    registry: dict[str, CustomCommand] = {}
    if not cmd_dir.is_dir():
        return registry
    for p in sorted(cmd_dir.glob("*.md")):
        try:
            cmd = load_one(p)
            registry[cmd.name] = cmd
        except (OSError, ValueError):
            continue
    return registry


def get_custom_command(name: str, *, project_dir: str | Path = ".") -> CustomCommand | None:
    """Return the authored :class:`CustomCommand` for ``name`` (or ``None``)."""
    return load_custom_commands(project_dir=project_dir).get((name or "").strip().lower())


def invoke_custom_command(name: str, arguments: str = "", *,
                          project_dir: str | Path = ".") -> dict[str, Any]:
    """Expand an authored command into its dispatchable flow/tool sequence (T-CTL-01 invoke side).

    Resolves the command, substitutes ``$ARGUMENTS`` in the body + step args, and returns the
    expansion the control plane dispatches: ``{found, name, flow, steps, prompt}``. Does NOT
    execute — it produces the request the control-plane agent runs (so spend stays gated by the
    M1 approval/plan-mode beat).
    """
    cmd = get_custom_command(name, project_dir=project_dir)
    if cmd is None:
        return {"found": False, "name": (name or "").strip().lower(),
                "error": f"no custom command named {name!r}"}
    args = arguments or ""

    def _sub(val: Any) -> Any:
        if isinstance(val, str):
            return val.replace("$ARGUMENTS", args)
        if isinstance(val, dict):
            return {k: _sub(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_sub(v) for v in val]
        return val

    expanded_steps = [_sub(dict(s)) for s in cmd.steps]
    return {
        "found": True,
        "name": cmd.name,
        "flow": cmd.flow,
        "steps": expanded_steps,
        "prompt": cmd.body.replace("$ARGUMENTS", args),
        "argument_hint": cmd.argument_hint,
    }
