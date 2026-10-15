"""CLI ``capability`` command group — the CLI surface over the M7 funnel (ADR-010).

A thin Click adapter so any registered flow/capability is runnable non-interactively from
the terminal — the same funnel Chat/TUI/Python-API use. NO flow logic here (ADR-010):
every subcommand delegates to ``control_plane.capability`` / ``runtime_session``.

    amlas capability list                       # discover (the flow registry)
    amlas capability run data_quality_flow -p dataset_uri=data/x.csv
    amlas capability plan deploy_flow            # Plan-Mode preview (no spend)
    amlas capability run data_quality_flow -p ... --print   # headless -p (CI/JSON)
    amlas capability resume my-session           # --resume analog
    amlas capability mention @dataset:silver/x   # @-mention

``-p key=value`` params are parsed into the flow's kwargs (ints/floats/bools coerced).
``--print``/``-P`` runs headless (the ``claude -p`` analog); ``--allow``/``--max-budget``/
``--bypass`` set the non-interactive approval policy.
"""

from __future__ import annotations

import json
from typing import Any

import click


def _coerce(value: str) -> Any:
    """Coerce a ``key=value`` string value to int/float/bool/json where possible."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except (TypeError, ValueError):
            pass
    if value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _parse_params(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse repeated ``-p key=value`` options into a params dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"param {pair!r} must be key=value")
        key, _, val = pair.partition("=")
        out[key.strip()] = _coerce(val)
    return out


@click.group(name="capability")
def capability() -> None:
    """Run any registered ML capability (flow) — the one-capability CLI surface (ADR-010)."""


@capability.command(name="list")
def list_cmd() -> None:
    """List every registered capability (id, gated, runnable) from the shared registry."""
    from ...control_plane.capability import list_capabilities

    caps = list_capabilities()
    click.echo(f"{len(caps)} capabilities:")
    for c in caps:
        gate = "GATED" if c["gated"] else "read "
        run = "ok " if c["runnable"] else "n/a"
        click.echo(f"  [{gate}|{run}|{c['milestone']:>2}] {c['capability_id']:<24} {c['description'][:80]}")


@capability.command(name="plan")
@click.argument("flow_id")
@click.option("--param", "-p", multiple=True, help="Flow param key=value (repeatable).")
def plan_cmd(flow_id: str, param: tuple[str, ...]) -> None:
    """Plan-Mode preview a capability (read-only; provisions nothing)."""
    from ...control_plane.capability import invoke_capability

    res = invoke_capability(capability_id=flow_id, params=_parse_params(param),
                            surface="cli", plan_only=True)
    click.echo(json.dumps(res.to_dict(), indent=2))


@capability.command(name="run")
@click.argument("flow_id")
@click.option("--param", "-p", multiple=True, help="Flow param key=value (repeatable).")
@click.option("--print", "-P", "print_mode", is_flag=True, default=False,
              help="Headless print-mode (the claude -p analog): non-interactive, JSON out.")
@click.option("--allow", multiple=True, help="Allowlist a capability for headless (repeatable).")
@click.option("--max-budget", type=float, default=None, help="Headless spend cap for allowlisted actions.")
@click.option("--bypass", is_flag=True, default=False, help="Headless trusted-automation bypass.")
def run_cmd(flow_id: str, param: tuple[str, ...], print_mode: bool,
            allow: tuple[str, ...], max_budget: float | None, bypass: bool) -> None:
    """Run a capability through the shared funnel (ADR-010); ``--print`` runs it headless."""
    params = _parse_params(param)
    if print_mode:
        from ...control_plane.runtime_session import run_headless

        out = run_headless(capability_id=flow_id, params=params,
                           allowed_tools=list(allow) or None, max_budget=max_budget, bypass=bypass)
        click.echo(json.dumps(out, indent=2))
        return
    from ...control_plane.capability import invoke_capability

    res = invoke_capability(capability_id=flow_id, params=params, surface="cli")
    click.echo(json.dumps(res.to_dict(), indent=2))


@capability.command(name="resume")
@click.argument("session_ref")
def resume_cmd(session_ref: str) -> None:
    """Resume a prior session (T-RUN-06; the --resume analog)."""
    from ...control_plane.runtime_session import resume_session

    click.echo(json.dumps(resume_session(session_ref=session_ref), indent=2, default=str))


@capability.command(name="mention")
@click.argument("ref")
def mention_cmd(ref: str) -> None:
    """Resolve an @-mention into context (T-LAKE-07; @dataset:/@spec:/@run:/@artifact:)."""
    from ...control_plane.mention import mention_context

    click.echo(json.dumps(mention_context(ref), indent=2, default=str))
