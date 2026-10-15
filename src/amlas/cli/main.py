"""Root click group and common options for the Amlas ML CLI.

Entry point: `amlas` command.
"""

from __future__ import annotations

import click

from .commands.capability import capability
from .commands.config import config
from .commands.data import data
from .commands.info import info
from .commands.mlops import mlops
from .commands.research import research
from .commands.results import results
from .commands.run import run
from .commands.spec import spec
from .commands.status import status


def _install_error_surface() -> None:
    """Initialize the centralized error surface for this CLI process.

    Points the process-wide ``ERROR_LOG`` at a real per-session JSONL file and installs
    the process-level uncaught-exception funnel. Idempotent + best-effort (a telemetry
    init failure must never block the CLI).
    """
    try:
        from .._runtime_guard import install_error_excepthook
        from ..telemetry.error_log import init_error_log

        init_error_log()
        install_error_excepthook()
    except Exception:  # nosec B110
        pass


@click.group(name="amlas")
@click.version_option(version="0.1.0", prog_name="amlas")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose/debug output.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Amlas ML Engineering Platform — Terminal Control System.

    Comprehensive CLI for managing the ML pipeline: spec creation,
    data quality, research, training, evaluation, and deployment.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    # Universal error handling: give the process-wide ERROR_LOG a real JSONL
    # file sink for this CLI session AND install the process-level uncaught-exception
    # funnel, so every captured/escaped error across the CLI lands in the one error file
    # (+ the CloudWatch mirror in managed modes). Both are idempotent + best-effort — a
    # telemetry init failure must never block the CLI.
    _install_error_surface()


# Register all command groups
cli.add_command(spec)
cli.add_command(data)
cli.add_command(research)
cli.add_command(run)
cli.add_command(status)
cli.add_command(results)
cli.add_command(config)
cli.add_command(info)
cli.add_command(mlops)
# M7 — the one-capability CLI surface (ADR-010): run any registered flow non-interactively.
cli.add_command(capability)
