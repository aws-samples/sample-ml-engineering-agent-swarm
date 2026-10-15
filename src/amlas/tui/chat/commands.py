"""Slash command definitions and handlers for the chat interface.

Commands:
  /help            — Show available commands
  /spec "desc"     — Generate a spec from description
  /run <spec> <data> — Start a pipeline run
  /status          — Show current pipeline status
  /leaderboard     — Show current leaderboard
  /research <query> — Search algorithms; `--web` adds external research
  /quality <path>  — Run data quality check
  /train <algo> <data> — Train a specific model
  /attach <path>   — Attach a file for context
  /data [path]     — List datasets or load one by path
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass


@dataclass
class SlashCommand:
    """Definition of a single slash command."""

    name: str
    description: str
    usage: str
    min_args: int = 0
    max_args: int = -1  # -1 = unlimited


# ─── Command Registry ─────────────────────────────────────────────────────────

COMMANDS: dict[str, SlashCommand] = {
    "help": SlashCommand(
        name="help",
        description="Show available commands",
        usage="/help",
    ),
    # Alias: the chat welcome text + input placeholder both advertise
    # ``/commands``, so register it (and the short ``/cmds``) as help aliases.
    # Without this a user who types the advertised ``/commands`` hits the
    # slash-vs-path fallback and sees a spurious "File not found".
    "commands": SlashCommand(
        name="commands",
        description="Show available commands (alias for /help)",
        usage="/commands",
    ),
    "cmds": SlashCommand(
        name="cmds",
        description="Show available commands (alias for /help)",
        usage="/cmds",
    ),
    "spec": SlashCommand(
        name="spec",
        description="Generate a spec, or `/spec list` / `/spec show <path>`",
        usage='/spec "predict wheel failures from sensor data" | /spec list | /spec show <path>',
        min_args=1,
    ),
    "run": SlashCommand(
        name="run",
        description="Start a pipeline run with spec and data paths",
        usage="/run specs/my-spec.yaml data/train.csv",
        min_args=2,
        max_args=2,
    ),
    "status": SlashCommand(
        name="status",
        description="Show current pipeline/system status",
        usage="/status",
    ),
    "leaderboard": SlashCommand(
        name="leaderboard",
        description="Show the current model leaderboard",
        usage="/leaderboard",
    ),
    "research": SlashCommand(
        name="research",
        description="Search algorithms; `--web` adds external research",
        usage="/research [--web] gradient boosting tabular",
        min_args=1,
    ),
    "quality": SlashCommand(
        name="quality",
        description="Run data quality assessment on a CSV file",
        usage="/quality data/train.csv",
        min_args=1,
        max_args=1,
    ),
    "train": SlashCommand(
        name="train",
        description="Train a specific algorithm on data",
        usage="/train xgboost data/train.csv",
        min_args=2,
        max_args=2,
    ),
    "attach": SlashCommand(
        name="attach",
        description="Attach a file to the chat context",
        usage="/attach path/to/file.csv",
        min_args=1,
        max_args=1,
    ),
    # ── Phase 11 — agentic Data-Engineering / MLOps slash-commands (§10) ──────
    "dqscan": SlashCommand(
        name="dqscan",
        description="Run the 25-pattern DQ catalog scan on a CSV",
        usage="/dqscan data/locomotive_1957.csv",
        min_args=1,
        max_args=1,
    ),
    "diagnose": SlashCommand(
        name="diagnose",
        description="Diagnose the missingness mechanism (MCAR/MAR/MNAR) of a CSV",
        usage="/diagnose data/locomotive_1957.csv",
        min_args=1,
        max_args=1,
    ),
    "impute": SlashCommand(
        name="impute",
        description="Impute missing values (auto/median/...); honors the /steer dial",
        usage="/impute data/x.csv [--method auto] [--llm] [--evaluate]",
        min_args=1,
    ),
    "synth": SlashCommand(
        name="synth",
        description="Generate synthetic rows (or --balance to augment); /steer thresholds",
        usage="/synth data/x.csv 2000 [--balance --target y] [--validate]",
        min_args=1,
    ),
    "synth-from": SlashCommand(
        name="synth-from",
        description="Build a SynthesisSpec from a natural-language requirement",
        usage='/synth-from "more class-3 fault rows under high speed"',
        min_args=1,
    ),
    "steer": SlashCommand(
        name="steer",
        description="Set/show the session domain-steering dial (forbid/pin/notes/thresholds)",
        usage="/steer set forbid=mean pin=datetime:time notes=...  |  /steer show",
        min_args=1,
    ),
    "lake": SlashCommand(
        name="lake",
        description="Medallion lake read-slice / write-back / catalog",
        usage="/lake read silver://db.table  |  /lake write local.csv s3://.../gold/",
        min_args=1,
    ),
    "mlops": SlashCommand(
        name="mlops",
        description="MLOps actions: pipeline | register | monitor | status",
        usage="/mlops status  |  /mlops pipeline build  |  /mlops register <artifact>",
        min_args=1,
    ),
    "data": SlashCommand(
        name="data",
        description="List available datasets or load a dataset by path",
        usage="/data [path]",
        min_args=0,
        max_args=1,
    ),
    # ─── Phase 5 — newly wired commands (G4, G5, G10, G12) ──────────────────
    "export": SlashCommand(
        name="export",
        description="Export a run's model artifact to a destination path",
        usage="/export <run-id> <dest-path>",
        min_args=2,
        max_args=2,
    ),
    "dashboard": SlashCommand(
        name="dashboard",
        description="Show the project dashboard (scheduler summary)",
        usage="/dashboard",
        min_args=0,
        max_args=0,
    ),
    "tune": SlashCommand(
        name="tune",
        description="Run Optuna hyperparameter optimization for an algorithm",
        usage="/tune xgboost data/train.csv",
        min_args=2,
        max_args=2,
    ),
    "experiment": SlashCommand(
        name="experiment",
        description="Run a multi-algorithm experiment plan from a spec",
        usage="/experiment specs/my-spec.md data/train.csv",
        min_args=2,
        max_args=2,
    ),
    "modelcard": SlashCommand(
        name="modelcard",
        description="Generate a model card for a recorded run",
        usage="/modelcard <run-id>",
        min_args=1,
        max_args=1,
    ),
    # ─── Phase 9 — EXPOSE-set parity commands (G21–G37) ─────────────────────
    "split": SlashCommand(
        name="split",
        description="Split a dataset into train/test sets (split_dataset)",
        usage="/split <data> <target> [test_size]",
        min_args=2,
        max_args=3,
    ),
    "describe": SlashCommand(
        name="describe",
        description="Full summary statistics for a dataset (describe_dataset)",
        usage="/describe <data> [target]",
        min_args=1,
        max_args=2,
    ),
    "anomalies": SlashCommand(
        name="anomalies",
        description="Detect anomalies/outliers in a dataset (detect_anomalies)",
        usage="/anomalies <data> [method]",
        min_args=1,
        max_args=2,
    ),
    "leakage": SlashCommand(
        name="leakage",
        description="Run train/test leakage checks (leakage_check_tool)",
        usage="/leakage <train> <test> <target>",
        min_args=3,
        max_args=3,
    ),
    "preprocess": SlashCommand(
        name="preprocess",
        description="Auto feature-engineer a dataset (auto_preprocess)",
        usage="/preprocess <data> [target]",
        min_args=1,
        max_args=2,
    ),
    "ensemble": SlashCommand(
        name="ensemble",
        description="Train a stacking/voting ensemble (train_ensemble)",
        usage="/ensemble <data> <algo1> <algo2> [...]",
        min_args=2,
    ),
    "best": SlashCommand(
        name="best",
        description="Show the current best model for a spec (get_best_model)",
        usage="/best [spec_id]",
        min_args=0,
        max_args=1,
    ),
    "recommend": SlashCommand(
        name="recommend",
        description="Deploy/iterate/reject recommendation (get_recommendation)",
        usage="/recommend [spec_id]",
        min_args=0,
        max_args=1,
    ),
    "nextsteps": SlashCommand(
        name="nextsteps",
        description="Generate a next-steps decision report (generate_next_steps_report)",
        usage="/nextsteps [spec_id]",
        min_args=0,
        max_args=1,
    ),
    "project": SlashCommand(
        name="project",
        description="Manage queued projects: submit | list | status",
        usage="/project submit <spec> <data> | /project list | /project status <id>",
        min_args=1,
    ),
    "queue": SlashCommand(
        name="queue",
        description="Run the project queue (ProjectScheduler.run_all)",
        usage="/queue run [max_concurrent]",
        min_args=1,
        max_args=2,
    ),
    "watch": SlashCommand(
        name="watch",
        description="Live run/store summary snapshot (MLPipeline.get_summary)",
        usage="/watch",
        min_args=0,
        max_args=0,
    ),
    "upload": SlashCommand(
        name="upload",
        description="Upload a local file to S3 (boto3, creds-gated)",
        usage="/upload <local-path> <s3-key>",
        min_args=2,
        max_args=2,
    ),
    "download": SlashCommand(
        name="download",
        description="Download an S3 object to a local path (boto3, creds-gated)",
        usage="/download <s3-key> <local-path>",
        min_args=2,
        max_args=2,
    ),
    "health": SlashCommand(
        name="health",
        description="Check AWS connectivity readiness (check_aws_health)",
        usage="/health",
        min_args=0,
        max_args=0,
    ),
    "plan": SlashCommand(
        name="plan",
        description="Preview an experiment plan from a spec (create_experiment_plan)",
        usage="/plan <spec> [data]",
        min_args=1,
        max_args=2,
    ),
    # ─── M7 — one-capability across surfaces (ADR-010): chat → the shared funnel ──
    "flows": SlashCommand(
        name="flows",
        description="List every registered ML capability (the shared flow registry)",
        usage="/flows",
        min_args=0,
        max_args=0,
    ),
    "flow": SlashCommand(
        name="flow",
        description="Run (or --plan) any registered capability via the shared funnel (ADR-010)",
        usage="/flow data_quality_flow dataset_uri=data/x.csv  |  /flow deploy_flow --plan",
        min_args=1,
    ),
    "mention": SlashCommand(
        name="mention",
        description="Resolve an @-mention into context (T-LAKE-07; @dataset:/@spec:/@run:/@artifact:)",
        usage="/mention @dataset:silver/sensors",
        min_args=1,
    ),
}


def parse_slash_command(message: str) -> tuple[str, list[str]] | None:
    """Parse a slash command from a message string.

    Returns (command_name, args) tuple or None if not a slash command.
    """
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None

    # Use shlex to handle quoted arguments
    try:
        parts = shlex.split(stripped)
    except ValueError:
        # Fallback to simple split if quotes are unbalanced
        parts = stripped.split()

    if not parts:
        return None

    cmd_name = parts[0][1:]  # Remove the leading /
    args = parts[1:]

    return (cmd_name, args)


def format_help_text() -> str:
    """Generate formatted help text for all available commands."""
    lines = [
        "📋 **Available Commands:**",
        "",
        "Each line shows the usage then a one-line description. Arguments in "  # nosemgrep
        "`<angle brackets>` are required; `[square brackets]` are optional.",
        "",
    ]
    for cmd in COMMANDS.values():
        lines.append(f"  `{cmd.usage}`")
        lines.append(f"    {cmd.description}")
        lines.append("")
    lines.append("💡 You can also chat naturally — ask questions, describe ML tasks,")
    lines.append("   or discuss results without using commands.")
    return "\n".join(lines)


def validate_command_args(cmd_name: str, args: list[str]) -> str | None:
    """Validate arguments for a command. Returns error message or None if valid."""
    cmd = COMMANDS.get(cmd_name)
    if cmd is None:
        available = ", ".join(f"/{c}" for c in COMMANDS)
        return f"Unknown command: /{cmd_name}\nAvailable: {available}"

    if len(args) < cmd.min_args:
        return f"/{cmd_name} requires at least {cmd.min_args} argument(s).\nUsage: {cmd.usage}"

    if cmd.max_args >= 0 and len(args) > cmd.max_args:
        return f"/{cmd_name} accepts at most {cmd.max_args} argument(s).\nUsage: {cmd.usage}"

    return None
