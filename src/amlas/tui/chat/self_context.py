"""TUI agent self-command context.

 ("Bug | ensure TUI agent has context of its own commands"): the chat
agent confused ``/dqscan`` with a nonexistent ``/data-quality``, could not tell
its TUI slash commands apart from the CLI / API surfaces, and had no awareness of
itself + its 7 agentcore/strands ML-automation agents.

Root cause: the agent system prompt (:data:`amlas.tui.chat.agent_tools.TUI_SYSTEM_PROMPT`)
described abilities in *prose* but never enumerated the REAL slash-command registry.
So the model guessed command names.

Fix: derive the self-context from the SINGLE SOURCES OF TRUTH —
:data:`amlas.tui.chat.commands.COMMANDS` (the exact registered slash commands) and
:data:`amlas.control_plane.token_cost.AGENTS` (the TUI agent + 7 leads) — and embed
:func:`build_self_context_block` into the system prompt. Because it is generated
from the registry, it can never drift from the commands that actually exist.

This block was originally embedded into TWO prompts: the tool agent's and the
raw-Converse ``bedrock_backend.SYSTEM_PROMPT``. The latter was deleted with its
module (issue #35, 6.2), so there is now one prompt and one embedding site
([`agent_tools.py:475`](agent_tools.py:475)) — the issue-#6 feature is unaffected,
since that site composes the block independently.
"""

from __future__ import annotations

from ...control_plane.token_cost import ML_AUTOMATION_AGENTS, TUI_AGENT
from .commands import COMMANDS

# Human-readable one-liners for the 7 agentcore/strands ML-automation leads, keyed
# on the canonical roster names in ``token_cost.ML_AUTOMATION_AGENTS`` (kept in the
# same Research→DataEng→FE→ML→MLOps→Deploy→Monitor lifecycle order).
_AGENT_ROLES: dict[str, str] = {
    "research_lead": "surveys algorithms/papers and recommends approaches",
    "data_engineering_lead": "data acquisition, quality, imputation, synthesis, lakehouse",
    "fe_lead": "feature engineering and preprocessing",
    "ml_execution_lead": "baseline training, HPO, ensembling",
    "mlops_lead": "pipelines, model registry, monitoring setup",
    "deploy_serving_lead": "endpoint deploy and serving",
    "monitoring_lead": "drift/quality monitoring and Clarify checks",
}


def list_tui_command_names() -> list[str]:
    """Return the exact list of registered TUI slash-command names (registry order).

    Sourced from :data:`amlas.tui.chat.commands.COMMANDS`, so it is ALWAYS the real
    command surface — no hand-maintained list that could drift ( root cause).
    """
    return list(COMMANDS)


def build_self_context_block() -> str:
    """Render the agent's self-context block for injection into the system prompts.

    Enumerates every real ``/command`` (name + one-line description straight from the
    registry), names the TUI agent + its 7 agentcore/strands leads, and draws the
    TUI-vs-CLI-vs-API surface distinction the issue calls out — so the agent stops
    inventing command names like ``/data-quality`` and knows what it (and its swarm)
    actually is.
    """
    lines: list[str] = [
        "## Your own command surface (authoritative — do NOT invent commands)",
        "You are the single **TUI** chat agent. These are your EXACT slash commands "  # nosemgrep: string-concat-in-list
        "(the ONLY valid ones — never invent or guess a command that is not listed):",
    ]
    for name, cmd in COMMANDS.items():
        lines.append(f"- `/{name}` — {cmd.description}")

    lines.append("")
    lines.append(
        "If a user names a command that is not in this list (e.g. `/data-quality`), it "
        "does NOT exist — map their intent to the closest REAL command above "
        "(data-quality → `/dqscan` or `/quality`) or answer conversationally. Never "
        "claim a made-up command."
    )
    lines.append("")
    lines.append("## Three distinct surfaces — do not conflate them")
    lines.append(
        "- **TUI** (you): the `/slash` commands above, run inside this terminal UI.\n"
        "- **CLI**: the `amlas ...` shell command (a separate entrypoint); TUI `/slash` "
        "commands are NOT CLI subcommands and vice-versa.\n"
        "- **API**: the programmatic Python/HTTP surface behind the shared capability "
        "funnel. Your `/flow` / `/flows` reach the same capabilities the CLI/API do, but "
        "the invocation syntax differs per surface."
    )
    lines.append("")
    lines.append("## You + your 7 agentcore/strands ML-automation agents")
    lines.append(
        f"You (`{TUI_AGENT}`) front the conversation. Heavy ML work is executed by "
        f"{len(ML_AUTOMATION_AGENTS)} specialist agentcore/strands leads you dispatch to "
        "(via `dispatch_cloud_experiment` / `run_ml_capability`):"
    )
    for agent in ML_AUTOMATION_AGENTS:
        role = _AGENT_ROLES.get(agent, "specialist ML-automation lead")
        lines.append(f"- `{agent}` — {role}")

    return "\n".join(lines)
