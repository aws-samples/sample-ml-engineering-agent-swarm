"""Canonical Strands Agent factory.

Every agent in the system — the 7 Leads, their subagents, the pipeline agent, and the
TUI's tool-bound chat agent — is constructed through :func:`build_agent`. What that
single seam owns:

    - **Strands version-drift guard** — :func:`_filter_agent_kwargs` forwards only the
      kwargs the installed ``Agent.__init__`` actually accepts.
    - **Per-role model tiers** (``planner``/``reviewer``/``specialist``/``bounded``/
      ``finetune``), overridable by an explicit ``model_id`` (the TUI chat tier).
    - **Memory-hook injection seam** — ``hooks_factory`` lets the Orchestrator attach a
      ``MemoryHookProvider`` scoped to ``(memory_id, actor_id, session_id)`` without
      threading those ids through every factory signature. When ``MEMORY_ID`` is unset
      the Orchestrator's factory yields no hooks. See :mod:`amlas.memory.hooks`.
    - **In-band Bedrock Guardrail** — :func:`_build_bedrock_model` applies
      ``guardrail_model_kwargs()`` on EVERY model turn, which is what makes
      WA-02/AGENTSEC04 filtering always-on rather than gate-time.
    - **CREDS-01 credential hygiene** — :func:`_build_bedrock_model` calls
      ``neutralize_stale_bearer_token`` before the boto3 session is built, so an
      inherited/expired ``AWS_BEARER_TOKEN_BEDROCK`` cannot outrank the profile's
      SigV4 chain on ANY agent build (not just at the TUI entrypoint).
    - **Adaptive boto3 retries + long read_timeout** for training-heavy tools.
    - **1M-context beta header** on all Sonnet 4.x + Opus 4.x (Haiku has no 1M profile).
      Note Sonnet 4.x now defaults to a 1M window on Bedrock, so the header is no longer
      what unlocks the large context for that tier.
    - **Centralized error capture** — an Agent-construction failure funnels into
      ``ERROR_LOG`` with ``source="agents.<name>"``, then re-raises unchanged.

Passing ``model=`` (a pre-built model) bypasses :func:`_build_bedrock_model`, which is
the seam offline validation use to construct agents without AWS credentials; passing ``None``
(the production default) builds the real ``BedrockModel``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .config import Config, load_config
from .config.aws_region import resolve_aws_profile, resolve_aws_region
from .config.runtime import supports_explicit_converse_cache, supports_temperature

logger = logging.getLogger(__name__)


def _filter_agent_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs the *installed* ``Agent.__init__`` actually accepts.

    STRANDS-DRIFT (root cause of the ``⚠️ Tool-agent error`` the TUI showed and
    of every real swarm build failing): Strands ``Agent.__init__`` evolves across
    versions. v0.3.0 dropped ``tool_executor`` (and ``strands.tools.executors`` no
    longer exists), but this factory historically passed ``tool_executor=…``
    unconditionally — raising ``Agent.__init__() got an unexpected keyword
    argument 'tool_executor'`` on EVERY real build (the TUI's tool-bound NL agent
    AND the leads/sub-agents). The hermetic compatibility shim hid it because its
    ``Agent(*a, **k)`` swallows any kwargs.

    Fix: introspect the installed signature ONCE per build and forward only the
    parameters it declares. If the signature exposes ``**kwargs`` (the shim, or a
    future permissive Agent), nothing is dropped — preserving the existing callers.
    A dropped non-None value is logged at DEBUG so version drift is observable.
    """
    try:
        sig = inspect.signature(Agent.__init__)
    except (TypeError, ValueError):  # builtins / C-extensions without a signature
        return kwargs

    params = sig.parameters.values()
    # A ``**kwargs`` (VAR_KEYWORD) parameter accepts anything → pass through whole.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs

    accepted = {
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.name != "self"
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = {k: v for k, v in kwargs.items() if k not in accepted and v is not None}
    if dropped:
        logger.debug(
            "build_agent: dropping kwargs unsupported by installed Strands "
            "Agent.__init__ (version drift): %s",
            sorted(dropped),
        )
    return filtered


def build_agent(
    *,
    name: str,
    system_prompt: str,
    tools: Sequence[Any] = (),
    model: Any | None = None,  # optional pre-built model; None → real Bedrock
    role: str = "planner",
    model_id: str | None = None,  # explicit model-id override (wins over role tier)
    max_tokens: int | None = None,  # explicit output-cap override (else cfg.bedrock)
    hooks: Sequence[Any] = (),
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    tool_executor: Any | None = None,
    description: str | None = None,
    messages: Sequence[Any] | None = None,
    config: Config | None = None,
) -> Agent:
    """Build a Strands Agent with Amlas canonical defaults.

    Args:
        name: Agent name — used in trace spans and a2a routing.
        system_prompt: Full system prompt text (usually from stack_prompts()).
        tools: Sequence of @tool-decorated callables or Agent instances (auto-wrapped via as_tool).
        model: Optional pre-built model instance. If None (the production
               default), we build a real BedrockModel from config — requires
               AWS credentials. Callers may pass a pre-built model, or omit it
               to use the configured Bedrock default.
        role: Model role selector for per-role model routing.
        hooks: HookProvider instances.
        hooks_factory: Optional zero-arg callable invoked exactly once per build whose
               return value is appended to ``hooks``. Enables the Orchestrator to
               inject a ``MemoryHookProvider`` scoped to ``(memory_id, actor_id=
               project_id, session_id=run_id)`` without plumbing those IDs through
               every lead/subagent factory signature. Defaulting to ``None``
               preserves the import-time behavior and every existing caller call
               site. The factory is invoked only here (never at module import).
        model_id: Optional explicit Bedrock model-id (e.g. the TUI chat tier
               ``AMLAS_CHAT_MODEL``=Sonnet 4.6). When provided AND no pre-built
               ``model`` is passed, it OVERRIDES the per-role tier so the TUI's
               tool-bound NL agent can run on Sonnet 4.6 while the swarm leads/
               sub-agents keep their Opus 4.8 ``role`` tiers.
        max_tokens: Optional explicit output-token cap for the model, honored on the
               SAME terms as ``model_id``: it is a MODEL-construction kwarg, so it
               applies only when no pre-built ``model`` is passed, and it never
               reaches ``Agent.__init__``. ``None`` (every swarm lead/subagent call
               site) resolves to ``cfg.bedrock.max_tokens``, so those builds are
               unchanged. The TUI chat tier passes
               :func:`~amlas.config.chat_model.chat_max_tokens` so chat can be capped
               without moving the swarm tier (issue #35, criteria 4.1/4.2).
        tool_executor: Optional ToolExecutor (SequentialToolExecutor for artifact-writing tools).
        description: Optional description for a2a / multi-agent routing.
        messages: Optional PRIOR conversation history to seed the Agent's
               ``.messages`` at construction — the canonical Strands multi-turn
               mechanism (``Agent(messages=[{"role":.., "content":[{"text":..}]}])``).
               Used by the TUI chat agent to replay prior turns so an affirmative
               like "yes to both" resolves against the assistant's previous
               question. Defaulting to ``None`` keeps every swarm lead/subagent
               call site byte-identical (they never pass it → single-turn as
               before); it is filtered out on any installed ``Agent`` that predates
               the ``messages`` kwarg (see ``_filter_agent_kwargs``).
        config: Optional Config override (defaults to load_config()).
    """
    if model is None:
        try:
            from .config.env_loader import load_layered_env

            load_layered_env("agentcore")
        except Exception as e:
            # SEC-15 / R-014: env overlay is best-effort and must never block a build,
            # but the swallowed failure is now observable at debug level so a broken
            # .env.agentcore overlay is diagnosable instead of failing silently.
            logger.debug("load_layered_env('agentcore') skipped (%s: %s)", type(e).__name__, e)
    cfg = config or load_config()
    # ``model_id``/``max_tokens`` are MODEL-construction kwargs and are consumed HERE.
    # They must never enter ``agent_kwargs`` below: ``_filter_agent_kwargs`` drops
    # unknown names QUIETLY, so an output cap routed through the Agent kwargs would
    # vanish without a trace and silently leave the model on its config default.
    resolved_model = (
        model
        if model is not None
        else _build_bedrock_model(cfg, role, model_id=model_id, max_tokens=max_tokens)
    )
    produced_hooks = list(hooks) + (list(hooks_factory()) if hooks_factory is not None else [])
    # Assemble the full desired kwarg set, then forward ONLY what the installed
    # Strands ``Agent.__init__`` accepts (see ``_filter_agent_kwargs`` — fixes the
    # ``tool_executor`` version-drift crash without changing the shim path).
    agent_kwargs: dict[str, Any] = {
        "name": name,
        "model": resolved_model,
        "tools": list(tools),
        "system_prompt": system_prompt,
        "hooks": produced_hooks,
        "tool_executor": tool_executor,
        "description": description,
        # Only include ``messages`` when non-empty so the default single-turn
        # build stays byte-identical (``Agent()`` starts with an empty history).
        "messages": list(messages) if messages else None,
    }
    # Universal error handling: this is the SHARED build path for ALL 7
    # sub-agents + the leads, so capturing an Agent-construction failure HERE funnels
    # every agent build error into the centralized ``ERROR_LOG`` with a per-agent
    # ``source="agents.<name>"`` — in one place, without hand-editing each factory. The
    # capture is purely additive: the exception is RE-RAISED unchanged (build semantics
    # preserved) and the capture is guarded so telemetry can never mask the real error.
    try:
        return Agent(**_filter_agent_kwargs(agent_kwargs))
    except Exception as exc:
        try:
            from .telemetry.error_log import capture_error

            capture_error(exc, source=f"agents.{name}")
        except Exception:  # nosec B110
            pass
        raise


def _render_problem_block(problem: Any) -> str:
    """Render the active problem framing (dict or ``ActiveProblem``) as a prompt block.

    Accepts either a plain dict (``{"problem_id":.., "kpi":.., "problem_card_path":..}``)
    or any object exposing those attributes (the TUI ``ActiveProblem`` snapshot). Returns
    an empty string when there is no problem signal so the no-problem path adds nothing.
    """
    if not problem:
        return ""

    def _get(key: str) -> Any:
        if isinstance(problem, dict):
            return problem.get(key)
        return getattr(problem, key, None)

    lines: list[str] = []
    pid = _get("problem_id")
    kpi = _get("kpi")
    card = _get("problem_card_path")
    if pid:
        lines.append(f"- Problem id: {pid}")
    if kpi:
        lines.append(f"- KPI / objective: {kpi}")
    if card:
        lines.append(f"- Problem card: {card}")
    if not lines:
        return ""
    return "## Active problem framing\n\n" + "\n".join(lines)


def context_preamble(*, steering: Any = None, problem: Any = None) -> str:
    """Compose a null-safe steering/problem PREAMBLE for a subagent system prompt (H2-6).

    Reuses :meth:`SteeringProfile.as_prompt_block` (duck-typed via ``as_prompt_block`` so
    this stays import-light and tolerant of ``None``) for the steering dial, and
    :func:`_render_problem_block` for the active problem framing. Returns an EMPTY string
    when neither carries a signal — so a subagent built with no steering/problem is
    byte-identical to today (the default swarm path is unchanged).
    """
    blocks: list[str] = []
    if steering is not None and hasattr(steering, "as_prompt_block"):
        try:
            block = steering.as_prompt_block()
        except Exception:  # a malformed dial must never break a build
            block = ""
        if block:
            blocks.append(block)
    problem_block = _render_problem_block(problem)
    if problem_block:
        blocks.append(problem_block)
    return "\n\n".join(blocks)


def compose_system_prompt(base: str, *, steering: Any = None, problem: Any = None) -> str:
    """Return ``base`` with a steering/problem preamble appended when there is signal.

    ADDITIVE + null-safe: with no steering/problem this returns ``base`` UNCHANGED, so
    the default subagent build is byte-identical. When there IS a dial/problem, the
    preamble is appended after a markdown HR (matching ``stack_prompts``' separator) so
    the delegated subagent's reasoning respects the session's HARD constraints + framing.
    """
    preamble = context_preamble(steering=steering, problem=problem)
    if not preamble:
        return base
    return f"{base}\n\n---\n\n{preamble}"


def _bedrock_boto_config(cfg: Config) -> Any:
    """Build the botocore ``Config`` for the agent BedrockModel client (PERF-02).

    Behavior-preserving: carries the SAME ``read_timeout`` + adaptive ``retries``
    as before, and ADDS ``max_pool_connections`` (default 50 via
    :class:`BedrockConfig`) so a single client can hold many concurrent
    Bedrock/tool calls in-flight instead of being capped at urllib3's default
    pool of 10. Pure connection-pool sizing — request/response bytes are
    byte-for-byte identical. Extracted as a pure helper so it is directly verifiable
    offline (no creds / no client construction).
    """
    from botocore.config import Config as BotoConfig

    return BotoConfig(
        read_timeout=cfg.bedrock.read_timeout,
        max_pool_connections=cfg.bedrock.max_pool_connections,
        retries={"max_attempts": cfg.bedrock.max_attempts, "mode": cfg.bedrock.retry_mode},
    )


def _effective_profile(cfg: Config) -> str | None:
    """The AWS profile CREDS-01 should read as the operator's explicit intent.

    Mirrors :attr:`Settings.aws.effective_profile` — an empty/whitespace profile
    resolves to ``None`` so boto3 falls back to its default credential chain.

    ``Config`` DOES carry a profile (``cfg.bedrock.profile``), but it is populated from
    ``AMLAS_AWS_PROFILE`` alone and never reads the STANDARD ``AWS_PROFILE`` var, which
    is precisely the shape of the CREDS-01b bug recorded at ``tui/app.py``: a bare
    ``Settings()`` did not read ``AWS_PROFILE``, ``effective_profile`` came back ``None``,
    and a still-VALID inherited bearer token was therefore KEPT — then went on to
    override the profile's SigV4 chain. So fall back to ``AWS_PROFILE`` here, with the
    same precedence ``Settings._apply_env_vars`` uses (``AMLAS_AWS_PROFILE`` wins over
    the standard name).

    ``None`` is a supported outcome, not an error: an expired or undecodable token is
    dropped regardless of the profile. Only the "valid token, but an explicit profile
    wins" case needs one.
    """
    return resolve_aws_profile(cfg.bedrock.profile)


def _build_bedrock_model(
    cfg: Config,
    role: str,
    *,
    model_id: str | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Build the configured BedrockModel for live runtime use.

    Layered-config: this is the REAL-model path (callers may inject ``model=`` and never
    reach here), so it's the right seam to overlay the AgentCore/Strands runtime
    ``.env.agentcore`` (runtime ARN, OTEL/STRANDS telemetry toggles, per-role
    model tiers, boto retry/timeout) on top of the shared-base root ``.env``
    BEFORE the boto session + BedrockModel are constructed. Idempotent +
    shell-wins (see ``config/env_loader``); best-effort so it never blocks a build.

    Also the CREDS-01 enforcement point for every ``BedrockModel`` in the system —
    see the comment on the ``neutralize_stale_bearer_token`` call below.

    ``max_tokens`` is the per-build output cap; ``None`` (every swarm lead/subagent
    call site) keeps ``cfg.bedrock.max_tokens`` exactly as before. It exists so the
    TUI chat tier can pass :func:`~amlas.config.chat_model.chat_max_tokens` and be
    capped WITHOUT moving the swarm tier — mirroring how ``model_id`` lets the same
    caller override the per-role model tier (issue #35, criteria 4.1/4.2).
    """
    # Deferred imports so offline validation don't need boto3 configured
    import boto3
    from strands.models.bedrock import BedrockModel

    # CREDS-01: drop an inherited/expired ``AWS_BEARER_TOKEN_BEDROCK`` BEFORE the boto3
    # session + BedrockModel are constructed, so boto3 uses the profile's SigV4 chain
    # rather than giving a dead bearer token precedence over it. Ordered relative to
    # ``boto3.Session`` exactly as the (since-deleted) ``BedrockChatBackend._init_client``
    # ordered it, and
    # placed AFTER the ``.env`` overlay so a token supplied by a ``.env`` file is covered
    # too.
    #
    # WHY here, in the SHARED factory: previously the guard ran in only two places — the
    # TUI entrypoint (``tui/app.py``) and the raw chat backend. So a token that expired
    # MID-SESSION was never re-neutralised (the TUI had already launched), and any
    # non-TUI ``build_agent`` caller — CLI, the AgentCore runtime entrypoint, the
    # Orchestrator's leads and subagents — had no protection at all. Applying it here
    # makes the protection total: every path that builds a ``BedrockModel`` goes through
    # this function.
    #
    # The call is idempotent, a no-op when the var is absent, and never raises (both
    # asserted in the devdocs unit suite), so running it on every build is safe and
    # cheap — and deliberately NOT wrapped in a bare ``except`` that would hide a real
    # regression in agent construction.
    from .config.bedrock_creds import neutralize_stale_bearer_token

    profile = _effective_profile(cfg)
    neutralize_stale_bearer_token(profile)

    session = boto3.Session(
        profile_name=profile,
        region_name=resolve_aws_region(cfg.bedrock.region, profile_name=profile),
    )
    boto_cfg = _bedrock_boto_config(cfg)
    # An explicit ``model_id`` override (e.g. the TUI chat tier) wins over the
    # per-role tier; else resolve the role's configured model.
    model_id = model_id or cfg.models.for_role(role)  # type: ignore[arg-type]
    # Filter model-specific sampling and cache fields at the shared native boundary.
    is_opus_4x = "opus-4" in model_id
    # User directive (2026-06-04): "1 million context endpoints on Bedrock for
    # all: sonnet + opus." Attach the 1M-context beta to EVERY Sonnet 4.x and
    # Opus 4.x call (not just Opus 4.7 as before). Haiku is excluded — it has no
    # 1M-context profile.
    wants_1m_context = ("sonnet-4" in model_id) or is_opus_4x
    kwargs: dict[str, Any] = dict(
        model_id=model_id,
        boto_session=session,
        boto_client_config=boto_cfg,
        # An explicit per-build cap (the chat tier) wins; ``None`` → the config
        # value, so every swarm lead/subagent build is byte-identical to before.
        max_tokens=max_tokens if max_tokens is not None else cfg.bedrock.max_tokens,
    )
    if supports_explicit_converse_cache(model_id):
        if cfg.bedrock.cache_prompt != "off":
            kwargs["cache_prompt"] = cfg.bedrock.cache_prompt
        if cfg.bedrock.cache_tools != "off":
            kwargs["cache_tools"] = cfg.bedrock.cache_tools
    if supports_temperature(model_id) and cfg.bedrock.temperature is not None:
        kwargs["temperature"] = cfg.bedrock.temperature
    # 1M context beta → all Sonnet 4.x + Opus 4.x.
    if wants_1m_context and cfg.bedrock.anthropic_betas:
        kwargs["additional_request_fields"] = {"anthropic_beta": list(cfg.bedrock.anthropic_betas)}
    # WA-02 (AGENTSEC04/08): bind the Bedrock Guardrail IN-BAND on every model turn when one
    # is configured (AMLAS_GUARDRAIL_ID, published to SSM by GuardrailStack). Returns {} when
    # unconfigured, so offline/dev builds are byte-for-byte unchanged. This is what makes
    # harmful-output / prompt-injection / PII filtering always-on, not just gate-time.
    from .config.guardrails import guardrail_model_kwargs

    kwargs.update(guardrail_model_kwargs())
    return BedrockModel(**kwargs)
