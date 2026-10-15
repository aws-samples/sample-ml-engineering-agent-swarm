"""Lead_Factory — route a Lead to its harness, or build it locally (T-HARN-04.3/.5).

This is the seam ADR-0007 phase 1 specified and did not land. Everything needed to run a
Lead server-side already existed — :func:`~amlas.harness.spec.harness_spec_for_lead`
configures it, ``HarnessStack`` provisions it, :func:`~amlas.harness.client.invoke_harness`
calls it, :func:`~amlas.harness.approvals.resolve_tool_calls` gates the paused calls — but
**nothing chose between the two paths**, so no production code path reached any of it. A
capability that nothing invokes is dead code, which is precisely the state the migration set
out to fix in the ``MEMORY_ID`` plumbing.

Two paths, one declaration:

* **Harness** — :func:`run_lead_turn` drives the server-side loop: invoke, execute the
  ``inline_function`` calls the harness pauses on, resume, repeat.
* **Local** — :func:`build_local_lead` delegates to the same ``build_*_lead`` the TUI uses.

Which one runs is a deliberate opt-in, never a guess. See :func:`harness_enabled`.

Contracts this module holds
---------------------------

**Opt-in, and OFF by default.** An unset flag means every Lead builds locally, exactly as
before. A migration that silently reroutes production the moment infrastructure appears is
not reviewable; ``HarnessStack`` being deployed must not by itself change how any Lead runs.

**A missing ARN never silently degrades to local.** :func:`harness_hosted` answers False so a
caller can route safely, but :func:`run_lead_turn` RAISES, naming the SSM key and the env var
it looked for. Falling back would turn a deploy mistake into a silent behaviour change, and
the two paths differ in exactly the ways that matter — managed memory, server-side
truncation, and where the guardrail is bound.

**The executor refuses compute and grading, even if a tool is somehow declared.** Defence in
depth behind the spec's own construction-time refusal: this is the process that would
actually run the call, so it checks ``HEAVY_COMPUTE_TOOL_NAMES`` (ADR-0002) and
``FORBIDDEN_EVALUATOR_TOOL_NAMES`` (ADR-0003) at execution time. A spec built some other way,
or a declaration edited without re-running the placement guard, still cannot execute compute
here.

**Tools come from the declaration, not from a rebuilt Agent.** ``direct_tools`` are unwrapped
with :func:`~amlas.backends.adapters._util.raw_tool` and called directly; a sub-agent is built
lazily, only when the harness actually calls it, from the ``builder`` its own
:class:`~amlas.leads.declarations.SubAgentDeclaration` carries. So the harness path and the
local path execute the same implementations without constructing a Lead ``Agent`` — and
nothing is built for a tool the model never calls.

**The loop is bounded.** ``max_tool_hops`` caps the invoke/resume cycle. A server-side loop
that keeps requesting tools would otherwise bill indefinitely with no local stack to blow.
Exhausting the budget returns the run with ``hops_exhausted`` set rather than raising, so a
caller still gets the transcript and the usage it paid for.

Import-time inert and AWS-free, like every other module here: resolving an ARN reads env
first and only then SSM through the injectable getter the offline suite already fakes.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..backends.adapters._util import raw_tool
from ..config.deployment import resolve_ssm_prefix
from ..config.externalized import SSM_NAMESPACE, MissingRequiredConfigError, SsmGetter
from ..config.externalized import _default_ssm_getter as default_ssm_getter
from ..config.externalized import resolve as resolve_config
from ..subagents.compute_placement import HEAVY_COMPUTE_TOOL_NAMES
from .approvals import ToolExecutor, build_resume_messages, resolve_tool_calls
from .client import HarnessSessions, HarnessToolUse, HarnessTurn, invoke_harness
from .registry import HARNESS_HOSTED_LEADS, harness_declaration, local_peer_builder
from .spec import FORBIDDEN_EVALUATOR_TOOL_NAMES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from strands import Agent

    from ..config.runtime import Config
    from ..control_plane.approvals import ApprovalPolicy

#: Opt-in flag. Unset/empty → every Lead builds locally. ``"all"`` (any case) → every
#: harness-hosted Lead. Otherwise a comma-separated list of lead keys.
HARNESS_LEADS_ENV = "AMLAS_HARNESS_LEADS"

#: Per-lead ARN override, e.g. ``AMLAS_HARNESS_ARN_RESEARCH_LEAD``. Checked before SSM so a
#: developer can point one Lead at a scratch harness without touching Parameter Store.
ARN_ENV_PREFIX = "AMLAS_HARNESS_ARN_"

#: Where ``HarnessStack`` publishes each id. Mirrors ``harness_stack.py``'s
#: ``f"{SSM_PREFIX}/harness/{lead_name.replace('_', '-')}-id"`` — the ONE place the two must
#: agree, so it is stated here once and asserted against the stack by the infra suite.
SSM_HARNESS_PREFIX = f"{SSM_NAMESPACE}/harness"

#: Default cap on invoke → execute → resume cycles for a single logical turn.
DEFAULT_MAX_TOOL_HOPS = 8


class HarnessLeadError(RuntimeError):
    """The harness path was requested for a Lead that cannot currently run on one."""


class HarnessToolRefusedError(RuntimeError):
    """A paused call named a tool this process refuses to execute (ADR-0002 / ADR-0003)."""


def harness_ssm_key(lead_name: str, *, env: Mapping[str, str] | None = None) -> str:
    """The SSM key ``HarnessStack`` publishes this Lead's harness id under."""
    return f"{resolve_ssm_prefix(env=env)}/harness/{lead_name.replace('_', '-')}-id"


def harness_arn_env_var(lead_name: str) -> str:
    """The per-lead ARN override variable name."""
    return f"{ARN_ENV_PREFIX}{lead_name.upper()}"


def harness_enabled(lead_name: str, *, env: Mapping[str, str] | None = None) -> bool:
    """Is the harness path opted in for ``lead_name``?

    OFF unless asked for. ``HarnessStack`` being deployed must not by itself reroute a Lead;
    the cutover is a decision someone makes, not a side effect of infrastructure existing.
    """
    source = os.environ if env is None else env
    raw = (source.get(HARNESS_LEADS_ENV) or "").strip()
    if not raw:
        return False
    if raw.lower() == "all":
        return lead_name in HARNESS_HOSTED_LEADS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return lead_name in wanted and lead_name in HARNESS_HOSTED_LEADS


def harness_arn_for(
    lead_name: str,
    *,
    env: Mapping[str, str] | None = None,
    ssm_getter: SsmGetter | None = None,
    region: str | None = None,
    account_id: str | None = None,
) -> str:
    """Resolve this Lead's harness ARN, or ``""`` when it cannot be resolved.

    Precedence is the repo's usual env → SSM. The env form is a full ARN; the SSM form is the
    harness *id* the stack publishes, which is composed into an ARN here. Returns ``""``
    rather than raising so a caller can ask "is this routable?" without handling an exception;
    :func:`run_lead_turn` is where an unresolvable ARN becomes a loud failure.

    ``ssm_getter`` DEFAULTS to the real reader rather than meaning "skip SSM". Requiring one
    made the SSM branch unreachable from production: ``app.py`` calls :func:`lead_for` without
    a getter, so the seven ids ``HarnessStack`` publishes were never read and opting a Lead in
    via ``AMLAS_HARNESS_LEADS`` alone left it silently local — which is the one failure mode a
    live cutover test cannot see, because the run succeeds either way. Injection is still how
    the suite stays offline; only the default changed. The read is reached ONLY for a Lead that
    is already opted in (:func:`harness_hosted` short-circuits on :func:`harness_enabled`
    first), and ``_default_ssm_getter`` is itself offline-safe — bounded by FIX-24's timeouts
    and returning ``""`` on any failure — so with the flag unset nothing is called, and with no
    credentials the answer is the same ``""`` as before.
    """
    source = os.environ if env is None else env

    override = (source.get(harness_arn_env_var(lead_name)) or "").strip()
    if override:
        return override

    getter = ssm_getter or default_ssm_getter
    eff_region = region or resolve_config("aws_region", ssm_getter=getter)
    harness_id = (getter(harness_ssm_key(lead_name, env=source), eff_region) or "").strip()
    if not harness_id:
        return ""

    # ``aws_account_id`` is a REQUIRED externalized key, so ``resolve`` raises rather than
    # returning "" when it is unset. That fail-loud behaviour is right for a deploy, but it
    # would make this function — and therefore :func:`harness_hosted`, which exists to be a
    # safe yes/no question — raise instead of answering. An account id we cannot determine
    # simply means the ARN is not resolvable, so it is reported the same way as a missing id.
    try:
        eff_region = region or resolve_config("aws_region", ssm_getter=ssm_getter)
        acct = account_id or resolve_config("aws_account_id", ssm_getter=ssm_getter)
    except MissingRequiredConfigError:
        return ""
    if not acct:
        return ""
    return f"arn:aws:bedrock-agentcore:{eff_region}:{acct}:harness/{harness_id}"


def harness_hosted(
    lead_name: str,
    *,
    env: Mapping[str, str] | None = None,
    ssm_getter: SsmGetter | None = None,
    region: str | None = None,
    account_id: str | None = None,
) -> bool:
    """True when ``lead_name`` is opted in AND its ARN resolves — i.e. actually routable."""
    if not harness_enabled(lead_name, env=env):
        return False
    return bool(
        harness_arn_for(
            lead_name,
            env=env,
            ssm_getter=ssm_getter,
            region=region,
            account_id=account_id,
        )
    )


def build_local_lead(lead_name: str, **kwargs: Any) -> Agent:
    """Build ``lead_name`` in-process via its public ``build_*_lead`` factory.

    The equivalence oracle T-HARN-04.3 requires, and the path every Lead still takes unless
    opted in. Delegates rather than reimplements, so there is no second construction path to
    drift.
    """
    return local_peer_builder(lead_name)(**kwargs)


def _refuse_if_forbidden(name: str) -> None:
    if name in HEAVY_COMPUTE_TOOL_NAMES:
        raise HarnessToolRefusedError(
            f"{name!r} is heavy compute (ADR-0002): agents orchestrate, they never run "
            "compute. Dispatch it to SageMaker/Glue/Lambda/Fargate instead of executing it "
            "in the orchestration process."
        )
    if name in FORBIDDEN_EVALUATOR_TOOL_NAMES:
        raise HarnessToolRefusedError(
            f"{name!r} would delegate grading (ADR-0003): the Arbiter grade is deterministic "
            "and never delegated to a model or a managed service."
        )


def default_tool_executor(
    lead_name: str,
    *,
    sub_agent_kwargs: Mapping[str, Any] | None = None,
) -> ToolExecutor:
    """Build the executor for ``lead_name`` from its declaration.

    Direct ``@tool`` objects are unwrapped to their raw callable and invoked with the paused
    call's input. A sub-agent tool builds its sub-agent lazily — only if the model actually
    calls it — and passes the prompt through, mirroring
    ``.as_tool(preserve_context=True)`` binding on the local path.
    """
    decl = harness_declaration(lead_name)
    extra = dict(sub_agent_kwargs or {})

    direct: dict[str, Callable[..., Any]] = {}
    for tool in decl.direct_tools:
        from ..leads.declarations import declared_tool_name

        direct[declared_tool_name(tool)] = raw_tool(tool)

    subs = {sub.tool_name: sub for sub in decl.sub_agents}

    def _execute(call: HarnessToolUse) -> Any:
        _refuse_if_forbidden(call.name)

        if call.name in direct:
            return direct[call.name](**dict(call.input))

        sub = subs.get(call.name)
        if sub is None:
            raise HarnessToolRefusedError(
                f"{call.name!r} is not declared by {lead_name!r}; declared: "
                f"{list(decl.declared_tool_names)}"
            )
        agent = sub.builder(**extra) if extra else sub.builder()
        payload = dict(call.input)
        prompt = payload.get("prompt") or payload.get("input") or payload.get("query") or ""
        return str(agent(prompt if isinstance(prompt, str) else str(payload)))

    return _execute


@dataclass
class LeadRun:
    """The outcome of one logical Lead turn on the harness — possibly several hops.

    Attributes:
        text: Assistant text from the FINAL hop, which is the answer.
        turns: Every :class:`~amlas.harness.client.HarnessTurn`, in order.
        tool_calls: Names executed, in order, across all hops.
        decisions: Raw approval decisions, the audit trail.
        hops: Invoke/resume cycles performed.
        hops_exhausted: True if the cap stopped the loop while tools were still pending.
        session_id: The ``runtimeSessionId`` used.
        error: Terminal error event from the final hop, or ``None``.
    """

    text: str = ""
    turns: list[HarnessTurn] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    hops: int = 0
    hops_exhausted: bool = False
    session_id: str = ""
    error: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def usage(self) -> dict[str, Any]:
        """Summed token usage across hops — a paused turn bills every hop, not just the last."""
        total: dict[str, Any] = {}
        for turn in self.turns:
            for key, value in (turn.usage or {}).items():
                if isinstance(value, int | float):
                    total[key] = total.get(key, 0) + value
        return total


class HarnessLead:
    """A harness-backed Lead that satisfies the Orchestrator's ``Callable[[str], Any]``.

    :meth:`~amlas.orchestrator.Orchestrator._call_lead` asks nothing of a Lead except that it
    be callable with one string and return something readable, so a harness-hosted Lead needs
    no Orchestrator change beyond being handed one of these instead of an ``Agent``. Keeping
    the adapter this thin is deliberate: the Orchestrator owns the retreat loop, the
    deterministic grade (ADR-0003) and the streaming contract, and none of that should learn
    where a Lead happens to run.

    A terminal error RAISES rather than returning its text. The Orchestrator distinguishes a
    Lead that failed from a Lead that answered badly, and handing back an error string as if
    it were an answer would send a fabricated response into grading.

    ``last_run`` keeps the full :class:`LeadRun` from the most recent call — hops, per-hop
    usage, approval decisions — because the Orchestrator's return channel is a single string
    and that detail is otherwise lost.
    """

    def __init__(
        self,
        lead_name: str,
        *,
        run_id: str = "",
        actor_id: str = "",
        sessions: HarnessSessions | None = None,
        **turn_kwargs: Any,
    ) -> None:
        self.lead_name = lead_name
        self.run_id = run_id
        self.actor_id = actor_id
        # ONE ledger per adapter, so every call within a run reuses the session and pays the
        # ~55s microVM cold start once rather than per turn.
        self.sessions = sessions if sessions is not None else HarnessSessions()
        self._turn_kwargs = turn_kwargs
        self.last_run: LeadRun | None = None

    def __call__(self, prompt: str) -> str:
        run = run_lead_turn(
            self.lead_name,
            prompt,
            run_id=self.run_id,
            actor_id=self.actor_id,
            sessions=self.sessions,
            **self._turn_kwargs,
        )
        self.last_run = run
        if run.is_error:  # nosemgrep: is-function-without-parentheses
            raise HarnessLeadError(f"{self.lead_name} harness turn failed: {run.error}")
        return run.text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"HarnessLead({self.lead_name!r}, run_id={self.run_id!r})"


def lead_for(
    lead_name: str,
    *,
    run_id: str = "",
    actor_id: str = "",
    env: Mapping[str, str] | None = None,
    ssm_getter: SsmGetter | None = None,
    region: str | None = None,
    account_id: str | None = None,
    sessions: HarnessSessions | None = None,
    **kwargs: Any,
) -> Any:
    """The routing entry point: a :class:`HarnessLead`, or the locally-built Agent.

    ``region`` / ``account_id`` are named explicitly rather than left to ``**kwargs``: they are
    ARN-resolution inputs, and ``**kwargs`` is forwarded to the LOCAL builder, which would
    reject them with a confusing ``TypeError`` on the fallback path.

    The ONE place the decision is made, so no caller repeats the flag check and the two paths
    cannot diverge on how they are selected.

    Note the deliberate asymmetry with :func:`run_lead_turn`: :func:`harness_hosted` requires
    the flag AND a resolvable ARN, so an opted-in Lead whose ARN is missing builds locally
    here, while `run_lead_turn` raises. A caller asking "give me the Lead" should get a
    working Lead; a caller explicitly asking for a harness turn should be told the harness is
    missing.
    """
    if harness_hosted(
        lead_name, env=env, ssm_getter=ssm_getter, region=region, account_id=account_id
    ):
        return HarnessLead(
            lead_name,
            run_id=run_id,
            actor_id=actor_id,
            sessions=sessions,
            env=env,
            ssm_getter=ssm_getter,
            **kwargs,
        )
    return build_local_lead(lead_name, **kwargs)


def run_lead_turn(
    lead_name: str,
    prompt: str,
    *,
    harness_arn: str | None = None,
    executor: ToolExecutor | None = None,
    run_id: str = "",
    actor_id: str = "",
    sessions: HarnessSessions | None = None,
    policy: ApprovalPolicy | None = None,
    cost_estimates: dict[str, float] | None = None,
    max_tool_hops: int = DEFAULT_MAX_TOOL_HOPS,
    sub_agent_kwargs: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    ssm_getter: SsmGetter | None = None,
    client: Any | None = None,
    config: Config | None = None,
    error_log: Any | None = None,
    sink: Callable[[dict[str, Any]], None] | None = None,
) -> LeadRun:
    """Run one logical turn for ``lead_name`` on its harness, gating every paused call.

    Raises:
        HarnessLeadError: If no ARN resolves — deliberately loud, never a silent fall back to
            the local path.
    """
    arn = harness_arn or harness_arn_for(lead_name, env=env, ssm_getter=ssm_getter)
    if not arn:
        raise HarnessLeadError(
            f"no harness ARN for {lead_name!r}: set {harness_arn_env_var(lead_name)} or "
            f"publish SSM {harness_ssm_key(lead_name)} (HarnessStack does the latter). "
            "Refusing to fall back to local construction, because the two paths differ in "
            "managed memory, truncation and guardrail binding."
        )

    run_executor = executor or default_tool_executor(
        lead_name, sub_agent_kwargs=sub_agent_kwargs
    )
    spec_role = harness_declaration(lead_name).role

    result = LeadRun()
    # ``list[dict]`` rather than the looser ``Sequence[Mapping]`` ``invoke_harness`` accepts,
    # because ``build_resume_messages`` needs the narrower type to append to it.
    messages: list[dict[str, Any]] | None = None

    for hop in range(max_tool_hops):
        turn = invoke_harness(
            harness_arn=arn,
            prompt=prompt if messages is None else None,
            messages=messages,
            lead_name=lead_name,
            role=spec_role,
            run_id=run_id,
            actor_id=actor_id,
            sessions=sessions,
            sink=sink,
            client=client,
            config=config,
            error_log=error_log,
        )
        result.turns.append(turn)
        result.hops = hop + 1
        result.session_id = turn.session_id
        result.text = turn.text
        result.error = turn.error

        if turn.is_error or not turn.tool_uses:  # nosemgrep: is-function-without-parentheses
            return result

        tool_results = resolve_tool_calls(
            turn.tool_uses,
            executor=run_executor,
            policy=policy,
            cost_estimates=cost_estimates,
            on_decision=result.decisions.append,
        )
        result.tool_calls.extend(call.name for call in turn.tool_uses)
        messages = build_resume_messages(
            prompt=prompt,
            tool_uses=turn.tool_uses,
            tool_results=tool_results,
            prior_messages=messages,
        )

    result.hops_exhausted = bool(result.turns and result.turns[-1].tool_uses)
    return result


__all__ = [
    "ARN_ENV_PREFIX",
    "DEFAULT_MAX_TOOL_HOPS",
    "HARNESS_LEADS_ENV",
    "SSM_HARNESS_PREFIX",
    "HarnessLead",
    "HarnessLeadError",
    "HarnessToolRefusedError",
    "LeadRun",
    "build_local_lead",
    "default_tool_executor",
    "harness_arn_env_var",
    "harness_arn_for",
    "harness_enabled",
    "harness_hosted",
    "harness_ssm_key",
    "lead_for",
    "run_lead_turn",
]
