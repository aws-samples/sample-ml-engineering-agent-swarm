"""HarnessSpec — the offline configuration object for one AgentCore harness.

ADR-0007 phase 1 moves the leaf Leads off the in-process Strands loop and onto a managed
harness. A harness is **configured**, not constructed: there is no ``Agent`` to inspect and
no ``model=`` seam to inject. Everything that used to be an assertion about a built
``Agent`` therefore has to become an assertion about this object instead, so it is
deliberately pure — constructing one makes no AWS call, opens no socket, and builds no
model (T-HARN-04.1, T-HARN-16.4).

The spec derives from :class:`amlas.leads.declarations.LeadDeclaration`, which is the SAME
object the local-construction peer consumes. That is what makes drift impossible
(T-HARN-04.5) rather than merely unlikely.

Three findings from the T-HARN-01 probe are encoded here as invariants, because each is a
silent failure mode rather than a visible one:

* **``allowedTools`` uses ``@*/<tool>`` and never ``["*"]``** (T-HARN-05.7/.8). A plain tool
  name matches builtins ONLY, so ``allowedTools=["my_tool"]`` matches nothing and silently
  leaves the agent with zero tools. ``*`` admits our tool but also makes the in-VM ``shell``
  builtin LLM-reachable, which ADR-0002 forbids. ``@*/<tool>`` — documented only as a glob
  across MCP servers — was measured to admit an ``inline_function`` while excluding the
  builtins. That behaviour is UNDOCUMENTED for this case, hence the regression tests.
* **Memory is create-time only** (T-HARN-06.2). ``UpdateHarness`` rejects the
  ``CreateHarness`` memory shape, so :meth:`HarnessSpec.to_update_payload` must not emit it.
* **Every tool is an ``inline_function``** (T-HARN-18). No Gateway exists, so the harness
  pauses and hands each call back to our process — which is also the seam the approval gate
  needs, since a server-side loop must never be able to self-approve a gated action.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args, runtime_checkable

from ..config.guardrails import converse_guardrail_config
from ..config.runtime import Config, load_config
from ..prompts import load_prompt, prompt_version


@runtime_checkable
class HarnessableDeclaration(Protocol):
    """What a spec needs from a declaration — satisfied by BOTH declaration types.

    :class:`~amlas.leads.declarations.LeadDeclaration` and
    :class:`~amlas.subagents.declarations.SubagentDeclaration` independently converged on this
    interface, so one spec builder serves both layers. Stating it as a Protocol makes that
    compatibility explicit and type-checked rather than a happy accident that a later edit to
    either side could quietly break.
    """

    # Read-only properties, not settable attributes: both implementations are FROZEN
    # dataclasses, so declaring these as variables would make the Protocol unsatisfiable.
    @property
    def key(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def declared_tool_names(self) -> tuple[str, ...]: ...

    @property
    def tool_contracts(self) -> tuple[tuple[str, str, dict[str, Any]], ...]: ...

    def system_prompt(self) -> str: ...

TruncationStrategy = Literal["sliding_window", "summarization", "none"]
MemoryStrategy = Literal["SEMANTIC", "SUMMARIZATION", "USER_PREFERENCE", "EPISODIC"]

#: The ONLY ``allowedTools`` form verified to admit an ``inline_function`` while leaving the
#: ``shell`` / ``file_operations`` builtins unreachable (T-HARN-01 Part C).
ALLOWED_TOOL_PREFIX = "@*/"

#: Long-term memory strategies bound to every harness-hosted lead (T-HARN-06.1). Replaces
#: the ``MEMORY_ID`` / ``MemoryHookProvider`` plumbing that is declared in CDK but never set
#: in any deployed environment, so leads run today with zero long-term memory.
DEFAULT_MEMORY_STRATEGIES: tuple[MemoryStrategy, ...] = (
    "SEMANTIC",
    "SUMMARIZATION",
    "USER_PREFERENCE",
)

#: Versioned prompt file backing ``summarizationSystemPrompt`` (T-HARN-08.3).
SUMMARIZATION_PROMPT_SOURCE = "harness/summarization"

DEFAULT_SUMMARY_RATIO = 0.3
DEFAULT_PRESERVE_RECENT_MESSAGES = 10

#: The tool type that would make a harness a compute path — ADR-0002's hard line.
CODE_INTERPRETER_TOOL_TYPE = "agentcore_code_interpreter"

#: Grading / evaluator tools that must never be reachable from a harness-hosted lead
#: (ADR-0003: the deterministic Arbiter grade is never delegated). ``grade_submission`` is a
#: real ``@tool`` in :mod:`amlas.tools.backend_neutral`; the rest are the shapes a managed
#: evaluator would plausibly arrive under, guarded ahead of time.
FORBIDDEN_EVALUATOR_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "grade_submission",
        "grade",
        "managed_evaluator",
        "llm_judge",
        "judge",
    }
)


class HarnessSpecError(ValueError):
    """A spec that would produce an unsafe or invalid harness — raised BEFORE any AWS call."""


def _flatten_for_env(text: str) -> str:
    """Collapse whitespace and strip control characters, for a value bound as an env var.

    See :attr:`SummarizationConfig.wire_prompt` for why this is required rather than tidy.
    ``str.split()`` splits on every whitespace run including newlines and tabs; the explicit
    filter then drops any remaining control character (e.g. ``0x7F``) that is not whitespace.
    """
    collapsed = " ".join(str(text).split())
    return "".join(ch for ch in collapsed if ord(ch) >= 0x20 and ord(ch) != 0x7F)


@dataclass(frozen=True)
class SummarizationConfig:
    """``truncation.summarizationConfiguration`` — all three fields verified to persist.

    Attributes:
        summary_ratio: Fraction of the window the summary may occupy.
        preserve_recent_messages: Recent turns kept verbatim, never summarized.
        summarization_system_prompt: The AMLAS summarization prompt body.
    """

    summary_ratio: float = DEFAULT_SUMMARY_RATIO
    preserve_recent_messages: int = DEFAULT_PRESERVE_RECENT_MESSAGES
    summarization_system_prompt: str = ""

    @classmethod
    def load(cls, **overrides: Any) -> SummarizationConfig:
        """Build the default config, reading the prompt body off disk (no AWS)."""
        kwargs: dict[str, Any] = {
            "summarization_system_prompt": load_prompt(SUMMARIZATION_PROMPT_SOURCE)
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def prompt_version(self) -> str:
        """Content-hash version of the SOURCE prompt body (WA-09 prompt-catalog seam).

        Hashes the source, not the flattened wire form, so the version tracks the file a human
        edits and reviews.
        """
        return prompt_version(self.summarization_system_prompt)

    @property
    def wire_prompt(self) -> str:
        """The prompt as the service will accept it — flattened to a single line.

        **Measured constraint, not a style choice.** The service implements
        ``summarizationSystemPrompt`` by injecting it into the harness container as the
        environment variable ``AWS_TRUNCATION_SUMMARIZATION_SYSTEM_PROMPT``, and env var values
        may not contain control characters. Sending multi-line markdown produced:

            Environment variable value contains invalid control characters (0x00-0x1F, 0x7F).
            Key: 'AWS_TRUNCATION_SUMMARIZATION_SYSTEM_PROMPT'

        and the harness reached ``CREATE_FAILED``. Nothing offline could catch this: the CFN
        schema types the field as a plain ``string``, and the API accepts the payload — the
        failure happens later, during provisioning.

        So the source file stays readable, diffable markdown (T-HARN-08.3) and normalisation
        happens HERE, at the wire boundary. Every run of control characters and whitespace
        collapses to one space, which preserves the sentence content the prompt relies on.
        """
        return _flatten_for_env(self.summarization_system_prompt)

    def to_payload(self) -> dict[str, Any]:
        return {
            "summaryRatio": self.summary_ratio,
            "preserveRecentMessages": self.preserve_recent_messages,
            "summarizationSystemPrompt": self.wire_prompt,
        }


@dataclass(frozen=True)
class HarnessSpec:
    """One harness's full configuration — pure, offline, and validated at construction.

    Attributes:
        name: Lead identity (the registry key, e.g. ``"research_lead"``).
        role: Model tier selector, as accepted by ``ModelConfig.for_role``.
        system_prompt: Byte-identical to the local peer's ``Agent.system_prompt``.
        declared_tool_names: Tool names, in order — the offline stand-in for
            ``Agent.tool_names``.
        memory_strategies: Managed long-term memory strategies (create-time only).
        truncation_strategy: Server-side context policy.
        summarization: Summarization parameters, used when the strategy is summarization.
        guardrail: The Converse-shaped guardrail block, or ``{}`` when unconfigured.
    """

    name: str
    role: str
    system_prompt: str
    declared_tool_names: tuple[str, ...]
    tool_contracts: tuple[tuple[str, str, dict[str, Any]], ...] = ()
    memory_strategies: tuple[MemoryStrategy, ...] = DEFAULT_MEMORY_STRATEGIES
    truncation_strategy: TruncationStrategy = "summarization"
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig.load)
    guardrail: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # ── validation — every failure mode here is silent if left unchecked ──────
    def _validate(self) -> None:
        if self.truncation_strategy not in get_args(TruncationStrategy):
            raise HarnessSpecError(
                f"{self.name}: invalid truncation strategy {self.truncation_strategy!r}; "
                f"expected one of {list(get_args(TruncationStrategy))}"
            )
        unknown = [s for s in self.memory_strategies if s not in get_args(MemoryStrategy)]
        if unknown:
            raise HarnessSpecError(
                f"{self.name}: unknown memory strategy {unknown}; "
                f"expected from {list(get_args(MemoryStrategy))}"
            )
        if CODE_INTERPRETER_TOOL_TYPE in self.declared_tool_names:
            raise HarnessSpecError(
                f"{self.name}: declares {CODE_INTERPRETER_TOOL_TYPE!r} — ADR-0002 forbids a "
                "harness from being a compute path"
            )
        if "*" in self.declared_tool_names:
            raise HarnessSpecError(
                f"{self.name}: '*' is not a tool name; allowedTools=['*'] would make the "
                "in-VM shell builtin LLM-reachable (ADR-0002)"
            )
        graders = sorted(set(self.declared_tool_names) & FORBIDDEN_EVALUATOR_TOOL_NAMES)
        if graders:
            raise HarnessSpecError(
                f"{self.name}: declares grading tool(s) {graders} — ADR-0003 keeps the "
                "deterministic Arbiter grade in the orchestration process, never delegated"
            )

    # ── derived views ────────────────────────────────────────────────────────
    @property
    def tool_names(self) -> list[str]:
        """Alias of :attr:`declared_tool_names` for the ADR-0002 guardrail helpers.

        :func:`amlas.subagents.compute_placement.agent_tool_names` probes ``tool_names``
        first. Without this alias it would find no attribute it recognises and return ``[]``,
        so ``compute_tools_in(spec)`` would pass VACUOUSLY even for a spec carrying
        ``train_model``. Exposing the real names makes that check bite.
        """
        return list(self.declared_tool_names)

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """``allowedTools`` in the one verified inline-function form (T-HARN-05.7)."""
        return tuple(f"{ALLOWED_TOOL_PREFIX}{name}" for name in self.declared_tool_names)

    @property
    def inline_function_tools(self) -> tuple[str, ...]:
        """Every tool, since without a Gateway a harness reaches ours no other way."""
        return self.declared_tool_names

    @property
    def approval_gated_tools(self) -> tuple[str, ...]:
        """Tools the Approval_Gate must evaluate before the client resumes (T-HARN-18.1).

        Derived from the REAL policy — :func:`amlas.control_plane.approvals.classify_action`
        — rather than a copied list, so a reclassification cannot leave this stale. Anything
        that is not a cheap read can gate (bounded compute gates once it exceeds its cap, and
        the cost estimate is not known at spec time), so cheap reads are the only exclusion.
        """
        from ..control_plane.approvals import ActionClass, classify_action

        return tuple(
            name
            for name in self.declared_tool_names
            if classify_action(name) is not ActionClass.CHEAP_READ
        )

    # ── AWS payloads ─────────────────────────────────────────────────────────
    def _tool_payload(self) -> list[dict[str, Any]]:
        """``tools`` — every entry an ``inline_function`` with its REQUIRED contract.

        ``inlineFunction`` requires BOTH ``description`` and ``inputSchema`` (verified against
        the ``AWS::BedrockAgentCore::Harness`` schema); a bare ``{}`` is rejected.
        """
        by_name = {name: (desc, schema) for name, desc, schema in self.tool_contracts}
        payload: list[dict[str, Any]] = []
        for name in self.declared_tool_names:
            description, schema = by_name.get(name, ("", {}))
            payload.append(
                {
                    "type": "inline_function",
                    "name": name,
                    "config": {
                        "inlineFunction": {
                            "description": description,
                            "inputSchema": schema,
                        }
                    },
                }
            )
        return payload

    def _truncation_payload(self) -> dict[str, Any]:
        """``truncation`` — the strategy block nests under ``config``, NOT as a sibling.

        Real shape: ``{"strategy": ..., "config": {"summarization": {...}}}``.
        """
        payload: dict[str, Any] = {"strategy": self.truncation_strategy}
        if self.truncation_strategy == "summarization":
            payload["config"] = {"summarization": self.summarization.to_payload()}
        return payload

    def model_payload(self, *, model_id: str) -> dict[str, Any]:
        """``model`` — and the ONLY place a guardrail can live.

        The harness API exposes **no** guardrail field: neither
        ``AWS::BedrockAgentCore::Harness`` nor ``CreateHarness`` has one, and
        ``bedrockModelConfig`` is ``additionalProperties: false``. The T-HARN-02 probe measured
        that ``additionalParams`` is spliced into the **Converse request itself** (its rejection
        message enumerates the Converse parameters), so a nested ``guardrailConfig`` there is
        genuinely enforced — confirmed by an intervention returning an unforgeable marker.

        The block is the one :func:`converse_guardrail_config` already produces, so this reuses
        the single source of truth rather than re-reading the environment.
        """
        cfg: dict[str, Any] = {"modelId": model_id, "apiFormat": "converse_stream"}
        if self.guardrail:
            cfg["additionalParams"] = {"guardrailConfig": dict(self.guardrail)}
        return {"bedrockModelConfig": cfg}

    def to_create_payload(
        self, *, execution_role_arn: str, model_id: str | None = None
    ) -> dict[str, Any]:
        """The ``CreateHarness`` request shape.

        Memory appears ONLY here: ``UpdateHarness`` rejects this shape, so a memory change
        means replacing the harness (T-HARN-06.2 / T-HARN-12.1).
        """
        return {
            "harnessName": self.name,
            "executionRoleArn": execution_role_arn,
            "model": self.model_payload(model_id=model_id or self.resolved_model_id()),
            "systemPrompt": [{"text": self.system_prompt}],
            "tools": self._tool_payload(),
            "allowedTools": list(self.allowed_tools),
            "truncation": self._truncation_payload(),
            "memory": {
                "managedMemoryConfiguration": {"strategies": list(self.memory_strategies)}
            },
        }

    def to_update_payload(self, *, model_id: str | None = None) -> dict[str, Any]:
        """The ``UpdateHarness`` request shape — deliberately carries NO memory key."""
        return {
            "model": self.model_payload(model_id=model_id or self.resolved_model_id()),
            "systemPrompt": [{"text": self.system_prompt}],
            "tools": self._tool_payload(),
            "allowedTools": list(self.allowed_tools),
            "truncation": self._truncation_payload(),
        }

    def resolved_model_id(self, *, config: Config | None = None) -> str:
        """The model id for this spec's role, from the same map ``_build`` reads."""
        cfg = config or load_config()
        return str(cfg.models.for_role(self.role))  # type: ignore[arg-type]

    def to_cfn_properties(
        self, *, execution_role_arn: str, model_id: str | None = None
    ) -> dict[str, Any]:
        """The ``AWS::BedrockAgentCore::Harness`` L1 properties (CloudFormation PascalCase).

        Deliberately written OUT IN FULL rather than machine-converted from
        :meth:`to_create_payload`. A recursive camelCase→PascalCase pass looks tempting and is
        **wrong**: the contents of ``AdditionalParams`` are passed to the model provider
        UNCHANGED, so ``guardrailConfig`` / ``guardrailIdentifier`` must stay camelCase even
        here. A blind converter would rename them and silently unbind the guardrail — the exact
        failure mode T-HARN-02 exists to prevent, and one no synth test would catch because
        CloudFormation accepts an opaque ``AdditionalParams`` map.
        """
        resolved_model = model_id or self.resolved_model_id()
        model_cfg: dict[str, Any] = {
            "ModelId": resolved_model,
            "ApiFormat": "converse_stream",
        }
        if self.guardrail:
            # camelCase INSIDE AdditionalParams — passed through to Converse verbatim.
            model_cfg["AdditionalParams"] = {"guardrailConfig": dict(self.guardrail)}

        tools: list[dict[str, Any]] = []
        by_name = {name: (desc, schema) for name, desc, schema in self.tool_contracts}
        for name in self.declared_tool_names:
            description, schema = by_name.get(name, ("", {}))
            tools.append(
                {
                    "Type": "inline_function",
                    "Name": name,
                    "Config": {
                        "InlineFunction": {
                            "Description": description,
                            "InputSchema": schema,
                        }
                    },
                }
            )

        truncation: dict[str, Any] = {"Strategy": self.truncation_strategy}
        if self.truncation_strategy == "summarization":
            truncation["Config"] = {
                "Summarization": {
                    "SummaryRatio": self.summarization.summary_ratio,
                    "PreserveRecentMessages": self.summarization.preserve_recent_messages,
                    # Flattened: the service binds this as an env var, which forbids
                    # control characters. See SummarizationConfig.wire_prompt.
                    "SummarizationSystemPrompt": self.summarization.wire_prompt,
                }
            }

        return {
            "HarnessName": self.name,
            "ExecutionRoleArn": execution_role_arn,
            "Model": {"BedrockModelConfig": model_cfg},
            "SystemPrompt": [{"Text": self.system_prompt}],
            "Tools": tools,
            "AllowedTools": list(self.allowed_tools),
            "Truncation": truncation,
            "Memory": {
                "ManagedMemoryConfiguration": {"Strategies": list(self.memory_strategies)}
            },
        }


def harness_spec_from_declaration(
    declaration: HarnessableDeclaration,
    *,
    config: Config | None = None,
) -> HarnessSpec:
    """Derive a spec from the declaration the local peer also consumes (T-HARN-04.5).

    Accepts a Lead OR a sub-agent declaration. ADR-0007 established the true harness LEAVES
    are the sub-agents, so both layers need specs; both satisfy
    :class:`HarnessableDeclaration`, so neither needs a bespoke builder.

    ``config`` is accepted for symmetry with the builder path and to let a caller pin the
    role tier resolution; the spec itself stores the role name, not a resolved model id, so
    per-invoke model overrides stay the client's job (T-HARN-09).
    """
    _ = config or None
    return HarnessSpec(
        name=declaration.key,
        role=declaration.role,
        system_prompt=declaration.system_prompt(),
        declared_tool_names=declaration.declared_tool_names,
        tool_contracts=declaration.tool_contracts,
        guardrail=converse_guardrail_config() or {},
    )


def harness_spec_for_subagent(key: str, *, config: Config | None = None) -> HarnessSpec:
    """Build the :class:`HarnessSpec` for sub-agent ``key`` — the true harness leaf.

    Derivable for ANY sub-agent, whether or not one is currently provisioned: deriving is free
    and offline, and phase 1 runs sub-agents in-process as ``inline_function`` calls rather than
    on their own harnesses (see ``HARNESS_HOSTED_SUBAGENTS``).
    """
    from .registry import subagent_harness_declaration

    return harness_spec_from_declaration(
        subagent_harness_declaration(key), config=config or _maybe_config()
    )


def harness_spec_for_lead(name: str, *, config: Config | None = None) -> HarnessSpec:
    """Build the :class:`HarnessSpec` for the harness-hosted lead ``name``.

    Raises:
        KeyError: If ``name`` is not a harness-hosted lead — notably ``"orchestrator"``,
            which stays code-defined on AgentCore Runtime (T-HARN-15).
    """
    from .registry import harness_declaration

    return harness_spec_from_declaration(
        harness_declaration(name), config=config or _maybe_config()
    )


def _maybe_config() -> Config | None:
    """Load config when it is available, but never make it a hard requirement.

    The spec does not need config today; keeping this soft means a spec still builds in the
    credential-free suite even if config resolution starts requiring more of the world.
    """
    try:
        return load_config()
    except Exception:
        return None


__all__ = [
    "ALLOWED_TOOL_PREFIX",
    "CODE_INTERPRETER_TOOL_TYPE",
    "DEFAULT_MEMORY_STRATEGIES",
    "DEFAULT_PRESERVE_RECENT_MESSAGES",
    "DEFAULT_SUMMARY_RATIO",
    "FORBIDDEN_EVALUATOR_TOOL_NAMES",
    "SUMMARIZATION_PROMPT_SOURCE",
    "HarnessSpec",
    "HarnessSpecError",
    "HarnessableDeclaration",
    "MemoryStrategy",
    "SummarizationConfig",
    "TruncationStrategy",
    "harness_spec_for_lead",
    "harness_spec_for_subagent",
    "harness_spec_from_declaration",
]
