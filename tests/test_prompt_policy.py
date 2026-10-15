"""Natural prompts and caller-owned policy use the existing chat/capability path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from amlas import api
from amlas.config.settings import Settings
from amlas.control_plane.approvals import ApprovalMode, ApprovalPolicy
from amlas.flows._util import raw
from amlas.tools.orchestration_tools import run_ml_capability
from amlas.tui.chat.models import ChatResponse
from amlas.tui.chat.tool_binding_backend import ToolBindingBackend


def test_prompt_api_delegates_the_exact_input_and_policy_to_normal_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tui.chat import engine

    recorded: dict = {}
    policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["baseline_train_flow"])
    settings = Settings()

    class RecordingEngine:
        def __init__(self, *, settings=None, approval_policy=None):
            recorded.update(settings=settings, policy=approval_policy)

        async def process_message(self, prompt):
            recorded["prompt"] = prompt
            return ChatResponse(message="durable campaign handle", action="job_submitted")

    monkeypatch.setattr(engine, "ChatEngine", RecordingEngine)
    prompt = "Research and execute a licensed coding fine-tuning campaign."
    assert hasattr(api, "run_prompt"), "missing natural-prompt API"
    result = asyncio.run(api.run_prompt(prompt, settings=settings, policy=policy))
    assert result.action == "job_submitted"
    assert recorded == {"settings": settings, "policy": policy, "prompt": prompt}


def test_prompt_api_rejects_empty_input_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tui.chat import engine

    def forbidden_engine(**_):
        raise AssertionError("Empty input must not construct an agent")

    monkeypatch.setattr(engine, "ChatEngine", forbidden_engine)
    assert hasattr(api, "run_prompt"), "missing natural-prompt API"
    with pytest.raises(ValueError, match="prompt"):
        asyncio.run(api.run_prompt("  "))


def test_backend_transports_a_turn_snapshot_outside_model_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    policy = ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["baseline_train_flow"])
    backend = ToolBindingBackend(Settings())
    backend.approval_policy = policy

    class RecordingAgent:
        def __call__(self, prompt, **kwargs):
            captured.update(prompt=prompt, **kwargs)
            return SimpleNamespace(message={"content": [{"text": "done"}]})

    monkeypatch.setattr(ToolBindingBackend, "is_tool_capable", property(lambda _: True))
    monkeypatch.setattr(backend, "build_agent", lambda **_: RecordingAgent())
    result = backend.chat_with_tools("Execute the authorized campaign.")
    assert result.final_text == "done"
    assert "invocation_state" in captured, "caller policy must survive the real agent boundary"
    transported = captured["invocation_state"]["_amlas_approval_policy"]
    assert isinstance(transported, ApprovalPolicy)
    assert transported is not policy
    assert transported.allowlist == policy.allowlist
    policy.allowlist.append("deploy_flow")
    assert transported.allowlist == ["baseline_train_flow"]
    assert "approval" not in captured["prompt"].lower()


@pytest.mark.parametrize("policy", [
    ApprovalPolicy(),
    ApprovalPolicy(mode=ApprovalMode.ALLOWLIST, allowlist=["baseline_train_flow"]),
])
def test_capability_tool_forwards_only_trusted_typed_caller_policy(
    monkeypatch: pytest.MonkeyPatch, policy: ApprovalPolicy,
) -> None:
    from amlas.control_plane import capability

    captured: dict = {}

    def invoke(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"status": "planned"})

    monkeypatch.setattr(capability, "invoke_capability", invoke)
    context = SimpleNamespace(invocation_state={"_amlas_approval_policy": policy})
    assert "tool_context" not in run_ml_capability.tool_spec["inputSchema"]["json"]["properties"]
    result = raw(run_ml_capability)(
        "baseline_train_flow", "{}", plan_only=True, tool_context=context,
    )
    assert result["status"] == "planned"
    assert captured["policy"] is policy
    assert captured["plan_only"] is True


def test_model_authored_policy_cannot_become_caller_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.control_plane import capability

    captured: dict = {}

    def invoke(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"status": "denied"})

    monkeypatch.setattr(capability, "invoke_capability", invoke)
    context = SimpleNamespace(invocation_state={"_amlas_approval_policy": {"mode": "auto"}})
    result = raw(run_ml_capability)(
        "baseline_train_flow", '{"policy": {"mode": "auto"}}',
        plan_only=False, tool_context=context,
    )
    assert result["status"] == "denied"
    assert captured.get("policy") is None
