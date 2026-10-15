"""Sonnet default, override and model-admission contracts."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import calls, invoke
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

from amlas.config import runtime
from amlas.config.chat_model import chat_model_id
from amlas.evaluation.judge import BedrockLLMJudge

SONNET = "global.anthropic.claude-sonnet-5-5"
REVIEWER = "us.openai.gpt-6.1-sol"
ROOT = Path(__file__).resolve().parents[1]
ROLES = ("planner", "reviewer", "specialist", "bounded", "finetune")


@pytest.fixture(autouse=True)
def clean_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("AMLAS_MODEL_", "AGENTICML_MODEL_")) or key in (
            "AMLAS_CHAT_MODEL",
            "AGENTICML_CHAT_MODEL",
        ):
            monkeypatch.delenv(key)


def test_execution_defaults_share_sonnet_and_reviewer_is_separate() -> None:
    assert chat_model_id() == SONNET
    for role in ROLES:
        expected = REVIEWER if role == "reviewer" else SONNET
        assert runtime.ModelConfig().for_role(role) == expected
        assert runtime.load_config().models.for_role(role) == expected


def test_role_and_chat_overrides_preserve_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    for role in ROLES:
        monkeypatch.setenv(f"AGENTICML_MODEL_{role.upper()}", runtime.OPUS_4_7)
        monkeypatch.setenv(f"AMLAS_MODEL_{role.upper()}", runtime.SONNET_4_6)
    monkeypatch.setenv("AGENTICML_CHAT_MODEL", runtime.OPUS_4_7)
    monkeypatch.setenv("AMLAS_CHAT_MODEL", runtime.SONNET_4_6)
    assert chat_model_id() == runtime.SONNET_4_6
    assert all(runtime.load_config().models.for_role(role) == runtime.SONNET_4_6 for role in ROLES)
    monkeypatch.delenv("AMLAS_MODEL_REVIEWER")
    with pytest.warns(DeprecationWarning):
        assert runtime.load_config().models.reviewer == runtime.OPUS_4_7


@pytest.mark.parametrize("template", [".env.example", ".env.tui.example", ".env.agentcore.example"])
def test_shipped_model_templates_use_sonnet_55(template: str) -> None:
    selectors = {
        key: value
        for key, value in (
            line.split("=", 1)
            for line in (ROOT / template).read_text().splitlines()
            if line.startswith(("AMLAS_CHAT_MODEL=", *[f"AMLAS_MODEL_{role.upper()}=" for role in ROLES]))
        )
    }
    assert selectors
    assert all(
        value == (REVIEWER if key == "AMLAS_MODEL_REVIEWER" else SONNET) for key, value in selectors.items()
    )


def test_setup_admits_and_preflights_default_profile(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--preflight-only", "--skip-harness")
    assert result.returncode == 0, result.stdout + result.stderr
    profiles = [
        call["args"][call["args"].index("--inference-profile-identifier") + 1]
        for call in calls(shell_workspace)
        if call["args"][:2] == ["bedrock", "get-inference-profile"]
    ]
    assert profiles == [SONNET, REVIEWER]


def test_setup_rejects_unadmitted_override_before_model_metadata(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    shell_workspace[1]["AMLAS_CHAT_MODEL"] = "global.anthropic.unreviewed-model"
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--preflight-only", "--skip-harness")
    assert result.returncode != 0
    assert "outside the runtime IAM allowlist" in result.stdout + result.stderr
    assert not any(
        call["args"][:2] == ["bedrock", "get-inference-profile"] for call in calls(shell_workspace)
    )


def test_runtime_role_policy_admits_sonnet_55(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())
    policy = state["policy"]
    resources = next(
        statement["Resource"]
        for statement in policy["Statement"]
        if "bedrock:InvokeModel" in statement["Action"]
    )
    assert f"arn:aws:bedrock:us-east-1:111122223333:inference-profile/{SONNET}" in resources
    assert "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5-5" in resources
    assert all(
        f"arn:aws:bedrock:us-east-1:111122223333:inference-profile/{profile}" in resources
        for profile in runtime.BEDROCK_INVOKE_PROFILES
    )


def test_cdk_model_resources_match_runtime_catalog() -> None:
    # Exercise the catalog conversion without constructing the stack's unrelated resources.
    from types import SimpleNamespace

    from infra.cdk_stacks.runtime_stack import RuntimeStack

    resources = RuntimeStack._bedrock_resources(
        SimpleNamespace(region_name="us-east-1", account="111122223333")
    )
    assert f"arn:aws:bedrock:us-east-1:111122223333:inference-profile/{SONNET}" in resources
    assert "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5-5" in resources


def test_runtime_shell_derives_iam_resources_from_shared_catalog(tmp_path: Path) -> None:
    source = (ROOT / "scripts/lib/deploy_agentcore_runtime.sh").read_text()
    start = source.index("BEDROCK_RESOURCES=")
    end = source.index("# NOTE (AgentCorePrimitives", start)
    assignment = source[start:end]
    fixture_home = tmp_path / "home"
    fixture_home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(ROOT / "src"),
        "TMPDIR": str(tmp_path),
        "HOME": str(fixture_home),
    }
    # Same interpreter, closed uv shim: execute the actual shell's stdin program.
    shim = tmp_path / "uv"
    shim.write_text(
        f"#!{sys.executable}\nimport os,sys\nos.execv(sys.executable,[sys.executable,*sys.argv[4:]])\n"
    )
    shim.chmod(0o755)
    env["PATH"] = str(tmp_path) + ":" + env["PATH"]
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
            "ACCT=111122223333\nREGION=us-east-1\n" + assignment + '\nprintf "%s" "$BEDROCK_RESOURCES"',
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    resources = json.loads(result.stdout)
    assert f"arn:aws:bedrock:us-east-1:111122223333:inference-profile/{SONNET}" in resources


def test_sonnet_55_factory_omits_legacy_sampling_and_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3
    import strands.models.bedrock

    from amlas._build import _build_bedrock_model

    captured: dict[str, Any] = {}

    def strict_model(**kwargs: Any) -> dict[str, Any]:
        # Measured native 5.5 contract: sampling and legacy context beta are not used.
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "anthropic_beta" not in kwargs.get("additional_request_fields", {})
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(boto3, "Session", lambda **kwargs: object())
    monkeypatch.setattr(strands.models.bedrock, "BedrockModel", strict_model)
    cfg = runtime.Config(bedrock=runtime.BedrockConfig(region="us-east-1", temperature=0.0))
    _build_bedrock_model(cfg, "planner", model_id=SONNET)
    assert captured["model_id"] == SONNET
    assert captured["max_tokens"] == cfg.bedrock.max_tokens


class StrictJudgeClient:
    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.request = kwargs
        assert "temperature" not in kwargs["inferenceConfig"]
        return {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "synthetic private reasoning"}}},
                        {"text": '{"criteria_scores":{"relevance":1.0},"reasons":["relevant"]}'},
                    ]
                }
            }
        }


def test_managed_judge_uses_reviewer_default_and_text_blocks() -> None:
    client = StrictJudgeClient()
    verdict = BedrockLLMJudge(client=client).evaluate(prompt="p", response="r")
    assert client.request["modelId"] == REVIEWER
    assert verdict.passed is True
    assert verdict.reasons == ["relevant"]


def test_managed_judge_honors_reviewer_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMLAS_MODEL_REVIEWER", runtime.OPUS_4_7)
    client = StrictJudgeClient()
    BedrockLLMJudge(client=client).evaluate(prompt="p", response="r")
    assert client.request["modelId"] == runtime.OPUS_4_7


def test_managed_judge_explicit_legacy_override_retains_sampling() -> None:
    class LegacyClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["modelId"] == runtime.HAIKU_4_5
            assert kwargs["inferenceConfig"]["temperature"] == 0.0
            return {"output": {"message": {"content": [{"text": '{"criteria_scores":{"relevance":1}}'}]}}}

    assert (
        BedrockLLMJudge(model_id=runtime.HAIKU_4_5, client=LegacyClient())
        .evaluate(prompt="p", response="r")
        .passed
    )


def test_role_dataclass_and_env_loader_cannot_drift() -> None:
    assert dataclasses.asdict(runtime.load_config().models) == dataclasses.asdict(runtime.ModelConfig())
