"""Separate GPT reviewer defaults and native Converse compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3
import pytest
import strands.models.bedrock
from test_runtime_deployment_advisory import run_runtime
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import calls, invoke
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

from amlas._build import _build_bedrock_model
from amlas.config.runtime import BedrockConfig, Config, bedrock_invoke_resources, load_config
from amlas.evaluation.judge import BedrockLLMJudge

GPT = "us.openai.gpt-6.1-sol"
SONNET = "global.anthropic.claude-sonnet-5-5"


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("temperature", [None, 0.0])
def test_gpt_reviewer_factory_omits_native_rejected_fields(
    monkeypatch: pytest.MonkeyPatch, explicit: bool, temperature: float | None
) -> None:
    captured: dict[str, Any] = {}

    def strict_model(**kwargs: Any) -> dict[str, Any]:
        # Measured ConverseStream rejects explicit cachePoint and temperature for GPT-6.1 Sol.
        assert kwargs["model_id"] == GPT
        assert "temperature" not in kwargs
        assert "cache_prompt" not in kwargs
        assert "cache_tools" not in kwargs
        assert "anthropic_beta" not in kwargs.get("additional_request_fields", {})
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(boto3, "Session", lambda **kwargs: object())
    monkeypatch.setattr(strands.models.bedrock, "BedrockModel", strict_model)
    monkeypatch.delenv("AMLAS_MODEL_REVIEWER", raising=False)
    monkeypatch.delenv("AGENTICML_MODEL_REVIEWER", raising=False)
    cfg = Config(bedrock=BedrockConfig(region="us-east-1", temperature=temperature))
    _build_bedrock_model(cfg, "reviewer", model_id=GPT if explicit else None)
    assert captured["model_id"] == GPT


def test_sonnet_execution_retains_supported_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: object())
    monkeypatch.setattr(strands.models.bedrock, "BedrockModel", lambda **kwargs: kwargs)
    model = _build_bedrock_model(Config(bedrock=BedrockConfig(region="us-east-1")), "planner")
    assert model["model_id"] == SONNET
    assert model["cache_prompt"] == model["cache_tools"] == "default"


def test_disabled_cache_selectors_are_not_wire_cachepoint_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: object())
    monkeypatch.setattr(strands.models.bedrock, "BedrockModel", lambda **kwargs: kwargs)
    config = Config(bedrock=BedrockConfig(region="us-east-1", cache_prompt="off", cache_tools="off"))
    model = _build_bedrock_model(config, "planner")
    assert "cache_prompt" not in model
    assert "cache_tools" not in model


def test_gpt_judge_uses_visible_text_without_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMLAS_MODEL_REVIEWER", raising=False)
    monkeypatch.delenv("AGENTICML_MODEL_REVIEWER", raising=False)

    class StrictClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["modelId"] == GPT
            assert kwargs["inferenceConfig"] == {"maxTokens": 512}
            return {
                "output": {
                    "message": {
                        "content": [
                            {"reasoningContent": {"reasoningText": {"text": "synthetic private reasoning"}}},
                            {"text": '{"criteria_scores":{"relevance":1},"reasons":["relevant"]}'},
                        ]
                    }
                }
            }

    result = BedrockLLMJudge(client=StrictClient()).evaluate(prompt="p", response="r")
    assert result.passed is True
    assert result.reasons == ["relevant"]


def test_explicit_judge_model_wins_over_reviewer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMLAS_MODEL_REVIEWER", GPT)
    judge = BedrockLLMJudge(model_id=SONNET, client=object())
    assert judge._model_id == SONNET
    assert load_config().models.reviewer == GPT


def test_us_reviewer_profile_has_exact_foundation_resource() -> None:
    resources = bedrock_invoke_resources("111122223333", "us-east-1")
    assert f"arn:aws:bedrock:us-east-1:111122223333:inference-profile/{GPT}" in resources
    assert "arn:aws:bedrock:*::foundation-model/openai.gpt-6.1-sol" in resources
    assert not any("foundation-model/us.openai." in resource for resource in resources)


def test_runtime_policy_admits_gpt_without_new_service_actions(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())
    statement = next(item for item in state["policy"]["Statement"] if "bedrock:InvokeModel" in item["Action"])
    assert set(statement["Action"]) == {"bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"}
    assert f"arn:aws:bedrock:us-east-1:111122223333:inference-profile/{GPT}" in statement["Resource"]
    assert "arn:aws:bedrock:*::foundation-model/openai.gpt-6.1-sol" in statement["Resource"]


def test_unverifiable_reviewer_profile_fails_preflight_before_writes(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    _repo, env, _ = shell_workspace
    env["AMLAS_MODEL_REVIEWER"] = GPT
    tools = Path(env["PATH"].split(":")[0])
    executable = tools / "aws"
    source = executable.read_text().replace(
        ' if args[:2] == ["bedrock", "get-inference-profile"]: print("ACTIVE"); sys.exit(0)',
        ' if args[:2] == ["bedrock", "get-inference-profile"]:\n'
        '  if "us.openai.gpt-6.1-sol" in args: print("AccessDeniedException",file=sys.stderr);sys.exit(254)\n'
        '  print("ACTIVE");sys.exit(0)',
    )
    executable.write_text(source)
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--preflight-only", "--skip-harness")
    assert result.returncode != 0
    assert "permission denied" in result.stdout + result.stderr
    assert not any(
        call["args"][:2] in (["cloudformation", "create-stack"], ["secretsmanager", "create-secret"])
        for call in calls(shell_workspace)
    )
