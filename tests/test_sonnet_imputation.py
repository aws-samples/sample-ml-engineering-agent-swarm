"""Direct Bedrock imputation preserves the shared model contract."""

from __future__ import annotations

from typing import Any

import boto3
import pytest

from amlas.config.runtime import HAIKU_4_5
from amlas.tools.imputation.llm_extrapolate import _default_bedrock_call


@pytest.mark.parametrize("legacy", [False, True])
def test_imputation_uses_bounded_config_and_supported_sampling(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    model = HAIKU_4_5 if legacy else "global.anthropic.claude-sonnet-5-5"
    monkeypatch.setenv("AMLAS_MODEL_BOUNDED", model)
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    class StrictClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            # Native Sonnet 5.5 rejects temperature. Legacy Haiku accepts it.
            assert kwargs["modelId"] == model
            if legacy:
                assert kwargs["inferenceConfig"]["temperature"] == 0.2
            else:
                assert "temperature" not in kwargs["inferenceConfig"]
            return {
                "output": {
                    "message": {
                        "content": [
                            {"reasoningContent": {"reasoningText": {"text": "synthetic private reasoning"}}},
                            {"text": '{"values":[1],"confidence":[0.9]}'},
                        ]
                    }
                }
            }

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: StrictClient())
    assert _default_bedrock_call("synthetic prompt") == '{"values":[1],"confidence":[0.9]}'
