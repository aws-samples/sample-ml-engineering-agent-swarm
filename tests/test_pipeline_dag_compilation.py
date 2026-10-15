"""The default flow compiler produces an inspectable DAG without AWS access."""
from pathlib import Path

import pytest

from amlas.backends.adapters.orchestration_pipeline import SageMakerPipelineOrchestrationBackend


def test_default_pipeline_template_compiles_without_aws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AMLAS_AWS_REGION",
        "AMLAS_AWS_PROFILE",
        "AGENTICML_AWS_REGION",
        "AGENTICML_AWS_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "empty-aws-config"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    class NoSageMakerCalls:
        def __getattr__(self, name: str) -> None:
            raise AssertionError(f"Offline DAG compilation attempted a managed client call: {name}")

    backend = SageMakerPipelineOrchestrationBackend(sm_client=NoSageMakerCalls())
    try:
        dag = backend.compile_flow(
            flow_id="baseline_train_flow",
            params={"metric_name": "f1_macro", "threshold": 0.8},
        )
    except ValueError as error:
        raise AssertionError("The default pipeline template must compile a complete DAG") from error

    assert dag.flow_id == "baseline_train_flow"
    assert dag.step_names == (
        "Preprocess",
        "Train",
        "QualityCheck",
        "ClarifyCheck",
        "ArbiterGate",
        "RegisterModel",
    )
    assert dag.steps[0].step_type == "Processing"
    assert dag.steps[0].depends_on == ()
    assert dag.condition_step == "ArbiterGate"
    gate = dag.arbiter_step
    assert gate is not None
    assert gate.depends_on == ("QualityCheck", "ClarifyCheck")
    assert gate.config["condition"] == "GreaterThanOrEqualTo"
    assert gate.config["metric"] == "f1_macro"
    assert gate.config["threshold"] == 0.8
    assert gate.config["arbiter_delegated"] is False
    assert dag.steps[-1].depends_on == ("ArbiterGate",)
