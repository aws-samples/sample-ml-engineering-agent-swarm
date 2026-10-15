"""Coding generation intent must survive survey and specification boundaries."""

from __future__ import annotations

import json

import pytest

from amlas.flows._util import raw
from amlas.flows.solution_survey_flow import run_solution_survey_flow
from amlas.spec.loader import SpecLoadError, load_spec_from_text
from amlas.tools import research_neutral
from amlas.tools.research_tools import recommend_approaches
from amlas.tools.spec_generation_tools import generate_spec_from_description, spec_dict_to_frontmatter
from amlas.tools.spec_heuristics import _extract_metric, _infer_task_type


@pytest.fixture(autouse=True)
def no_model_hub_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research_neutral, "list_jumpstart_models", lambda **_: {"models": []})


def test_generation_survey_infers_text_and_excludes_classifier_options() -> None:
    result = run_solution_survey_flow(problem_type="text_generation")
    assert result.status == "completed"
    assert result.modality == "text"
    options = json.dumps(result.managed_options).lower()
    assert "sagemaker" in options
    for wrong in ("autopilot", "xgboost", "tabtransformer", "bert", "comprehend"):
        assert wrong not in options
    explicit = run_solution_survey_flow(problem_type="text_generation", modality="text")
    assert explicit.managed_options == result.managed_options


def test_generation_survey_rejects_explicit_conflicting_modality() -> None:
    result = run_solution_survey_flow(problem_type="text_generation", modality="tabular")
    assert result.status == "failed"
    assert result.shortlist == []
    assert "modality" in result.detail["error"].lower()


def test_generation_survey_filters_the_real_mixed_model_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.backends.adapters.modelhub import LocalModelHubBackend

    catalog = LocalModelHubBackend()
    requests: list[str | None] = []

    def list_models(*, task: str | None = None, query: str | None = None) -> dict:
        requests.append(task)
        return {"models": catalog.list_models(task=task, query=query)}

    monkeypatch.setattr(research_neutral, "list_jumpstart_models", list_models)
    result = run_solution_survey_flow(problem_type="text_generation")
    assert result.status == "completed"
    assert requests == ["text-generation"]
    assert result.hub_models
    assert all(model["task"] == "text-generation" for model in result.hub_models)
    assert not any(
        word in json.dumps(result.shortlist).lower()
        for word in ("xgboost", "bert", "resnet", "tabtransformer")
    )


def test_existing_tabular_survey_still_defaults_to_tabular() -> None:
    result = run_solution_survey_flow(problem_type="classification")
    assert result.status == "completed"
    assert result.modality == "tabular"
    assert "autopilot" in json.dumps(result.managed_options).lower()


@pytest.mark.parametrize(
    ("description", "task", "metric"),
    [
        ("Fine-tune an open Qwen coding model on coding trajectories; measure held-out pass@1.",
         "text_generation", "pass_at_1"),
        ("Train a language model for text generation and evaluate perplexity.",
         "text_generation", "perplexity"),
        ("Fine-tune BERT to classify sentiment with F1 macro.",
         "classification", "f1_macro"),
        ("Forecast demand values with RMSE.", "regression", "rmse"),
    ],
)
def test_description_preserves_task_and_explicit_metric(
    description: str, task: str, metric: str,
) -> None:
    assert _infer_task_type(description) == task
    assert _extract_metric(description) == metric
    generated = raw(generate_spec_from_description)(description)
    assert generated["task_type"] == task
    assert generated["primary_metric"] == metric


def test_generation_without_a_metric_reports_required_research() -> None:
    result = raw(generate_spec_from_description)(
        "Fine-tune an open language model for coding and text generation."
    )
    assert result["task_type"] == "text_generation"
    assert not result.get("primary_metric")
    assert result["success_threshold"] is None
    assert "primary_metric" in result["needs_research"]


def test_generation_spec_roundtrip_preserves_real_metadata_and_unmeasured_baseline() -> None:
    original = {
        "project_id": "coding_campaign",
        "title": "Coding campaign",
        "task_type": "text_generation",
        "target": "messages",
        "primary_metric": "pass_at_1",
        "golden_dataset": "s3://trial-blue-lake/coding/eval.jsonl",
        "splits": {"strategy": "group", "group_col": "repo_id"},
        "constraints": {"inference_latency_ms": 5000},
        "compute_budget": {"trials": 5, "wall_clock_minutes": 0, "gpu_hours": 0},
        "baseline": {"type": "untouched_base", "pass_at_1": None},
        "acceptance": {"primary_metric_delta_vs_baseline": ">= 0.01"},
        "description": "Train and evaluate coding generation without fabricating a baseline.",
    }
    try:
        loaded = load_spec_from_text(spec_dict_to_frontmatter(original))
    except SpecLoadError as error:
        pytest.fail(f"Generation intent must validate without classification coercion: {error}")
    assert loaded.task_type == "text_generation"
    assert loaded.primary_metric == "pass_at_1"
    assert loaded.golden_dataset == original["golden_dataset"]
    assert loaded.splits.strategy == "group"
    assert loaded.splits.group_col == "repo_id"
    assert loaded.compute_budget.wall_clock_minutes == 0
    assert loaded.compute_budget.trials == 5
    assert loaded.constraints.inference_latency_ms == 5000
    assert loaded.baseline.model_dump()["pass_at_1"] is None


def test_generation_recommendation_never_falls_back_to_tabular_training() -> None:
    result = raw(recommend_approaches)(json.dumps({
        "task_type": "text_generation", "target": "messages", "primary_metric": "pass_at_1",
    }))
    assert result.get("data_type") == "text", "generation must not assume tabular data"
    text = json.dumps(result).lower()
    for wrong in ("gradient boosting", "ensemble methods", "linear baseline", "imputation"):
        assert wrong not in text


def test_language_model_implementation_does_not_change_forecasting_objective() -> None:
    plain = "Forecast demand values with RMSE."
    implementation = "Use a language model to forecast demand values with RMSE."
    assert _infer_task_type(plain) == "regression"
    assert _infer_task_type(implementation) == "regression"
    assert raw(generate_spec_from_description)(implementation)["primary_metric"] == "rmse"


@pytest.mark.parametrize("target", ["values", "prices", "amounts"])
def test_language_model_preserves_plural_numeric_prediction_objective(target: str) -> None:
    description = f"Use a language model to predict numeric {target}."
    assert _infer_task_type(description) == "regression"
    generated = raw(generate_spec_from_description)(description)
    assert generated["task_type"] == "regression"
    assert generated["primary_metric"] == "rmse"


def test_negated_classification_does_not_change_generation_objective() -> None:
    description = "Train a coding model for text generation, not classification."
    assert _infer_task_type(description) == "text_generation"
    assert raw(generate_spec_from_description)(description)["task_type"] == "text_generation"


@pytest.mark.parametrize("alias", ["code_generation", "language_modeling", "text-generation"])
def test_generation_alias_recommendation_uses_same_safe_contract(
    monkeypatch: pytest.MonkeyPatch, alias: str,
) -> None:
    from amlas.tools import research_tools

    def forbidden_rank(**_):
        raise AssertionError("Generation aliases must not enter the tabular algorithm catalog")

    monkeypatch.setattr(research_tools, "rank_algorithms_for_task", forbidden_rank)
    result = raw(recommend_approaches)(json.dumps({"task_type": alias}))
    assert result.get("data_type") == "text"
    assert result["recommended_algorithms"] == []
