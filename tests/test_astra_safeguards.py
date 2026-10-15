"""Closed-boundary advisory reproductions against an immutable candidate."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ROOT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace


def _shell(workspace, code, *args):
    repo, env, _ = workspace
    return subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1"; shift; ' + code,
            "probe",
            str(repo / "scripts/lib/deploy_common.sh"),
            *args,
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def test_selected_profile_matches_real_botocore_cli_resolution(shell_workspace, tmp_path):
    _, env, _ = shell_workspace
    config = tmp_path / "aws-config"
    config.write_text("[profile profile-a]\nregion=us-east-1\n[profile profile-b]\nregion=us-west-1\n")
    env.update(AWS_PROFILE="profile-a", AWS_DEFAULT_PROFILE="profile-b", AWS_CONFIG_FILE=str(config))
    code = 'export_aws_env profile-a 111122223333 us-east-1; "$TEST_PYTHON" -c \'import botocore.session;print("PROFILE="+str(botocore.session.Session().get_config_variable("profile")))\''
    result = _shell(shell_workspace, code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROFILE=profile-a" in result.stdout


@pytest.mark.parametrize(
    "old,value",
    [
        ("AWS_PROFILE = old\n", "new-profile"),
        ("AWS_PROFILE=old\n", r"team\tools"),
        ("AWS_PROFILE=old\n", "team #identity"),
    ],
)
def test_persisted_selectors_round_trip_literal_values(shell_workspace, old, value):
    repo, env, _ = shell_workspace
    (repo / ".env").write_text(old)
    env["PERSIST_VALUE"] = value
    result = _shell(
        shell_workspace,
        'persist_env_value "$1/.env" AWS_PROFILE "$PERSIST_VALUE"; unset AWS_PROFILE; load_env_file "$1"; printf "VALUE=%s\\n" "$AWS_PROFILE"',
        str(repo),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALUE=" + value in result.stdout.splitlines()


@pytest.mark.parametrize("failure", ["s3-encryption", "ddb-encryption"])
def test_unverifiable_encryption_fails_before_iam_writes(runtime_workspace, failure):
    _, env, _ = runtime_workspace
    env["TEST_FAILURE"] = failure
    tool = Path(env["PATH"].split(":")[0]) / "aws"
    text = tool.read_text()
    old = 'if operation in (("s3api","get-bucket-encryption"),("dynamodb","describe-table")): emit("None")'
    new = 'if operation in (("s3api","get-bucket-encryption"),("dynamodb","describe-table")):\n if (failure == "s3-encryption" and operation[0] == "s3api") or (failure == "ddb-encryption" and operation[0] == "dynamodb"): denied()\n emit("None")'
    assert old in text
    tool.write_text(text.replace(old, new))
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not any(
        c["args"][:2]
        in (
            ["iam", "create-role"],
            ["iam", "put-role-policy"],
            ["bedrock-agentcore-control", "create-agent-runtime"],
            ["bedrock-agentcore-control", "update-agent-runtime"],
        )
        and "--generate-cli-skeleton" not in c["args"]
        for c in runtime_calls(runtime_workspace)
    )


def _control():
    path = ROOT / "scripts/lib/agentcore_runtime_control.py"
    spec = importlib.util.spec_from_file_location("astra_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converging_endpoint_does_not_receive_another_update(monkeypatch):
    module = _control()
    monkeypatch.setenv("AMLAS_RUNTIME_POLL_ATTEMPTS", "2")
    monkeypatch.setenv("AMLAS_RUNTIME_POLL_SECONDS", "0")
    control = module.RuntimeControl("us-east-1")
    calls = []
    responses = iter(
        [
            {
                "status": "UPDATING",
                "targetVersion": "2",
                "liveVersion": "1",
                "agentRuntimeArn": "example-arn",
            },
            {"status": "READY", "targetVersion": "2", "liveVersion": "2", "agentRuntimeArn": "example-arn"},
        ]
    )

    def call(operation, payload):
        calls.append(operation)
        assert operation == "get-agent-runtime-endpoint", (
            "requested version is already converging; update is redundant"
        )
        return next(responses)

    monkeypatch.setattr(control, "call", call)
    control.bind_default("example-runtime", "2", "example-arn")
    assert calls == ["get-agent-runtime-endpoint", "get-agent-runtime-endpoint"]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_poll_interval_uses_existing_single_pass(monkeypatch, value):
    from amlas.tui.screens import dashboard

    monkeypatch.setenv("AMLAS_TRACE_POLL_INTERVAL_S", value)
    interval = dashboard._trace_poll_interval_s()
    assert math.isfinite(interval)
    assert interval <= 0


def test_nan_does_not_busy_poll_an_active_job(monkeypatch):
    from amlas.tui.screens import dashboard

    monkeypatch.setenv("AMLAS_TRACE_POLL_INTERVAL_S", "nan")
    worker = SimpleNamespace(is_cancelled=False, cancelled_event=Event())
    monkeypatch.setattr(dashboard, "get_current_worker", lambda: worker)
    pane = dashboard.DashboardPane()
    pane._poll_run_id = "example-run"
    monkeypatch.setattr(pane, "_bus", lambda: object())
    calls = []
    monkeypatch.setattr(pane, "_poll_once", lambda bus, run_id: calls.append(run_id))
    monkeypatch.setattr(pane, "_any_job_active", lambda bus: len(calls) < 4)
    dashboard.DashboardPane._run_trace_poller.__wrapped__(pane)
    assert calls == ["example-run"]


def _region_env(monkeypatch, tmp_path):
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AMLAS_AWS_REGION",
        "AGENTICML_AWS_REGION",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AMLAS_AWS_PROFILE",
        "AGENTICML_AWS_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "aws-config"
    config.write_text("[default]\nregion=us-west-2\n[profile other-profile]\nregion=eu-west-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))


def _factory_spies(monkeypatch):
    import boto3
    from strands.models import bedrock

    from amlas import _build

    sessions = []

    class Session:
        def __init__(self, **kwargs):
            sessions.append(kwargs)

    class Model:
        stateful = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(boto3, "Session", Session)
    monkeypatch.setattr(bedrock, "BedrockModel", Model)
    monkeypatch.setattr(_build, "Agent", lambda **kwargs: kwargs)
    return _build, sessions


def test_profile_only_constructor_resolves_region_from_that_profile(monkeypatch, tmp_path):
    from amlas.config import env_loader
    from amlas.config.runtime import BedrockConfig, Config

    _region_env(monkeypatch, tmp_path)
    monkeypatch.setattr(env_loader, "load_layered_env", lambda *args: None)
    build, sessions = _factory_spies(monkeypatch)
    build.build_agent(
        name="probe", system_prompt="probe", config=Config(bedrock=BedrockConfig(profile="other-profile"))
    )
    assert sessions[-1] == {"profile_name": "other-profile", "region_name": "eu-west-1"}


def test_implicit_config_is_constructed_after_agentcore_overlay(monkeypatch, tmp_path):
    from amlas.config import env_loader

    _region_env(monkeypatch, tmp_path)

    def overlay(*args):
        monkeypatch.setenv("AMLAS_AWS_PROFILE", "other-profile")

    monkeypatch.setattr(env_loader, "load_layered_env", overlay)
    build, sessions = _factory_spies(monkeypatch)
    build.build_agent(name="probe", system_prompt="probe")
    assert sessions[-1] == {"profile_name": "other-profile", "region_name": "eu-west-1"}


def _setup_stub(workspace, *, reject_role=False):
    repo, env, _ = workspace
    tool = Path(env["PATH"].split(":")[0]) / "aws"
    source = tool.read_text().replace(
        ' print("UNEXPECTED-AWS-CALL", file=sys.stderr); sys.exit(9)',
        ' if args[:2] == ["iam", "get-role"]: print(json.dumps({"Role":{"Arn":"arn:aws:iam::111122223333:role/historical-role","AssumeRolePolicyDocument":{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole","Condition":{"StringEquals":{"aws:SourceAccount":"111122223333"},"ArnLike":{"aws:SourceArn":"arn:aws:bedrock-agentcore:us-west-2:111122223333:*"}}}]}}}));sys.exit(0)\n'
        ' if args[:2] == ["ssm", "describe-parameters"]: print("0");sys.exit(0)\n'
        ' if args[:2] == ["ssm", "put-parameter"]: assert "--tags" in args and "--overwrite" not in args;print(json.dumps({"Version":1}));sys.exit(0)\n'
        ' if args[:2] == ["ssm", "list-tags-for-resource"]: print(json.dumps({"TagList":[]}));sys.exit(0)\n'
        ' if args[0] == "secretsmanager": print("None");sys.exit(0)\n'
        ' print("UNEXPECTED-AWS-CALL", file=sys.stderr); sys.exit(9)',
    )
    tool.write_text(source)
    # These tests isolate selectors/key preservation/publication; ledger ownership
    # and foreign-resource refusal are verified in their native boundary suite.
    (repo / "scripts/lib/deployment_manifest.py").write_text(
        '#!/usr/bin/env python3\nimport json,sys\nif sys.argv[1]=="init": print(json.dumps({"deployment_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}))\nelse: print(json.dumps({"resources":[],"refresh_errors":[],"remaining":[]}))\n'
    )
    (repo / "scripts/lib/build_runtime_image.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    helper = "#!/usr/bin/env bash\n"
    if reject_role:
        helper += "echo 'Existing role is incompatible; choose a distinct role.' >&2\nexit 1\n"
    else:
        helper += "printf '%s\\n' 'AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/example' 'FINAL_STATUS=READY' 'AGENTCORE_RUNTIME_DEPLOY_DONE' 'RESOLVED_SESSION_BUCKET=current-bucket'\n"
    (repo / "scripts/lib/deploy_agentcore_runtime.sh").write_text(helper)
    return repo, env


def _setup(workspace):
    from test_shell_entrypoint_advisory import invoke

    return invoke(workspace, "setup.sh", "--yes", "--skip-deps", "--agent-runtime-only", "--skip-harness")


def test_existing_incompatible_role_rejected_before_main_cloud_writes(shell_workspace):
    _, env = _setup_stub(shell_workspace, reject_role=True)
    env["AGENTCORE_EXEC_ROLE_NAME"] = "historical-role"
    result = _setup(shell_workspace)
    assert result.returncode != 0
    records = [json.loads(s) for s in shell_workspace[2].read_text().splitlines()]
    assert not any(
        r["tool"] == "aws"
        and r["args"][:2]
        in (
            ["secretsmanager", "put-secret-value"],
            ["secretsmanager", "create-secret"],
            ["ssm", "put-parameter"],
        )
        for r in records
    )


def test_missing_replacement_key_preserves_existing_secret(shell_workspace):
    _setup_stub(shell_workspace)
    result = _setup(shell_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(s) for s in shell_workspace[2].read_text().splitlines()]
    assert not any(
        r["tool"] == "aws" and r["args"][:2] == ["secretsmanager", "put-secret-value"] for r in records
    )


def test_published_session_bucket_is_the_runtime_verified_bucket(shell_workspace):
    _, env = _setup_stub(shell_workspace)
    env["AMLAS_SESSION_BUCKET"] = "stale-bucket"
    result = _setup(shell_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(s) for s in shell_workspace[2].read_text().splitlines()]
    call = next(
        r
        for r in records
        if r["tool"] == "aws"
        and r["args"][:2] == ["ssm", "put-parameter"]
        and r["args"][r["args"].index("--name") + 1] == "/amlas/session-bucket"
    )
    assert call["args"][call["args"].index("--value") + 1] == "current-bucket"


def test_cli_calls_explicitly_select_the_named_profile(shell_workspace):
    _, env, _ = shell_workspace
    env["AWS_PROFILE"] = "profile-a"
    result = _shell(
        shell_workspace,
        "export_aws_env profile-a 111122223333 us-east-1; aws sts get-caller-identity --query Account --output text",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    call = [json.loads(s) for s in shell_workspace[2].read_text().splitlines()][-1]
    assert "--profile=profile-a" in call.get("raw_args", call["args"])
