"""Fresh-process real CLI discovery from setup-style dotenv; closed AWS boundaries."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""import json, os, sys
from pathlib import Path
import boto3
from click.testing import CliRunner
account = '111122223333'
region = 'us-east-2'
prefix = '/trial-cli'
arn = 'arn:aws:bedrock-agentcore:' + region + ':' + account + ':runtime/example-runtime'
bucket = 'trial-cli-session'
requests = []
class SSM:
    def get_parameter(self, *, Name):
        requests.append(('ssm', Name))
        assert Name.startswith(prefix + '/'), 'wrong deployment namespace'
        values = {prefix + '/agentcore-runtime-arn': arn, prefix + '/session-bucket': bucket}
        assert Name in values, 'unexpected parameter'
        return {'Parameter': {'Value': values[Name]}}
class S3:
    def upload_file(self, filename, selected_bucket, key):
        assert selected_bucket == bucket, 'wrong session bucket'
        assert Path(filename).is_file()
        requests.append(('s3', selected_bucket))
class Runtime:
    def invoke_agent_runtime(self, **kwargs):
        assert kwargs['agentRuntimeArn'] == arn, 'wrong runtime account or identity'
        assert len(kwargs['runtimeSessionId']) >= 33
        requests.append(('invoke', arn))
        return {'response': [json.dumps({'event':'run_completed','status':'accepted'}).encode()]}
def factory(service, *, region_name, **kwargs):
    # Real native configuration parsing, before any credential/network lookup.
    session = boto3.session.Session()
    assert session.profile_name == 'default', 'wrong selected credential chain'
    assert region_name == region, 'wrong selected region'
    assert os.environ.get('AMLAS_AWS_ACCOUNT_ID') == account, 'wrong selected account'
    return {'ssm': SSM, 's3': S3, 'bedrock-agentcore': Runtime}[service]()
boto3.client = factory
from amlas.config.settings import Settings
from amlas.cli.main import cli
settings = Settings.load()
result = CliRunner().invoke(cli, ['run','cloud','Predict the target using a classifier','--dataset','dataset.csv','--target','target','--project-id','cli_fixture','--json'])
print(result.output)
print(json.dumps({'exit_code':result.exit_code,'exception_class':type(result.exception).__name__ if result.exception else None,'requests':requests,'settings_region':settings.aws.region,'settings_profile':settings.aws.effective_profile}))
assert result.exit_code == 0, 'real CLI failed before a valid invocation'
ssm_attempts = [value for service, value in requests if service == 'ssm']
assert sorted(ssm_attempts) == sorted([prefix + '/agentcore-runtime-arn', prefix + '/session-bucket']), 'unexpected SSM lookup attempt'
assert [value for service, value in requests if service == 's3'] == [bucket], 'required S3 upload omitted or wrong bucket'
assert ('invoke', arn) in requests
"""


def _run_cli_probe(tmp_path, empty_key, probe):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/amlas", project / "src/amlas")
    (project / ".env").write_text(
        "\n".join(
            [
                empty_key + "=",
                "AWS_REGION=us-east-2",
                "AMLAS_AWS_PROFILE=",
                "AMLAS_AWS_ACCOUNT_ID=111122223333",
                "AMLAS_DEPLOYMENT_PREFIX=trial-cli",
                "AMLAS_SSM_PREFIX=/trial-cli",
                "",
            ]
        )
    )
    (project / "dataset.csv").write_text("feature,target\n1,0\n2,1\n")
    config = tmp_path / "aws-config"
    config.write_text("")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_", "AGENTCORE_"))
    }
    env.update(
        PYTHONPATH=str(project / "src"),
        AWS_CONFIG_FILE=str(config),
        AWS_SHARED_CREDENTIALS_FILE=str(tmp_path / "absent-credentials"),
        AWS_EC2_METADATA_DISABLED="true",
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=project, env=env, capture_output=True, text=True, timeout=30
    )
    return completed


@pytest.mark.parametrize("empty_key", ["AWS_PROFILE", "AWS_DEFAULT_PROFILE"])
def test_real_cli_discovers_persisted_namespace_with_default_credential_chain(tmp_path, empty_key):
    completed = _run_cli_probe(tmp_path, empty_key, PROBE)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("mutation", ["suppressed-extra-lookup", "omitted-upload"])
def test_probe_rejects_mutations_after_application_suppression(tmp_path, mutation):
    if mutation == "suppressed-extra-lookup":
        change = "from amlas.config.externalized import _default_ssm_getter\n_default_ssm_getter('/unexpected/parameter', region)\n"
        expected = "unexpected SSM lookup attempt"
    else:
        change = "from amlas.cloud import dispatch\ndispatch.upload_dataset_to_session = lambda *args, **kwargs: 's3://' + bucket + '/fixture.csv'\n"
        expected = "required S3 upload omitted or wrong bucket"
    probe = PROBE.replace("settings = Settings.load()", change + "settings = Settings.load()")
    completed = _run_cli_probe(tmp_path, "AWS_PROFILE", probe)
    assert completed.returncode != 0
    assert expected in completed.stderr, completed.stdout + completed.stderr
    assert '"exit_code": 0' in completed.stdout, (
        "mutation must reach successful app return before oracle rejects it"
    )


@pytest.mark.parametrize("empty_key", ["AWS_PROFILE", "AWS_DEFAULT_PROFILE"])
def test_native_sdk_rejects_empty_default_profile_without_network(tmp_path, empty_key):
    config = tmp_path / "aws-config"
    config.write_text("")
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(
        **{empty_key: ""},
        AWS_CONFIG_FILE=str(config),
        AWS_SHARED_CREDENTIALS_FILE=str(tmp_path / "absent-credentials"),
        AWS_EC2_METADATA_DISABLED="true",
    )
    code = "import boto3; from botocore.exceptions import ProfileNotFound;\ntry: boto3.Session()\nexcept ProfileNotFound: print('EXPECTED_PROFILE_NOT_FOUND')\nelse: raise AssertionError('native SDK unexpectedly accepted empty profile')"
    completed = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "EXPECTED_PROFILE_NOT_FOUND"


@pytest.mark.parametrize("shell_profile", [None, " shell identity "])
def test_root_loader_preserves_nonempty_literal_profile_and_shell_precedence(tmp_path, shell_profile):
    project = tmp_path / "profile-project"
    (project / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src/amlas", project / "src/amlas")
    literal = " persisted identity "
    (project / ".env").write_text('AWS_PROFILE="' + literal + '"\nAWS_REGION=us-east-2\n')
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "AMLAS_", "AGENTICML_", "AGENTCORE_"))
    }
    if shell_profile is not None:
        env["AWS_PROFILE"] = shell_profile
    env.update(
        PYTHONPATH=str(project / "src"),
        EXPECTED_PROFILE=shell_profile if shell_profile is not None else literal,
        AWS_CONFIG_FILE=str(tmp_path / "untouched-config"),
    )
    code = "import amlas,os;assert os.environ['AWS_PROFILE']==os.environ['EXPECTED_PROFILE'];assert os.environ['AWS_CONFIG_FILE'].endswith('untouched-config')"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=project, env=env, capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
