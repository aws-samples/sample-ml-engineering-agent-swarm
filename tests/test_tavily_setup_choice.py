"""Optional provider setup and secret transport never expose a key in argv."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import secrets
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import boto3
from botocore.stub import Stubber

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_PATH = "/".join(("trial-blue", "tavily-api-key"))  # Synthetic resource name, never a credential.


class TavilySetupChoice(unittest.TestCase):
    def test_explicit_hidden_prompt_option_is_documented(self):
        result = subprocess.run(
            ["bash", "scripts/setup.sh", "--yes", "--ask-tavily-key", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--ask-tavily-key", result.stdout)

    def test_noninteractive_setup_keeps_key_optional(self):
        env = dict(os.environ)
        env.pop("AMLAS_TAVILY_API_KEY", None)
        result = subprocess.run(
            [
                "bash",
                "-c",
                '. scripts/lib/deploy_common.sh; NON_INTERACTIVE=1; prompt_secret_with_default AMLAS_TAVILY_API_KEY "Optional Tavily key" "$TAVILY_PLACEHOLDER_VALUE"; [ "$AMLAS_TAVILY_API_KEY" = "$TAVILY_PLACEHOLDER_VALUE" ]',
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_key_uses_stdin_not_arguments_environment_or_output(self):
        for operation in ("create", "put"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                calls = path / "calls.json"
                fake = path / "uv"
                fake.write_text(
                    "#!/usr/bin/env python3\nimport hashlib,json,os,sys\nfrom pathlib import Path\n"
                    "payload=json.load(sys.stdin)\nsecret=payload.pop('SecretString')\n"
                    "data={'args':sys.argv[1:],'payload':payload,"
                    "'secret_matches':hashlib.sha256(secret.encode()).hexdigest()==os.environ['EXPECTED_DIGEST'],"
                    "'key_in_environment':'AMLAS_TAVILY_API_KEY' in os.environ,"
                    "'profile':os.environ.get('AWS_PROFILE'),'region':os.environ.get('AWS_REGION')}\n"
                    "Path(os.environ['CALLS']).write_text(json.dumps(data))\n"
                )
                fake.chmod(0o700)
                legacy = path / "aws"
                legacy.write_text("#!/bin/sh\nexit 252\n")
                legacy.chmod(0o700)
                # Per-run, nonfunctional machine-key sentinel; never persisted as plaintext.
                key = "tvly-" + secrets.token_urlsafe(32)
                env = dict(os.environ)
                env.update(
                    PATH=str(path) + ":" + env["PATH"],
                    CALLS=str(calls),
                    EXPECTED_DIGEST=hashlib.sha256(key.encode()).hexdigest(),
                    AMLAS_TAVILY_API_KEY=key,
                    AMLAS_DEPLOYMENT_PREFIX="trial-blue",
                    AMLAS_DEPLOYMENT_ID="00000000-0000-4000-8000-aaaaaaaaaaa1",
                    AWS_PROFILE="selected-local-profile",
                    AWS_REGION="us-east-1",
                )
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        '. scripts/lib/deploy_common.sh; write_tavily_secret "$1" "trial-blue/tavily-api-key"',
                        "test",
                        operation,
                    ],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(calls.read_text())
                self.assertEqual(data["args"][:4], ["run", "--frozen", "--no-sync", "python"])
                self.assertTrue(data["args"][4].endswith("scripts/lib/write_research_secret.py"))
                self.assertEqual(data["args"][5:], [operation])
                self.assertNotIn(key, json.dumps(data) + result.stdout + result.stderr)
                self.assertTrue(data["secret_matches"])
                self.assertFalse(data["key_in_environment"])
                self.assertEqual(data["profile"], "selected-local-profile")
                self.assertEqual(data["region"], "us-east-1")
                if operation == "create":
                    self.assertEqual(data["payload"]["Name"], "trial-blue/tavily-api-key")
                    self.assertEqual(data["payload"]["Tags"], self._tags())
                else:
                    expected_name = "/".join(("trial-blue", "tavily-api-key"))
                    self.assertEqual(data["payload"], {"SecretId": expected_name})

    @staticmethod
    def _tags():
        return [
            {"Key": "amlas:deployment-prefix", "Value": "trial-blue"},
            {"Key": "amlas:deployment-id", "Value": "00000000-0000-4000-8000-aaaaaaaaaaa1"},
        ]

    @staticmethod
    def _helper():
        spec = importlib.util.spec_from_file_location(
            "research_secret_helper", ROOT / "scripts/lib/write_research_secret.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _client():
        # Synthetic SDK credentials under Stubber: no credential discovery or cloud calls.
        env = dict(os.environ)
        env.pop("AWS_PROFILE", None)
        with patch.dict(os.environ, env, clear=True):
            return boto3.Session(
                aws_access_key_id=secrets.token_hex(12),
                aws_secret_access_key=secrets.token_urlsafe(32),
                region_name="us-east-1",
            ).client("secretsmanager")

    def test_sdk_create_preserves_uuid_tags_and_explicit_profile_region(self):
        helper = self._helper()
        payload = {
            "Name": "trial-blue/tavily-api-key",
            "SecretString": secrets.token_urlsafe(32),
            "Tags": self._tags(),
        }
        client = self._client()
        session = Mock()
        session.client.return_value = client
        with Stubber(client) as stub:
            stub.add_response("create_secret", {"Name": payload["Name"]}, payload)
            with (
                patch.dict(
                    os.environ, {"AWS_PROFILE": " selected-local-profile ", "AWS_REGION": "us-east-1"}
                ),
                patch.object(helper.boto3, "Session", return_value=session) as factory,
            ):
                helper.write_secret("create", payload)
                factory.assert_called_once_with(
                    profile_name=" selected-local-profile ", region_name="us-east-1"
                )
                session.client.assert_called_once_with("secretsmanager")
            stub.assert_no_pending_responses()

    def test_sdk_put_preserves_payload_and_default_credential_chain(self):
        helper = self._helper()
        payload = {"SecretId": RESOURCE_PATH, "SecretString": secrets.token_urlsafe(32)}
        client = self._client()
        session = Mock()
        session.client.return_value = client

        def default_session(**kwargs):
            self.assertFalse("AWS_PROFILE" in os.environ, "native SDK rejects an empty profile selector")
            return session

        with Stubber(client) as stub:
            stub.add_response("put_secret_value", {}, payload)
            with (
                patch.dict(os.environ, {"AWS_PROFILE": "", "AWS_REGION": "us-east-1"}),
                patch.object(helper.boto3, "Session", side_effect=default_session) as factory,
            ):
                helper.write_secret("put", payload)
                factory.assert_called_once_with(profile_name=None, region_name="us-east-1")
            stub.assert_no_pending_responses()

    def test_real_sdk_empty_profile_uses_default_chain_without_network(self):
        helper = self._helper()
        payload = {"SecretId": RESOURCE_PATH, "SecretString": secrets.token_urlsafe(32)}
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            credentials = Path(directory) / "credentials"
            config.write_text("")
            credentials.write_text("")
            env = {
                "AWS_PROFILE": "",
                "AWS_REGION": "us-east-1",
                "AWS_CONFIG_FILE": str(config),
                "AWS_SHARED_CREDENTIALS_FILE": str(credentials),
                "AWS_EC2_METADATA_DISABLED": "true",
                "AWS_ACCESS_KEY_ID": secrets.token_hex(12),
                "AWS_SECRET_ACCESS_KEY": secrets.token_urlsafe(32),
            }
            # Real Session and client construction; stop before the native API call.
            with (
                patch.dict(os.environ, env, clear=True),
                patch("botocore.client.BaseClient._make_api_call", return_value={}) as api,
            ):
                helper.write_secret("put", payload)
                self.assertEqual(api.call_count, 1)
                self.assertEqual(api.call_args.args[0], "PutSecretValue")
                self.assertFalse("AWS_PROFILE" in os.environ)

    def test_helper_cli_sanitizes_sdk_error_without_secret_or_body(self):
        helper = self._helper()
        key = secrets.token_urlsafe(32)
        payload = {"SecretId": RESOURCE_PATH, "SecretString": key}
        client = self._client()
        session = Mock()
        session.client.return_value = client
        stderr = io.StringIO()
        stdout = io.StringIO()
        with Stubber(client) as stub:
            stub.add_client_error(
                "put_secret_value",
                service_error_code="AccessDeniedException",
                service_message="untrusted-body " + key,
                expected_params=payload,
            )
            with (
                patch.dict(os.environ, {"AWS_REGION": "us-east-1"}),
                patch.object(helper.boto3, "Session", return_value=session),
                patch.object(helper.sys, "stdin", io.StringIO(json.dumps(payload))),
                contextlib.redirect_stderr(stderr),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(helper.main(["put"]), 1)
        self.assertNotIn(key, stderr.getvalue() + stdout.getvalue())
        self.assertNotIn("untrusted-body", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Research secret write failed", stderr.getvalue())

    def test_malformed_stdin_never_calls_sdk_or_echoes_input(self):
        helper = self._helper()
        key = secrets.token_urlsafe(32)
        stderr = io.StringIO()
        with (
            patch.object(helper.boto3, "Session") as factory,
            patch.object(helper.sys, "stdin", io.StringIO('{"SecretString":"' + key)),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(helper.main(["create"]), 1)
            factory.assert_not_called()
        self.assertNotIn(key, stderr.getvalue())

    def test_real_project_uv_reads_stdin_json_with_native_sdk_stubber(self):
        payload = {
            "Name": "trial-blue/tavily-api-key",
            "SecretString": secrets.token_urlsafe(32),
            "Tags": self._tags(),
        }
        code = """import importlib.util,io,json,os,secrets,sys
from unittest.mock import Mock,patch
import boto3
from botocore.stub import Stubber
raw=sys.stdin.read()
payload=json.loads(raw)
os.environ.pop('AWS_PROFILE',None)
os.environ['AWS_REGION']='us-east-1'
client=boto3.Session(aws_access_key_id=secrets.token_hex(12),aws_secret_access_key=secrets.token_urlsafe(32),region_name='us-east-1').client('secretsmanager')
spec=importlib.util.spec_from_file_location('helper','scripts/lib/write_research_secret.py')
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
session=Mock()
session.client.return_value=client
with Stubber(client) as stub:
    stub.add_response('create_secret',{'Name':payload['Name']},payload)
    with patch.object(helper.boto3,'Session',return_value=session),patch.object(helper.sys,'stdin',io.StringIO(raw)):
        result=helper.main(['create'])
    stub.assert_no_pending_responses()
raise SystemExit(result)
"""
        result = subprocess.run(
            ["uv", "run", "--frozen", "--no-sync", "python", "-c", code],
            cwd=ROOT,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn(payload["SecretString"], result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
