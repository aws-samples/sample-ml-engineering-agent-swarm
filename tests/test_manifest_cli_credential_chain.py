"""Manifest-selected profiles must control the native CLI credential provider."""

import importlib.util
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manifest_cli_credentials", ROOT / "scripts/lib/deployment_manifest.py"
)
LEDGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEDGER)


class ManifestCliCredentialChainTest(unittest.TestCase):
    def invoke(self, recorded_profile, ambient, override=None):
        captured = {}

        def native_call(arguments, **kwargs):
            captured["arguments"] = list(arguments)
            captured["environment"] = dict(kwargs.get("env", os.environ))
            return subprocess.CompletedProcess(arguments, 0, stdout="{}", stderr="")

        with mock.patch.dict(os.environ, ambient, clear=True):
            with mock.patch.object(LEDGER.subprocess, "run", side_effect=native_call):
                result = LEDGER.AwsCli(
                    {"region": "us-west-2", "profile": recorded_profile}, profile=override
                ).call("sts", "get-caller-identity")
        self.assertEqual(result, {})
        return captured

    def test_default_manifest_removes_empty_environment_profile_selectors(self):
        captured = self.invoke(
            "",
            {
                "AWS_PROFILE": "",
                "AWS_DEFAULT_PROFILE": "",
                "AWS_EC2_METADATA_DISABLED": "false",
                "AWS_CONFIG_FILE": "/synthetic/provider/config",
            },
        )
        self.assertNotIn("--profile", captured["arguments"])
        self.assertNotIn("AWS_PROFILE", captured["environment"])
        self.assertNotIn("AWS_DEFAULT_PROFILE", captured["environment"])
        self.assertEqual(captured["environment"]["AWS_EC2_METADATA_DISABLED"], "false")
        self.assertEqual(captured["environment"]["AWS_CONFIG_FILE"], "/synthetic/provider/config")

    def test_default_manifest_ignores_stale_named_ambient_profiles(self):
        captured = self.invoke(
            "",
            {
                "AWS_PROFILE": "stale-ambient-profile",
                "AWS_DEFAULT_PROFILE": "stale-shadow-profile",
                "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/synthetic/provider/credentials",
            },
        )
        self.assertNotIn("--profile", captured["arguments"])
        self.assertNotIn("AWS_PROFILE", captured["environment"])
        self.assertNotIn("AWS_DEFAULT_PROFILE", captured["environment"])
        self.assertEqual(
            captured["environment"]["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"],
            "/synthetic/provider/credentials",
        )

    def test_recorded_literal_profile_matches_cli_and_child_environment(self):
        literal = "recorded profile.[literal]"
        captured = self.invoke(
            literal,
            {"AWS_PROFILE": "stale-profile", "AWS_DEFAULT_PROFILE": "shadow-profile"},
        )
        arguments = captured["arguments"]
        self.assertEqual(arguments[arguments.index("--profile") + 1], literal)
        self.assertEqual(captured["environment"]["AWS_PROFILE"], literal)
        self.assertNotIn("AWS_DEFAULT_PROFILE", captured["environment"])

    def test_explicit_literal_profile_override_matches_child_environment(self):
        literal = "selected profile:[literal]"
        captured = self.invoke(
            "recorded-profile",
            {"AWS_PROFILE": "stale-profile", "AWS_DEFAULT_PROFILE": "shadow-profile"},
            override=literal,
        )
        arguments = captured["arguments"]
        self.assertEqual(arguments[arguments.index("--profile") + 1], literal)
        self.assertEqual(captured["environment"]["AWS_PROFILE"], literal)
        self.assertNotIn("AWS_DEFAULT_PROFILE", captured["environment"])


if __name__ == "__main__":
    unittest.main()
