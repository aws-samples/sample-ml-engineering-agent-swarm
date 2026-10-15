"""Run setup preflight-to-ledger handoff without cloud or dependency writes."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARN = "arn:aws:iam::111122223333:role/verified-runtime-role"
# Synthetic native-format identifier assembled from obvious fixture parts.
ROLE_ID = "AROA" + "FIXTUREIDENTITY" + "123"


class SetupReuseWitnessTests(unittest.TestCase):
    def run_handoff(self, output):
        source = (ROOT / "scripts/setup.sh").read_text()
        start = source.index('if [ "$SYNTH_ONLY" != "1" ]; then\n  normalize_model_env')
        end = source.index("  MANIFEST_INIT_OUT=", start)
        block = source[start:end] + '\nfi\nprintf "ARG=%s\\n" "${MANIFEST_INIT_ARGS[@]}"\n'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            helper = base / "scripts/lib/deploy_agentcore_runtime.sh"
            helper.parent.mkdir(parents=True)
            helper.write_text('#!/usr/bin/env bash\nprintf "%s\\n" ' + shlex.quote(output) + "\n")
            script = "\n".join(
                [
                    "set -e",
                    ". scripts/lib/deploy_common.sh",
                    "SYNTH_ONLY=0; PREFLIGHT_ONLY=0; SKIP_AGENT_RUNTIME=0; AGENT_RUNTIME_ONLY=0",
                    "AMLAS_AWS_ACCOUNT_ID=111122223333; AWS_REGION=test-region-1",
                    "AMLAS_DEPLOYMENT_PREFIX=trial-blue; configure_deployment_namespace",
                    "AGENTCORE_RUNTIME_NAME=trial_blue; AGENTCORE_EXEC_ROLE_NAME=verified-runtime-role",
                    "TAVILY_SECRET_ID=" + "trial-blue/" + "tavily-api-key",  # Synthetic resource selector.
                    "SCRIPT_DIR=" + shlex.quote(str(base / "scripts")),
                    "REPO_ROOT=" + shlex.quote(str(base / "repo")),
                    "normalize_model_env() { :; }",
                    'resolve_model_profiles() { printf "{}\\n"; }',
                    "preflight_deployment_services() { :; }",
                "preflight_cloudtrail_capacity() { :; }",
                    '_amlas_host_root() { printf "fixture-tools\\n"; }',
                    block,
                ]
            )
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("AMLAS_", "AGENTCORE_", "TAVILY_", "AWS_"))
            }
            return subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, text=True, capture_output=True)

    def test_verified_existing_role_reaches_native_ledger(self):
        result = self.run_handoff(
            "RESOLVED_REUSED_RUNTIME_ROLE_ARN="
            + ARN
            + "\nRESOLVED_REUSED_RUNTIME_ROLE_ID="
            + ROLE_ID
            + "\nRUNTIME-ROLE-PREFLIGHT PASS"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = [line[4:] for line in result.stdout.splitlines() if line.startswith("ARG=")]
        self.assertIn("--reused-role-arn", args)
        self.assertEqual(args[args.index("--reused-role-arn") + 1], ARN)
        self.assertIn("--reused-role-id", args)
        self.assertEqual(args[args.index("--reused-role-id") + 1], ROLE_ID)

    def test_absent_role_has_no_reuse_claim(self):
        result = self.run_handoff("RUNTIME-ROLE-PREFLIGHT PASS")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ARG=--reused-role-", result.stdout)

    def test_incomplete_or_duplicate_witness_fails_before_ledger(self):
        for output in (
            "RESOLVED_REUSED_RUNTIME_ROLE_ARN=" + ARN,
            "RESOLVED_REUSED_RUNTIME_ROLE_ID=" + ROLE_ID,
            "RESOLVED_REUSED_RUNTIME_ROLE_ARN="
            + ARN
            + "\nRESOLVED_REUSED_RUNTIME_ROLE_ARN="
            + ARN
            + "\nRESOLVED_REUSED_RUNTIME_ROLE_ID="
            + ROLE_ID,
        ):
            with self.subTest(output=output):
                result = self.run_handoff(output)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("ARG=", result.stdout)


if __name__ == "__main__":
    unittest.main()
