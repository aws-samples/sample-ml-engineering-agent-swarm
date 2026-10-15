"""Closed AWS state validates atomic publication and actual physical stack checks."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "333af729-c6bd-4283-95a4-fe5139e24276"


class OwnedPublicationTests(unittest.TestCase):
    def run_shell(self, command, mode="missing"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            log = path / "calls.jsonl"
            fake = path / "aws"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                + r"""
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ["CALL_LOG"],"a") as f: f.write(json.dumps(args)+"\n")
mode=os.environ["FAKE_MODE"]
if args[:2]==["ssm","describe-parameters"]: print("0" if mode=="missing" else "1")
elif args[:2]==["ssm","list-tags-for-resource"]:
 print(json.dumps({"TagList":[{"Key":"amlas:deployment-id","Value":os.environ["AMLAS_DEPLOYMENT_ID"] if mode=="owned" else "foreign"},{"Key":"amlas:deployment-prefix","Value":"trial-blue"}]}))
elif args[:2]==["ssm","put-parameter"]: print(json.dumps({"Version":1}))
elif args[:2]==["ssm","add-tags-to-resource"]: raise SystemExit("Tagging after creation must never be required")
elif args[:2]==["cloudformation","describe-stacks"]:
 name=args[args.index("--stack-name")+1]
 print("CREATE_COMPLETE" if name=="CDKToolkit" or mode=="healthy" or not name.startswith("trial-blue-") else "UPDATE_ROLLBACK_COMPLETE")
else: raise SystemExit("Unspecified AWS operation")
"""
            )
            fake.chmod(0o700)
            env = dict(os.environ)
            env.update(
                PATH=str(path) + ":" + env.get("PATH", ""),
                CALL_LOG=str(log),
                FAKE_MODE=mode,
                AMLAS_DEPLOYMENT_PREFIX="trial-blue",
                AMLAS_SSM_PREFIX="/trial-blue",
                AMLAS_DEPLOYMENT_ID=OWNER,
                AWS_REGION="test-region-1",
            )
            result = subprocess.run(
                ["bash", "-c", ". scripts/lib/deploy_common.sh; configure_deployment_namespace; " + command],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_creation_is_atomically_owned_without_overwrite(self):
        result, calls = self.run_shell('publish_owned_parameter "/trial-blue/session-bucket" "bucket-name"')
        self.assertEqual(result.returncode, 0, result.stderr)
        put = next(c for c in calls if c[:2] == ["ssm", "put-parameter"])
        self.assertIn("--tags", put)
        self.assertNotIn("--overwrite", put)
        self.assertTrue(any(OWNER in value for value in put))
        self.assertFalse(any(c[:2] == ["ssm", "add-tags-to-resource"] for c in calls))

    def test_owned_update_keeps_existing_tags(self):
        result, calls = self.run_shell(
            'publish_owned_parameter "/trial-blue/session-bucket" "next-bucket"', "owned"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        put = next(c for c in calls if c[:2] == ["ssm", "put-parameter"])
        self.assertIn("--overwrite", put)
        self.assertNotIn("--tags", put)

    def test_foreign_parameter_is_not_changed_or_retagged(self):
        result, calls = self.run_shell(
            'publish_owned_parameter "/trial-blue/session-bucket" "bucket-name"', "foreign"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(
            any(c[:2] in (["ssm", "put-parameter"], ["ssm", "add-tags-to-resource"]) for c in calls)
        )

    def test_namespaced_failed_stack_cannot_borrow_legacy_success(self):
        result, calls = self.run_shell(
            'AMLAS_STACKS_DEPLOY_ORDER="LakeStack"; verify_deployed_stacks', "failed"
        )
        self.assertNotEqual(result.returncode, 0)
        names = [
            c[c.index("--stack-name") + 1] for c in calls if c[:2] == ["cloudformation", "describe-stacks"]
        ]
        self.assertIn("trial-blue-LakeStack", names)
        self.assertNotIn("LakeStack", names)

    def test_only_successful_terminal_states_pass(self):
        result, calls = self.run_shell(
            'AMLAS_STACKS_DEPLOY_ORDER="LakeStack"; verify_deployed_stacks', "healthy"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
