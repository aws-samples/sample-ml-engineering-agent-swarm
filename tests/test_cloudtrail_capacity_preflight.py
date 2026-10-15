"""Closed native-metadata boundary for the five-trail CloudTrail quota."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "1111" + "2222" + "3333"  # Synthetic AWS documentation account.
REGION = "eu-west-1"
PREFIX = "fixture-deployment"


def trail(name, *, account=ACCOUNT, region=REGION):
    return {
        "Name": name,
        "HomeRegion": region,
        "TrailARN": "arn:aws:cloudtrail:" + region + ":" + account + ":trail/" + name,
    }


class CloudTrailCapacityTests(unittest.TestCase):
    def run_preflight(self, payload, *, denied=False):
        with tempfile.TemporaryDirectory(prefix="cloudtrail-capacity-") as tmp:
            tools = Path(tmp)
            callfile = tools / "calls.json"
            fake = tools / "aws"
            fake.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "Path(os.environ['TEST_CALLFILE']).write_text(json.dumps(args))\n"
                "if args and args[0].startswith('--profile='): args = args[1:]\n"
                "if args[:2] != ['cloudtrail', 'list-trails']: sys.exit(9)\n"
                "if os.environ['TEST_DENIED'] == '1':\n"
                " print('AccessDeniedException', file=sys.stderr); sys.exit(254)\n"
                "print(os.environ['TEST_PAYLOAD'])\n"
            )
            fake.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "AMLAS_", "AGENTICML_"))}
            env.update(
                PATH=str(tools) + ":" + env["PATH"],
                AWS_PROFILE="fixture profile",
                AWS_REGION=REGION,
                AMLAS_AWS_ACCOUNT_ID=ACCOUNT,
                AMLAS_DEPLOYMENT_PREFIX=PREFIX,
                TEST_CALLFILE=str(callfile),
                TEST_PAYLOAD=payload if isinstance(payload, str) else json.dumps(payload),
                TEST_DENIED="1" if denied else "0",
            )
            result = subprocess.run(
                [
                    "bash",
                    "-e",
                    "-c",
                    '. "$1"; preflight_cloudtrail_capacity; printf "AFTER_CAPACITY\\n"',
                    "fixture",
                    str(ROOT / "scripts/lib/deploy_common.sh"),
                ],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            args = json.loads(callfile.read_text()) if callfile.exists() else []
            return result, args

    def test_five_trails_block_before_followup_with_actionable_remediation(self):
        result, args = self.run_preflight({"Trails": [trail("other-" + str(i)) for i in range(5)]})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("AFTER_CAPACITY", result.stdout)
        self.assertIn("5", result.stderr)
        self.assertIn("cannot be increased", result.stderr.lower())
        self.assertIn("approved", result.stderr.lower())
        self.assertIn("list-trails", args)

    def test_unverifiable_metadata_fails_closed(self):
        normal = [trail("other-" + str(i)) for i in range(4)]
        malformed = (
            "not-json",
            {},
            {"Trails": -1},
            {"Trails": [None]},
            {"Trails": [*normal, normal[0]]},
            {"Trails": [{"Name": "missing-native-identity"}]},
            {"Trails": normal, "NextToken": "uncollected-page"},
        )
        for payload in malformed:
            with self.subTest(payload=payload):
                result, _ = self.run_preflight(payload)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("AFTER_CAPACITY", result.stdout)
                self.assertIn("CloudTrail", result.stderr)
        result, args = self.run_preflight({"Trails": []}, denied=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("AFTER_CAPACITY", result.stdout)
        self.assertIn("verify", result.stderr.lower())
        self.assertIn("list-trails", args)

    def test_four_trails_allow_exact_profile_region_and_count_shadow_trails(self):
        trails = [trail("other-" + str(i)) for i in range(3)]
        trails.append(trail("multi-region-shadow", region="ap-southeast-2"))
        result, args = self.run_preflight({"Trails": trails})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("4/5", result.stdout)
        self.assertIn("AFTER_CAPACITY", result.stdout)
        self.assertEqual(args[0], "--profile=fixture profile")
        self.assertEqual(args[1:3], ["cloudtrail", "list-trails"])
        self.assertEqual(args[args.index("--region") + 1], REGION)
        self.assertEqual(args[args.index("--output") + 1], "json")
        self.assertNotIn("--no-paginate", args)

    def test_existing_selected_home_trail_reuses_capacity_without_ownership_claim(self):
        other = [trail("other-" + str(i)) for i in range(4)]
        name = PREFIX + "-security-trail"
        result, _ = self.run_preflight({"Trails": [*other, trail(name)]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AFTER_CAPACITY", result.stdout)
        self.assertNotIn("owned", result.stdout.lower())
        for wrong in (trail(name, region="ap-southeast-2"), trail(name, account="9999" + "8888" + "7777")):
            with self.subTest(wrong_identity=wrong):
                result, _ = self.run_preflight({"Trails": [*other, wrong]})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("AFTER_CAPACITY", result.stdout)


if __name__ == "__main__":
    unittest.main()
