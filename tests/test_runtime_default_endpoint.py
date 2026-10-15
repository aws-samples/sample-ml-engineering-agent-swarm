"""Native DEFAULT endpoint regression: automatic update, authoritative liveVersion."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "default_runtime_control", ROOT / "scripts/lib/agentcore_runtime_control.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/example-runtime"


class NativeDefaultContract:
    """Shape from native GetDEFAULT: targetVersion and agentRuntimeVersion absent."""

    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.operations = []

    def call(self, operation, payload):
        self.operations.append(operation)
        if operation == "update-agent-runtime-endpoint":
            raise MODULE.RuntimeDeployError(
                "ConflictException: Default endpoints are managed through agent updates. "
                "Please use the update agent operation."
            )
        if operation != "get-agent-runtime-endpoint":
            raise AssertionError("unexpected control operation")
        if payload != {"agentRuntimeId": "example-runtime", "endpointName": "DEFAULT"}:
            raise AssertionError("endpoint selector changed")
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


def snapshot(live="2", status="READY", **extra):
    value = {
        "liveVersion": live,
        "agentRuntimeEndpointArn": RUNTIME_ARN + "/runtime-endpoint/DEFAULT",
        "agentRuntimeArn": RUNTIME_ARN,
        "status": status,
        "createdAt": "2026-01-01T00:00:00+00:00",
        "lastUpdatedAt": "2026-01-01T00:01:00+00:00",
        "name": "DEFAULT",
        "id": "DEFAULT",
    }
    value.update(extra)
    return value


class DefaultEndpointTests(unittest.TestCase):
    def control(self, snapshots):
        service = NativeDefaultContract(snapshots)
        control = MODULE.RuntimeControl("us-east-1")
        control.attempts = 3
        control.delay = 0
        control.call = service.call
        return control, service

    def test_native_ready_default_without_target_is_usable_without_direct_update(self):
        control, service = self.control([snapshot()])
        control.bind_default("example-runtime", "2", RUNTIME_ARN)
        self.assertNotIn("update-agent-runtime-endpoint", service.operations)

    def test_nullable_default_target_is_not_a_required_current_version(self):
        control, service = self.control([snapshot(targetVersion=None)])
        control.bind_default("example-runtime", "2", RUNTIME_ARN)
        self.assertNotIn("update-agent-runtime-endpoint", service.operations)

    def test_default_follows_runtime_update_and_polls_until_requested_live_version(self):
        control, service = self.control([snapshot("1"), snapshot("2", "UPDATING"), snapshot()])
        control.bind_default("example-runtime", "2", RUNTIME_ARN)
        self.assertEqual(service.operations, ["get-agent-runtime-endpoint"] * 3)

    def test_missing_null_invalid_or_wrong_live_version_never_passes(self):
        documents = [snapshot(live) for live in (None, "0", "1", 2)]
        absent = snapshot()
        del absent["liveVersion"]
        documents.append(absent)
        for document in documents:
            with self.subTest(live=document.get("liveVersion"), present="liveVersion" in document):
                control, service = self.control([document])
                with self.assertRaises(MODULE.RuntimeDeployError):
                    control.bind_default("example-runtime", "2", RUNTIME_ARN)
                self.assertNotIn("update-agent-runtime-endpoint", service.operations)

    def test_ready_default_on_wrong_target_or_runtime_never_passes(self):
        for document in (snapshot(targetVersion="3"), snapshot(agentRuntimeArn=RUNTIME_ARN + "-foreign")):
            with self.subTest(document=document):
                control, service = self.control([document])
                with self.assertRaises(MODULE.RuntimeDeployError):
                    control.bind_default("example-runtime", "2", RUNTIME_ARN)
                self.assertNotIn("update-agent-runtime-endpoint", service.operations)

    def test_failed_default_does_not_pass(self):
        control, service = self.control([snapshot(status="UPDATE_FAILED")])
        with self.assertRaises(MODULE.RuntimeDeployError):
            control.bind_default("example-runtime", "2", RUNTIME_ARN)
        self.assertNotIn("update-agent-runtime-endpoint", service.operations)


if __name__ == "__main__":
    unittest.main()
