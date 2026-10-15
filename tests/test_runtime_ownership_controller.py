"""Stdlib-only controller ownership regressions; no AWS or credentials required."""

from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ownership_runtime_control", ROOT / "scripts/lib/agentcore_runtime_control.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ACCOUNT = "111122223333"  # AWS documentation account, never an execution target.


class ClosedRuntimeService:
    """Enforce AgentCore request shapes; tags are create-only.

    https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_CreateAgentRuntime.html
    https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_ListTagsForResource.html
    """

    def __init__(self, existing=False, tags=None):
        self.existing = existing
        self.tags = tags or {}
        self.requests = []
        self.runtime_id = "example-runtime"
        self.arn = "arn:aws:bedrock-agentcore:us-east-1:" + ACCOUNT + ":runtime/" + self.runtime_id
        self.endpoint_version = "1"
        self.current = {
            "agentRuntimeId": self.runtime_id,
            "agentRuntimeArn": self.arn,
            "agentRuntimeVersion": "1",
            "status": "READY",
            "platformVersion": "1.0.0",
        }

    def call(self, operation, payload):
        self.requests.append((operation, dict(payload)))
        if operation == "list-agent-runtimes":
            return {
                "agentRuntimes": [{"agentRuntimeName": "owned_runtime", "agentRuntimeId": self.runtime_id}]
                if self.existing
                else []
            }
        if operation == "list-tags-for-resource":
            assert payload == {"resourceArn": self.arn}
            return {"tags": dict(self.tags)}
        if operation in ("create-agent-runtime", "update-agent-runtime"):
            assert not (operation == "create-agent-runtime" and "metadataConfiguration" in payload)
            assert not (operation == "update-agent-runtime" and "tags" in payload)
            assert not set(payload) - {
                "agentRuntimeName",
                "agentRuntimeId",
                "agentRuntimeArtifact",
                "roleArn",
                "networkConfiguration",
                "protocolConfiguration",
                "environmentVariables",
                "clientToken",
                "platformVersion",
                "metadataConfiguration",
                "tags",
            }
            self.current.update(payload)
            self.current["agentRuntimeVersion"] = "1" if operation.startswith("create") else "2"
            self.endpoint_version = self.current["agentRuntimeVersion"]
            if "tags" in payload:
                self.tags = dict(payload["tags"])
            return {
                "agentRuntimeId": self.runtime_id,
                "agentRuntimeArn": self.arn,
                "agentRuntimeVersion": self.current["agentRuntimeVersion"],
                "status": "CREATING",
            }
        if operation == "get-agent-runtime":
            return dict(self.current)
        if operation == "get-agent-runtime-endpoint":
            return {
                "agentRuntimeArn": self.arn,
                "status": "READY",
                "liveVersion": self.endpoint_version,
            }
        if operation == "update-agent-runtime-endpoint":
            if payload["endpointName"] == "DEFAULT":
                raise MODULE.RuntimeDeployError(
                    "Default endpoints are managed through agent updates. Please use the update agent operation."
                )
            self.endpoint_version = payload["agentRuntimeVersion"]
            return {"targetVersion": self.endpoint_version}
        raise AssertionError("Unexpected service operation: " + operation)


class OwnershipControllerTests(unittest.TestCase):
    def options(self, deployment_id):
        return argparse.Namespace(
            region="us-east-1",
            name="owned_runtime",
            image="example-image",
            role="arn:aws:iam::" + ACCOUNT + ":role/example",
            network_mode="PUBLIC",
            protocol="HTTP",
            environment_json="{}",
            platform_version="1.0.0",
            tags_json=json.dumps({"amlas:deployment-prefix": "owned", "amlas:deployment-id": deployment_id}),
        )

    def test_create_ownership_tags_are_part_of_the_atomic_request(self):
        deployment_id = str(uuid.uuid4())
        service = ClosedRuntimeService()
        control = MODULE.RuntimeControl("us-east-1")
        control.call = service.call
        control.deploy(self.options(deployment_id))
        create = next(
            payload for operation, payload in service.requests if operation == "create-agent-runtime"
        )
        self.assertEqual(
            create.get("tags"), {"amlas:deployment-prefix": "owned", "amlas:deployment-id": deployment_id}
        )

    def test_foreign_uuid_refuses_every_runtime_update(self):
        service = ClosedRuntimeService(
            existing=True, tags={"amlas:deployment-prefix": "owned", "amlas:deployment-id": str(uuid.uuid4())}
        )
        control = MODULE.RuntimeControl("us-east-1")
        control.call = service.call
        with self.assertRaises(MODULE.RuntimeDeployError):
            control.deploy(self.options(str(uuid.uuid4())))
        self.assertFalse(any(operation == "update-agent-runtime" for operation, _ in service.requests))

    def test_matching_uuid_preserves_tags_and_default_endpoint_verification(self):
        deployment_id = str(uuid.uuid4())
        tags = {"amlas:deployment-prefix": "owned", "amlas:deployment-id": deployment_id}
        service = ClosedRuntimeService(existing=True, tags=tags)
        control = MODULE.RuntimeControl("us-east-1")
        control.call = service.call
        runtime_id, arn, version = control.deploy(self.options(deployment_id))
        self.assertEqual((runtime_id, arn, version), (service.runtime_id, service.arn, "2"))
        self.assertEqual(service.tags, tags)
        self.assertEqual(service.endpoint_version, "2")
        self.assertTrue(service.current["metadataConfiguration"]["requireMMDSV2"])
        operations = [operation for operation, _ in service.requests]
        update_index = operations.index("update-agent-runtime")
        verified_reads = [
            index
            for index, (operation, payload) in enumerate(service.requests)
            if index > update_index
            and operation == "get-agent-runtime"
            and payload.get("agentRuntimeVersion") == version
        ]
        endpoint_reads = [
            index for index, operation in enumerate(operations) if operation == "get-agent-runtime-endpoint"
        ]
        self.assertTrue(verified_reads, "accepted runtime update must be read back")
        self.assertTrue(endpoint_reads, "successful deploy must read back DEFAULT")
        self.assertGreater(endpoint_reads[0], verified_reads[-1])
        self.assertFalse(
            any(
                operation == "update-agent-runtime-endpoint" and payload.get("endpointName") == "DEFAULT"
                for operation, payload in service.requests
            )
        )

    def test_untagged_legacy_runtime_without_uuid_remains_supported(self):
        service = ClosedRuntimeService(existing=True)
        control = MODULE.RuntimeControl("us-east-1")
        control.call = service.call
        options = self.options(str(uuid.uuid4()))
        options.tags_json = "{}"
        _, _, version = control.deploy(options)
        self.assertEqual(version, "2")
        self.assertEqual(service.tags, {})


if __name__ == "__main__":
    unittest.main()
