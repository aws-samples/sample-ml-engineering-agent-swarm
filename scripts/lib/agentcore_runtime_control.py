#!/usr/bin/env python3
"""Verify a managed runtime and its DEFAULT endpoint before returning a usable ARN."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid


class RuntimeDeployError(RuntimeError):
    """A required control operation or readback failed."""


class RuntimeControl:
    def __init__(self, region: str) -> None:
        self.region = region
        self.attempts = int(os.environ.get("AMLAS_RUNTIME_POLL_ATTEMPTS", "60"))
        self.delay = float(os.environ.get("AMLAS_RUNTIME_POLL_SECONDS", "10"))
        if not 1 <= self.attempts <= 60 or not 0 <= self.delay <= 60:
            raise RuntimeDeployError("Runtime poll settings require 1–60 attempts and 0–60 seconds.")

    def call(self, operation: str, payload: dict) -> dict:
        result = subprocess.run(
            [
                "aws",
                *(["--profile=" + os.environ["AWS_PROFILE"]] if os.environ.get("AWS_PROFILE") else []),
                "bedrock-agentcore-control",
                operation,
                "--region",
                self.region,
                "--cli-input-json",
                json.dumps(payload),
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeDeployError(f"{operation} failed: {result.stderr.strip() or result.stdout.strip()}")
        try:
            document = json.loads(result.stdout)
        except (ValueError, TypeError) as error:
            raise RuntimeDeployError(f"{operation} returned invalid JSON.") from error
        if not isinstance(document, dict):
            raise RuntimeDeployError(f"{operation} returned an invalid response object.")
        return document

    @staticmethod
    def version(document: dict) -> str:
        value = document.get("agentRuntimeVersion")
        if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]{0,4}", value):
            raise RuntimeDeployError("Runtime response is missing a valid version.")
        return value

    def wait_runtime(
        self,
        runtime_id: str,
        version: str,
        expected: dict,
        platform: str | None = None,
        require_mmds: bool = False,
    ) -> dict:
        for attempt in range(self.attempts):
            actual = self.call(
                "get-agent-runtime", {"agentRuntimeId": runtime_id, "agentRuntimeVersion": version}
            )
            status = actual.get("status")
            if status == "READY":
                if self.version(actual) != version:
                    raise RuntimeDeployError("READY runtime has the wrong requested version.")
                for key in (
                    "agentRuntimeArtifact",
                    "roleArn",
                    "environmentVariables",
                    "networkConfiguration",
                    "protocolConfiguration",
                ):
                    if actual.get(key) != expected[key]:
                        raise RuntimeDeployError(f"READY runtime does not match requested {key}.")
                if platform is not None and actual.get("platformVersion") != platform:
                    raise RuntimeDeployError("READY runtime has the wrong pinned platform.")
                if require_mmds and actual.get("metadataConfiguration", {}).get("requireMMDSV2") is not True:
                    raise RuntimeDeployError("READY runtime did not enforce MMDSv2.")
                return actual
            if status not in {"CREATING", "UPDATING"}:
                raise RuntimeDeployError(f"Runtime is not usable: status={status!r}.")
            if attempt + 1 < self.attempts:
                time.sleep(self.delay)
        raise RuntimeDeployError("Timed out waiting for the requested runtime version to become READY.")

    def bind_default(self, runtime_id: str, version: str, runtime_arn: str) -> None:
        # DEFAULT follows UpdateAgentRuntime; the service rejects direct endpoint
        # updates. Its native response can omit targetVersion, so liveVersion is
        # authoritative. Keep this behavior confined to the DEFAULT selector.
        request = {"agentRuntimeId": runtime_id, "endpointName": "DEFAULT"}
        for attempt in range(self.attempts):
            actual = self.call("get-agent-runtime-endpoint", request)
            if actual.get("agentRuntimeArn") != runtime_arn:
                raise RuntimeDeployError("DEFAULT endpoint runtime changed during verification.")
            status = actual.get("status")
            if status not in {"READY", "CREATING", "UPDATING"}:
                raise RuntimeDeployError(f"DEFAULT endpoint is not usable: {status!r}.")
            if (
                status == "READY"
                and actual.get("liveVersion") == version
                and actual.get("targetVersion") in (None, version)
            ):
                return
            if attempt + 1 < self.attempts:
                time.sleep(self.delay)
        raise RuntimeDeployError("DEFAULT endpoint did not become READY on the requested version.")

    def verify_ownership_tags(self, runtime_arn: str, expected: dict, legacy_default: bool = False) -> None:
        actual = self.call("list-tags-for-resource", {"resourceArn": runtime_arn}).get("tags")
        if not isinstance(actual, dict):
            raise RuntimeDeployError("Runtime ownership tags could not be verified.")
        expected_id = expected.get("amlas:deployment-id")
        actual_id = actual.get("amlas:deployment-id")
        if (expected_id or actual_id) and expected_id != actual_id:
            raise RuntimeDeployError(
                "Selected runtime belongs to a different deployment UUID; no runtime update was attempted."
            )
        expected_prefix = expected.get("amlas:deployment-prefix")
        actual_prefix = actual.get("amlas:deployment-prefix")
        if expected_prefix and expected_prefix != actual_prefix:
            if not (
                legacy_default
                and expected_prefix == "amlas"
                and not actual_prefix
                and not expected_id
                and not actual_id
            ):
                raise RuntimeDeployError(
                    "Selected runtime deployment prefix is incompatible; no runtime update was attempted."
                )

    def deploy(self, options: argparse.Namespace) -> tuple[str, str, str]:
        environment = json.loads(options.environment_json)
        if not isinstance(environment, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in environment.items()
        ):
            raise RuntimeDeployError("Runtime environment must be a JSON string map.")
        if {
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AMLAS_AWS_PROFILE",
            "AGENTICML_AWS_PROFILE",
        } & environment.keys():
            raise RuntimeDeployError("Developer profiles cannot be injected into a managed runtime.")
        tags = json.loads(getattr(options, "tags_json", "{}"))
        if not isinstance(tags, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in tags.items()
        ):
            raise RuntimeDeployError("Runtime ownership tags must be a JSON string map.")
        desired = {
            "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": options.image}},
            "roleArn": options.role,
            "networkConfiguration": {"networkMode": options.network_mode},
            "protocolConfiguration": {"serverProtocol": options.protocol},
            "environmentVariables": environment,
        }
        inventory = self.call("list-agent-runtimes", {})
        matches = [
            item
            for item in inventory.get("agentRuntimes", [])
            if item.get("agentRuntimeName") == options.name
        ]
        if len(matches) > 1:
            raise RuntimeDeployError("Multiple runtimes match the selected name; resolve the ambiguity.")
        platform = options.platform_version
        if matches:
            runtime_id = matches[0].get("agentRuntimeId")
            if not runtime_id:
                raise RuntimeDeployError("Runtime discovery returned no valid ID.")
            current = self.call("get-agent-runtime", {"agentRuntimeId": runtime_id})
            current_arn = current.get("agentRuntimeArn")
            if not isinstance(current_arn, str) or not current_arn:
                raise RuntimeDeployError("Selected runtime discovery returned no ownership ARN.")
            self.verify_ownership_tags(current_arn, tags, legacy_default=True)
            platform = platform or current.get("platformVersion")
        else:
            create = dict(desired, agentRuntimeName=options.name, clientToken=str(uuid.uuid4()))
            if tags:
                create["tags"] = tags
            if platform:
                create["platformVersion"] = platform
            response = None
            for attempt in range(1, 11):
                try:
                    response = self.call("create-agent-runtime", create)
                    break
                except RuntimeDeployError as error:
                    message = str(error).lower()
                    if "validating ecr uri" not in message or "access denied" not in message or attempt == 10:
                        raise
                    time.sleep(attempt * 10)
            if response is None:
                raise RuntimeDeployError("Runtime creation returned no response.")
            runtime_id = response.get("agentRuntimeId")
            if not runtime_id:
                raise RuntimeDeployError("Runtime creation returned no ID.")
            created = self.wait_runtime(runtime_id, self.version(response), desired, platform)
            if tags:
                created_arn = created.get("agentRuntimeArn")
                if not isinstance(created_arn, str) or not created_arn:
                    raise RuntimeDeployError("Created runtime returned no ownership ARN.")
                self.verify_ownership_tags(created_arn, tags)
            platform = platform or created.get("platformVersion")
        if not isinstance(platform, str) or not platform:
            raise RuntimeDeployError(
                "Platform version could not be discovered; upgrade the CLI or select AGENTCORE_PLATFORM_VERSION."
            )
        update = dict(
            desired,
            agentRuntimeId=runtime_id,
            platformVersion=platform,
            metadataConfiguration={"requireMMDSV2": True},
            clientToken=str(uuid.uuid4()),
        )
        response = self.call("update-agent-runtime", update)
        version = self.version(response)
        actual = self.wait_runtime(runtime_id, version, desired, platform, require_mmds=True)
        runtime_arn = actual.get("agentRuntimeArn")
        if (
            not isinstance(runtime_arn, str)
            or not runtime_arn
            or runtime_arn != response.get("agentRuntimeArn")
        ):
            raise RuntimeDeployError("Verified runtime ARN does not match the accepted update response.")
        self.bind_default(runtime_id, version, runtime_arn)
        return runtime_id, runtime_arn, version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("region", "name", "image", "role", "network-mode", "protocol", "environment-json"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--platform-version", default=os.environ.get("AGENTCORE_PLATFORM_VERSION") or None)
    parser.add_argument(
        "--tags-json",
        default="{}",
        help="Ownership string map; tags are applied atomically at runtime creation.",
    )
    options = parser.parse_args()
    try:
        runtime_id, arn, version = RuntimeControl(options.region).deploy(options)
    except (RuntimeDeployError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"AGENTCORE_RUNTIME_ID={runtime_id}")
    print(f"AGENTCORE_RUNTIME_ARN={arn}")
    print(f"AGENTCORE_RUNTIME_VERSION={version}")
    print("AGENTCORE_RUNTIME_ENDPOINT=DEFAULT")
    print("FINAL_STATUS=READY")
    print("AGENTCORE_RUNTIME_DEPLOY_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
