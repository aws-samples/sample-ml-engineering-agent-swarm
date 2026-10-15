"""Exact-identity cleanup tests; all AWS boundaries use a closed native-state fake."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "0" * 12
REGION = "us-east-1"
PREFIX = "trial-blue"
ACTOR = f"arn:aws:iam::{ACCOUNT}:role/cdk-example-cfn-exec"


def manifest_module() -> Any:
    path = ROOT / "scripts/lib/deployment_manifest.py"
    assert path.is_file(), "missing stdlib exact-identity deployment manifest module"
    spec = importlib.util.spec_from_file_location("cleanup_manifest_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NativeAWS:
    """Reject unknown calls and foreign deletes; retain native IDs and all object versions."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.account = ACCOUNT
        self.calls: list[list[str]] = []
        self.stacks: dict[str, dict[str, Any]] = {}
        self.buckets: dict[str, dict[str, Any]] = {}
        self.tables: dict[str, dict[str, Any]] = {}
        self.repositories: dict[str, dict[str, Any]] = {}
        self.training_jobs: dict[str, dict[str, Any]] = {}
        self.processing_jobs: dict[str, dict[str, Any]] = {}
        self.actor_role_id = "actor-native-id"
        self.delete_failed_stack = False
        self.keys: dict[str, dict[str, Any]] = {}
        self.groups: dict[str, dict[str, Any]] = {}
        self.params: dict[str, dict[str, Any]] = {}
        self.logs: dict[str, dict[str, Any]] = {}
        self.secret: dict[str, Any] | None = None
        self.runtime: dict[str, Any] | None = None
        self.owner = ""
        self.fail: tuple[str, str] | None = None
        self.deleted_versions: list[dict[str, str]] = []

    @property
    def mutations(self) -> list[list[str]]:
        return [x for x in self.calls if x[1].startswith(("delete-", "schedule-", "stop-", "batch-delete-"))]

    def missing(self) -> None:
        raise self.module.AwsError("ResourceNotFoundException", missing=True)

    def __call__(self, args: list[str]) -> dict[str, Any]:
        self.calls.append(args)
        op = tuple(args[:2])

        def value(flag: str) -> str:
            assert flag in args, (op, flag, args)
            return args[args.index(flag) + 1]

        if op == self.fail:
            raise self.module.AwsError("AccessDeniedException")
        if op == ("sts", "get-caller-identity"):
            return {
                "Account": self.account,
                "Arn": f"arn:aws:sts::{self.account}:assumed-role/example/session",
            }
        if op == ("cloudformation", "describe-stacks"):
            key = value("--stack-name")
            found = next((s for name, s in self.stacks.items() if key in (name, s["StackId"])), None)
            if found is None:
                self.missing()
            return {"Stacks": [{k: v for k, v in found.items() if k != "members"}]}
        if op == ("cloudformation", "list-stack-resources"):
            key = value("--stack-name")
            found = next((s for name, s in self.stacks.items() if key in (name, s["StackId"])), None)
            if found is None:
                self.missing()
            return {"StackResourceSummaries": found["members"]}
        if op == ("cloudformation", "delete-stack"):
            key = value("--stack-name")
            assert value("--role-arn") == ACTOR, "cleanup changed the stack's service actor"
            found = next(((name, s) for name, s in self.stacks.items() if key == s["StackId"]), None)
            assert found is not None, "cleanup must delete by recorded native StackId"
            assert self.tags(found[1]["Tags"]).get("amlas:deployment-id") == self.owner
            if self.delete_failed_stack:
                self.stacks[found[0]]["StackStatus"] = "DELETE_FAILED"
            else:
                del self.stacks[found[0]]
            return {}
        if op == ("iam", "get-role"):
            name = value("--role-name")
            if name == "cdk-example-cfn-exec":
                return {"Role": {"RoleName": name, "Arn": ACTOR, "RoleId": self.actor_role_id}}
            self.missing()
        if op == ("bedrock-agentcore-control", "list-agent-runtimes"):
            return {"agentRuntimes": [self.runtime] if self.runtime else []}
        if op == ("bedrock-agentcore-control", "get-agent-runtime"):
            if not self.runtime or value("--agent-runtime-id") != self.runtime["agentRuntimeId"]:
                self.missing()
            return self.runtime
        if op == ("bedrock-agentcore-control", "list-tags-for-resource"):
            return {"tags": self.runtime["tags"] if self.runtime else {}}
        if op == ("bedrock-agentcore", "stop-runtime-session"):
            assert self.runtime and value("--agent-runtime-arn") == self.runtime["agentRuntimeArn"]
            assert value("--runtime-session-id") == "owned-session-id-00000000000000000001"
            return {}
        if op == ("bedrock-agentcore-control", "delete-agent-runtime"):
            assert self.runtime and value("--agent-runtime-id") == self.runtime["agentRuntimeId"]
            assert self.runtime["tags"]["amlas:deployment-id"] == self.owner
            self.runtime = None
            return {}
        if op == ("secretsmanager", "describe-secret"):
            if self.secret is None or value("--secret-id") not in (self.secret["Name"], self.secret["ARN"]):
                self.missing()
            return self.secret
        if op == ("secretsmanager", "delete-secret"):
            assert self.secret and value("--secret-id") == self.secret["ARN"]
            assert self.tags(self.secret["Tags"])["amlas:deployment-id"] == self.owner
            self.secret = None
            return {}
        if op == ("ssm", "describe-parameters"):
            return {"Parameters": list(self.params.values())}
        if op == ("ssm", "list-tags-for-resource"):
            return {"TagList": self.params[value("--resource-id")]["Tags"]}
        if op == ("ssm", "delete-parameter"):
            name = value("--name")
            assert self.tags(self.params[name]["Tags"])["amlas:deployment-id"] == self.owner
            del self.params[name]
            return {}
        if op == ("s3api", "get-bucket-location"):
            name = value("--bucket")
            if name not in self.buckets:
                self.missing()
            assert value("--expected-bucket-owner") == ACCOUNT
            return {
                "LocationConstraint": self.buckets[name]["region"]
                if self.buckets[name]["region"] != "us-east-1"
                else None
            }
        if op == ("s3api", "get-bucket-tagging"):
            return {"TagSet": self.buckets[value("--bucket")]["Tags"]}
        if op == ("s3api", "list-object-versions"):
            bucket = self.buckets[value("--bucket")]
            return {"Versions": list(bucket["versions"]), "DeleteMarkers": list(bucket["markers"])}
        if op == ("s3api", "list-objects-v2"):
            return {"Contents": []}
        if op == ("s3api", "delete-objects"):
            bucket = self.buckets[value("--bucket")]
            assert self.tags(bucket["Tags"])["amlas:deployment-id"] == self.owner
            objects = json.loads(value("--delete"))["Objects"]
            self.deleted_versions.extend(objects)
            for kind in ("versions", "markers"):
                bucket[kind] = [
                    v for v in bucket[kind] if {"Key": v["Key"], "VersionId": v["VersionId"]} not in objects
                ]
            return {}
        if op == ("s3api", "delete-bucket"):
            name = value("--bucket")
            bucket = self.buckets[name]
            assert not bucket["versions"] and not bucket["markers"]
            assert self.tags(bucket["Tags"])["amlas:deployment-id"] == self.owner
            del self.buckets[name]
            return {}
        if op == ("dynamodb", "describe-table"):
            name = value("--table-name")
            if name not in self.tables:
                self.missing()
            return {"Table": self.tables[name]}
        if op == ("dynamodb", "list-tags-of-resource"):
            return {
                "Tags": next(
                    t["Tags"] for t in self.tables.values() if t["TableArn"] == value("--resource-arn")
                )
            }
        if op == ("dynamodb", "delete-table"):
            name = value("--table-name")
            assert self.tags(self.tables[name]["Tags"])["amlas:deployment-id"] == self.owner
            del self.tables[name]
            return {}
        if op == ("kms", "describe-key"):
            key = value("--key-id")
            found = next((k for kid, k in self.keys.items() if key in (kid, k["Arn"])), None)
            if found is None:
                self.missing()
            return {"KeyMetadata": found}
        if op == ("kms", "list-resource-tags"):
            return {
                "Tags": next(
                    k["Tags"] for kid, k in self.keys.items() if value("--key-id") in (kid, k["Arn"])
                )
            }
        if op == ("kms", "schedule-key-deletion"):
            key = value("--key-id")
            found = next(k for kid, k in self.keys.items() if key in (kid, k["Arn"]))
            assert self.tags(found["Tags"])["amlas:deployment-id"] == self.owner
            assert value("--pending-window-in-days") == "7"
            found.update(KeyState="PendingDeletion", DeletionDate="2030-01-08T00:00:00Z")
            return {
                "KeyId": found["KeyId"],
                "KeyState": "PendingDeletion",
                "DeletionDate": found["DeletionDate"],
            }
        if op in (("sagemaker", "list-training-jobs"), ("sagemaker", "list-processing-jobs")):
            training = op[1] == "list-training-jobs"
            jobs = self.training_jobs if training else self.processing_jobs
            key = "TrainingJobSummaries" if training else "ProcessingJobSummaries"
            return {key: list(jobs.values())}
        if op in (("sagemaker", "describe-training-job"), ("sagemaker", "describe-processing-job")):
            training = op[1] == "describe-training-job"
            jobs = self.training_jobs if training else self.processing_jobs
            flag = "--training-job-name" if training else "--processing-job-name"
            name = value(flag)
            if name not in jobs:
                self.missing()
            return jobs[name]
        if op in (("sagemaker", "stop-training-job"), ("sagemaker", "stop-processing-job")):
            training = op[1] == "stop-training-job"
            jobs = self.training_jobs if training else self.processing_jobs
            flag = "--training-job-name" if training else "--processing-job-name"
            job = jobs[value(flag)]
            assert self.tags(job["Tags"])["amlas:deployment-id"] == self.owner
            job["TrainingJobStatus" if training else "ProcessingJobStatus"] = "Stopped"
            return {}
        if op == ("sagemaker", "describe-model-package-group"):
            name = value("--model-package-group-name")
            if name not in self.groups:
                self.missing()
            return self.groups[name]
        if op == ("sagemaker", "list-tags"):
            arn = value("--resource-arn")
            for collection, arn_key in (
                (self.groups, "ModelPackageGroupArn"),
                (self.training_jobs, "TrainingJobArn"),
                (self.processing_jobs, "ProcessingJobArn"),
            ):
                for item in collection.values():
                    if item[arn_key] == arn:
                        return {"Tags": item["Tags"]}
            self.missing()
        if op == ("sagemaker", "list-model-packages"):
            return {"ModelPackageSummaryList": []}
        if op == ("sagemaker", "delete-model-package-group"):
            name = value("--model-package-group-name")
            assert self.tags(self.groups[name]["Tags"])["amlas:deployment-id"] == self.owner
            del self.groups[name]
            return {}
        if op == ("ecr", "describe-repositories"):
            name = value("--repository-names")
            if name not in self.repositories:
                self.missing()
            return {"repositories": [self.repositories[name]]}
        if op == ("ecr", "list-tags-for-resource"):
            return {
                "tags": next(
                    r["tags"]
                    for r in self.repositories.values()
                    if r["repositoryArn"] == value("--resource-arn")
                )
            }
        if op == ("ecr", "list-images"):
            return {"imageIds": list(self.repositories[value("--repository-name")]["images"])}
        if op == ("ecr", "batch-delete-image"):
            repo = self.repositories[value("--repository-name")]
            assert self.tags(repo["tags"])["amlas:deployment-id"] == self.owner
            image_ids = json.loads(value("--image-ids"))
            # AWS BatchDeleteImage accepts at most 100 image IDs per request:
            # https://docs.aws.amazon.com/AmazonECR/latest/APIReference/API_BatchDeleteImage.html
            assert 1 <= len(image_ids) <= 100
            repo["images"] = [image for image in repo["images"] if image not in image_ids]
            return {"failures": []}
        if op == ("ecr", "delete-repository"):
            repo = self.repositories[value("--repository-name")]
            assert self.tags(repo["tags"])["amlas:deployment-id"] == self.owner
            assert "--force" in args
            del self.repositories[value("--repository-name")]
            return {}
        if op == ("logs", "describe-log-groups"):
            return {"logGroups": list(self.logs.values())}
        if op == ("logs", "list-tags-for-resource"):
            return {
                "tags": next(
                    g["tags"] for g in self.logs.values() if g["logGroupArn"] == value("--resource-arn")
                )
            }
        if op == ("logs", "delete-log-group"):
            name = value("--log-group-name")
            del self.logs[name]
            return {}
        raise AssertionError(f"unsupported AWS call {op}")

    @staticmethod
    def tags(items: Any) -> dict[str, str]:
        if isinstance(items, dict):
            return items
        return {t.get("Key", t.get("TagKey")): t.get("Value", t.get("TagValue")) for t in items}

    def populate_owned(self, owner: str) -> None:
        self.owner = owner
        tags = [
            {"Key": "amlas:deployment-id", "Value": owner},
            {"Key": "amlas:deployment-prefix", "Value": PREFIX},
        ]
        bucket_name = PREFIX + "-native-lake"
        table_name = PREFIX + "-training-jobs"
        group_name = PREFIX + "-models"
        key_id = "00000000-0000-0000-0000-aaaaaaaaaaa3"
        members = [
            ("AWS::S3::Bucket", bucket_name),
            ("AWS::DynamoDB::Table", table_name),
            ("AWS::KMS::Key", key_id),
            ("AWS::SageMaker::ModelPackageGroup", group_name),
        ]
        for index, name in enumerate((PREFIX + "-LakeStack", PREFIX + "-runtime-role")):
            self.stacks[name] = {
                "StackName": name,
                "StackId": f"arn:aws:cloudformation:{REGION}:{ACCOUNT}:stack/{name}/00000000-0000-0000-0000-{index + 1:012d}",
                "StackStatus": "CREATE_COMPLETE",
                "RoleARN": ACTOR,
                "Tags": tags,
                "Outputs": [],
                "members": [
                    {
                        "LogicalResourceId": f"Resource{i}",
                        "PhysicalResourceId": native,
                        "ResourceType": kind,
                        "ResourceStatus": "CREATE_COMPLETE",
                    }
                    for i, (kind, native) in enumerate(members)
                ]
                if index == 0
                else [],
            }
        self.buckets[bucket_name] = {
            "region": REGION,
            "Tags": tags,
            "versions": [{"Key": "artifact", "VersionId": "one"}, {"Key": "artifact", "VersionId": "two"}],
            "markers": [{"Key": "artifact", "VersionId": "marker"}],
        }
        self.tables[table_name] = {
            "TableName": table_name,
            "TableId": "owned-table-id",
            "TableArn": f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{table_name}",
            "Tags": tags,
        }
        self.keys[key_id] = {
            "KeyId": key_id,
            "Arn": f"arn:aws:kms:{REGION}:{ACCOUNT}:key/{key_id}",
            "KeyManager": "CUSTOMER",
            "KeyState": "Enabled",
            "Tags": [{"TagKey": t["Key"], "TagValue": t["Value"]} for t in tags],
        }
        self.groups[group_name] = {
            "ModelPackageGroupName": group_name,
            "ModelPackageGroupArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package-group/{group_name}",
            "CreationTime": "2030-01-01T00:00:00Z",
            "Tags": tags,
        }
        self.secret = {
            "Name": PREFIX + "/tavily-api-key",
            "ARN": f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:{PREFIX}/tavily-api-key-native",
            "CreatedDate": "2030-01-01T00:00:00Z",
            "Tags": tags,
        }
        self.params["/" + PREFIX + "/agentcore-runtime-arn"] = {
            "Name": "/" + PREFIX + "/agentcore-runtime-arn",
            "Version": 1,
            "LastModifiedDate": "2030-01-01T00:00:00Z",
            "Tags": tags,
        }
        runtime_id = "trial_blue-0000000001"
        self.runtime = {
            "agentRuntimeId": runtime_id,
            "agentRuntimeName": "trial_blue",
            "agentRuntimeArn": f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/{runtime_id}",
            "status": "READY",
            "tags": {"amlas:deployment-id": owner, "amlas:deployment-prefix": PREFIX},
        }


def initialized(tmp_path: Path) -> tuple[Any, NativeAWS, Path]:
    module = manifest_module()
    aws = NativeAWS(module)
    path = tmp_path / "deployment-manifest.json"
    result = module.init_manifest(
        path,
        account=ACCOUNT,
        region=REGION,
        prefix=PREFIX,
        stacks=[PREFIX + "-LakeStack"],
        secret_id=PREFIX + "/tavily-api-key",
        runtime_name="trial_blue",
        role_stack=PREFIX + "-runtime-role",
        runner=aws,
    )
    aws.populate_owned(result["deployment_id"])
    module.refresh_manifest(path, runner=aws)
    aws.calls.clear()
    return module, aws, path


class DeploymentManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_init_is_read_only_and_reuses_manifest_uuid(self) -> None:
        module, aws, path = initialized(self.path)
        self.assertFalse(aws.mutations)
        original = json.loads(path.read_text())["deployment_id"]
        result = module.init_manifest(
            path,
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[PREFIX + "-LakeStack"],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            role_stack=PREFIX + "-runtime-role",
            runner=aws,
        )
        self.assertEqual(result["deployment_id"], original)
        self.assertFalse(aws.mutations)

    def test_nondefault_init_refuses_occupied_unowned_target(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        aws.buckets[f"{PREFIX}-lake-{ACCOUNT}-{REGION}"] = {
            "region": REGION,
            "Tags": [{"Key": "amlas:deployment-id", "Value": "foreign"}],
            "versions": [],
            "markers": [],
        }
        with self.assertRaisesRegex(module.LedgerError, "occupied|ownership"):
            module.init_manifest(
                self.path / "manifest.json",
                account=ACCOUNT,
                region=REGION,
                prefix=PREFIX,
                stacks=[],
                secret_id=PREFIX + "/tavily-api-key",
                runtime_name="trial_blue",
                runner=aws,
            )
        self.assertFalse(aws.mutations)

    def check_guard(self, change: str) -> None:
        module, aws, path = initialized(self.path)
        options: dict[str, Any] = {}
        if change == "account":
            aws.account = "1" * 12
        elif change == "region":
            options["region"] = "eu-west-1"
        elif change == "identity":
            aws.tables[PREFIX + "-training-jobs"]["TableId"] = "replacement-table-id"
        else:
            aws.buckets[PREFIX + "-native-lake"]["Tags"] = [
                {"Key": "amlas:deployment-id", "Value": "foreign"}
            ]
        with self.assertRaises(module.LedgerError):
            module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None, **options)
        self.assertFalse(aws.mutations)

    def test_wrong_account_fails_before_first_delete(self) -> None:
        self.check_guard("account")

    def test_wrong_region_fails_before_first_delete(self) -> None:
        self.check_guard("region")

    def test_changed_native_identity_fails_before_first_delete(self) -> None:
        self.check_guard("identity")

    def test_foreign_tag_fails_before_first_delete(self) -> None:
        self.check_guard("tag")

    def test_plan_is_read_only_and_contains_runtime_auxiliary_stack_and_native_resources(self) -> None:
        module, aws, path = initialized(self.path)
        result = module.plan_cleanup(path, runner=aws)
        self.assertFalse(aws.mutations)
        encoded = json.dumps(result)
        self.assertIn("runtime", encoded)
        self.assertIn("runtime-role", encoded)
        self.assertIn("owned-table-id", encoded)
        self.assertNotIn("CDKToolkit", encoded)

    def test_refresh_preserves_handles_when_later_native_read_fails(self) -> None:
        module, aws, path = initialized(self.path)
        aws.fail = ("s3api", "get-bucket-tagging")
        with self.assertRaises(module.LedgerError):
            module.refresh_manifest(path, runner=aws)
        text = path.read_text()
        self.assertIn("stack/trial-blue-LakeStack/", text)
        self.assertIn("stack/trial-blue-runtime-role/", text)
        self.assertIn("trial_blue-0000000001", text)

    def test_cleanup_only_owned_native_ids_and_all_versions(self) -> None:
        module, aws, path = initialized(self.path)
        foreign = [{"Key": "amlas:deployment-id", "Value": "foreign"}]
        aws.buckets[PREFIX + "-foreign"] = {"region": REGION, "Tags": foreign, "versions": [], "markers": []}
        aws.groups[PREFIX + "-foreign"] = {
            "ModelPackageGroupName": PREFIX + "-foreign",
            "ModelPackageGroupArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package-group/{PREFIX}-foreign",
            "CreationTime": "2030-01-01T00:00:00Z",
            "Tags": foreign,
        }
        aws.keys["foreign-key"] = {
            "KeyId": "foreign-key",
            "Arn": f"arn:aws:kms:{REGION}:{ACCOUNT}:key/foreign-key",
            "KeyManager": "CUSTOMER",
            "Description": "Amlas misleading foreign description",
            "KeyState": "Enabled",
            "Tags": [],
        }
        aws.params["/" + PREFIX + "/foreign"] = {
            "Name": "/" + PREFIX + "/foreign",
            "Version": 1,
            "LastModifiedDate": "2030-01-01T00:00:00Z",
            "Tags": foreign,
        }
        module.record_session(
            path, runtime_id="trial_blue-0000000001", session_id="owned-session-id-00000000000000000001"
        )
        result = module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertIn(PREFIX + "-foreign", aws.buckets)
        self.assertIn(PREFIX + "-foreign", aws.groups)
        self.assertIn("foreign-key", aws.keys)
        self.assertIn("/" + PREFIX + "/foreign", aws.params)
        self.assertEqual(aws.keys["00000000-0000-0000-0000-aaaaaaaaaaa3"]["KeyState"], "PendingDeletion")
        self.assertEqual(len(result["pending_kms_keys"]), 1)
        self.assertEqual(result["remaining_owned_resources"], [])
        self.assertEqual({x["VersionId"] for x in aws.deleted_versions}, {"one", "two", "marker"})
        self.assertFalse(aws.runtime or aws.stacks or aws.secret)
        operations = [tuple(x[:2]) for x in aws.mutations]
        self.assertLess(
            operations.index(("bedrock-agentcore", "stop-runtime-session")),
            operations.index(("bedrock-agentcore-control", "delete-agent-runtime")),
        )
        self.assertLess(
            operations.index(("bedrock-agentcore-control", "delete-agent-runtime")),
            operations.index(("cloudformation", "delete-stack")),
        )
        self.assertTrue(all(x[:2] not in [["iam", "delete-role"], ["iam", "create-role"]] for x in aws.calls))
        self.assertFalse(any(x[:2] in [["s3api", "list-buckets"], ["kms", "list-keys"]] for x in aws.calls))

    def test_teardown_requires_manifest_and_never_nukes_shared_bootstrap(self) -> None:
        script = (ROOT / "scripts/teardown.sh").read_text()
        self.assertIn("deployment_manifest.py", script)
        self.assertIn("--manifest", script)
        self.assertNotIn("discover_session_buckets", script)
        self.assertNotIn("discover_amlas_cmks", script)
        self.assertNotIn("cdk destroy", script)
        self.assertIn("not owned", script)

    def runtime_log(self, aws: NativeAWS, *, foreign: bool = False) -> str:
        # Synthetic native runtime log identity, with the exact original value.
        name = "/aws/bedrock-agentcore/" + "runtimes/trial_blue-" + "0000000001-DEFAULT"
        aws.logs[name] = {
            "logGroupName": name,
            "logGroupArn": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{name}",
            "creationTime": 1893456000000,
            "tags": {"amlas:deployment-id": "foreign"} if foreign else {},
        }
        return name

    def test_late_runtime_logs_are_bound_to_the_recorded_runtime_identity(self) -> None:
        module, aws, path = initialized(self.path)
        name = self.runtime_log(aws)
        result = module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertNotIn(name, aws.logs)
        self.assertEqual(result["remaining_owned_resources"], [])

    def test_explicit_foreign_uuid_on_matching_runtime_log_prefix_is_protected(self) -> None:
        module, aws, path = initialized(self.path)
        name = self.runtime_log(aws, foreign=True)
        module.refresh_manifest(path, runner=aws)
        module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertIn(name, aws.logs)
        self.assertFalse(any(x[:2] == ["logs", "delete-log-group"] for x in aws.mutations))

    def test_lambda_logs_use_exact_native_function_membership(self) -> None:
        module, aws, path = initialized(self.path)
        function_name = PREFIX + "-native-callback"
        aws.stacks[PREFIX + "-LakeStack"]["members"].append(
            {
                "ResourceType": "AWS::Lambda::Function",
                "PhysicalResourceId": function_name,
                "LogicalResourceId": "Callback",
                "ResourceStatus": "CREATE_COMPLETE",
            }
        )
        name = "/aws/lambda/" + function_name
        aws.logs[name] = {
            "logGroupName": name,
            "logGroupArn": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{name}",
            "creationTime": 1893456000000,
            "tags": {},
        }
        module.refresh_manifest(path, runner=aws)
        module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertNotIn(name, aws.logs)

    def test_incomplete_refresh_verification_blocks_plan_before_mutation(self) -> None:
        module, aws, path = initialized(self.path)
        aws.fail = ("s3api", "get-bucket-tagging")
        with self.assertRaises(module.LedgerError):
            module.refresh_manifest(path, runner=aws)
        aws.fail = None
        with self.assertRaisesRegex(module.LedgerError, "refresh|verification|incomplete"):
            module.plan_cleanup(path, runner=aws)
        self.assertFalse(aws.mutations)

    def test_legacy_default_preexisting_resources_are_never_adopted(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        aws.groups["amlas-models"] = {
            "ModelPackageGroupName": "amlas-models",
            "ModelPackageGroupArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package-group/amlas-models",
            "CreationTime": "2030-01-01T00:00:00Z",
            "Tags": [],
        }
        path = self.path / "legacy.json"
        with self.assertRaisesRegex(module.LedgerError, "occupied|ownership"):
            module.init_manifest(
                path,
                account=ACCOUNT,
                region=REGION,
                prefix="amlas",
                stacks=[],
                secret_id="amlas/tavily-api-key",
                runtime_name="amlas",
                runner=aws,
            )
        self.assertFalse(aws.mutations)
        self.assertIn("amlas-models", aws.groups)

    def test_first_refresh_requires_tags_on_expected_native_resources(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        path = self.path / "untagged.json"
        result = module.init_manifest(
            path,
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[PREFIX + "-LakeStack"],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            role_stack=PREFIX + "-runtime-role",
            runner=aws,
        )
        aws.populate_owned(result["deployment_id"])
        aws.buckets[PREFIX + "-native-lake"]["Tags"] = []
        with self.assertRaisesRegex(module.LedgerError, "ownership|tag|verification"):
            module.refresh_manifest(path, runner=aws)
        self.assertIn(PREFIX + "-native-lake", path.read_text())
        with self.assertRaises(module.LedgerError):
            module.plan_cleanup(path, runner=aws)
        self.assertFalse(aws.mutations)

    def test_changed_ssm_publication_revision_fails_before_first_delete(self) -> None:
        module, aws, path = initialized(self.path)
        aws.params["/" + PREFIX + "/agentcore-runtime-arn"]["Version"] = 2
        with self.assertRaises(module.LedgerError):
            module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertFalse(aws.mutations)

    def test_retained_ecr_uses_native_identity_and_service_batch_limits(self) -> None:
        module, aws, path = initialized(self.path)
        name = PREFIX + "-runtime"
        tags = [
            {"Key": "amlas:deployment-id", "Value": aws.owner},
            {"Key": "amlas:deployment-prefix", "Value": PREFIX},
        ]
        aws.repositories[name] = {
            "repositoryName": name,
            "repositoryArn": f"arn:aws:ecr:{REGION}:{ACCOUNT}:repository/{name}",
            "createdAt": "2030-01-01T00:00:00Z",
            "tags": tags,
            "images": [{"imageDigest": "sha256:" + str(index).zfill(64)} for index in range(101)],
        }
        aws.stacks[PREFIX + "-LakeStack"]["members"].append(
            {
                "ResourceType": "AWS::ECR::Repository",
                "PhysicalResourceId": name,
                "LogicalResourceId": "Repository",
                "ResourceStatus": "CREATE_COMPLETE",
            }
        )
        module.refresh_manifest(path, runner=aws)
        module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertNotIn(name, aws.repositories)
        batches = [x for x in aws.mutations if x[:2] == ["ecr", "batch-delete-image"]]
        self.assertEqual([len(json.loads(x[x.index("--image-ids") + 1])) for x in batches], [100, 1])

    def test_changed_cloudformation_actor_identity_blocks_first_delete(self) -> None:
        module, aws, path = initialized(self.path)
        aws.actor_role_id = "replacement-actor-role-id"
        with self.assertRaises(module.LedgerError):
            module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertFalse(aws.mutations)

    def test_failed_cloudformation_delete_preserves_receipt_and_never_claims_success(self) -> None:
        module, aws, path = initialized(self.path)
        aws.delete_failed_stack = True
        with self.assertRaisesRegex(module.LedgerError, "deletion failed"):
            module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None, poll_attempts=2)
        self.assertIn("stack/trial-blue-runtime-role/", path.read_text())
        self.assertIn(PREFIX + "-runtime-role", aws.stacks)
        self.assertNotIn('"remaining_owned_resources": []', path.read_text())

    def test_actual_ssm_root_and_model_group_are_immutable_selectors(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        path = self.path / "selectors.json"
        module.init_manifest(
            path,
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            ssm_prefix="/selected/root",
            model_group="selected-model-group",
            runner=aws,
        )
        data = json.loads(path.read_text())
        self.assertEqual(data["selectors"]["ssm_prefix"], "/selected/root")
        self.assertEqual(data["selectors"]["model_group"], "selected-model-group")
        with self.assertRaisesRegex(module.LedgerError, "selectors"):
            module.init_manifest(
                path,
                account=ACCOUNT,
                region=REGION,
                prefix=PREFIX,
                stacks=[],
                secret_id=PREFIX + "/tavily-api-key",
                runtime_name="trial_blue",
                ssm_prefix="/different",
                model_group="selected-model-group",
                runner=aws,
            )
        self.assertFalse(aws.mutations)

    def test_actual_selected_model_group_collision_fails_before_writes(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        aws.groups["selected-model-group"] = {
            "ModelPackageGroupName": "selected-model-group",
            "ModelPackageGroupArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package-group/selected-model-group",
            "CreationTime": "2030-01-01T00:00:00Z",
            "Tags": [],
        }
        with self.assertRaisesRegex(module.LedgerError, "occupied|ownership"):
            module.init_manifest(
                self.path / "occupied.json",
                account=ACCOUNT,
                region=REGION,
                prefix=PREFIX,
                stacks=[],
                secret_id=PREFIX + "/tavily-api-key",
                runtime_name="trial_blue",
                ssm_prefix="/selected/root",
                model_group="selected-model-group",
                runner=aws,
            )
        self.assertFalse(aws.mutations)

    def test_actual_ssm_root_is_used_for_direct_publication_discovery(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        path = self.path / "selected-root.json"
        result = module.init_manifest(
            path,
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            ssm_prefix="/selected/root",
            runner=aws,
        )
        tags = [
            {"Key": "amlas:deployment-id", "Value": result["deployment_id"]},
            {"Key": "amlas:deployment-prefix", "Value": PREFIX},
        ]
        aws.params["/selected/root/agentcore-runtime-arn"] = {
            "Name": "/selected/root/agentcore-runtime-arn",
            "Version": 1,
            "LastModifiedDate": "2030-01-01T00:00:00Z",
            "Tags": tags,
        }
        module.refresh_manifest(path, runner=aws)
        self.assertIn("/selected/root/agentcore-runtime-arn", path.read_text())

    def test_preexisting_untagged_lambda_log_incarnation_is_protected(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        name = "/aws/lambda/" + PREFIX + "-native-callback"
        aws.logs[name] = {
            "logGroupName": name,
            "logGroupArn": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{name}",
            "creationTime": int(datetime(2000, 1, 1, tzinfo=UTC).timestamp() * 1000),
            "tags": {},
        }
        path = self.path / "older-log.json"
        result = module.init_manifest(
            path,
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[PREFIX + "-LakeStack"],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            role_stack=PREFIX + "-runtime-role",
            runner=aws,
        )
        aws.populate_owned(result["deployment_id"])
        aws.stacks[PREFIX + "-LakeStack"]["members"].append(
            {
                "ResourceType": "AWS::Lambda::Function",
                "PhysicalResourceId": PREFIX + "-native-callback",
                "LogicalResourceId": "Callback",
                "ResourceStatus": "CREATE_COMPLETE",
            }
        )
        module.refresh_manifest(path, runner=aws)
        module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertIn(name, aws.logs)
        self.assertFalse(any(x[:2] == ["logs", "delete-log-group"] for x in aws.mutations))

    def test_owned_native_ml_jobs_stop_before_supporting_cleanup_and_foreign_jobs_survive(self) -> None:
        module, aws, path = initialized(self.path)
        tags = [
            {"Key": "amlas:deployment-id", "Value": aws.owner},
            {"Key": "amlas:deployment-prefix", "Value": PREFIX},
            {"Key": "amlas:runtime-session-id", "Value": "native-correlated-session"},
        ]
        for collection, stem, field in (
            (aws.training_jobs, "training", "Training"),
            (aws.processing_jobs, "processing", "Processing"),
        ):
            name = PREFIX + "-" + stem
            collection[name] = {
                field + "JobName": name,
                field + "JobArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:{stem}-job/{name}",
                field + "JobStatus": "InProgress",
                "CreationTime": "2030-01-01T00:00:00Z",
                "RoleArn": f"arn:aws:iam::{ACCOUNT}:role/owned-execution",
                "Tags": tags,
            }
            foreign_name = PREFIX + "-foreign-" + stem
            collection[foreign_name] = {
                field + "JobName": foreign_name,
                field + "JobArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:{stem}-job/{foreign_name}",
                field + "JobStatus": "InProgress",
                "CreationTime": "2030-01-01T00:00:00Z",
                "RoleArn": f"arn:aws:iam::{ACCOUNT}:role/foreign",
                "Tags": [{"Key": "amlas:deployment-id", "Value": "foreign"}],
            }
        result = module.delete_manifest_resources(path, runner=aws, sleep=lambda _: None)
        self.assertEqual(aws.training_jobs[PREFIX + "-training"]["TrainingJobStatus"], "Stopped")
        self.assertEqual(aws.processing_jobs[PREFIX + "-processing"]["ProcessingJobStatus"], "Stopped")
        self.assertEqual(aws.training_jobs[PREFIX + "-foreign-training"]["TrainingJobStatus"], "InProgress")
        self.assertEqual(len(result["terminal_managed_jobs"]), 2)
        operations = [tuple(x[:2]) for x in aws.mutations]
        last_stop = max(
            operations.index(("sagemaker", "stop-training-job")),
            operations.index(("sagemaker", "stop-processing-job")),
        )
        self.assertLess(last_stop, operations.index(("cloudformation", "delete-stack")))
        self.assertLess(last_stop, operations.index(("s3api", "delete-objects")))

    def old_empty_manifest(self, path: Path, *, resources: dict | None = None) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "deployment_id": "00000000-0000-0000-0000-aaaaaaaaaa33",
                    "account": ACCOUNT,
                    "region": REGION,
                    "prefix": PREFIX,
                    "profile": "",
                    "selectors": {
                        "stacks": [],
                        "role_stack": "",
                        "role_name": PREFIX + "-runtime-exec-" + REGION,
                        "secret_id": PREFIX + "/tavily-api-key",
                        "runtime_name": "trial_blue",
                    },
                    "created_at": "2026-10-13T00:00:00+00:00",
                    "resources": resources or {},
                    "baseline": {"runtime:trial_blue": None},
                    "sessions": [],
                }
            )
        )

    def test_explicit_empty_schema_upgrade_preserves_uuid_and_rechecks_actual_selectors(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        path = self.path / "old-empty.json"
        self.old_empty_manifest(path)
        result = module.init_manifest(
            path,
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            ssm_prefix="/selected/root",
            model_group="selected-model-group",
            runner=aws,
        )
        self.assertEqual(result["deployment_id"], "00000000-0000-0000-0000-aaaaaaaaaa33")
        self.assertEqual(json.loads(path.read_text())["selectors"]["ssm_prefix"], "/selected/root")
        self.assertFalse(aws.mutations)

    def test_schema_upgrade_refuses_any_existing_resource_receipt(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        path = self.path / "old-nonempty.json"
        self.old_empty_manifest(path, resources={"unknown": {"protected": True}})
        with self.assertRaisesRegex(module.LedgerError, "empty schema"):
            module.init_manifest(
                path,
                account=ACCOUNT,
                region=REGION,
                prefix=PREFIX,
                stacks=[],
                secret_id=PREFIX + "/tavily-api-key",
                runtime_name="trial_blue",
                ssm_prefix="/selected/root",
                model_group="selected-model-group",
                runner=aws,
            )
        self.assertFalse(aws.calls)

    def test_existing_role_requires_explicit_native_compatibility_witness_and_stays_protected(self) -> None:
        module = manifest_module()
        aws = NativeAWS(module)
        common = dict(
            account=ACCOUNT,
            region=REGION,
            prefix=PREFIX,
            stacks=[],
            secret_id=PREFIX + "/tavily-api-key",
            runtime_name="trial_blue",
            role_name="cdk-example-cfn-exec",
            runner=aws,
        )
        with self.assertRaisesRegex(module.LedgerError, "occupied"):
            module.init_manifest(self.path / "without-witness.json", **common)
        path = self.path / "witness.json"
        module.init_manifest(path, reused_role_arn=ACTOR, reused_role_id=aws.actor_role_id, **common)
        record = json.loads(path.read_text())["resources"]["iam_role:cdk-example-cfn-exec"]
        self.assertTrue(record["preexisting"] and record["protected"])
        self.assertFalse(aws.mutations)


if __name__ == "__main__":
    unittest.main()
