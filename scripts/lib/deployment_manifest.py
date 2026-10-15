#!/usr/bin/env python3
"""Dependency-free, native-identity ownership ledger for setup and teardown.

All AWS commands use argv (never a shell). The injected runner is the isolated
validation seam. Metadata and tags are read; secret values and SSM values are not.
Existing resources are protected. A name prefix is never evidence of ownership.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

OWNER_TAG = "amlas:deployment-id"
SESSION_TAG = "amlas:runtime-session-id"
TERMINAL_JOBS = frozenset({"Completed", "Failed", "Stopped"})
PREFIX_TAG = "amlas:deployment-prefix"
SCHEMA = 1
CFN_TYPES = {
    "AWS::S3::Bucket": "s3",
    "AWS::ECR::Repository": "ecr",
    "AWS::DynamoDB::Table": "dynamodb",
    "AWS::KMS::Key": "kms",
    "AWS::KMS::Alias": "kms_alias",
    "AWS::SageMaker::ModelPackageGroup": "model_group",
    "AWS::SSM::Parameter": "ssm",
    "AWS::Logs::LogGroup": "log_group",
    "AWS::IAM::Role": "iam_role",
}


class LedgerError(RuntimeError):
    """An identity or ownership gate failed; cleanup must stop."""


class AwsError(LedgerError):
    def __init__(self, code: str, *, missing: bool = False) -> None:
        self.code = code
        self.missing = missing
        super().__init__("AWS metadata/control request failed (" + code + ")")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017 -- CLI runs on Python 3.7 before dependency installation.


def _tags(items: Any) -> dict:
    if isinstance(items, dict):
        return items
    return {t.get("Key", t.get("TagKey")): t.get("Value", t.get("TagValue")) for t in items or []}


def _owned(data: dict, tags: dict) -> bool:
    return tags.get(OWNER_TAG) == data["deployment_id"] and tags.get(PREFIX_TAG) == data["prefix"]


def _load(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise LedgerError(
            "A readable deployment manifest is required; no prefix adoption is permitted"
        ) from error
    if data.get("schema") != SCHEMA or not isinstance(data.get("resources"), dict):
        raise LedgerError("Unsupported or incomplete deployment manifest")
    try:
        uuid.UUID(data["deployment_id"])
    except (KeyError, ValueError, TypeError) as error:
        raise LedgerError("Invalid deployment manifest ownership identifier") from error
    return data


def _save(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    descriptor = os.open(str(temporary), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


class AwsCli:
    def __init__(self, data: dict, runner: Callable | None = None, profile: str | None = None) -> None:
        self.data = data
        self.runner = runner
        self.profile = data.get("profile", "") if profile is None else profile

    def call(self, service: str, operation: str, *arguments: str) -> dict:
        args = [service, operation, *arguments, "--region", self.data["region"], "--output", "json"]
        if self.profile:
            args.extend(["--profile", self.profile])
        if self.runner is not None:
            return self.runner(args)
        try:
            binary = str(Path(self.data.get("tool_bin", "")) / "aws") if self.data.get("tool_bin") else "aws"
            if binary != "aws" and not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
                binary = "aws"
            # The saved selector controls authentication; preserve the remaining credential chain.
            environment = os.environ.copy()
            environment.pop("AWS_DEFAULT_PROFILE", None)
            if self.profile:
                environment["AWS_PROFILE"] = self.profile
            else:
                environment.pop("AWS_PROFILE", None)
            result = subprocess.run(
                [binary, *args], capture_output=True, text=True, timeout=120, check=False, env=environment
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LedgerError("AWS CLI unavailable or metadata/control request timed out") from error
        if result.returncode:
            match = re.search(r"An error occurred \(([A-Za-z0-9_]+)\)", result.stderr)
            code = match.group(1) if match else "CliFailure"
            absent_codes = {
                "ResourceNotFoundException",
                "ResourceNotFound",
                "RepositoryNotFoundException",
                "NoSuchBucket",
                "NoSuchEntity",
                "NotFoundException",
                "ParameterNotFound",
            }
            missing = code in absent_codes or (
                code in {"ValidationError", "ValidationException"}
                and bool(re.search(r"does not exist|not found|could not find", result.stderr, re.I))
            )
            # Error responses can contain sensitive values; keep only the error class.
            raise AwsError(code, missing=missing)
        try:
            return json.loads(result.stdout or "{}")
        except ValueError as error:
            raise LedgerError("AWS CLI returned malformed metadata/control JSON") from error

    def maybe(self, service: str, operation: str, *arguments: str) -> dict | None:
        try:
            return self.call(service, operation, *arguments)
        except AwsError as error:
            if error.missing:
                return None
            raise


def _target(data: dict, cli: AwsCli, *, account: str | None = None, region: str | None = None) -> None:
    if account is not None and account != data["account"]:
        raise LedgerError("Selected account does not match the manifest account")
    if region is not None and region != data["region"]:
        raise LedgerError("Selected region does not match the manifest region")
    identity = cli.call("sts", "get-caller-identity")
    if identity.get("Account") != data["account"]:
        raise LedgerError("Live AWS account does not match the manifest account")


def _arn(data: dict, value: str) -> None:
    if not value.startswith("arn:"):
        raise LedgerError("Native resource ARN is missing")
    fields = value.split(":", 5)
    if len(fields) != 6 or fields[4] not in ("", data["account"]):
        raise LedgerError("Native identity belongs to another account")
    if fields[3] not in ("", data["region"]):
        raise LedgerError("Native identity belongs to another region")


def _record(kind: str, name: str, native_id: str, identity: dict, tags: Any, **fields: Any) -> dict:
    ownership = _tags(tags)
    ownership = {key: ownership[key] for key in (OWNER_TAG, PREFIX_TAG, SESSION_TAG) if key in ownership}
    return dict(kind=kind, name=name, native_id=native_id, identity=identity, tags=ownership, **fields)


def _observe(cli: AwsCli, kind: str, name: str, native_id: str | None = None) -> dict | None:
    account, region = cli.data["account"], cli.data["region"]
    if kind == "stack":
        result = cli.maybe("cloudformation", "describe-stacks", "--stack-name", native_id or name)
        if not result or not result.get("Stacks"):
            return None
        stack = result["Stacks"][0]
        if stack.get("StackStatus") == "DELETE_COMPLETE":
            return None
        actor = stack.get("RoleARN", "")
        actor_id = ""
        if actor:
            _arn(cli.data, actor)
            role = cli.maybe("iam", "get-role", "--role-name", actor.rsplit("/", 1)[-1])
            if role:
                actor_id = role["Role"].get("RoleId", "")
        return _record(
            kind,
            stack["StackName"],
            stack["StackId"],
            {"stack_id": stack["StackId"], "role_arn": actor, "actor_role_id": actor_id},
            stack.get("Tags"),
            status=stack.get("StackStatus"),
            role_arn=actor,
            outputs={o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])},
        )
    if kind == "runtime":
        runtime_id = native_id.rsplit("/", 1)[-1] if native_id else ""
        if not runtime_id:
            runtimes = cli.call("bedrock-agentcore-control", "list-agent-runtimes").get("agentRuntimes", [])
            matches = [r for r in runtimes if r.get("agentRuntimeName") == name]
            if len(matches) > 1:
                raise LedgerError("Runtime name is ambiguous; select an exact native identity")
            if not matches:
                return None
            runtime_id = matches[0]["agentRuntimeId"]
        runtime = cli.maybe(
            "bedrock-agentcore-control", "get-agent-runtime", "--agent-runtime-id", runtime_id
        )
        if not runtime:
            return None
        arn = runtime["agentRuntimeArn"]
        tags = cli.maybe("bedrock-agentcore-control", "list-tags-for-resource", "--resource-arn", arn) or {}
        return _record(
            kind,
            runtime.get("agentRuntimeName", name),
            arn,
            {"runtime_id": runtime_id, "arn": arn, "created_at": runtime.get("createdAt")},
            tags.get("tags"),
            runtime_id=runtime_id,
            status=runtime.get("status"),
        )
    if kind == "secret":
        secret = cli.maybe("secretsmanager", "describe-secret", "--secret-id", native_id or name)
        if not secret:
            return None
        return _record(
            kind,
            secret["Name"],
            secret["ARN"],
            {"arn": secret["ARN"], "created_at": secret.get("CreatedDate")},
            secret.get("Tags"),
        )
    if kind == "s3":
        location = cli.maybe(
            "s3api", "get-bucket-location", "--bucket", name, "--expected-bucket-owner", account
        )
        if location is None:
            return None
        bucket_region = location.get("LocationConstraint") or "us-east-1"
        if bucket_region == "EU":
            bucket_region = "eu-west-1"
        if bucket_region != region:
            raise LedgerError("Recorded bucket is in another region")
        try:
            tags = cli.call(
                "s3api", "get-bucket-tagging", "--bucket", name, "--expected-bucket-owner", account
            ).get("TagSet", [])
        except AwsError as error:
            if error.code != "NoSuchTagSet":
                raise
            tags = []
        arn = "arn:aws:s3:::" + name
        return _record(kind, name, arn, {"arn": arn, "region": bucket_region, "account": account}, tags)
    if kind == "dynamodb":
        response = cli.maybe("dynamodb", "describe-table", "--table-name", name)
        if not response:
            return None
        table = response["Table"]
        tags = cli.maybe("dynamodb", "list-tags-of-resource", "--resource-arn", table["TableArn"]) or {}
        return _record(
            kind,
            table["TableName"],
            table["TableArn"],
            {
                "arn": table["TableArn"],
                "table_id": table.get("TableId"),
                "created_at": table.get("CreationDateTime"),
            },
            tags.get("Tags"),
        )
    if kind in ("kms", "kms_alias"):
        result = cli.maybe(
            "kms", "describe-key", "--key-id", native_id if kind == "kms" and native_id else name
        )
        if not result:
            return None
        key = result["KeyMetadata"]
        tags = cli.maybe("kms", "list-resource-tags", "--key-id", key["KeyId"]) or {}
        if kind == "kms_alias":
            arn = "arn:aws:kms:" + region + ":" + account + ":" + name
            return _record(
                kind,
                name,
                arn,
                {"arn": arn, "target_key_id": key["KeyId"]},
                tags.get("Tags"),
                ownership_source="target-key",
            )
        return _record(
            kind,
            key["KeyId"],
            key["Arn"],
            {"arn": key["Arn"], "key_id": key["KeyId"], "manager": key.get("KeyManager")},
            tags.get("Tags"),
            key_state=key.get("KeyState"),
            deletion_date=key.get("DeletionDate"),
        )
    if kind == "model_group":
        group = cli.maybe("sagemaker", "describe-model-package-group", "--model-package-group-name", name)
        if not group:
            return None
        tags = cli.maybe("sagemaker", "list-tags", "--resource-arn", group["ModelPackageGroupArn"]) or {}
        return _record(
            kind,
            group["ModelPackageGroupName"],
            group["ModelPackageGroupArn"],
            {"arn": group["ModelPackageGroupArn"], "created_at": group.get("CreationTime")},
            tags.get("Tags"),
        )
    if kind == "ecr":
        response = cli.maybe(
            "ecr", "describe-repositories", "--repository-names", name, "--registry-id", account
        )
        if not response or not response.get("repositories"):
            return None
        repo = response["repositories"][0]
        tags = cli.maybe("ecr", "list-tags-for-resource", "--resource-arn", repo["repositoryArn"]) or {}
        return _record(
            kind,
            repo["repositoryName"],
            repo["repositoryArn"],
            {"arn": repo["repositoryArn"], "created_at": repo.get("createdAt")},
            tags.get("tags"),
        )
    if kind == "ssm":
        parameters = cli.call(
            "ssm", "describe-parameters", "--parameter-filters", "Key=Name,Option=Equals,Values=" + name
        ).get("Parameters", [])
        parameter = next((p for p in parameters if p.get("Name") == name), None)
        if not parameter:
            return None
        tags = (
            cli.maybe("ssm", "list-tags-for-resource", "--resource-type", "Parameter", "--resource-id", name)
            or {}
        )
        arn = "arn:aws:ssm:" + region + ":" + account + ":parameter" + name
        return _record(
            kind,
            name,
            arn,
            {"arn": arn},
            tags.get("TagList"),
            revision={"version": parameter.get("Version"), "modified_at": parameter.get("LastModifiedDate")},
        )
    if kind == "log_group":
        groups = cli.call("logs", "describe-log-groups", "--log-group-name-prefix", name).get("logGroups", [])
        group = next((g for g in groups if g.get("logGroupName") == name), None)
        if not group:
            return None
        arn = group.get("logGroupArn") or group.get("arn", "")
        if arn.endswith(":*"):
            arn = arn[:-2]
        tags = cli.maybe("logs", "list-tags-for-resource", "--resource-arn", arn) or {}
        return _record(
            kind, name, arn, {"arn": arn, "created_at": group.get("creationTime")}, tags.get("tags")
        )
    if kind in ("training_job", "processing_job"):
        training = kind == "training_job"
        stem = "Training" if training else "Processing"
        operation = "describe-training-job" if training else "describe-processing-job"
        flag = "--training-job-name" if training else "--processing-job-name"
        job = cli.maybe("sagemaker", operation, flag, name)
        if not job:
            return None
        arn = job[stem + "JobArn"]
        tags = cli.maybe("sagemaker", "list-tags", "--resource-arn", arn) or {}
        tag_values = _tags(tags.get("Tags"))
        session = tag_values.get(SESSION_TAG, "")
        return _record(
            kind,
            job[stem + "JobName"],
            arn,
            {
                "arn": arn,
                "created_at": job.get("CreationTime"),
                "session_id": session,
                "role_arn": job.get("RoleArn"),
            },
            tag_values,
            job_status=job[stem + "JobStatus"],
            runtime_session_id=session,
        )
    if kind == "iam_role":
        response = cli.maybe("iam", "get-role", "--role-name", name)
        if not response:
            return None
        role = response["Role"]
        return _record(
            kind,
            role["RoleName"],
            role["Arn"],
            {"arn": role["Arn"], "role_id": role.get("RoleId")},
            role.get("Tags"),
            cleanup_mode="cloudformation",
        )
    raise LedgerError("Unsupported native resource kind")


def _key(record: dict) -> str:
    return record["kind"] + ":" + record["name"]


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 1000 if float(value) > 10**11 else float(value)
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise LedgerError("Native creation timestamp is invalid") from error
    raise LedgerError("Native creation timestamp is unavailable")


def _older_log(data: dict, record: dict) -> bool:
    baseline = data.get("log_baseline", {}).get(record["name"])
    if baseline == record["identity"]:
        return True
    return _epoch(record["identity"].get("created_at")) < _epoch(data["created_at"])


def _capture_log_baseline(data: dict, cli: AwsCli) -> None:
    prefixes = {"/aws/lambda/" + data["prefix"] + "-"}
    prefixes.update("/aws/lambda/" + stack for stack in data["selectors"]["stacks"])
    baseline = data.setdefault("log_baseline", {})
    for prefix in sorted(prefixes):
        for group in cli.call("logs", "describe-log-groups", "--log-group-name-prefix", prefix).get(
            "logGroups", []
        ):
            name = group.get("logGroupName", "")
            if name.startswith(prefix):
                observed = _observe(cli, "log_group", name)
                if observed:
                    baseline[name] = observed["identity"]


def _inventory_jobs(data: dict, cli: AwsCli) -> None:
    for kind, operation, collection, stem in (
        ("training_job", "list-training-jobs", "TrainingJobSummaries", "Training"),
        ("processing_job", "list-processing-jobs", "ProcessingJobSummaries", "Processing"),
    ):
        summaries = cli.call("sagemaker", operation, "--name-contains", data["prefix"] + "-").get(
            collection, []
        )
        for summary in summaries:
            name = summary.get(stem + "JobName", "")
            if not name.startswith(data["prefix"] + "-"):
                continue
            observed = _observe(cli, kind, name)
            if not observed or not _owned(data, observed["tags"]):
                continue
            if not observed.get("runtime_session_id"):
                raise LedgerError("Owned native job is missing its session correlation tag")
            if _epoch(observed["identity"].get("created_at")) < _epoch(data["created_at"]):
                raise LedgerError("Owned native job predates this manifest and cannot be adopted")
            _bind(data, observed)


def _empty_upgrade_allowed(data: dict) -> bool:
    return (
        not data["resources"]
        and not data.get("sessions")
        and not data.get("stack_inventory")
        and all(value is None for value in data.get("baseline", {}).values())
    )


def _refuse_protected_mutable_targets(data: dict) -> None:
    for record in data["resources"].values():
        if not (record.get("protected") or record.get("preexisting")):
            continue
        if record["kind"] == "log_group" and record.get("ownership_source") in ("stack", "runtime"):
            continue
        witness = data.get("reused_role", {})
        if record["kind"] == "iam_role" and (record["native_id"], record["identity"].get("role_id")) == (
            witness.get("arn"),
            witness.get("role_id"),
        ):
            continue
        raise LedgerError("Mutable target is occupied without owned manifest proof: " + _key(record))


def _bind(
    data: dict,
    observed: dict,
    *,
    preexisting: bool | None = None,
    source_stack: str | None = None,
    derived_runtime: str | None = None,
    derived_stack: str | None = None,
) -> dict:
    _arn(data, observed["native_id"])
    key = _key(observed)
    previous = data["resources"].get(key)
    if previous and previous["identity"] != observed["identity"]:
        raise LedgerError("Native resource identity changed for " + key)
    has_foreign_tag = bool(observed["tags"].get(OWNER_TAG) and not _owned(data, observed["tags"]))
    if previous and not previous.get("protected"):
        derived = previous.get("derived_runtime") or previous.get("derived_stack")
        if has_foreign_tag or (not derived and not _owned(data, observed["tags"])):
            raise LedgerError("Ownership changed for " + key)
    if observed["kind"] == "log_group" and (derived_stack or derived_runtime) and _older_log(data, observed):
        preexisting = True
    existed = (
        previous.get("preexisting", False)
        if previous
        else (preexisting if preexisting is not None else False)
    )
    original_owner = (
        previous.get("original_owner_id", previous.get("tags", {}).get(OWNER_TAG))
        if previous
        else observed["tags"].get(OWNER_TAG)
    )
    foreign_origin = bool(original_owner and original_owner != data["deployment_id"])
    protected = bool(existed or foreign_origin or not _owned(data, observed["tags"]))
    observed["original_owner_id"] = original_owner
    if (derived_runtime or derived_stack) and not existed and not has_foreign_tag and not foreign_origin:
        protected = False
    if derived_runtime:
        observed["ownership_source"] = "runtime"
        observed["derived_runtime"] = derived_runtime
    if derived_stack:
        observed["ownership_source"] = "stack"
        observed["derived_stack"] = derived_stack
    observed.update(preexisting=bool(existed), protected=protected, captured_at=_now())
    if source_stack:
        observed["source_stack"] = source_stack
    if previous and (previous.get("derived_runtime") or previous.get("derived_stack")):
        for field in ("derived_runtime", "derived_stack", "ownership_source", "protected"):
            if field in previous:
                observed[field] = previous[field]
    data["resources"][key] = observed
    return observed


def _require_created_ownership(record: dict) -> None:
    if not record.get("preexisting") and record.get("protected"):
        raise LedgerError(
            "New expected native resource has missing or foreign ownership tags: " + _key(record)
        )


def _summary(data: dict, path: Path) -> dict:
    return {
        "file": str(path),
        "deployment_id": data["deployment_id"],
        "account": data["account"],
        "region": data["region"],
        "prefix": data["prefix"],
        "ownership_tag": {"Key": OWNER_TAG, "Value": data["deployment_id"]},
        "recorded_resources": len(data["resources"]),
        "protected_resources": sum(bool(r.get("protected")) for r in data["resources"].values()),
    }


def init_manifest(
    path: Path,
    *,
    account: str,
    region: str,
    prefix: str,
    stacks: list,
    secret_id: str,
    runtime_name: str,
    role_stack: str = "",
    role_name: str = "",
    profile: str = "",
    ssm_prefix: str = "",
    model_group: str = "",
    tool_bin: str = "",
    reused_role_arn: str = "",
    reused_role_id: str = "",
    runner: Callable | None = None,
) -> dict:
    path = Path(path)
    if (
        not re.fullmatch(r"\d{12}", account)
        or not region
        or len(prefix) > 24
        or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", prefix)
    ):
        raise LedgerError("Invalid account, region or deployment prefix")
    if "CDKToolkit" in stacks or role_stack == "CDKToolkit":
        raise LedgerError("Shared bootstrap resources are not owned by this manifest")
    ssm_prefix = (ssm_prefix or "/" + prefix).rstrip("/")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", ssm_prefix):
        raise LedgerError("Invalid selected SSM prefix")
    model_group = model_group or prefix + "-models"
    selectors = dict(
        stacks=list(stacks),
        secret_id=secret_id,
        runtime_name=runtime_name,
        role_stack=role_stack,
        role_name=role_name or prefix + "-runtime-exec-" + region,
        ssm_prefix=ssm_prefix,
        model_group=model_group,
    )
    if path.exists():
        data = _load(path)
        if (data["account"], data["region"], data["prefix"]) != (account, region, prefix):
            raise LedgerError("Manifest selectors do not match this deployment")
        missing = {"ssm_prefix", "model_group"} - set(data["selectors"])
        if missing:
            if not _empty_upgrade_allowed(data) or any(
                selectors.get(key) != value for key, value in data["selectors"].items()
            ):
                raise LedgerError("Manifest selectors require an explicitly empty schema upgrade")
            data["selectors"] = selectors
            data["baseline"] = {}
        else:
            if data["selectors"] != selectors:
                raise LedgerError("Manifest selectors do not match this deployment")
            if tool_bin:
                data["tool_bin"] = tool_bin
                _save(path, data)
            refresh_manifest(path, runner=runner, profile=profile or None)
            data = _load(path)
            _refuse_protected_mutable_targets(data)
            return _summary(data, path)
    else:
        data = dict(
            schema=SCHEMA,
            deployment_id=str(uuid.uuid4()),
            account=account,
            region=region,
            prefix=prefix,
            profile=profile,
            selectors=selectors,
            created_at=_now(),
            resources={},
            baseline={},
            sessions=[],
        )
    data["tool_bin"] = tool_bin
    cli = AwsCli(data, runner)
    _target(data, cli)
    candidates = [("stack", name) for name in stacks]
    if role_stack:
        candidates.append(("stack", role_stack))
    candidates.extend(
        [
            ("runtime", runtime_name),
            ("secret", secret_id),
            ("iam_role", selectors["role_name"]),
            ("s3", prefix + "-lake-" + account + "-" + region),
            ("s3", prefix + "-cloudtrail-" + account + "-" + region),
            ("dynamodb", prefix + "-training-jobs"),
            ("model_group", model_group),
            ("ecr", prefix + "-runtime"),
        ]
    )
    candidates.extend(
        ("kms", "alias/" + prefix + "-" + suffix) for suffix in ("lake", "detections", "async-training")
    )
    occupied = []
    for kind, name in candidates:
        observed = _observe(cli, kind, name)
        data["baseline"][kind + ":" + name] = observed["native_id"] if observed else None
        if observed:
            _bind(data, observed, preexisting=True)
            if (
                kind == "iam_role"
                and reused_role_arn
                and reused_role_id
                and (observed["native_id"], observed["identity"].get("role_id"))
                == (reused_role_arn, reused_role_id)
            ):
                data["reused_role"] = {
                    "arn": reused_role_arn,
                    "role_id": reused_role_id,
                    "proof": "explicit compatible preflight witness",
                }
            else:
                occupied.append(kind + ":" + name)
    namespace = ssm_prefix + "/"
    for parameter in cli.call(
        "ssm", "describe-parameters", "--parameter-filters", "Key=Name,Option=BeginsWith,Values=" + namespace
    ).get("Parameters", []):
        name = parameter.get("Name", "")
        if name.startswith(namespace):
            observed = _observe(cli, "ssm", name)
            if observed:
                _bind(data, observed, preexisting=True)
                data["baseline"]["ssm:" + name] = observed["native_id"]
                occupied.append("ssm:" + name)
    if occupied:
        raise LedgerError(
            "Deployment target is occupied without matching manifest ownership: " + ", ".join(occupied)
        )
    _capture_log_baseline(data, cli)
    _inventory_jobs(data, cli)
    _save(path, data)
    return _summary(data, path)


def _capture_runtime_logs(data: dict, cli: AwsCli, runtime: dict) -> None:
    if runtime.get("protected") or runtime.get("preexisting"):
        return
    prefix = "/aws/bedrock-agentcore/runtimes/" + runtime["runtime_id"]
    groups = cli.call("logs", "describe-log-groups", "--log-group-name-prefix", prefix).get("logGroups", [])
    for group in groups:
        name = group.get("logGroupName", "")
        if name == prefix or name.startswith(prefix + "-") or name.startswith(prefix + "/"):
            observed = _observe(cli, "log_group", name)
            if observed:
                _bind(data, observed, derived_runtime=runtime["native_id"])


def refresh_manifest(path: Path, *, runner: Callable | None = None, profile: str | None = None) -> dict:
    data = _load(path)
    cli = AwsCli(data, runner, profile)
    _target(data, cli)
    failures = []
    selectors = data["selectors"]
    stacks = selectors["stacks"] + ([selectors["role_stack"]] if selectors["role_stack"] else [])
    for name in stacks:
        try:
            previous = data["resources"].get("stack:" + name)
            stack = _observe(cli, "stack", name, previous["native_id"] if previous else None)
            if not stack:
                continue
            bound = _bind(data, stack, preexisting=bool(data["baseline"].get("stack:" + name)))
            _require_created_ownership(bound)
            members = cli.call(
                "cloudformation", "list-stack-resources", "--stack-name", stack["native_id"]
            ).get("StackResourceSummaries", [])
            data.setdefault("stack_inventory", {})[stack["native_id"]] = [
                {
                    key: member[key]
                    for key in ("LogicalResourceId", "PhysicalResourceId", "ResourceType", "ResourceStatus")
                    if key in member
                }
                for member in members
            ]
            for member in members:
                native = member.get("PhysicalResourceId")
                kind = CFN_TYPES.get(member.get("ResourceType"))
                if (
                    native
                    and member.get("ResourceType") == "AWS::Lambda::Function"
                    and not bound["protected"]
                ):
                    function_name = native.rsplit(":", 1)[-1] if native.startswith("arn:") else native
                    log = _observe(cli, "log_group", "/aws/lambda/" + function_name)
                    if log:
                        _bind(data, log, derived_stack=stack["native_id"])
                if not native or not kind:
                    continue
                try:
                    observed = _observe(cli, kind, native)
                    if observed:
                        member_record = _bind(
                            data, observed, preexisting=bound["preexisting"], source_stack=stack["native_id"]
                        )
                        _require_created_ownership(member_record)
                except LedgerError as error:
                    failures.append(str(error))
        except LedgerError as error:
            failures.append(str(error))
    for kind, name in (("secret", selectors["secret_id"]), ("runtime", selectors["runtime_name"])):
        try:
            previous = data["resources"].get(kind + ":" + name)
            observed = _observe(cli, kind, name, previous["native_id"] if previous else None)
            if observed:
                bound = _bind(data, observed, preexisting=bool(data["baseline"].get(kind + ":" + name)))
                _require_created_ownership(bound)
                if kind == "runtime":
                    _capture_runtime_logs(data, cli, bound)
        except LedgerError as error:
            failures.append(str(error))
    namespace = data["selectors"].get("ssm_prefix", "/" + data["prefix"]) + "/"
    try:
        parameters = cli.call(
            "ssm",
            "describe-parameters",
            "--parameter-filters",
            "Key=Name,Option=BeginsWith,Values=" + namespace,
        ).get("Parameters", [])
        for parameter in parameters:
            name = parameter.get("Name", "")
            if name.startswith(namespace):
                observed = _observe(cli, "ssm", name)
                if observed:
                    parameter_record = _bind(
                        data, observed, preexisting=bool(data["baseline"].get("ssm:" + name))
                    )
                    if name in (namespace + "agentcore-runtime-arn", namespace + "session-bucket"):
                        _require_created_ownership(parameter_record)
    except LedgerError as error:
        failures.append(str(error))
    try:
        _inventory_jobs(data, cli)
    except LedgerError as error:
        failures.append(str(error))
    data["refreshed_at"] = _now()
    data["refresh_errors"] = failures
    _save(path, data)  # Preserve partial native handles even when a later read fails.
    if failures:
        raise LedgerError(
            "Refresh preserved partial handles but metadata verification failed: "
            + "; ".join(sorted(set(failures)))
        )
    return _summary(data, Path(path))


def _guard(data: dict, cli: AwsCli, record: dict) -> dict | None:
    if record.get("protected") or record.get("preexisting"):
        return None
    _arn(data, record["native_id"])
    observed = _observe(cli, record["kind"], record["name"], record["native_id"])
    if not observed:
        return None
    if observed["identity"] != record["identity"] or observed.get("revision") != record.get("revision"):
        raise LedgerError("Native identity or publication revision changed for " + _key(record))
    if observed["tags"].get(OWNER_TAG) and not _owned(data, observed["tags"]):
        raise LedgerError("Explicit foreign ownership tag on recorded resource")
    if record.get("ownership_source") in ("runtime", "stack"):
        parent_kind = record["ownership_source"]
        parent_id = record["derived_runtime"] if parent_kind == "runtime" else record["derived_stack"]
        parent = next(
            (
                r
                for r in data["resources"].values()
                if r["kind"] == parent_kind and r["native_id"] == parent_id and not r.get("protected")
            ),
            None,
        )
        if not parent:
            raise LedgerError("Generated runtime log group has no recorded owned parent")
        live_parent = _observe(cli, parent_kind, parent["name"], parent["native_id"])
        if live_parent and not _owned(data, live_parent["tags"]):
            raise LedgerError("Runtime log group parent ownership changed")
        if not live_parent and not parent.get("cleanup_absent"):
            raise LedgerError("Runtime log group parent absence has not been verified by this cleanup")
    elif not _owned(data, observed["tags"]):
        raise LedgerError("Ownership tag changed or is missing for " + _key(record))
    if record["kind"] == "stack" and (
        not observed.get("role_arn") or not observed["identity"].get("actor_role_id")
    ):
        raise LedgerError("CloudFormation service actor is missing or unverified")
    return observed


def plan_cleanup(
    path: Path,
    *,
    runner: Callable | None = None,
    account: str | None = None,
    region: str | None = None,
    profile: str | None = None,
) -> dict:
    data = _load(path)
    if data.get("refresh_errors"):
        raise LedgerError("Resolve incomplete refresh verification errors before planning cleanup")
    cli = AwsCli(data, runner, profile)
    _target(data, cli, account=account, region=region)
    _inventory_jobs(data, cli)
    _save(path, data)
    actions = []
    protected = []
    for record in data["resources"].values():
        if record.get("preexisting") or record.get("protected"):
            protected.append({"kind": record["kind"], "native_id": record["native_id"]})
            continue
        observed = _guard(data, cli, record)
        if observed:
            actions.append(
                {
                    "kind": record["kind"],
                    "name": record["name"],
                    "native_id": record["native_id"],
                    "identity": record["identity"],
                    "role_arn": record.get("role_arn", ""),
                }
            )
    return dict(_summary(data, Path(path)), actions=actions, protected=protected, dry_run=True)


def record_session(path: Path, *, runtime_id: str, session_id: str, qualifier: str = "DEFAULT") -> dict:
    data = _load(path)
    if not 33 <= len(session_id) <= 256 or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", qualifier):
        raise LedgerError("Invalid runtime session id or endpoint qualifier")
    runtime = next(
        (
            r
            for r in data["resources"].values()
            if r["kind"] == "runtime" and r.get("runtime_id") == runtime_id and not r.get("protected")
        ),
        None,
    )
    if not runtime:
        raise LedgerError("Session runtime is not an owned manifest identity")
    session = dict(runtime_id=runtime_id, session_id=session_id, qualifier=qualifier)
    if session not in data["sessions"]:
        data["sessions"].append(session)
    _save(path, data)
    return dict(_summary(data, Path(path)), recorded_sessions=len(data["sessions"]))


def _wait_absent(cli: AwsCli, record: dict, sleep: Callable, attempts: int, seconds: float) -> None:
    for attempt in range(attempts):
        observed = _observe(cli, record["kind"], record["name"], record["native_id"])
        if not observed:
            return
        if record["kind"] == "stack" and observed.get("status") == "DELETE_FAILED":
            raise LedgerError("CloudFormation exact stack deletion failed")
        if attempt + 1 < attempts:
            sleep(seconds)
    raise LedgerError("Native deletion did not reach verified absence within the polling bound")


def _empty_bucket(data: dict, cli: AwsCli, record: dict) -> None:
    if not _guard(data, cli, record):
        return
    for operation, collection in (
        ("list-object-versions", ("Versions", "DeleteMarkers")),
        ("list-objects-v2", ("Contents",)),
    ):
        for _ in range(10000):
            result = cli.call(
                "s3api",
                operation,
                "--bucket",
                record["name"],
                "--expected-bucket-owner",
                data["account"],
                "--no-paginate",
            )
            objects = []
            for key in collection:
                for obj in result.get(key, []) or []:
                    entry = {"Key": obj["Key"]}
                    if "VersionId" in obj:
                        entry["VersionId"] = obj["VersionId"]
                    objects.append(entry)
            if not objects:
                break
            for offset in range(0, len(objects), 1000):
                _guard(data, cli, record)
                deleted = cli.call(
                    "s3api",
                    "delete-objects",
                    "--bucket",
                    record["name"],
                    "--expected-bucket-owner",
                    data["account"],
                    "--delete",
                    json.dumps({"Objects": objects[offset : offset + 1000], "Quiet": True}),
                )
                if deleted.get("Errors"):
                    raise LedgerError("S3 object/version deletion returned errors")
        else:
            raise LedgerError("S3 object/version cleanup exceeded its bounded batch count")


def _empty_ecr(data: dict, cli: AwsCli, record: dict) -> None:
    if not _guard(data, cli, record):
        return
    for _ in range(10000):
        images = cli.call(
            "ecr",
            "list-images",
            "--repository-name",
            record["name"],
            "--registry-id",
            data["account"],
            "--no-paginate",
        ).get("imageIds", [])
        if not images:
            return
        for offset in range(0, len(images), 100):
            deleted = cli.call(
                "ecr",
                "batch-delete-image",
                "--repository-name",
                record["name"],
                "--registry-id",
                data["account"],
                "--image-ids",
                json.dumps(images[offset : offset + 100]),
            )
            if deleted.get("failures"):
                raise LedgerError("ECR image deletion returned failures")
    raise LedgerError("ECR image cleanup exceeded its bounded batch count")


def delete_manifest_resources(
    path: Path,
    *,
    runner: Callable | None = None,
    account: str | None = None,
    region: str | None = None,
    profile: str | None = None,
    sleep: Callable | None = None,
    poll_attempts: int = 240,
    poll_seconds: float = 5,
) -> dict:
    if not 1 <= poll_attempts <= 720 or not 0 <= poll_seconds <= 30:
        raise LedgerError("Invalid cleanup polling bounds")
    plan_cleanup(
        path, runner=runner, account=account, region=region, profile=profile
    )  # Validate every owned identity before the FIRST mutation.
    data = _load(path)
    cli = AwsCli(data, runner, profile)
    pause = sleep or time.sleep
    owned = [r for r in data["resources"].values() if not r.get("preexisting") and not r.get("protected")]
    for runtime in (r for r in owned if r["kind"] == "runtime"):
        if _guard(data, cli, runtime):
            for session in (s for s in data["sessions"] if s["runtime_id"] == runtime["runtime_id"]):
                for attempt in range(poll_attempts):
                    try:
                        cli.call(
                            "bedrock-agentcore",
                            "stop-runtime-session",
                            "--agent-runtime-arn",
                            runtime["native_id"],
                            "--runtime-session-id",
                            session["session_id"],
                            "--qualifier",
                            session["qualifier"],
                        )
                        break
                    except AwsError as error:
                        if error.missing:
                            break
                        if error.code != "RetryableConflictException" or attempt + 1 == poll_attempts:
                            raise
                        pause(poll_seconds)
            cli.call(
                "bedrock-agentcore-control",
                "delete-agent-runtime",
                "--agent-runtime-id",
                runtime["runtime_id"],
            )
            _wait_absent(cli, runtime, pause, poll_attempts, poll_seconds)
        runtime["cleanup_absent"] = True
        _capture_runtime_logs(data, cli, runtime)
        _save(path, data)
    owned = [r for r in data["resources"].values() if not r.get("preexisting") and not r.get("protected")]
    terminal_jobs = []
    _inventory_jobs(data, cli)
    owned = [r for r in data["resources"].values() if not r.get("preexisting") and not r.get("protected")]
    for job in (record for record in owned if record["kind"] in ("training_job", "processing_job")):
        current = _guard(data, cli, job)
        if not current:
            raise LedgerError("Owned accepted ML job could not be observed before supporting cleanup")
        training = job["kind"] == "training_job"
        if current["job_status"] not in TERMINAL_JOBS:
            operation = "stop-training-job" if training else "stop-processing-job"
            flag = "--training-job-name" if training else "--processing-job-name"
            cli.call("sagemaker", operation, flag, job["name"])
        for attempt in range(poll_attempts):
            current = _guard(data, cli, job)
            if current and current["job_status"] in TERMINAL_JOBS:
                break
            if attempt + 1 == poll_attempts:
                raise LedgerError(
                    "Owned ML job did not reach a terminal state; supporting cleanup is blocked"
                )
            pause(poll_seconds)
        job["job_status"] = current["job_status"]
        terminal_jobs.append(
            {
                "native_id": job["native_id"],
                "status": current["job_status"],
                "runtime_session_id": job["runtime_session_id"],
            }
        )
        _save(path, data)
    # Empty exact owned data before CFN deletion so DESTROY resources cannot fail as non-empty.
    for record in owned:
        if record["kind"] == "s3":
            _empty_bucket(data, cli, record)
        elif record["kind"] == "ecr":
            _empty_ecr(data, cli, record)
    stack_order = ([data["selectors"]["role_stack"]] if data["selectors"]["role_stack"] else []) + list(
        reversed(data["selectors"]["stacks"])
    )
    for name in stack_order:
        stack = data["resources"].get("stack:" + name)
        if stack and _guard(data, cli, stack):
            cli.call(
                "cloudformation",
                "delete-stack",
                "--stack-name",
                stack["native_id"],
                "--role-arn",
                stack["role_arn"],
            )
            _wait_absent(cli, stack, pause, poll_attempts, poll_seconds)
            stack["cleanup_absent"] = True
            _save(path, data)
    pending = []
    # Keep keys usable until all encrypted data, logs, and secrets are gone.
    retained = sorted(
        (
            r
            for r in owned
            if r["kind"] not in ("stack", "runtime", "iam_role", "training_job", "processing_job")
        ),
        key=lambda r: r["kind"] == "kms",
    )
    for record in retained:
        current = _guard(data, cli, record)
        if not current:
            continue
        kind, name = record["kind"], record["name"]
        if kind == "s3":
            _empty_bucket(data, cli, record)
            cli.call("s3api", "delete-bucket", "--bucket", name, "--expected-bucket-owner", data["account"])
        elif kind == "ecr":
            cli.call(
                "ecr",
                "delete-repository",
                "--repository-name",
                name,
                "--registry-id",
                data["account"],
                "--force",
            )
        elif kind == "dynamodb":
            cli.call("dynamodb", "delete-table", "--table-name", name)
        elif kind == "model_group":
            for _ in range(10000):
                packages = cli.call(
                    "sagemaker", "list-model-packages", "--model-package-group-name", name, "--no-paginate"
                ).get("ModelPackageSummaryList", [])
                if not packages:
                    break
                for package in packages:
                    package_arn = package["ModelPackageArn"]
                    _arn(data, package_arn)
                    if not package_arn.startswith(
                        "arn:aws:sagemaker:"
                        + data["region"]
                        + ":"
                        + data["account"]
                        + ":model-package/"
                        + name
                        + "/"
                    ):
                        raise LedgerError("Model package does not belong to recorded group")
                    cli.call("sagemaker", "delete-model-package", "--model-package-name", package_arn)
            else:
                raise LedgerError("Model package cleanup exceeded its bounded batch count")
            cli.call("sagemaker", "delete-model-package-group", "--model-package-group-name", name)
        elif kind == "secret":
            cli.call(
                "secretsmanager",
                "delete-secret",
                "--secret-id",
                record["native_id"],
                "--force-delete-without-recovery",
            )
        elif kind == "ssm":
            cli.call("ssm", "delete-parameter", "--name", name)
        elif kind == "log_group":
            cli.call("logs", "delete-log-group", "--log-group-name", name)
        elif kind == "kms_alias":
            cli.call("kms", "delete-alias", "--alias-name", name)
        elif kind == "kms":
            if current["identity"].get("manager") != "CUSTOMER":
                raise LedgerError("AWS-managed KMS keys are never cleanup targets")
            if current.get("key_state") != "PendingDeletion":
                cli.call(
                    "kms",
                    "schedule-key-deletion",
                    "--key-id",
                    record["native_id"],
                    "--pending-window-in-days",
                    "7",
                )
            verified = _observe(cli, "kms", name, record["native_id"])
            if not verified or verified.get("key_state") != "PendingDeletion":
                raise LedgerError("KMS pending deletion was not verified")
            pending.append(
                {
                    "native_id": record["native_id"],
                    "state": "PendingDeletion",
                    "deletion_date": verified.get("deletion_date"),
                    "minimum_wait_days": 7,
                }
            )
            continue
        _wait_absent(cli, record, pause, poll_attempts, poll_seconds)
        record["cleanup_absent"] = True
        _save(path, data)
    remaining = []
    for record in owned:
        observed = _observe(cli, record["kind"], record["name"], record["native_id"])
        retained_terminal_job = (
            observed
            and record["kind"] in ("training_job", "processing_job")
            and observed.get("job_status") in TERMINAL_JOBS
        )
        if (
            observed
            and not retained_terminal_job
            and not (record["kind"] == "kms" and observed.get("key_state") == "PendingDeletion")
        ):
            remaining.append({"kind": record["kind"], "native_id": record["native_id"]})
    data["cleanup"] = dict(
        verified_at=_now(),
        pending_kms_keys=pending,
        remaining_owned_resources=remaining,
        terminal_managed_jobs=terminal_jobs,
    )
    _save(path, data)
    if remaining:
        raise LedgerError("Owned native resources remain after cleanup")
    return dict(
        _summary(data, Path(path)),
        pending_kms_keys=pending,
        remaining_owned_resources=remaining,
        protected_resources_preserved=True,
        terminal_managed_jobs=terminal_jobs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name in ("init", "refresh", "plan", "delete", "record-session"):
        command = commands.add_parser(name)
        command.add_argument("--file", type=Path, required=True)
        if name != "record-session":
            command.add_argument("--profile")
            command.add_argument("--account")
            command.add_argument("--region")
        if name == "init":
            command.add_argument("--prefix", required=True)
            command.add_argument("--stacks", nargs="*", default=[])
            command.add_argument("--secret-id", required=True)
            command.add_argument("--runtime-name", required=True)
            command.add_argument("--role-stack", default="")
            command.add_argument("--role-name", "--runtime-role-name", default="")
            command.add_argument("--ssm-prefix", default="")
            command.add_argument("--model-group", default="")
            command.add_argument("--tool-bin", default="")
            command.add_argument("--reused-role-arn", default="")
            command.add_argument("--reused-role-id", default="")
        if name == "record-session":
            command.add_argument("--runtime-id", required=True)
            command.add_argument("--session-id", required=True)
            command.add_argument("--qualifier", default="DEFAULT")
        if name == "delete":
            command.add_argument("--poll-attempts", type=int, default=240)
            command.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    try:
        if args.command == "init":
            if not args.account or not args.region:
                raise LedgerError("init requires --account and --region")
            result = init_manifest(
                args.file,
                account=args.account,
                region=args.region,
                prefix=args.prefix,
                stacks=args.stacks,
                secret_id=args.secret_id,
                runtime_name=args.runtime_name,
                role_stack=args.role_stack,
                role_name=args.role_name,
                profile=args.profile or "",
                ssm_prefix=args.ssm_prefix,
                model_group=args.model_group,
                tool_bin=args.tool_bin,
                reused_role_arn=args.reused_role_arn,
                reused_role_id=args.reused_role_id,
            )
        elif args.command == "refresh":
            result = refresh_manifest(args.file, profile=args.profile)
        elif args.command == "record-session":
            result = record_session(
                args.file, runtime_id=args.runtime_id, session_id=args.session_id, qualifier=args.qualifier
            )
        elif args.command == "plan":
            result = plan_cleanup(args.file, account=args.account, region=args.region, profile=args.profile)
        else:
            result = delete_manifest_resources(
                args.file,
                account=args.account,
                region=args.region,
                profile=args.profile,
                poll_attempts=args.poll_attempts,
                poll_seconds=args.poll_seconds,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except LedgerError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
