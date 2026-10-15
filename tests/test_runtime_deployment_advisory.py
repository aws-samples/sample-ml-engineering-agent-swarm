"""Closed CLI fixtures for verified AgentCore deployment, without AWS credentials.

Runtime/endpoint shapes follow the AWS AgentCore control API models. Invoke's qualifier
is an endpoint name; targetVersion/liveVersion identify its bound runtime version.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_shell_entrypoint_advisory import ACCOUNT, ROOT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace


@pytest.fixture
def runtime_workspace(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> tuple[Path, dict[str, str], Path]:
    repo, env, calls = shell_workspace
    env.pop("TAVILY_SECRET_ID", None)
    for key in tuple(env):
        if key.startswith("AGENTCORE_"):
            env.pop(key)
    tools = Path(env["PATH"].split(":")[0])
    state = repo.parent / "runtime-state.json"
    env.update(
        AMLAS_RUNTIME_POLL_ATTEMPTS="2",
        AMLAS_RUNTIME_POLL_SECONDS="0",
        TEST_STATE=str(state),
        AWS_ACCOUNT_ID=ACCOUNT,
        AWS_REGION="us-east-1",
        AMLAS_HARNESS_LEADS="research_lead,business_understanding_lead",
        AGENTICML_MODEL_PLANNER="global.anthropic.claude-opus-4-7",
    )
    driver = r"""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
raw_args = list(args)
while args and args[0].startswith("--profile="): args = args[1:]
operation = tuple(args[:2])
failure = os.environ.get("TEST_FAILURE", "")
account = os.environ["TEST_ACCOUNT"]
region = os.environ["AWS_REGION"]
rid = "amlas-example123"
arn = f"arn:aws:bedrock-agentcore:{region}:{account}:runtime/{rid}"
role_name = args[args.index("--role-name")+1] if "--role-name" in args else os.environ.get("AGENTCORE_EXEC_ROLE_NAME", os.environ.get("AMLAS_DEPLOYMENT_PREFIX", "amlas") + "-runtime-exec-" + region)
role = f"arn:aws:iam::{account}:role/{role_name}"
state_file = Path(os.environ["TEST_STATE"])
state = json.loads(state_file.read_text()) if state_file.exists() else {}
with open(os.environ["TEST_CALLS"], "a") as stream:
 stream.write(json.dumps({"tool": "aws", "args": args, "raw_args": raw_args}) + "\n")
def emit(value):
 if "--query" in args and isinstance(value, dict):
  query = args[args.index("--query")+1]
  if query in ("Role.Arn",): value = value["Role"]["Arn"]
  elif query in ("status", "agentRuntimeVersion", "agentRuntimeArn"): value = value.get(query, "None")
  elif "agentRuntimes" in query: value = rid if value.get("agentRuntimes") else ""
  else: raise RuntimeError("unsupported fixture query: " + query)
 print(json.dumps(value) if isinstance(value, (dict,list)) else value)
 sys.exit(0)
def denied():
 print("An error occurred (AccessDeniedException): fixture operation denied", file=sys.stderr); sys.exit(254)
def request():
 if "--cli-input-json" in args: return json.loads(args[args.index("--cli-input-json")+1])
 result = {}
 mapping = {"--agent-runtime-id":"agentRuntimeId", "--agent-runtime-name":"agentRuntimeName", "--agent-runtime-version":"agentRuntimeVersion", "--role-arn":"roleArn", "--endpoint-name":"endpointName", "--platform-version":"platformVersion", "--resource-arn":"resourceArn"}
 for flag,key in mapping.items():
  if flag in args: result[key] = args[args.index(flag)+1]
 if "--agent-runtime-artifact" in args:
  text=args[args.index("--agent-runtime-artifact")+1]
  result["agentRuntimeArtifact"]={"containerConfiguration":{"containerUri":text.split("containerUri=",1)[1].removesuffix("}")}}
 if "--environment-variables" in args:
  text=args[args.index("--environment-variables")+1]
  if text.startswith("{"): result["environmentVariables"] = json.loads(text)
  else:
   pairs=text.split(",")
   if any("=" not in pair for pair in pairs):
    print("CLI shorthand cannot encode the comma-containing environment value",file=sys.stderr);sys.exit(252)
   result["environmentVariables"] = dict(pair.split("=",1) for pair in pairs)
 return result
if operation == ("configure","get"): emit(region)
if operation == ("sts","get-caller-identity"): emit(account)
# CloudFormation CreateStack/Wait contracts: https://docs.aws.amazon.com/AWSCloudFormation/latest/APIReference/API_CreateStack.html
prefix = os.environ.get("AMLAS_DEPLOYMENT_PREFIX", "amlas")
bootstrap_name = f"cdk-hnb659fds-cfn-exec-role-{account}-{region}"
bootstrap_arn = f"arn:aws:iam::{account}:role/{bootstrap_name}"
stack_name = f"{prefix}-runtime-role"
stack_arn = f"arn:aws:cloudformation:{region}:{account}:stack/{stack_name}/fixture-stack"
route_failure = os.environ.get("TEST_ROUTE_FAILURE", "")
if operation == ("cloudformation","list-stack-resources"):
 assert args[args.index("--stack-name")+1] == "CDKToolkit"
 if route_failure == "bootstrap-read": denied()
 emit({"StackResourceSummaries": [] if route_failure == "bootstrap-missing" else [{"LogicalResourceId":"CloudFormationExecutionRole","PhysicalResourceId":bootstrap_name,"ResourceType":"AWS::IAM::Role","ResourceStatus":"CREATE_COMPLETE"}]})
if operation == ("iam","get-role") and role_name == bootstrap_name:
 trust={"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"cloudformation.amazonaws.com"},"Action":"sts:AssumeRole"}]}
 if route_failure == "bootstrap-trust": trust["Statement"][0]["Principal"]={"AWS":f"arn:aws:iam::{account}:root"}
 result={"Role":{"Arn":bootstrap_arn if route_failure != "bootstrap-account" else f"arn:aws:iam::{'0'*12}:role/{bootstrap_name}","AssumeRolePolicyDocument":trust}}
 if route_failure == "bootstrap-boundary": result["Role"]["PermissionsBoundary"]={"PermissionsBoundaryArn":f"arn:aws:iam::{account}:policy/restricted"}
 emit(result)
if operation == ("cloudformation","describe-stacks"):
 selected=args[args.index("--stack-name")+1]
 if selected not in (stack_name,stack_arn):
  physical_prefix="" if prefix == "amlas" else prefix + "-"
  binding_failure=os.environ.get("TEST_BINDING_FAILURE","")
  if selected == physical_prefix + "AsyncTrainingStack": emit("example-session-bucket")
  lake=f"{prefix}-lake-{account}-{region}"
  if selected == physical_prefix + "LakeStack":
   values={"LakeBucketName":lake,"LakeSilverPrefix":f"s3://{lake}/silver/","LakeGoldPrefix":f"s3://{lake}/gold/"}
   if binding_failure == "lake-silver": values.pop("LakeSilverPrefix")
   if binding_failure == "lake-gold": values.pop("LakeGoldPrefix")
  elif selected == physical_prefix + "GuardrailStack":
   gid="guardrailex01"
   values={"GuardrailIdOut":gid,"GuardrailArnOut":f"arn:aws:bedrock:{region}:{account}:guardrail/{gid}","GuardrailVersionOut":"DRAFT"}
   if binding_failure == "guardrail-id": values.pop("GuardrailIdOut")
   if binding_failure == "guardrail-version": values.pop("GuardrailVersionOut")
   if binding_failure == "guardrail-arn": values["GuardrailArnOut"]=f"arn:aws:bedrock:{region}:{'0'*12}:guardrail/{gid}"
  else: raise AssertionError("Unexpected selected stack: " + selected)
  emit({"Stacks":[{"StackName":selected,"StackStatus":"CREATE_COMPLETE","Outputs":[{"OutputKey":key,"OutputValue":value} for key,value in values.items()]}]})
 if not state.get("role_created") and route_failure != "stack-collision":
  print(f"An error occurred (ValidationError): Stack with id {selected} does not exist",file=sys.stderr);sys.exit(254)
 result={"Stacks":[{"StackId":stack_arn,"StackName":stack_name,"StackStatus":"CREATE_COMPLETE","RoleARN":bootstrap_arn,"Tags":state.get("stack_tags",[{"Key":"amlas:deployment-prefix","Value":prefix}]),"Outputs":[{"OutputKey":"RuntimeRoleArn","OutputValue":role}]}]}
 if failure == "cfn-readback": result["Stacks"][0]["StackStatus"]="ROLLBACK_COMPLETE"
 if "--query" in args:
  query=args[args.index("--query")+1]
  if "RuntimeRoleArn" in query: emit(role)
  if query == "Stacks[0].StackStatus": emit(result["Stacks"][0]["StackStatus"])
 emit(result)
if operation == ("cloudformation","validate-template"):
 if failure == "cfn-template": denied()
 template=json.loads(args[args.index("--template-body")+1]);assert template["Resources"]["RuntimeExecutionRole"]["Type"] == "AWS::IAM::Role"
 emit({"Capabilities":["CAPABILITY_NAMED_IAM"]})
if operation == ("cloudformation","create-stack"):
 if failure == "cfn-create": denied()
 assert args[args.index("--stack-name")+1] == stack_name
 assert args[args.index("--role-arn")+1] == bootstrap_arn
 assert "CAPABILITY_NAMED_IAM" in args
 template=json.loads(args[args.index("--template-body")+1]);properties=template["Resources"]["RuntimeExecutionRole"]["Properties"]
 assert properties["RoleName"] == role_name
 assert properties["AssumeRolePolicyDocument"]["Statement"][0]["Principal"] == {"Service":"bedrock-agentcore.amazonaws.com"}
 state["trust"]=properties["AssumeRolePolicyDocument"];state["policy"]=properties["Policies"][0]["PolicyDocument"];state["role_tags"]=properties["Tags"];state["stack_tags"]=json.loads(args[args.index("--tags")+1]);state["role_created"]=True;state_file.write_text(json.dumps(state))
 if "--query" in args: assert args[args.index("--query")+1] == "StackId";emit(stack_arn)
 emit({"StackId":stack_arn})
if operation == ("cloudformation","wait"):
 assert args[2] == "stack-create-complete"
 assert args[args.index("--stack-name")+1] == stack_arn
 if failure == "cfn-wait": denied()
 emit({})
if operation == ("secretsmanager","describe-secret"):
 selected=args[args.index("--secret-id")+1]
 if os.environ.get("TEST_BINDING_FAILURE") == "secret-read": denied()
 assert selected == os.environ.get("TAVILY_SECRET_ID",f"{prefix}/tavily-api-key")
 emit({"Name":selected,"ARN":f"arn:aws:secretsmanager:{region}:{account}:secret:{selected}-ABC123"})
if operation in (("s3api","get-bucket-encryption"),("dynamodb","describe-table")): emit("None")
if operation == ("iam","get-role"):
 if failure == "iam-get": denied()
 if failure == "iam-policy-read": emit({"Role":{"Arn":role,"AssumeRolePolicyDocument":{}}})
 existing=os.environ.get("TEST_EXISTING_ROLE", "")
 if failure == "cfn-role-readback" and state.get("role_created"):
  print("NoSuchEntity",file=sys.stderr);sys.exit(254)
 if existing == "captured" or state.get("role_created"): emit({"Role":{"Arn":role,"RoleId":"AROA" + "X"*16,"AssumeRolePolicyDocument":state["trust"],"Tags":state.get("role_tags",[])}})
 if existing == "west2":
  emit({"Role":{"Arn":role,"AssumeRolePolicyDocument":{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole","Condition":{"StringEquals":{"aws:SourceAccount":account},"ArnLike":{"aws:SourceArn":f"arn:aws:bedrock-agentcore:us-west-2:{account}:*"}}}]}}})
 print("NoSuchEntity",file=sys.stderr);sys.exit(254)
if operation == ("iam","get-role-policy"):
 if failure == "iam-policy-read": denied()
 emit({"RoleName":role_name,"PolicyName":"amlas-runtime-exec-policy","PolicyDocument":state.get("policy",{"Version":"2012-10-17","Statement":[]})})
if operation == ("iam","create-role"):
 if failure == "iam-create": denied()
 state["trust"]=json.loads(args[args.index("--assume-role-policy-document")+1]);state_file.write_text(json.dumps(state))
 emit({"Role":{"Arn":role}})
if operation == ("iam","update-assume-role-policy"):
 raise RuntimeError("Existing role trust must not be replaced")
if operation == ("iam","put-role-policy"):
 if failure == "iam-policy": denied()
 state["policy"]=json.loads(args[args.index("--policy-document")+1]);state_file.write_text(json.dumps(state))
 emit({})
if operation == ("bedrock-agentcore-control","update-agent-runtime") and "--generate-cli-skeleton" in args:
 emit({"agentRuntimeId":"", "metadataConfiguration":{"requireMMDSV2":False}, "platformVersion":""} if failure != "missing-cli-fields" else {"agentRuntimeId":""})
if operation == ("bedrock-agentcore-control","update-agent-runtime-endpoint") and "--generate-cli-skeleton" in args:
 emit({"agentRuntimeId":"", "endpointName":"", "agentRuntimeVersion":""})
if operation == ("bedrock-agentcore-control","list-agent-runtimes"):
 if failure == "runtime-list": denied()
 emit({"agentRuntimes":[] if os.environ.get("TEST_CREATE") == "1" or failure == "runtime-create" else [{"agentRuntimeName":"amlas","agentRuntimeId":rid}]})
if operation in (("bedrock-agentcore-control","create-agent-runtime"),("bedrock-agentcore-control","update-agent-runtime")):
 if failure == "runtime-create" and operation[1] == "create-agent-runtime": denied()
 if failure == "runtime-update" and operation[1] == "update-agent-runtime": denied()
 payload=request()
 allowed={"agentRuntimeName","agentRuntimeId","agentRuntimeArtifact","roleArn","networkConfiguration","protocolConfiguration","environmentVariables","description","clientToken","metadataConfiguration","platformVersion","tags"}
 assert not set(payload)-allowed, "unexpected control input field"
 assert not (operation[1] == "update-agent-runtime" and "tags" in payload), "Update does not accept tags"
 assert not (operation[1] == "create-agent-runtime" and "metadataConfiguration" in payload), "Create does not accept MMDS configuration"
 assert isinstance(payload.get("environmentVariables"),dict), "runtime environment must be a map"
 assert payload.get("roleArn") == role, "role ARN required"
 version="1" if operation[1] == "create-agent-runtime" else "2"
 state.update(payload, agentRuntimeVersion=version, agentRuntimeId=rid, agentRuntimeArn=arn,
              status="READY", platformVersion=payload.get("platformVersion","1.0.0"))
 state["endpointVersion"]=version
 state_file.write_text(json.dumps(state))
 emit({"agentRuntimeId":rid,"agentRuntimeArn":arn,"agentRuntimeVersion":version,"status":"CREATING" if version == "1" else "UPDATING"})
if operation == ("bedrock-agentcore-control","list-tags-for-resource"):
 if failure == "runtime-tags-read": denied()
 payload=request();assert payload.get("resourceArn") == arn
 emit({"tags":json.loads(os.environ["TEST_EXISTING_RUNTIME_TAGS"]) if os.environ.get("TEST_EXISTING_RUNTIME_TAGS") else state.get("tags",{})})
if operation == ("bedrock-agentcore-control","get-agent-runtime"):
 if failure == "runtime-get": denied()
 value=dict(state) if "agentRuntimeVersion" in state else {"agentRuntimeId":rid,"agentRuntimeArn":arn,"agentRuntimeVersion":"1","status":"READY","platformVersion":"1.0.0","roleArn":role,"environmentVariables":{},"agentRuntimeArtifact":{"containerConfiguration":{"containerUri":"previous-image"}}}
 if failure == "runtime-failed": value["status"]="UPDATE_FAILED"
 if failure == "runtime-timeout": value["status"]="UPDATING"
 if "agentRuntimeVersion" in state:
  if failure == "wrong-image": value["agentRuntimeArtifact"]={"containerConfiguration":{"containerUri":"previous-image"}}
  if failure == "wrong-env": value["environmentVariables"]={}
  if failure == "wrong-role": value["roleArn"]="previous-role"
  if failure == "wrong-version": value["agentRuntimeVersion"]="1"
 emit(value)
if operation == ("bedrock-agentcore-control","get-agent-runtime-endpoint"):
 if failure == "endpoint-get": denied()
 version=state.get("endpointVersion","1")
 emit({"name":"DEFAULT","agentRuntimeArn":arn,"liveVersion":version,"status":"UPDATING" if failure == "endpoint-timeout" else "READY"})
if operation == ("bedrock-agentcore-control","update-agent-runtime-endpoint"):
 payload=request()
 if payload.get("endpointName") == "DEFAULT":
  print("An error occurred (ConflictException): Default endpoints are managed through agent updates. Please use the update agent operation.",file=sys.stderr);sys.exit(254)
 assert set(payload) <= {"agentRuntimeId","endpointName","agentRuntimeVersion","description","clientToken"}
 assert payload["endpointName"] == "DEFAULT"
 assert payload["agentRuntimeVersion"] == state["agentRuntimeVersion"]
 state["endpointVersion"]=payload["agentRuntimeVersion"];state_file.write_text(json.dumps(state))
 emit({"agentRuntimeArn":arn,"liveVersion":"1","targetVersion":state["endpointVersion"],"status":"UPDATING"})
print("UNEXPECTED-AWS-CALL " + " ".join(args),file=sys.stderr);sys.exit(9)
"""
    (tools / "aws").write_text(f"#!{sys.executable}\n" + driver)
    (tools / "aws").chmod(0o755)
    sleep = tools / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    sleep.chmod(0o755)
    return repo, env, calls


def run_runtime(workspace: tuple[Path, dict[str, str], Path]) -> subprocess.CompletedProcess[str]:
    repo, env, _ = workspace
    return subprocess.run(
        ["bash", str(repo / "scripts/lib/deploy_agentcore_runtime.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def runtime_calls(workspace: tuple[Path, dict[str, str], Path]) -> list[dict]:
    path = workspace[2]
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize(
    "failure",
    [
        "iam-get",
        "cfn-create",
        "iam-policy-read",
        "cfn-wait",
        "runtime-list",
        "runtime-create",
        "runtime-update",
        "runtime-get",
        "runtime-failed",
        "runtime-timeout",
        "wrong-image",
        "wrong-env",
        "wrong-role",
        "wrong-version",
        "endpoint-get",
        "endpoint-timeout",
    ],
)
def test_required_failure_never_emits_success(
    runtime_workspace: tuple[Path, dict[str, str], Path], failure: str
) -> None:
    runtime_workspace[1]["TEST_FAILURE"] = failure
    runtime_workspace[1]["AMLAS_HARNESS_LEADS"] = "research_lead"
    result = run_runtime(runtime_workspace)
    expected = {
        "iam-get": ("iam", "get-role"),
        "cfn-create": ("cloudformation", "create-stack"),
        "iam-policy-read": ("iam", "get-role-policy"),
        "cfn-wait": ("cloudformation", "wait"),
        "runtime-list": ("bedrock-agentcore-control", "list-agent-runtimes"),
        "runtime-create": ("bedrock-agentcore-control", "create-agent-runtime"),
        "runtime-update": ("bedrock-agentcore-control", "update-agent-runtime"),
        "endpoint-get": ("bedrock-agentcore-control", "get-agent-runtime-endpoint"),
        "endpoint-timeout": ("bedrock-agentcore-control", "get-agent-runtime-endpoint"),
    }.get(failure, ("bedrock-agentcore-control", "get-agent-runtime"))
    assert any(
        tuple(call["args"][:2]) == expected and "--generate-cli-skeleton" not in call["args"]
        for call in runtime_calls(runtime_workspace)
    ), result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" not in result.stdout
    assert "AGENTCORE_RUNTIME_ARN=" not in result.stdout


def test_missing_cli_fields_fail_before_iam_writes(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    runtime_workspace[1]["TEST_FAILURE"] = "missing-cli-fields"
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert "upgrade" in (result.stdout + result.stderr).lower()
    assert not any(
        call["args"][:2]
        in (["iam", "create-role"], ["iam", "update-assume-role-policy"], ["iam", "put-role-policy"])
        for call in runtime_calls(runtime_workspace)
    )


@pytest.mark.parametrize("create", [False, True])
def test_success_pins_platform_mmds_and_default_endpoint(
    runtime_workspace: tuple[Path, dict[str, str], Path], create: bool
) -> None:
    if create:
        runtime_workspace[1]["TEST_CREATE"] = "1"
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    state_path = Path(runtime_workspace[1]["TEST_STATE"])
    assert state_path.exists(), result.stdout + result.stderr
    state = json.loads(state_path.read_text())
    assert state.get("metadataConfiguration", {}).get("requireMMDSV2") is True
    assert state["platformVersion"] == "1.0.0"
    assert state["endpointVersion"] == state["agentRuntimeVersion"] == "2"
    assert state["environmentVariables"]["AMLAS_HARNESS_LEADS"] == "research_lead,business_understanding_lead"
    assert state["environmentVariables"]["AMLAS_MODEL_PLANNER"] == "global.anthropic.claude-opus-4-7"
    assert not {"AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AMLAS_AWS_PROFILE", "AGENTICML_AWS_PROFILE"} & set(
        state["environmentVariables"]
    )
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" in result.stdout
    control_calls = [
        call["args"]
        for call in runtime_calls(runtime_workspace)
        if call["args"][:1] == ["bedrock-agentcore-control"] and "--generate-cli-skeleton" not in call["args"]
    ]
    update_index = next(
        index for index, args in enumerate(control_calls) if args[1] == "update-agent-runtime"
    )
    verified_reads = [
        index
        for index, args in enumerate(control_calls)
        if index > update_index
        and args[1] == "get-agent-runtime"
        and json.loads(args[args.index("--cli-input-json") + 1]).get("agentRuntimeVersion") == "2"
    ]
    endpoint_reads = [
        index for index, args in enumerate(control_calls) if args[1] == "get-agent-runtime-endpoint"
    ]
    assert verified_reads, "accepted runtime update must be read back"
    assert endpoint_reads, "successful deploy must read back DEFAULT"
    assert endpoint_reads[0] > verified_reads[-1]
    assert not any(
        args[1] == "update-agent-runtime-endpoint"
        and json.loads(args[args.index("--cli-input-json") + 1]).get("endpointName") == "DEFAULT"
        for args in control_calls
    )


def test_shared_iam_policy_temp_files_are_removed() -> None:
    source = (ROOT / "scripts/lib/deploy_agentcore_runtime.sh").read_text()
    assert "/tmp/p5_" not in source
