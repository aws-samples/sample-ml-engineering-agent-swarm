"""Lambda handler — EventBridge → AgentCore Runtime resume.

Port of `sample-mlops-agent/lambda/callback/handler.py` (verbatim pattern;
paths + names adapted for Amlas).

Trigger: EventBridge rule on source=aws.sagemaker, detail-type="SageMaker Training
Job State Change". Handler pulls the DynamoDB row for the job, extracts the
runtime_session_id, and calls `bedrock-agentcore:InvokeAgentRuntime` so our agent
resumes in the same session.

Deploy as a Lambda with env vars:
    AGENTCORE_RUNTIME_ARN  = arn:aws:bedrock-agentcore:...:runtime/<id>
    JOBS_TABLE             = amlas-training-jobs
    AWS_REGION             = <runtime AWS region provided by Lambda>
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

# ── Pure builders (no AWS) ────────────────────────────────────────────────────
# These construct the EXACT kwargs / payload / envelopes the handler passes to
# boto3 / returns. They are pure functions (string/dict construction only — no
# network, no boto3) so the byte-identical-contract property callers can exercise
# them DIRECTLY without faking any external service (Phase 8 de-mock).


def build_resume_payload(job_name: str, status: str) -> bytes:
    """The utf-8 JSON payload sent to ``invoke_agent_runtime`` (pure)."""
    return json.dumps(
        {"event": "training_complete", "job_name": job_name, "status": status}
    ).encode("utf-8")


def build_invoke_kwargs(
    runtime_arn: str, session_id: str, job_name: str, status: str
) -> dict[str, Any]:
    """The exact kwargs for ``bedrock-agentcore.invoke_agent_runtime`` (pure)."""
    return {
        "agentRuntimeArn": runtime_arn,
        "runtimeSessionId": session_id,
        "qualifier": "DEFAULT",
        "payload": build_resume_payload(job_name, status),
    }


def build_update_item_kwargs(
    jobs_table: str, job_name: str, status: str
) -> dict[str, Any]:
    """The exact kwargs for ``dynamodb.update_item`` (pure)."""
    return {
        "TableName": jobs_table,
        "Key": {"job_name": {"S": job_name}},
        "UpdateExpression": "SET #s = :s",
        "ExpressionAttributeNames": {"#s": "status"},
        "ExpressionAttributeValues": {":s": {"S": status}},
    }


def build_success_envelope(session_id: str, job_name: str) -> dict[str, Any]:
    """The 200-path response envelope (pure)."""
    return {
        "statusCode": 200,
        "body": f"resumed session={session_id} for job={job_name}",
    }


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point. Non-blocking — drains response stream then returns."""
    detail = event.get("detail", {}) or {}
    job_name = detail.get("TrainingJobName")
    status = detail.get("TrainingJobStatus")
    if not job_name or not status:
        return {"statusCode": 400, "body": "event missing TrainingJobName or TrainingJobStatus"}

    # Lambda provides AWS_REGION. Keep the callback stdlib + provided boto3 only:
    # importing amlas.config would transitively require YAML/other optional deps.
    region = os.environ.get("AWS_REGION", "").strip()
    if not region:
        raise ValueError("AWS_REGION is required for the training callback; Lambda normally provides it.")
    jobs_table = os.environ["JOBS_TABLE"]
    runtime_arn = os.environ["AGENTCORE_RUNTIME_ARN"]

    ddb = boto3.client("dynamodb", region_name=region)
    row = ddb.get_item(TableName=jobs_table, Key={"job_name": {"S": job_name}}).get("Item")
    if not row:
        return {"statusCode": 404, "body": f"no DynamoDB row for {job_name}"}

    session_id = row.get("runtime_session_id", {}).get("S")
    if not session_id:
        return {"statusCode": 500, "body": f"row missing runtime_session_id for {job_name}"}

    # Update status row first — belt-and-braces for observability
    ddb.update_item(**build_update_item_kwargs(jobs_table, job_name, status))

    # Invoke the runtime with the preserved session ID
    client = boto3.client("bedrock-agentcore", region_name=region)
    response = client.invoke_agent_runtime(
        **build_invoke_kwargs(runtime_arn, session_id, job_name, status)
    )

    # Drain the response body so the agent's training_complete handling is fully
    # observed before we return (LAMB-01). The SDK returns the body under the
    # "response" key as a StreamingBody (has .read()); align with the proven
    # dispatch.parse_invoke_response handling rather than iterating chunks. Fall
    # back to iterating chunk lines if it isn't a StreamingBody.
    body = response.get("response")
    if body is not None:
        reader = getattr(body, "read", None)
        if callable(reader):
            reader()  # fully consume the StreamingBody
        else:
            # Intentional full drain (LAMB-01): the invoke response is a finite HTTP
            # stream and the Lambda itself is bounded at 2 minutes (CDK timeout).
            for _ in body:  # nosemgrep: pass-body-range
                pass
    return {"statusCode": 200, "body": f"resumed session={session_id} for job={job_name}"}
