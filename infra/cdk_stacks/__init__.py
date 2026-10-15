"""CDK stack modules — composable per concern for easier TDD.

Standing (low/zero idle cost) stacks added in Phase 2 (real-infra de-mock):
    - AgentCoreRuntimeStack   — ECR repo for the runtime image
    - LakeStack               — medallion S3 bucket + Glue DB
    - ModelRegistryStack      — SageMaker model-package-group (amlas-models)
    - SecretsStack            — Tavily secret (imported) + SSM id

Pre-existing stacks (extended, not replaced):
    - RuntimeStack       — AgentCore Runtime + Memory + CI + session bucket + DDB + Lambda
    - AsyncTrainingStack  — async-training plane (already deployed in default)
"""

from .agentcore_runtime_stack import AgentCoreRuntimeStack
from .async_training_stack import AsyncTrainingStack
from .backend_iam_stack import BackendIamStack
from .detections_stack import DetectionsStack
from .guardrail_stack import GuardrailStack
from .harness_stack import HarnessStack
from .lake_stack import LakeStack
from .model_registry_stack import ModelRegistryStack
from .network_stack import NetworkStack
from .research_scan_stack import ResearchScanScheduleStack
from .runtime_stack import RuntimeStack
from .secrets_stack import SecretsStack

__all__ = [
    "AgentCoreRuntimeStack",
    "AsyncTrainingStack",
    "BackendIamStack",
    "DetectionsStack",
    "GuardrailStack",
    "HarnessStack",
    "LakeStack",
    "ModelRegistryStack",
    "NetworkStack",
    "ResearchScanScheduleStack",
    "RuntimeStack",
    "SecretsStack",
]
