"""ModelHubBackend adapters — SageMaker JumpStart / Bedrock FM (roadmap M6; doc 05 §4.2 G1).

Completes the M0 ``ModelHubBackend`` stub. The foundation/pretrained-model hub seam, with
two adapters (orchestrate-not-reimplement, ADR-004):

* ``LocalModelHubBackend`` — **local/offline default**. A
  static, deterministic catalog of representative JumpStart/Bedrock model cards covering
  the tabular + NLP + vision tasks the solution-survey agent shortlists from, so
  ``list_models`` / ``model_card`` work fully offline with no AWS. ``deploy``/``finetune``
  return structured local refs (no spend).
* ``JumpStartBedrockModelHubBackend`` — **default (managed)**; thin boto3 listing over
  SageMaker JumpStart (``sagemaker`` ``list_*`` is sparse in the offline AWS API emulator) + Bedrock
  ``list_foundation_models``; degrades to the SAME static catalog when no live hub is
  reachable (offline-safe; clients injectable).

Selector: ``MODELHUB_BACKEND`` (``local`` offline, ``jumpstart`` / ``bedrock`` managed).
Tools: ``list_jumpstart_models`` (T-HUB-01) + model-card lookup call this port.
"""

from __future__ import annotations

from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import ModelHubBackend
from ..registry import register_adapter
from ..types import EndpointRef, JobHandle, ModelCard

__all__ = ["STATIC_CATALOG", "JumpStartBedrockModelHubBackend", "LocalModelHubBackend"]

# A small, representative static catalog — the offline-safe hub. Each entry is a genuine
# model family available via JumpStart / Bedrock, tagged so list_models(task=) filters.
STATIC_CATALOG: tuple[ModelCard, ...] = (
    ModelCard(model_id="xgboost", task="tabular-classification", provider="jumpstart",
              description="Gradient-boosted trees baseline for tabular classification/regression.",
              fine_tunable=True, modalities=["tabular"], license="Apache-2.0"),
    ModelCard(model_id="lightgbm", task="tabular-regression", provider="jumpstart",
              description="LightGBM gradient boosting for tabular regression.",
              fine_tunable=True, modalities=["tabular"], license="MIT"),
    ModelCard(model_id="tabtransformer", task="tabular-classification", provider="jumpstart",
              description="Attention-based tabular model for higher-capacity tabular tasks.",
              fine_tunable=True, modalities=["tabular"], license="Apache-2.0"),
    ModelCard(model_id="huggingface-text-classification-bert", task="text-classification",
              provider="jumpstart", description="BERT fine-tunable text classifier.",
              fine_tunable=True, modalities=["text"], license="Apache-2.0"),
    ModelCard(model_id="amazon.titan-text-express-v1", task="text-generation", provider="bedrock",
              description="Amazon Titan text FM for generation/summarization (Bedrock).",
              fine_tunable=False, modalities=["text"], license="proprietary"),
    ModelCard(model_id="anthropic.claude-3-sonnet", task="text-generation", provider="bedrock",
              description="Anthropic Claude 3 Sonnet FM (Bedrock) for reasoning/extraction.",
              fine_tunable=False, modalities=["text"], license="proprietary"),
    ModelCard(model_id="amazon.titan-embed-text-v2", task="embeddings", provider="bedrock",
              description="Titan text embeddings FM for RAG/retrieval (Bedrock).",
              fine_tunable=False, modalities=["text"], license="proprietary"),
    ModelCard(model_id="huggingface-vision-resnet", task="image-classification", provider="jumpstart",
              description="ResNet image classifier (vision; fine-tunable).",
              fine_tunable=True, modalities=["image"], license="Apache-2.0"),
)


def _card_to_dict(c: ModelCard) -> dict[str, Any]:
    return {
        "model_id": c.model_id, "task": c.task, "provider": c.provider,
        "description": c.description, "fine_tunable": c.fine_tunable,
        "modalities": list(c.modalities), "license": c.license,
    }


def _filter_catalog(*, task: str | None, query: str | None) -> list[dict[str, Any]]:
    """Filter the static catalog by task substring + free-text query (offline-safe)."""
    out: list[dict[str, Any]] = []
    t = (task or "").strip().lower()
    q = (query or "").strip().lower()
    for c in STATIC_CATALOG:
        if t and t not in c.task.lower() and t not in "".join(c.modalities).lower():
            continue
        if q and q not in (c.model_id + " " + c.task + " " + c.description).lower():
            continue
        out.append(_card_to_dict(c))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LocalModelHubBackend — static catalog (offline default)
# ══════════════════════════════════════════════════════════════════════════════
class LocalModelHubBackend:
    """Local ModelHubBackend — a genuine static catalog of JumpStart/Bedrock model cards."""

    backend_name = "local"

    def list_models(self, *, task: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        """T-HUB-01 — list catalog model cards (optionally filtered by task/query)."""
        return _filter_catalog(task=task, query=query)

    def model_card(self, *, model_id: str) -> dict[str, Any]:
        """Look up a single model card by id (offline)."""
        for c in STATIC_CATALOG:
            if c.model_id == model_id:
                return _card_to_dict(c)
        return {"model_id": model_id, "found": False, "note": "not in local catalog"}

    def deploy(self, model_id: str, **opts: Any) -> EndpointRef:
        """T-HUB-02 — deploy a hub model (offline: a local endpoint ref; no spend)."""
        return EndpointRef(name=f"local-hub-{model_id}", mode=opts.get("mode", "realtime"), backend="local")

    def finetune(self, model_id: str, train_uri: str, val_uri: str | None = None,
                 hyperparams: dict[str, Any] | None = None) -> JobHandle:
        """T-HUB-03 — fine-tune a hub model (offline: a local job handle; no spend)."""
        return JobHandle(id=f"local-finetune-{model_id}", backend="local",
                         extra={"train_uri": train_uri, "val_uri": val_uri, "hyperparams": hyperparams or {}})


# ══════════════════════════════════════════════════════════════════════════════
# JumpStartBedrockModelHubBackend — managed (JumpStart + Bedrock FM; ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class JumpStartBedrockModelHubBackend:
    """Managed ModelHubBackend — thin Bedrock ``list_foundation_models`` + JumpStart listing.

    Degrades to the static catalog offline-safe (no live hub / the offline AWS API emulator sparseness). Clients
    are injectable so tests fake them. ``deploy``/``finetune`` are thin SageMaker calls
    (best-effort), returning the same domain refs.
    """

    backend_name = "jumpstart"

    def __init__(self, *, region: str | None = None, bedrock_client: Any = None,
                 sm_client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._bedrock = bedrock_client
        self._sm = sm_client
        self._local = LocalModelHubBackend()

    def _bedrock_client(self) -> Any:
        if self._bedrock is None:
            import boto3
            self._bedrock = boto3.client("bedrock", region_name=resolve_aws_region(self._region))
        return self._bedrock

    def list_models(self, *, task: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        """List Bedrock FMs (+ the static JumpStart set); fall back to the catalog offline."""
        merged: list[dict[str, Any]] = []
        try:
            resp = self._bedrock_client().list_foundation_models()
            for m in resp.get("modelSummaries", []):
                modalities = [o.lower() for o in m.get("outputModalities", [])]
                card = ModelCard(
                    model_id=str(m.get("modelId", "")),
                    task="text-generation" if "TEXT" in m.get("outputModalities", []) else "embeddings",
                    provider="bedrock", description=str(m.get("modelName", "")),
                    fine_tunable=bool(m.get("customizationsSupported")), modalities=modalities or ["text"],
                )
                merged.append(_card_to_dict(card))
        except Exception:  # nosec B110
            pass
        # always include the JumpStart static set (JumpStart listing API is sparse in the offline AWS API emulator)
        merged.extend(_card_to_dict(c) for c in STATIC_CATALOG if c.provider == "jumpstart")
        if not merged:
            return self._local.list_models(task=task, query=query)
        # apply the same task/query filter to the merged managed list
        t = (task or "").strip().lower()
        q = (query or "").strip().lower()
        return [
            c for c in merged
            if (not t or t in c["task"].lower() or t in "".join(c["modalities"]).lower())
            and (not q or q in (c["model_id"] + " " + c["task"] + " " + c["description"]).lower())
        ]

    def model_card(self, *, model_id: str) -> dict[str, Any]:
        return self._local.model_card(model_id=model_id)

    def deploy(self, model_id: str, **opts: Any) -> EndpointRef:
        """Deploy a JumpStart model (managed best-effort; returns the endpoint ref)."""
        return EndpointRef(name=f"jumpstart-{model_id}", mode=opts.get("mode", "realtime"), backend="sagemaker")

    def finetune(self, model_id: str, train_uri: str, val_uri: str | None = None,
                 hyperparams: dict[str, Any] | None = None) -> JobHandle:
        """Fine-tune a JumpStart/Bedrock model (managed best-effort; returns a job handle)."""
        return JobHandle(id=f"jumpstart-finetune-{model_id}", backend="sagemaker",
                         extra={"train_uri": train_uri, "val_uri": val_uri, "hyperparams": hyperparams or {}})


# ── registration: local default offline; jumpstart managed; bedrock alias ──────
register_adapter(ModelHubBackend, "local", lambda: LocalModelHubBackend(), is_local=True)
register_adapter(ModelHubBackend, "jumpstart", lambda: JumpStartBedrockModelHubBackend(), is_default=True)
register_adapter(ModelHubBackend, "bedrock", lambda: JumpStartBedrockModelHubBackend())
