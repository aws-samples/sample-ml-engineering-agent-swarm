"""GovernanceBackend + GuardrailBackend adapters — minimal real (roadmap M4; doc 05 §4.2 G4).

M4 stands these two ports up just enough for the **Stage-9 readiness gates** (the
Security/Privacy/Safety/Explainability gates of ``readiness_flow`` consult them) and the
``governance_agent`` model-card/lineage touchpoints. Full breadth (SageMaker Model Cards,
LakeFormation access, Bedrock Guardrails create/bind/monitor) is M6/M8 — these adapters
keep ONLY the thin, deterministic, offline-safe surface M4 needs (no AWS in the local path).

* ``GovernanceBackend`` →
    - ``LocalGovernanceBackend`` (**local/offline default**) — wraps the EXISTING local
      markdown model-card IP (``decisions/model_card.py`` when present) + a JSON-trace
      lineage stub; ``check_access`` defaults to allow-with-note offline.
    - ``SageMakerGovernanceBackend`` (managed default) — thin boto3 ``sagemaker``
      ``create_model_card`` + ``query_lineage`` orchestration (injectable client).
* ``GuardrailBackend`` →
    - ``InProcessGuardrailBackend`` (**local/offline default**) — a deterministic content
      policy filter (denylist + PII-ish regex) so the Safety gate runs offline.
    - ``BedrockGuardrailBackend`` (managed default) — thin ``bedrock`` ``apply_guardrail``
      orchestration (injectable client).

Selectors: ``GOVERNANCE_BACKEND`` (``local`` / ``sagemaker``), ``GUARDRAIL_BACKEND``
(``inprocess`` / ``bedrock``). Offline auto → the local adapters.
"""

from __future__ import annotations

import re
from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import GovernanceBackend, GuardrailBackend
from ..registry import register_adapter
from ..types import ModelRef


# ══════════════════════════════════════════════════════════════════════════════
# GovernanceBackend
# ══════════════════════════════════════════════════════════════════════════════
class LocalGovernanceBackend:
    """Local/offline GovernanceBackend — markdown model card + JSON lineage (no AWS).

    ``model_card`` composes a deterministic model-card dict (delegating to the EXISTING
    local markdown card IP when importable); ``lineage`` returns a simple upstream/
    downstream view; ``check_access`` allows-with-note offline (LakeFormation is managed).
    """

    backend_name = "local"

    def model_card(self, *, model_ref: ModelRef, content: dict[str, Any]) -> dict[str, Any]:
        """Build a local model card for ``model_ref`` from ``content`` (intended-use/risk/metrics)."""
        card = {
            "model": model_ref.uri or model_ref.algo or "local-model",
            "algorithm": model_ref.algo,
            "intended_use": content.get("intended_use", ""),
            "risk_rating": content.get("risk_rating", "low"),
            "metrics": dict(content.get("metrics", model_ref.metrics or {})),
            "observations": content.get("observations", ""),
            "backend": "local",
        }
        # Deterministic markdown card (the local governance touchpoint; the SageMaker Model
        # Cards adapter is the managed alternative). Kept self-contained + offline-safe.
        card["markdown"] = (
            f"# Model Card: {card['model']}\n\n"
            f"- Algorithm: {card['algorithm']}\n- Intended use: {card['intended_use']}\n"
            f"- Risk rating: {card['risk_rating']}\n- Metrics: {card['metrics']}\n"
            f"- Observations: {card['observations']}\n"
        )
        return {"card_arn": f"local:model-card:{card['model']}", "card": card, "backend": "local"}

    def lineage(self, *, ref: str) -> dict[str, Any]:
        """Return a simple lineage view (offline trace stub)."""
        return {"ref": ref, "upstream": [], "downstream": [], "backend": "local",
                "note": "local lineage stub (SageMaker Lineage is the managed adapter; M6 breadth)"}

    def check_access(self, *, principal: str, resource: str) -> dict[str, Any]:
        """Offline access check — allow-with-note (LakeFormation enforcement is managed)."""
        return {"principal": principal, "resource": resource, "granted_permissions": ["DESCRIBE", "SELECT"],
                "column_filters": [], "backend": "local",
                "note": "local allow-with-note; managed LakeFormation gate lands in M6"}


class SageMakerGovernanceBackend:
    """Managed GovernanceBackend — thin SageMaker Model Cards + Lineage orchestration."""

    backend_name = "sagemaker"

    def __init__(self, *, region: str | None = None, sm_client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._sm = sm_client

    def _client(self) -> Any:
        if self._sm is None:
            import boto3

            self._sm = boto3.client("sagemaker", region_name=resolve_aws_region(self._region))
        return self._sm

    def model_card(self, *, model_ref: ModelRef, content: dict[str, Any]) -> dict[str, Any]:
        """Create a SageMaker Model Card (thin orchestration; ADR-004)."""
        import json

        sm = self._client()
        name = re.sub(r"[^a-zA-Z0-9-]+", "-", (model_ref.algo or "model")).strip("-")[:60] or "model-card"
        card_content = {
            "model_overview": {"model_artifact": [model_ref.uri]},
            "intended_uses": {"purpose_of_model": content.get("intended_use", "")},
            "evaluation_details": [{"metric_groups": [{"metrics": [
                {"name": k, "value": v} for k, v in (content.get("metrics", model_ref.metrics) or {}).items()
            ]}]}],
        }
        out = sm.create_model_card(ModelCardName=name, Content=json.dumps(card_content),
                                   ModelCardStatus=content.get("status", "Draft"))
        return {"card_arn": str(out.get("ModelCardArn", "")), "card": card_content, "backend": "sagemaker"}

    def lineage(self, *, ref: str) -> dict[str, Any]:
        """Query SageMaker Lineage for ``ref`` (best-effort; falls back to empty view)."""
        sm = self._client()
        try:
            assoc = sm.list_associations(SourceArn=ref)
            return {"ref": ref, "associations": assoc.get("AssociationSummaries", []), "backend": "sagemaker"}
        except Exception as e:
            return {"ref": ref, "upstream": [], "downstream": [], "error": str(e), "backend": "sagemaker"}

    def check_access(self, *, principal: str, resource: str) -> dict[str, Any]:
        """LakeFormation access check is M6 breadth — M4 surfaces a managed-not-yet note."""
        return {"principal": principal, "resource": resource, "granted_permissions": [],
                "backend": "sagemaker", "note": "managed LakeFormation check lands in M6"}


# ══════════════════════════════════════════════════════════════════════════════
# GuardrailBackend
# ══════════════════════════════════════════════════════════════════════════════
# A tiny deterministic content policy: blocked terms + a multi-type PII detector. Enough for the
# Stage-9 Safety gate to run offline; Bedrock Guardrails is the managed adapter.
_DEFAULT_BLOCKED = ("malware", "exploit", "ssn", "password")
_PII_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")  # US-SSN-shaped, the canonical PII probe


# WA-04 (privacy): the audit found PII handling was a SINGLE SSN regex. ``PIIRedactor`` widens
# this to a configurable, dependency-free multi-entity detector/redactor used BOTH by the
# offline guardrail (so the Safety gate flags real PII) and by the telemetry layer (to scrub
# message content before it is ever logged). The managed path remains Bedrock Guardrails' PII
# policy (provisioned by GuardrailStack); this is the offline-safe equivalent.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # credit-card BEFORE phone so 16-digit groups aren't partially eaten by the phone pattern.
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
# Order matters: longer/more-specific entities first so they win overlapping matches.
_PII_ORDER = ("email", "ssn", "credit_card", "phone", "ip")


# WA-17 (prompt-injection defense): the audit found NO input-side prompt-injection filtering on
# untrusted research/web content. ``detect_prompt_injection`` is the offline heuristic (the
# managed path is the Bedrock Guardrail prompt-attack filter, provisioned by GuardrailStack).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+|the\s+|any\s+)?previous\s+(instructions|prompts?|context)", re.I),
    re.compile(r"disregard\s+(the\s+|all\s+|any\s+)?(system\s+)?(prompt|instructions?|rules?)", re.I),
    re.compile(r"forget\s+(everything|all|your)\s+(above|previous|prior|instructions)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a\s+|an\s+)?\w+", re.I),  # role-override ("you are now DAN")
    re.compile(r"\bDAN\b|do\s+anything\s+now", re.I),
    re.compile(r"reveal\s+(the\s+|your\s+)?(system\s+)?(prompt|instructions?)", re.I),
    re.compile(r"(print|repeat|output)\s+(the\s+|your\s+)?(system\s+)?(prompt|instructions?)", re.I),
    re.compile(r"override\s+(the\s+|your\s+|all\s+)?(safety|guardrails?|instructions?|rules?)", re.I),
)


def detect_prompt_injection(text: str) -> list[str]:
    """Return the prompt-injection pattern names matched in ``text`` (WA-17; empty = clean).

    Offline heuristic over untrusted input (research papers / web content the agent ingests),
    flagging instruction-override / role-hijack / prompt-disclosure attacks. Used by the
    in-process guardrail's input-side check; the Bedrock Guardrail PROMPT_ATTACK filter is the
    always-on managed equivalent.
    """
    hits: list[str] = []
    for i, pat in enumerate(_INJECTION_PATTERNS):
        if pat.search(text or ""):
            hits.append(f"injection:{i}")
    return hits


class PIIRedactor:
    """Configurable multi-entity PII detector + redactor (WA-04; offline, dependency-free).

    ``redact(text)`` replaces every detected PII span with a ``[REDACTED:<type>]`` placeholder;
    ``detect(text)`` returns the set of entity types present. ``entities`` selects which types to
    act on (default: all). Used by the guardrail Safety gate AND telemetry content scrubbing.
    """

    def __init__(self, *, entities: tuple[str, ...] | None = None,
                 placeholder: str = "[REDACTED:{kind}]") -> None:
        self._entities = entities or _PII_ORDER
        self._placeholder = placeholder

    def detect(self, text: str) -> set[str]:
        return {k for k in self._entities if _PII_PATTERNS[k].search(text or "")}

    def redact(self, text: str) -> str:
        out = text or ""
        for kind in self._entities:
            out = _PII_PATTERNS[kind].sub(self._placeholder.format(kind=kind), out)
        return out


class InProcessGuardrailBackend:
    """Local/offline GuardrailBackend — a deterministic content policy filter (no AWS)."""

    backend_name = "inprocess"

    def __init__(self) -> None:
        self._redactor = PIIRedactor()

    def check(
        self, *, text: str, policy: dict[str, Any] | None = None, untrusted: bool = False
    ) -> dict[str, Any]:
        """Return ``{"action": "NONE"|"GUARDRAIL_INTERVENED", "violations": [...]}``.

        ``untrusted`` (SEC-A16): set True when ``text`` originates from an UNTRUSTED
        research/web/Tavily path the agent ingests into its context. On the untrusted
        path, prompt-injection detection runs by DEFAULT (OWASP LLM01 indirect
        prompt-injection); on the trusted path it stays off so internal/system prompts
        are not false-flagged. An explicit ``policy["detect_injection"]`` ALWAYS wins
        (parameterized opt-in on trusted / opt-out on untrusted).
        """
        policy = policy or {}
        blocked = [b.lower() for b in policy.get("blocked_terms", _DEFAULT_BLOCKED)]
        lowered = (text or "").lower()
        violations = [b for b in blocked if b in lowered]
        if policy.get("detect_pii", True):
            # WA-04: detect multiple PII entity types (not just SSN) via the shared redactor.
            for kind in sorted(self._redactor.detect(text or "")):
                violations.append(f"pii:{kind}")
        # SEC-A16 / WA-17: input-side prompt-injection detection. The default now follows the
        # trust boundary — DEFAULT-ON for ``untrusted`` content (research/web/Tavily), default-off
        # for trusted internal prompts (no false-flagging). An explicit policy ``detect_injection``
        # overrides the boundary default either way. The managed Bedrock Guardrail PROMPT_ATTACK
        # filter is the always-on counterpart.
        if policy.get("detect_injection", untrusted):
            violations.extend(detect_prompt_injection(text or ""))
        intervened = bool(violations)
        return {
            "action": "GUARDRAIL_INTERVENED" if intervened else "NONE",
            "passed": not intervened,
            "violations": violations,
            "backend": "inprocess",
        }


class BedrockGuardrailBackend:
    """Managed GuardrailBackend — thin Bedrock Guardrails ``apply_guardrail`` orchestration."""

    backend_name = "bedrock"

    def __init__(self, *, region: str | None = None, guardrail_id: str = "", bedrock_client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._guardrail_id = guardrail_id
        self._client_ = bedrock_client

    def _client(self) -> Any:
        if self._client_ is None:
            import boto3

            self._client_ = boto3.client("bedrock-runtime", region_name=resolve_aws_region(self._region))
        return self._client_

    def check(self, *, text: str, policy: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply a Bedrock Guardrail to ``text`` (thin orchestration; ADR-004)."""
        policy = policy or {}
        client = self._client()
        out = client.apply_guardrail(
            guardrailIdentifier=policy.get("guardrail_id", self._guardrail_id),
            guardrailVersion=policy.get("guardrail_version", "DRAFT"),
            source="OUTPUT",
            content=[{"text": {"text": text}}],
        )
        action = str(out.get("action", "NONE"))
        return {"action": action, "passed": action == "NONE",
                "assessments": out.get("assessments", []), "backend": "bedrock"}


# ── self-registration ─────────────────────────────────────────────────────────
register_adapter(GovernanceBackend, "sagemaker", lambda: SageMakerGovernanceBackend(), is_default=True)
register_adapter(GovernanceBackend, "local", lambda: LocalGovernanceBackend(), is_local=True)
register_adapter(GuardrailBackend, "bedrock", lambda: BedrockGuardrailBackend(), is_default=True)
register_adapter(GuardrailBackend, "inprocess", lambda: InProcessGuardrailBackend(), is_local=True)
register_adapter(GuardrailBackend, "local", lambda: InProcessGuardrailBackend())
