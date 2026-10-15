"""WA-08 — LLM-as-judge / model-evaluation hook for agent-output quality (AGENTOPS06).

The audit found the deterministic **Arbiter** grades ML metrics (golden-set, non-LLM) but
there was **no LLM-as-judge / Bedrock model-evaluation** for the *generative/agent* output
quality + goal-alignment. This module adds that seam with the same port/adapter shape the
rest of the project uses:

* ``LocalHeuristicJudge`` (offline default) — a deterministic, dependency-free judge so the
  eval gate runs offline + in CI with no AWS. It scores relevance/groundedness/completeness
  with transparent heuristics and returns a structured :class:`JudgeVerdict`.
* ``BedrockLLMJudge`` (managed) — a thin Bedrock ``converse`` LLM-as-judge behind the SAME
  ``Judge`` protocol (injectable client; the live grader). Selected via
  ``AMLAS_JUDGE_BACKEND=bedrock``.

``resolve_judge`` picks the backend (default ``local`` — offline-safe). The verdict is a
plain dataclass so evaluation/ + flows can gate on ``passed`` / ``score`` without LLM access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from amlas.config._env_compat import env_with_compat

from ..config.aws_region import resolve_aws_region
from ..config.runtime import load_config, supports_temperature

__all__ = [
    "BedrockLLMJudge",
    "Judge",
    "JudgeVerdict",
    "LocalHeuristicJudge",
    "resolve_judge",
]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: extract the first balanced ``{...}`` JSON object from model text.

    P4-JUDGE-01: real Claude/Bedrock models wrap the verdict JSON in a ```json
    markdown fence or leading prose, so a bare ``json.loads`` fails on every live
    reply. This tolerates a fenced / prose-wrapped object (the same tolerance the
    orchestrator's ``_parse_json_or_raise`` applies) by (1) trying a direct parse,
    then (2) scanning for the first balanced ``{...}`` and parsing it. Returns
    ``None`` when no object parses.
    """
    import json

    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, ValueError):
                        break
        start = s.find("{", start + 1)
    return None


@dataclass
class JudgeVerdict:
    """The structured result of an LLM-as-judge evaluation.

    ``score`` is a 0..1 quality score; ``passed`` is the gate decision (score >= threshold);
    ``reasons`` is a human-readable rationale list; ``criteria_scores`` is the per-criterion
    breakdown; ``backend`` records which judge produced it.
    """

    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    criteria_scores: dict[str, float] = field(default_factory=dict)
    backend: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "criteria_scores": dict(self.criteria_scores),
            "backend": self.backend,
        }


@runtime_checkable
class Judge(Protocol):
    """An agent-output judge — evaluate a (prompt, response) pair against criteria."""

    backend: str

    def evaluate(self, *, prompt: str, response: str,
                 criteria: list[str] | None = None, threshold: float = 0.6) -> JudgeVerdict: ...


class LocalHeuristicJudge:
    """Offline, deterministic agent-output judge (no AWS) — the eval default.

    Transparent heuristics per criterion (each 0..1), averaged into the overall score:
      * ``relevance``     — token overlap between prompt and response (is it on-topic?);
      * ``groundedness``  — presence of concrete grounding signals (numbers/metrics/quotes);
      * ``completeness``  — response length adequacy vs a minimal floor;
      * ``coherence``     — sentence structure sanity (non-empty, multi-token sentences).
    An empty response always fails with score 0.0 (the obvious degenerate case).
    """

    backend = "local"

    def evaluate(self, *, prompt: str, response: str,
                 criteria: list[str] | None = None, threshold: float = 0.6) -> JudgeVerdict:
        criteria = criteria or ["relevance", "groundedness", "completeness", "coherence"]
        resp = (response or "").strip()
        if not resp:
            return JudgeVerdict(score=0.0, passed=False, backend="local",
                                reasons=["empty response — automatic fail"],
                                criteria_scores={c: 0.0 for c in criteria})

        scores: dict[str, float] = {}
        reasons: list[str] = []
        for crit in criteria:
            s, why = self._score_criterion(crit, prompt or "", resp)
            scores[crit] = s
            reasons.append(f"{crit}: {s:.2f} — {why}")

        overall = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
        return JudgeVerdict(score=overall, passed=overall >= threshold, backend="local",
                            reasons=reasons, criteria_scores=scores)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import re

        return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2}

    def _score_criterion(self, crit: str, prompt: str, resp: str) -> tuple[float, str]:
        if crit == "relevance":
            p, r = self._tokens(prompt), self._tokens(resp)
            if not p:
                return 0.7, "no prompt tokens to compare; neutral"
            overlap = len(p & r) / len(p)
            return round(min(1.0, overlap * 1.5), 4), f"prompt-token overlap {overlap:.2f}"
        if crit == "groundedness":
            import re

            has_num = bool(re.search(r"\d", resp))
            has_quote = '"' in resp or "'" in resp
            score = 0.4 + (0.4 if has_num else 0.0) + (0.2 if has_quote else 0.0)
            return round(min(1.0, score), 4), f"numbers={has_num} quotes={has_quote}"
        if crit == "completeness":
            words = len(resp.split())
            return round(min(1.0, words / 30.0), 4), f"{words} words vs 30-word floor"
        if crit == "coherence":
            sentences = [s for s in resp.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            multi = sum(1 for s in sentences if len(s.split()) >= 3)
            ratio = multi / len(sentences) if sentences else 0.0
            return round(ratio, 4), f"{multi}/{len(sentences)} well-formed sentences"
        # Unknown criterion → neutral (don't fail an output on an unrecognized axis).
        return 0.6, "unknown criterion; neutral default"


class BedrockLLMJudge:
    """Managed LLM-as-judge — a thin Bedrock ``converse`` grader (ADR-004; injectable client).

    Uses the configured reviewer-tier model to grade the response and parse a JSON verdict.
    The client is injectable for offline validation; the offline default is
    :class:`LocalHeuristicJudge`.
    """

    backend = "bedrock"

    def __init__(self, *, region: str | None = None, model_id: str | None = None,
                 client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._model_id = model_id if model_id is not None else load_config().models.reviewer
        self._client = client

    def _bedrock(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=resolve_aws_region(self._region))
        return self._client

    def evaluate(self, *, prompt: str, response: str,
                 criteria: list[str] | None = None, threshold: float = 0.6) -> JudgeVerdict:
        criteria = criteria or ["relevance", "groundedness", "completeness"]
        judge_prompt = (
            "You are a strict evaluator. Score the ASSISTANT RESPONSE to the USER PROMPT on "
            f"each criterion in {criteria} from 0.0 to 1.0. Respond ONLY with compact JSON: "
            '{"criteria_scores": {"<criterion>": <float>}, "reasons": ["..."]}.\n\n'
            f"USER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}"
        )
        inference: dict[str, Any] = {"maxTokens": 512}
        if supports_temperature(self._model_id):
            inference["temperature"] = 0.0
        out = self._bedrock().converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": judge_prompt}]}],
            inferenceConfig=inference,
        )
        text = "\n".join(
            block["text"] for block in out["output"]["message"]["content"] if "text" in block
        )
        # ROBUST PARSE (P4-JUDGE-01): real Claude models habitually wrap the verdict
        # JSON in a ```json markdown fence or leading prose, so a bare ``json.loads``
        # failed on EVERY live reply (conservative 0.0 fail → the LLM-as-judge was
        # effectively non-functional against real Bedrock). Extract the first balanced
        # ``{...}`` object (same tolerance the orchestrator's ``_parse_json_or_raise``
        # uses) so a well-formed-but-fenced verdict is honored.
        parsed = _extract_json_object(text)
        if parsed is None:
            return JudgeVerdict(score=0.0, passed=False, backend="bedrock",
                                reasons=["unparseable judge reply (content withheld from logs)"])
        cscores = {k: float(v) for k, v in parsed.get("criteria_scores", {}).items()}
        overall = round(sum(cscores.values()) / len(cscores), 4) if cscores else 0.0
        return JudgeVerdict(score=overall, passed=overall >= threshold, backend="bedrock",
                            reasons=list(parsed.get("reasons", [])), criteria_scores=cscores)


def resolve_judge() -> Judge:
    """Return the configured judge — default offline ``local`` (no AWS), ``bedrock`` opt-in."""
    sel = env_with_compat("AMLAS_JUDGE_BACKEND", "local").strip().lower()
    if sel == "bedrock":
        return BedrockLLMJudge()
    return LocalHeuristicJudge()
