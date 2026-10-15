"""DeploymentBackend adapters — THE priority GAP (doc 05 §4.2 G3 / roadmap M4; ADR-007).

Closes RECON §10.4 / GAP-ADAPTER-1: RECON had **no real endpoint/batch deploy**. This
module lands the two MVP adapters behind the
:class:`~amlas.backends.ports.DeploymentBackend` port so deploy is born
backend-neutral (the same flow runs offline ↔ on AWS by flipping ``DEPLOYMENT_BACKEND``):

* ``LocalServeBackend`` — the **local/offline default** (auto-resolves offline + under
  offline mode). A genuine *in-process server*: ``package`` captures a deployable, ``deploy``
  binds a real predict callable into a process-wide endpoint table, ``invoke`` returns a
  REAL prediction (loads the reviewed Skops estimator via the SEC-02-guarded
  ``load_model_artifact`` OR a caller-supplied ``predict_fn``), ``batch_score`` scores a
  CSV, ``update_canary``/``rollback`` keep a per-endpoint config history, ``delete`` tears
  down. This is what makes the **MVP end-to-end lifecycle pass** runnable fully offline.
* ``SageMakerDeploymentBackend`` — the **default (managed)**; thin boto3 ``sagemaker``
  orchestration (ADR-004, never reimplements hosting): ``create_model`` →
  ``create_endpoint_config`` (realtime / serverless / async / multi-model) →
  ``create_endpoint`` / ``describe_endpoint``; ``invoke_endpoint`` (sagemaker-runtime);
  ``create_transform_job`` (batch); ``update_endpoint`` with **deployment guardrails**
  (canary / blue-green) + shadow variants; ``create_inference_recommendations_job``
  (Inference Recommender sizing); ``create_compilation_job`` (Neo). Clients are injectable
  so callers may use the offline AWS API emulator where supported (``create_model``/``endpoint_config``/``endpoint``/
  ``transform``/``invoke_endpoint``/``compilation``) and a recording fake-client where the offline AWS API emulator
  lacks the call (``update_endpoint``, ``create_inference_recommendations_job``).

Serving modes (the parameterized ``mode=`` on T-DEPLOY-01): ``realtime`` (instance-backed),
``serverless`` (``ServerlessConfig``), ``async`` (``AsyncInferenceConfig``), ``multimodel``
(MME — multiple ``model_refs`` behind one endpoint, ``target_model`` on invoke). The local
adapter realizes each as a local equivalent.

Selector (``DEPLOYMENT_BACKEND``): ``local`` (offline server), ``sagemaker`` (managed).
Offline auto-resolves to ``local`` (doc 05 §4.3); deploy/delete/canary/rollback are
irreversible/expensive → gated by the M1 approval gate before they ever run (doc 05 §8.8).
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from amlas.config._env_compat import env_with_compat

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region
from ...config.sagemaker_images import validate_image_region, xgboost_image
from ..ports import DeploymentBackend
from ..registry import register_adapter
from ..types import EndpointRef, JobHandle, ModelRef, PackageRef, SizingRec
from ._util import _csv_safe_cell

# Metric-name substrings whose objective is minimized — informational for sizing.
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9]+")


def _sanitize(prefix: str, raw: str) -> str:
    """A SageMaker-safe resource name (alnum + hyphen, no underscores)."""
    base = _SAFE_NAME.sub("-", raw or prefix).strip("-")[:24] or prefix
    return f"{prefix}-{base}-{uuid.uuid4().hex[:8]}"


def _trusted_model_roots() -> list[Path]:
    """Resolve the independent, configured bases for estimator loads (SEC-02 / R-B-001).

    The SEC-02 ``load_model_artifact`` guard protects artifact integrity only when its
    ``trusted_root`` is an authority the artifact path cannot also choose.
    Deriving it from the artifact's own parent (the AMLAS-SEC-B-001 bug) made the containment
    check a tautology. This resolver therefore returns roots that are independent of any caller-
    supplied artifact URI:

      * ``AMLAS_TRUSTED_MODEL_ROOT`` — an explicit, operator-configured staging/registry root
        (``os.pathsep``-separated list allowed). When set, ONLY these roots are trusted (the
        tightest, recommended production posture).
      * otherwise, the process working directory (where flow run/output dirs live) PLUS the
        system temp dir (``tempfile.gettempdir()``) — the run-scoped defaults the offline flows
        and caller-provided temporary artifacts fall under, so trusted in-tree models keep loading
        while an artifact written outside both is refused.

    A ``.skops`` artifact is acceptable iff it resolves under at least one of these roots;
    containment, extension, file, and serialized-type checks remain in ``load_model_artifact``.
    """
    configured = env_with_compat("AMLAS_TRUSTED_MODEL_ROOT", "").strip()
    if configured:
        return [Path(p).resolve() for p in configured.split(os.pathsep) if p.strip()]
    return [Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve()]


def _json_dumps(obj: Any) -> str:
    """Compact JSON encode (edge recipe / preset-deployment config bodies)."""
    import json

    return json.dumps(obj)


def _as_rows(payload: Any) -> list[list[float]]:
    """Normalize an invoke payload into a 2-D list of float feature rows.

    Accepts a single row (``[1, 2, 3]``), a batch (``[[1,2],[3,4]]``), a CSV string,
    or a dict ``{"instances": [...]}`` / ``{"features": [...]}``. Always returns a 2-D
    list so a sklearn ``.predict`` receives a 2-D array.
    """
    if isinstance(payload, dict):
        payload = payload.get("instances", payload.get("features", payload.get("data", [])))
    if isinstance(payload, str):
        rows = [r for r in payload.strip().splitlines() if r.strip()]
        return [[float(x) for x in r.split(",")] for r in rows]
    if not isinstance(payload, (list, tuple)) or len(payload) == 0:
        raise ValueError("invoke payload must be a non-empty list / list-of-lists / CSV string")
    first = payload[0]
    if isinstance(first, (list, tuple)):
        return [[float(x) for x in row] for row in payload]
    return [[float(x) for x in payload]]  # a single flat row → one 2-D row


# ══════════════════════════════════════════════════════════════════════════════
# LocalServeBackend — the genuine in-process server (offline default + MVP demo)
# ══════════════════════════════════════════════════════════════════════════════
class LocalServeBackend:
    """Local/offline DeploymentBackend — an in-process model server (no AWS).

    Endpoint + package state is held in **process-wide class dicts** (the registry builds
    a fresh adapter instance per ``resolve``, so per-instance state would not survive
    across a deploy→invoke flow). ``deploy`` resolves a real predict callable and binds it
    to a named endpoint; ``invoke`` runs it for a genuine prediction; ``update_canary`` /
    ``rollback`` keep a per-endpoint config-version history; ``delete`` removes it.
    """

    backend_name = "local"

    # process-wide tables (shared across instances) + a lock for thread-safety.
    _ENDPOINTS: ClassVar[dict[str, dict[str, Any]]] = {}
    _PACKAGES: ClassVar[dict[str, dict[str, Any]]] = {}
    _LOCK = threading.Lock()

    # ── package (T-DEPLOY-09) ───────────────────────────────────────────────────
    def package(self, *, model_ref: ModelRef, image: str | None = None,
                inference_code: str | None = None, env: dict[str, Any] | None = None) -> PackageRef:
        """Capture a deployable package ref for ``model_ref`` (records the predict source)."""
        arn = f"local:package:{uuid.uuid4().hex[:10]}"
        with self._LOCK:
            self._PACKAGES[arn] = {
                "model_ref": model_ref,
                "image": image or "local-inproc",
                "inference_code": inference_code,
                "env": dict(env or {}),
            }
        return PackageRef(arn=arn, group=str(model_ref.algo or "local-model"), version=1,
                          approval_status="Packaged", metrics=dict(model_ref.metrics or {}))

    # ── deploy (T-DEPLOY-01, mode=) ─────────────────────────────────────────────
    def deploy(self, *, model_ref: ModelRef, mode: str = "realtime",
               resources: dict[str, Any] | None = None, autoscaling: dict[str, Any] | None = None) -> EndpointRef:
        """Bind a real predict callable into a named local endpoint (any ``mode``)."""
        res = dict(resources or {})
        name = res.get("endpoint_name") or _sanitize("local-ep", model_ref.algo or "model")
        predict = self._resolve_predict(model_ref, res)
        # multimodel: also accept a mapping of {target_model: model_ref} via resources.
        model_table: dict[str, Callable[[Any], Any]] = {}
        if mode == "multimodel" and isinstance(res.get("model_refs"), dict):
            model_table = {str(k): self._resolve_predict(v, res) for k, v in res["model_refs"].items()}
        with self._LOCK:
            self._ENDPOINTS[name] = {
                "predict": predict,
                "model_table": model_table,
                "mode": mode,
                "model_ref": model_ref,
                "history": [],  # config-version stack for rollback
                "version": 1,
                "autoscaling": dict(autoscaling or {}),
            }
        return EndpointRef(name=name, arn=f"local:endpoint:{name}", mode=mode, backend="local")

    # ── invoke (T-DEPLOY-02) ────────────────────────────────────────────────────
    def invoke(self, *, endpoint: EndpointRef, payload: Any, target_model: str | None = None) -> dict[str, Any]:
        """Invoke the local endpoint and return a REAL ``{"prediction": ...}``.

        For a ``streaming`` endpoint (G3) the in-process server realizes response-streaming
        as a **queue-backed predict loop**: each input row is enqueued, scored, and yielded as
        an ordered chunk — the local equivalent of a SageMaker response-streaming /
        async-inference endpoint. The result still carries a ``prediction`` key (so the
        ``deploy_flow`` smoke-invoke is mode-agnostic) plus a ``stream`` list + ``streamed=True``.
        """
        ep = self._ENDPOINTS.get(endpoint.name)
        if ep is None:
            raise RuntimeError(f"local endpoint {endpoint.name!r} not deployed (or already deleted)")
        predict = ep["predict"]
        if target_model and ep.get("model_table"):
            if target_model not in ep["model_table"]:
                raise RuntimeError(f"multi-model endpoint has no target_model {target_model!r}")
            predict = ep["model_table"][target_model]
        rows = _as_rows(payload)

        if ep.get("mode") == "streaming":
            # queue-backed predict loop: score row-by-row, emit ordered chunks (real streaming).
            stream: list[Any] = []
            for row in rows:
                stream.extend(self._to_list(predict([row])))
            single = stream[0] if len(stream) == 1 else stream
            return {"prediction": single, "predictions": stream, "stream": stream,
                    "streamed": True, "endpoint": endpoint.name, "mode": "streaming",
                    "target_model": target_model, "backend": "local"}

        preds = predict(rows)
        preds_list = self._to_list(preds)
        single = preds_list[0] if len(preds_list) == 1 else preds_list
        return {"prediction": single, "predictions": preds_list, "endpoint": endpoint.name,
                "mode": ep["mode"], "target_model": target_model, "backend": "local"}

    # ── canary / blue-green (T-DEPLOY-03) ───────────────────────────────────────
    def update_canary(self, *, endpoint: EndpointRef, new_config: dict[str, Any],
                      canary_pct: int, alarms: list[str] | None = None, mode: str = "canary") -> dict[str, Any]:
        """Shift the endpoint to ``new_config``'s predictor, keeping the prior in history."""
        ep = self._ENDPOINTS.get(endpoint.name)
        if ep is None:
            raise RuntimeError(f"local endpoint {endpoint.name!r} not deployed")
        new_ref = new_config.get("model_ref")
        new_predict = self._resolve_predict(new_ref, new_config) if new_ref is not None else (
            new_config.get("predict_fn") or ep["predict"]
        )
        with self._LOCK:
            ep["history"].append({"version": ep["version"], "predict": ep["predict"]})
            ep["predict"] = new_predict
            ep["version"] += 1
        return {"deployment_status": "Completed", "strategy": mode, "canary_pct": int(canary_pct),
                "endpoint": endpoint.name, "version": ep["version"], "alarms": list(alarms or []),
                "backend": "local"}

    # ── batch transform (T-DEPLOY-04) ───────────────────────────────────────────
    def batch_score(self, *, model_ref: ModelRef, input_uri: str, output_uri: str,
                    resources: dict[str, Any] | None = None) -> JobHandle:
        """Score every row of the ``input_uri`` CSV locally and write predictions."""
        predict = self._resolve_predict(model_ref, dict(resources or {}))
        rows: list[list[float]] = []
        with open(input_uri, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header_seen = False
            for row in reader:
                if not row:
                    continue
                if not header_seen and any(not _is_floatable(c) for c in row):
                    header_seen = True  # skip a header line
                    continue
                rows.append([float(c) for c in row])
        preds = self._to_list(predict(rows))
        out = Path(output_uri)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)  # nosemgrep: use-defusedcsv
            writer.writerow(["prediction"])
            for p in preds:
                writer.writerow([_csv_safe_cell(p)])
        return JobHandle(id=_sanitize("local-batch", model_ref.algo or "score"), backend="local",
                         extra={"output_uri": str(out), "n_scored": len(preds), "status": "Completed"})

    # ── rollback (T-DEPLOY-07) ──────────────────────────────────────────────────
    def rollback(self, *, endpoint: EndpointRef, to_version: str | None = None) -> dict[str, Any]:
        """Restore the endpoint's previous predictor from the config history."""
        ep = self._ENDPOINTS.get(endpoint.name)
        if ep is None:
            raise RuntimeError(f"local endpoint {endpoint.name!r} not deployed")
        if not ep["history"]:
            return {"status": "NoPriorVersion", "endpoint": endpoint.name, "restored_version": ep["version"],
                    "backend": "local"}
        with self._LOCK:
            prior = ep["history"].pop()
            ep["predict"] = prior["predict"]
            restored = prior["version"]
            ep["version"] = restored
        return {"status": "RolledBack", "endpoint": endpoint.name, "restored_version": restored,
                "backend": "local"}

    # ── delete (T-DEPLOY-08) ────────────────────────────────────────────────────
    def delete(self, *, endpoint: EndpointRef) -> dict[str, Any]:
        """Tear the local endpoint down (irreversible)."""
        with self._LOCK:
            existed = self._ENDPOINTS.pop(endpoint.name, None) is not None
        return {"status": "Deleted" if existed else "NotFound", "endpoint": endpoint.name, "backend": "local"}

    # ── inference-config sizing (T-DEPLOY-11) ───────────────────────────────────
    def recommend_config(self, *, model_ref: ModelRef, sample: str,
                         sla: dict[str, Any] | None = None) -> list[SizingRec]:
        """Return a small deterministic local sizing shortlist (no AWS)."""
        sla = sla or {}
        # A tiny deterministic catalog ordered cheapest-first; latency/throughput are
        # heuristic estimates (the managed adapter returns Inference Recommender data).
        catalog = [
            SizingRec(instance_type="ml.t2.medium", est_cost_per_hour=0.065, est_latency_ms=120.0, est_throughput=80.0),
            SizingRec(instance_type="ml.m5.large", est_cost_per_hour=0.134, est_latency_ms=45.0, est_throughput=320.0),
            SizingRec(instance_type="ml.c5.xlarge", est_cost_per_hour=0.238, est_latency_ms=22.0, est_throughput=640.0),
        ]
        sla_ms = sla.get("latency_sla_ms")
        if sla_ms:
            catalog = [r for r in catalog if r.est_latency_ms <= float(sla_ms)] or catalog
        return catalog

    # ── edge / embedded deploy (T-DEPLOY-10b, G3) ───────────────────────────────
    def deploy_edge(self, *, model_ref: ModelRef, compile_output: dict[str, Any] | None = None,
                    target_device: str = "jetson_nano", framework: str = "PYTORCH",
                    greengrass_target: str | None = None, component_name: str | None = None,
                    bundle_dir: str | None = None, data_shape: str = '{"input":[1,3,224,224]}') -> dict[str, Any]:
        """Bundle a compiled edge artifact locally (the offline equivalent of Neo→Edge→Greengrass).

        The local server has no SageMaker Neo / Edge Manager / Greengrass, so it realizes the edge
        path as a REAL on-disk bundle: it writes a ``manifest.json`` (compilation job id, target
        device, framework, component name/version) + a ``model.edge`` placeholder artifact into a
        bundle directory — the local analogue of the packaged Greengrass V2 component. When a
        ``greengrass_target`` is supplied the bundle is marked ``deployed`` with a synthetic
        deployment id (the local analogue of ``greengrassv2:create_deployment``).
        """
        comp = component_name or _sanitize("edge-comp", model_ref.algo or "edge")
        compile_job = (compile_output or {}).get("compilation_job") if compile_output else None
        compile_job = compile_job or _sanitize("neo", model_ref.algo or "compile")
        pkg_job = _sanitize("edge-pkg", model_ref.algo or "edge")
        # write a real local bundle (the packaged compiled-artifact + manifest).
        base = Path(bundle_dir) if bundle_dir else Path(tempfile.mkdtemp(prefix="amlas-edge-"))
        base.mkdir(parents=True, exist_ok=True)
        component_version = "1.0.0"
        manifest = {
            "component_name": comp,
            "component_version": component_version,
            "compilation_job": compile_job,
            "edge_packaging_job": pkg_job,
            "target_device": target_device,
            "framework": framework.upper(),
            "data_shape": data_shape,
            "model_ref": model_ref.uri or model_ref.algo,
            "greengrass_target": greengrass_target,
        }
        import json as _json

        (base / "manifest.json").write_text(_json.dumps(manifest, indent=2))
        (base / "model.edge").write_text(f"compiled:{model_ref.uri or model_ref.algo}:{target_device}\n")
        deployed = greengrass_target is not None
        deployment_id = _sanitize("ggv2-deploy", comp) if deployed else ""
        return {
            "compilation_job": compile_job,
            "edge_packaging_job": pkg_job,
            "component_name": comp,
            "component_version": component_version,
            "bundle_uri": str(base),
            "target_device": target_device,
            "greengrass_target": greengrass_target,
            "deployment_id": deployment_id,
            "deployed": deployed,
            "edge_endpoint": EndpointRef(name=comp, arn=f"local:edge:{comp}", mode="edge", backend="local"),
            "backend": "local",
        }

    # ── internals ───────────────────────────────────────────────────────────────
    def _resolve_predict(self, model_ref: ModelRef | None, opts: dict[str, Any]) -> Callable[[Any], Any]:
        """Build a real predict callable from a ``predict_fn``, a Skops estimator, or fail.

        Resolution order (genuine-serving-first):
          1. an explicit ``predict_fn`` callable in ``opts`` or ``model_ref.extra``;
          2. a Skops estimator at ``model_ref.uri`` (``.skops``) loaded via the
             SEC-02-guarded ``load_model_artifact`` → its ``.predict``;
          3. a ``model`` object with ``.predict`` in ``model_ref.extra``.
        Raises if none is resolvable (no fake constant predictor — the server is real).
        """
        extra = dict(getattr(model_ref, "extra", {}) or {})
        fn = opts.get("predict_fn") or extra.get("predict_fn")
        if callable(fn):
            return lambda rows: fn(rows)

        obj = extra.get("model")
        if obj is not None and hasattr(obj, "predict"):
            return lambda rows: obj.predict(rows)

        uri = getattr(model_ref, "uri", "") if model_ref is not None else ""
        if uri and str(uri).endswith(".skops"):
            from ...security.model_artifacts import UntrustedArtifactError, load_model_artifact

            # The trusted root must be independent of the caller-supplied artifact path.
            roots = _trusted_model_roots()
            last_err: UntrustedArtifactError | None = None
            for root in roots:
                try:
                    est = load_model_artifact(uri, trusted_root=root)
                    return lambda rows: est.predict(rows)
                except UntrustedArtifactError as error:
                    last_err = error
            raise UntrustedArtifactError(
                f"refusing to load .skops {uri!r}: it fails every configured trusted root "
                f"{[str(root) for root in roots]} (set AMLAS_TRUSTED_MODEL_ROOT to the "
                "artifact staging/registry directory)"
            ) from last_err

        raise RuntimeError(
            "LocalServeBackend cannot resolve a predict callable: pass model_ref.extra['predict_fn'] "
            "(callable) or model_ref.extra['model'] (has .predict) or a .skops model_ref.uri"
        )

    @staticmethod
    def _to_list(preds: Any) -> list[Any]:
        """Coerce a prediction result (numpy array / list / scalar) into a plain list."""
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        if not isinstance(preds, (list, tuple)):
            return [preds]
        return list(preds)


def _is_floatable(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SageMakerDeploymentBackend — thin boto3 orchestration (managed default, ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class SageMakerDeploymentBackend:
    """Managed DeploymentBackend — orchestrates SageMaker hosting (never reimplements it).

    A thin wrapper over the boto3 ``sagemaker`` + ``sagemaker-runtime`` clients (injectable
    for the offline AWS API emulator / fake-client tests). Each method authors the managed call sequence and maps
    the vendor response back to the domain types — the SageMaker service does the hosting
    (ADR-004 scope guard). Modes map to the create_endpoint_config shape:
    realtime→instance variant, serverless→``ServerlessConfig``, async→``AsyncInferenceConfig``,
    multimodel→``Mode=MultiModel`` container.
    """

    backend_name = "sagemaker"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        default_image: str = "",
        s3_model_bucket: str = "",
        sagemaker_client: Any = None,
        runtime_client: Any = None,
        greengrass_client: Any = None,
        autoscaling_client: Any = None,
    ) -> None:
        # LIVE-3 (same class as the LIVE-1 Clarify / LIVE-2 IR cross-account-pass-role fix):
        # a real ``create_model`` / IR / Neo job rejects the ``000000000000`` placeholder with
        # ``RoleArn: Cross-account pass role is not allowed`` (its account never matches the
        # caller's). Resolve the SageMaker exec role from the explicit arg → the SEC-01
        # env (the ``AMLAS_SAGEMAKER_ROLE_ARN`` environment setting) → the placeholder.
        # So when the managed (``DEPLOYMENT_BACKEND=sagemaker``) adapter is auto-built by the
        # registry's no-arg factory, it still passes a real same-account role on the live path.
        import os as _os
        self._region = region
        self._role = (
            role_arn
            or _os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "")
            or _os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "")
            or "arn:aws:iam::000000000000:role/deploy"
        )
        self._default_image = default_image
        self._bucket = s3_model_bucket or "<your-model-bucket>"  # invalid-by-construction: cannot be squatted
        self._sm = sagemaker_client
        self._rt = runtime_client
        self._gg = greengrass_client
        self._aas = autoscaling_client

    def _client(self) -> Any:
        if self._sm is None:
            import boto3

            self._sm = boto3.client("sagemaker", region_name=resolve_aws_region(self._region))
        return self._sm

    def _runtime(self) -> Any:
        if self._rt is None:
            import boto3

            self._rt = boto3.client("sagemaker-runtime", region_name=resolve_aws_region(self._region))
        return self._rt

    def _autoscaling(self) -> Any:
        """Resolve the Application Auto Scaling client lazily (WA-03 — only when scaling)."""
        if self._aas is None:
            import boto3

            self._aas = boto3.client("application-autoscaling", region_name=resolve_aws_region(self._region))
        return self._aas

    def _greengrass(self) -> Any:
        """Resolve the Greengrass V2 client lazily (only when an edge deploy actually deploys)."""
        if self._gg is None:
            import boto3

            self._gg = boto3.client("greengrassv2", region_name=resolve_aws_region(self._region))
        return self._gg

    # ── package: create_model (T-DEPLOY-09) ─────────────────────────────────────
    def package(self, *, model_ref: ModelRef, image: str | None = None,
                inference_code: str | None = None, env: dict[str, Any] | None = None) -> PackageRef:
        """Create a SageMaker Model from the artifact (the deployable container ref)."""
        sm = self._client()
        model_name = _sanitize("model", model_ref.algo or "pkg")
        override = image or model_ref.extra.get("image") or self._default_image
        region = resolve_aws_client_region(sm, self._region, required=not bool(override))
        container: dict[str, Any] = {
            "Image": validate_image_region(str(override or xgboost_image(region)), region),
            "ModelDataUrl": model_ref.uri or f"s3://{self._bucket}/{model_name}/model.tar.gz",
        }
        if env:
            container["Environment"] = {str(k): str(v) for k, v in env.items()}
        out = sm.create_model(ModelName=model_name, ExecutionRoleArn=self._role, PrimaryContainer=container)
        return PackageRef(arn=str(out.get("ModelArn", f"arn:aws:sagemaker:{region}::model/{model_name}")),
                          group=model_name, version=1, approval_status="Packaged",
                          metrics=dict(model_ref.metrics or {}))

    # ── deploy: endpoint_config → endpoint (T-DEPLOY-01, mode=) ─────────────────
    def deploy(self, *, model_ref: ModelRef, mode: str = "realtime",
               resources: dict[str, Any] | None = None, autoscaling: dict[str, Any] | None = None) -> EndpointRef:
        """create_model (if needed) → create_endpoint_config(mode) → create_endpoint."""
        sm = self._client()
        res = dict(resources or {})
        model_name = res.get("model_name")
        if not model_name:
            pkg = self.package(model_ref=model_ref, image=res.get("image"), env=res.get("env"))
            model_name = pkg.group
        cfg_name = _sanitize("ep-config", model_name)
        ep_name = res.get("endpoint_name") or _sanitize("endpoint", model_name)
        variant = self._build_variant(model_name, mode, res)
        cfg_kwargs: dict[str, Any] = {"EndpointConfigName": cfg_name, "ProductionVariants": [variant]}
        if mode == "async":
            cfg_kwargs["AsyncInferenceConfig"] = {
                "OutputConfig": {"S3OutputPath": res.get("async_output_uri", f"s3://{self._bucket}/async-out")}
            }
        elif mode == "streaming":
            # G3: the managed streaming realization is an async-inference endpoint with a
            # ClientConfig (response-streaming wiring) + an optional event-source notification
            # (SNS success/error topics). The ClientConfig is the marker that distinguishes a
            # streaming endpoint from a plain async one (master plan §G3).
            async_cfg: dict[str, Any] = {
                "OutputConfig": {"S3OutputPath": res.get("async_output_uri", f"s3://{self._bucket}/stream-out")},
                "ClientConfig": {
                    "MaxConcurrentInvocationsPerInstance": int(res.get("max_concurrent_invocations", 4)),
                },
            }
            notif = res.get("notification_config") or {}
            if notif.get("success_topic") or notif.get("error_topic"):
                async_cfg["OutputConfig"]["NotificationConfig"] = {
                    k: v for k, v in (
                        ("SuccessTopic", notif.get("success_topic")),
                        ("ErrorTopic", notif.get("error_topic")),
                    ) if v
                }
            cfg_kwargs["AsyncInferenceConfig"] = async_cfg
        sm.create_endpoint_config(**cfg_kwargs)
        sm.create_endpoint(EndpointName=ep_name, EndpointConfigName=cfg_name)
        desc = sm.describe_endpoint(EndpointName=ep_name)
        # WA-03 (MLREL05-BP01): honor the previously-ignored ``autoscaling`` dict by registering
        # an Application Auto Scaling target + target-tracking policy for instance-backed
        # endpoints. Serverless endpoints scale natively (MaxConcurrency) so they are skipped.
        if autoscaling and mode != "serverless":
            self._register_autoscaling(ep_name, res.get("variant_name", "AllTraffic"), autoscaling)
        return EndpointRef(name=ep_name, arn=str(desc.get("EndpointArn", "")),
                           mode=mode, backend="sagemaker")

    def _register_autoscaling(self, endpoint_name: str, variant_name: str,
                              autoscaling: dict[str, Any]) -> dict[str, Any]:
        """Register an Application Auto Scaling target + target-tracking policy (WA-03).

        Translates the neutral ``autoscaling`` dict (``min_capacity`` / ``max_capacity`` /
        ``target_value`` / optional cooldowns + ``metric``) into the
        ``application-autoscaling`` ``register_scalable_target`` + ``put_scaling_policy`` calls
        SageMaker endpoints use to scale ``DesiredInstanceCount`` on
        ``SageMakerVariantInvocationsPerInstance`` (the canonical endpoint scaling metric).
        Returns a provider-neutral summary so the tool surface can report it.
        """
        aas = self._autoscaling()
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
        min_cap = int(autoscaling.get("min_capacity", 1))
        max_cap = int(autoscaling.get("max_capacity", max(min_cap, 2)))
        target_value = float(autoscaling.get("target_value", 70.0))
        predefined = autoscaling.get("metric", "SageMakerVariantInvocationsPerInstance")
        aas.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=min_cap,
            MaxCapacity=max_cap,
        )
        policy_cfg: dict[str, Any] = {
            "TargetValue": target_value,
            "PredefinedMetricSpecification": {"PredefinedMetricType": predefined},
            "ScaleInCooldown": int(autoscaling.get("scale_in_cooldown", 300)),
            "ScaleOutCooldown": int(autoscaling.get("scale_out_cooldown", 60)),
        }
        aas.put_scaling_policy(
            PolicyName=f"{endpoint_name}-target-tracking",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration=policy_cfg,
        )
        return {"resource_id": resource_id, "min_capacity": min_cap, "max_capacity": max_cap,
                "target_value": target_value, "metric": predefined}

    def _build_variant(self, model_name: str, mode: str, res: dict[str, Any]) -> dict[str, Any]:
        """Translate the neutral ``mode`` + resources into a SageMaker ProductionVariant."""
        variant: dict[str, Any] = {"VariantName": res.get("variant_name", "AllTraffic"), "ModelName": model_name}
        if mode == "serverless":
            variant["ServerlessConfig"] = {
                "MemorySizeInMB": int(res.get("memory_mb", 2048)),
                "MaxConcurrency": int(res.get("max_concurrency", 5)),
            }
        else:  # realtime / async / multimodel → instance-backed
            variant["InitialInstanceCount"] = int(res.get("instance_count", 1))
            variant["InstanceType"] = res.get("instance_type", "ml.m5.large")
            variant["InitialVariantWeight"] = float(res.get("variant_weight", 1.0))
        return variant

    # ── invoke (T-DEPLOY-02) ────────────────────────────────────────────────────
    def invoke(self, *, endpoint: EndpointRef, payload: Any, target_model: str | None = None) -> dict[str, Any]:
        """Invoke via sagemaker-runtime; return the decoded ``{"prediction": ...}``."""
        rt = self._runtime()
        body = payload if isinstance(payload, (bytes, bytearray, str)) else self._to_csv(payload)
        kwargs: dict[str, Any] = {"EndpointName": endpoint.name, "Body": body,
                                  "ContentType": "text/csv", "Accept": "text/csv"}
        if target_model:
            kwargs["TargetModel"] = target_model
        out = rt.invoke_endpoint(**kwargs)
        raw = out.get("Body")
        decoded = raw.read() if hasattr(raw, "read") else raw
        if isinstance(decoded, (bytes, bytearray)):
            decoded = decoded.decode("utf-8", errors="replace")
        return {"prediction": decoded, "endpoint": endpoint.name, "target_model": target_model,
                "backend": "sagemaker"}

    # ── canary / blue-green (T-DEPLOY-03) ───────────────────────────────────────
    def update_canary(self, *, endpoint: EndpointRef, new_config: dict[str, Any],
                      canary_pct: int, alarms: list[str] | None = None, mode: str = "canary") -> dict[str, Any]:
        """Build a new endpoint-config + update_endpoint with deployment guardrails.

        Uses ``DeploymentConfig`` (canary/linear traffic-shift + CloudWatch auto-rollback
        alarms). the offline AWS API emulator does not implement ``update_endpoint`` — callers may inject a recording
        fake-client and assert the call shape (the orchestration), not a live transition.
        """
        sm = self._client()
        model_name = new_config.get("model_name") or new_config.get("ModelName")
        cfg_name = _sanitize("ep-config", model_name or endpoint.name)
        variant = self._build_variant(model_name or endpoint.name, new_config.get("mode", "realtime"), new_config)
        sm.create_endpoint_config(EndpointConfigName=cfg_name, ProductionVariants=[variant])
        traffic_type = "ALL_AT_ONCE" if mode == "blue_green" else "CANARY"
        deployment_config: dict[str, Any] = {
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": {
                    "Type": traffic_type,
                    "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": int(canary_pct)},
                    "WaitIntervalInSeconds": int(new_config.get("wait_interval", 300)),
                },
            },
        }
        if alarms:
            deployment_config["AutoRollbackConfiguration"] = {"Alarms": [{"AlarmName": a} for a in alarms]}
        sm.update_endpoint(EndpointName=endpoint.name, EndpointConfigName=cfg_name,
                           DeploymentConfig=deployment_config)
        return {"deployment_status": "Updating", "strategy": mode, "canary_pct": int(canary_pct),
                "endpoint": endpoint.name, "endpoint_config": cfg_name, "alarms": list(alarms or []),
                "backend": "sagemaker"}

    # ── batch transform (T-DEPLOY-04) ───────────────────────────────────────────
    def batch_score(self, *, model_ref: ModelRef, input_uri: str, output_uri: str,
                    resources: dict[str, Any] | None = None) -> JobHandle:
        """Submit a SageMaker Batch Transform job (create_transform_job)."""
        sm = self._client()
        res = dict(resources or {})
        model_name = res.get("model_name")
        if not model_name:
            model_name = self.package(model_ref=model_ref, image=res.get("image")).group
        job = _sanitize("transform", model_name)
        sm.create_transform_job(
            TransformJobName=job, ModelName=model_name,
            TransformInput={"DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_uri}},
                            "ContentType": res.get("content_type", "text/csv"),
                            "SplitType": res.get("split_type", "Line")},
            TransformOutput={"S3OutputPath": output_uri},
            TransformResources={"InstanceType": res.get("instance_type", "ml.m5.large"),
                                "InstanceCount": int(res.get("instance_count", 1))},
        )
        desc = sm.describe_transform_job(TransformJobName=job)
        return JobHandle(id=job, backend="sagemaker", arn=str(desc.get("TransformJobArn", "")),
                         extra={"status": desc.get("TransformJobStatus", "InProgress"), "output_uri": output_uri})

    # ── rollback (T-DEPLOY-07) ──────────────────────────────────────────────────
    def rollback(self, *, endpoint: EndpointRef, to_version: str | None = None) -> dict[str, Any]:
        """Roll the endpoint back to a prior endpoint-config (update_endpoint)."""
        sm = self._client()
        prior_cfg = to_version or f"{endpoint.name}-prior"
        sm.update_endpoint(EndpointName=endpoint.name, EndpointConfigName=prior_cfg)
        return {"status": "RolledBack", "endpoint": endpoint.name, "restored_version": prior_cfg,
                "backend": "sagemaker"}

    # ── delete (T-DEPLOY-08) ────────────────────────────────────────────────────
    def delete(self, *, endpoint: EndpointRef) -> dict[str, Any]:
        """Delete the endpoint (delete_endpoint) — irreversible."""
        sm = self._client()
        sm.delete_endpoint(EndpointName=endpoint.name)
        return {"status": "Deleting", "endpoint": endpoint.name, "backend": "sagemaker"}

    # ── inference recommender (T-DEPLOY-11) ─────────────────────────────────────
    def recommend_config(self, *, model_ref: ModelRef, sample: str,
                         sla: dict[str, Any] | None = None,
                         job_name: str | None = None,
                         wait_seconds: int = 0, poll_interval: float = 30.0) -> list[SizingRec]:
        """Submit an Inference Recommender job + parse its recommendations.

        the offline AWS API emulator does not implement ``create_inference_recommendations_job`` — callers may inject a
        recording fake-client that returns a canned ``InferenceRecommendations`` describe.

        G6 live-injection seam (ADDITIVE, default-preserving): a real IR ``Default`` job runs
        asynchronously (~30-45 min) and the FIRST ``describe`` returns ``PENDING``/``IN_PROGRESS``
        with no recommendations, whereas the recording fake returns ``COMPLETED`` instantly. So:

          * ``job_name`` — an explicit, predictable job name the operator-driven live test supplies
            so it can poll / stop / tear down the real job (default ``None`` → the original
            auto-sanitized ``ir-<algo>-<uuid>`` name — byte-identical to today);
          * ``wait_seconds`` / ``poll_interval`` — when ``wait_seconds>0`` the call POLLS
            ``describe_inference_recommendations_job`` until the job reaches a terminal status
            (``COMPLETED``/``FAILED``/``STOPPED``) or the budget elapses, THEN parses. Default
            ``wait_seconds=0`` → a single describe + parse, exactly as before (the recording-fake
            test stays green).
        """
        import time as _time

        sm = self._client()
        sla = sla or {}
        job = job_name or _sanitize("ir", model_ref.algo or "rec")
        pkg_arn = model_ref.extra.get("model_package_arn") if isinstance(model_ref.extra, dict) else None
        sm.create_inference_recommendations_job(
            JobName=job, JobType="Default", RoleArn=self._role,
            InputConfig={"ModelPackageVersionArn": pkg_arn or model_ref.uri or "arn:aws:sagemaker:::model-package/p/1"},
        )
        desc = sm.describe_inference_recommendations_job(JobName=job)
        if wait_seconds > 0:
            # live-poll path: keep describing until terminal (or budget exhausted) before parsing.
            _terminal = {"COMPLETED", "FAILED", "STOPPED"}
            deadline = _time.time() + wait_seconds
            while str(desc.get("Status", "")).upper() not in _terminal and _time.time() < deadline:
                if poll_interval > 0:
                    # Deadline-bounded poll (``deadline`` from ``wait_seconds``) — P3 audit H-06.
                    _time.sleep(poll_interval)  # nosemgrep: arbitrary-sleep
                desc = sm.describe_inference_recommendations_job(JobName=job)
        recs: list[SizingRec] = []
        for r in desc.get("InferenceRecommendations", []) or []:
            ep = r.get("EndpointConfiguration", {})
            metr = r.get("Metrics", {})
            recs.append(SizingRec(
                instance_type=str(ep.get("InstanceType", "ml.m5.large")),
                est_cost_per_hour=float(metr.get("CostPerHour", 0.0) or 0.0),
                est_latency_ms=float(metr.get("ModelLatency", 0.0) or 0.0),
                est_throughput=float(metr.get("MaxInvocations", 0.0) or 0.0),
            ))
        return recs

    # ── compile / Neo (T-DEPLOY-10) — extra method, used by the compile_model tool ──
    def compile(self, *, model_ref: ModelRef, target_device: str = "ml_c5",
                framework: str = "PYTORCH", dtype: str = "fp32",
                data_shape: str = '{"input":[1,3,224,224]}') -> dict[str, Any]:
        """Submit a SageMaker Neo compilation job (create_compilation_job)."""
        sm = self._client()
        job = _sanitize("neo", model_ref.algo or "compile")
        out_location = f"s3://{self._bucket}/neo-out"
        sm.create_compilation_job(
            CompilationJobName=job, RoleArn=self._role,
            InputConfig={"S3Uri": model_ref.uri or f"s3://{self._bucket}/model.tar.gz",
                         "DataInputConfig": data_shape, "Framework": framework.upper()},
            OutputConfig={"S3OutputLocation": out_location, "TargetDevice": target_device},
            StoppingCondition={"MaxRuntimeInSeconds": 900},
        )
        return {"compilation_job": job, "target_device": target_device, "dtype": dtype,
                "output_uri": out_location, "framework": framework.upper(), "backend": "sagemaker"}

    # ── edge / embedded deploy (T-DEPLOY-10b, G3) ───────────────────────────────
    def deploy_edge(self, *, model_ref: ModelRef, compile_output: dict[str, Any] | None = None,
                    target_device: str = "jetson_nano", framework: str = "PYTORCH",
                    greengrass_target: str | None = None, component_name: str | None = None,
                    component_version: str = "1.0.0",
                    data_shape: str = '{"input":[1,3,224,224]}') -> dict[str, Any]:
        """Compile (Neo) → package (Edge Manager) → optionally deploy (Greengrass V2) for an edge device.

        The managed edge path (master plan §G3) extends the Neo ``compile`` (T-DEPLOY-10) into an
        actual edge/embedded deploy:

          1. **compile** — reuse a supplied ``compile_output`` (from a prior :meth:`compile`) or
             submit a Neo ``create_compilation_job`` for ``target_device``;
          2. **package** — ``create_edge_packaging_job`` (SageMaker Edge Manager) with a
             ``PresetDeploymentType=GreengrassV2Component`` so the compiled artifact becomes a
             deployable Greengrass V2 component;
          3. **deploy** — when a ``greengrass_target`` (thing/thing-group ARN) is supplied AND a
             Greengrass V2 client is available, ``create_component_version`` + ``create_deployment``
             roll the component onto the fleet.

        the offline AWS API emulator implements ``create_compilation_job`` but NOT ``create_edge_packaging_job`` /
        Greengrass V2 — callers may inject recording fakes and assert the call shape (the orchestration),
        exactly like ``update_endpoint`` / Inference Recommender. Returns the SAME contract the
        local adapter returns so the two are interchangeable.
        """
        sm = self._client()
        # ── 1. compile (reuse a supplied output, else submit Neo) ──
        if compile_output and compile_output.get("compilation_job"):
            compile_job = str(compile_output["compilation_job"])
            model_artifact = compile_output.get("output_uri") or f"s3://{self._bucket}/neo-out"
        else:
            compiled = self.compile(model_ref=model_ref, target_device=target_device,
                                    framework=framework, data_shape=data_shape)
            compile_job = compiled["compilation_job"]
            # describe to resolve the compiled artifact location (best-effort under fakes/the offline AWS API emulator).
            try:
                cdesc = sm.describe_compilation_job(CompilationJobName=compile_job)
                model_artifact = cdesc.get("ModelArtifacts", {}).get("S3ModelArtifacts") \
                    or compiled.get("output_uri")
            except Exception:
                model_artifact = compiled.get("output_uri")

        # ── 2. package as a Greengrass V2 component (Edge Manager) ──
        comp = component_name or _sanitize("edge-comp", model_ref.algo or "edge")
        pkg_job = _sanitize("edge-pkg", model_ref.algo or "edge")
        sm.create_edge_packaging_job(
            EdgePackagingJobName=pkg_job,
            CompilationJobName=compile_job,
            ModelName=_sanitize("edge-model", model_ref.algo or "edge"),
            ModelVersion=component_version,
            RoleArn=self._role,
            OutputConfig={
                "S3OutputLocation": f"s3://{self._bucket}/edge-out",
                "PresetDeploymentType": "GreengrassV2Component",
                "PresetDeploymentConfig": _json_dumps({"ComponentName": comp,
                                                        "ComponentVersion": component_version}),
            },
        )

        out: dict[str, Any] = {
            "compilation_job": compile_job,
            "edge_packaging_job": pkg_job,
            "component_name": comp,
            "component_version": component_version,
            "model_artifact": model_artifact,
            "target_device": target_device,
            "greengrass_target": greengrass_target,
            "deployment_id": "",
            "deployed": False,
            "backend": "sagemaker",
        }

        # ── 3. deploy onto the fleet via Greengrass V2 (only when a target + client are set) ──
        if greengrass_target and (self._gg is not None):
            gg = self._greengrass()
            comp_resp = gg.create_component_version(
                inlineRecipe=_json_dumps({
                    "RecipeFormatVersion": "2020-01-25",
                    "ComponentName": comp,
                    "ComponentVersion": component_version,
                    "Manifests": [{"Platform": {"os": "linux"},
                                   "Artifacts": [{"Uri": model_artifact}]}],
                }).encode("utf-8"),
            )
            comp_arn = comp_resp.get("arn", "")
            deploy_resp = gg.create_deployment(
                targetArn=greengrass_target,
                deploymentName=_sanitize("edge-deploy", comp),
                components={comp: {"componentVersion": component_version}},
            )
            out.update({
                "deployed": True,
                "deployment_id": deploy_resp.get("deploymentId", ""),
                "component_arn": comp_arn,
                "iot_job_arn": deploy_resp.get("iotJobArn", ""),
            })
        return out

    @staticmethod
    def _to_csv(payload: Any) -> str:
        """Encode a list / list-of-lists payload as a CSV body for invoke_endpoint."""
        rows = _as_rows(payload)
        return "\n".join(",".join(str(x) for x in row) for row in rows)


# ── self-registration ─────────────────────────────────────────────────────────
# SageMaker is the AWS default; LocalServe is the offline/local default. Offline
# auto → local (doc 05 §4.3); scale/budget selects managed for production hosting.
register_adapter(DeploymentBackend, "sagemaker", lambda: SageMakerDeploymentBackend(), is_default=True)
register_adapter(DeploymentBackend, "local", lambda: LocalServeBackend(), is_local=True)
