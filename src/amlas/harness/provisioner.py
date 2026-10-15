"""Programmatic harness lifecycle — create / update / replace / delete.

[`HarnessStack`](../../../infra/cdk_stacks/harness_stack.py:1) owns DECLARATIVE provisioning.
This module owns the operations CloudFormation cannot express, each of which is a way to wedge
an environment if handled naively:

* **Memory is create-time only.** ``UpdateHarness`` rejects the ``CreateHarness`` memory shape,
  so changing the strategy set means REPLACING the harness. The CFN schema is actively
  misleading here — it marks only ``HarnessName`` and
  ``Memory/ManagedMemoryConfiguration/EncryptionKeyArn`` as create-only, so reading the schema
  alone suggests an in-place memory update is fine. It is not (T-HARN-12.1).
* **Deletion is asynchronous, takes >5 minutes, and cascades to the managed Memory.** Treating
  it as synchronous is what wedges things: the next create collides with a name that is still
  draining (T-HARN-12.2).
* **A ``DELETING`` harness is neither updatable nor deletable.** It is a dead end — report the
  state and touch nothing (T-HARN-12.3).

Provisioning failures funnel into ``ERROR_LOG`` with ``source="harness.<name>"``, peer to the
agent factory's ``agents.<name>``, and the original exception re-raises unchanged
(T-HARN-03.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spec import HarnessSpec

#: A harness in this state cannot be updated or deleted — only waited out.
DELETING = "DELETING"

#: Deletion is asynchronous and slow. Reported to callers so a 5-minute wait is expected
#: rather than mistaken for a hang.
DELETE_MINIMUM_SECONDS = 300


class HarnessProvisionerError(RuntimeError):
    """A lifecycle operation that must not be attempted, refused BEFORE calling AWS."""


class HarnessDeletingError(HarnessProvisionerError):
    """The harness is mid-deletion; nothing can be done with it until it drains."""


@dataclass(frozen=True)
class HarnessState:
    """The subset of a harness record the lifecycle decisions actually depend on."""

    name: str
    harness_id: str
    arn: str
    status: str
    version: str
    memory_strategies: tuple[str, ...] = ()

    @property
    def is_deleting(self) -> bool:
        return self.status == DELETING


@dataclass(frozen=True)
class DeleteOutcome:
    """What a delete request actually did, and what the caller should expect afterwards."""

    harness_id: str
    requested: bool
    already_deleting: bool = False
    asynchronous: bool = True
    expected_minimum_seconds: int = DELETE_MINIMUM_SECONDS
    cascades_to_memory: bool = True


def _capture(error_log: Any, exc: BaseException, *, name: str) -> None:
    """Funnel a provisioning failure, guarded so telemetry can NEVER mask the real error.

    Mirrors [`_build.py`](../_build.py:171): an exception raised from the capture path would
    replace the exception the caller needs to see.
    """
    try:
        if error_log is not None:
            error_log.capture(exc, source=f"harness.{name}")
        else:
            from ..telemetry.error_log import capture_error

            capture_error(exc, source=f"harness.{name}")
    except Exception:  # nosec B110
        pass


def _state(record: dict[str, Any]) -> HarnessState:
    memory = (record.get("memory") or {}).get("managedMemoryConfiguration") or {}
    return HarnessState(
        name=str(record.get("harnessName") or ""),
        harness_id=str(record.get("harnessId") or ""),
        arn=str(record.get("arn") or ""),
        status=str(record.get("status") or ""),
        version=str(record.get("harnessVersion") or ""),
            memory_strategies=tuple(memory.get("strategies") or ()),
    )


def find_harness(client: Any, name: str) -> HarnessState | None:
    """Return the current state of harness ``name``, or ``None`` when it does not exist."""
    for record in client.list_harnesses().get("harnesses", []):
        if record.get("harnessName") == name:
            return _state(record)
    return None


def memory_differs(spec: HarnessSpec, existing: HarnessState | None) -> bool:
    """True when the live memory strategy set differs from what the spec declares.

    Order-insensitive: the service may return the strategies in any order, and treating a
    reordering as a change would trigger a pointless (and slow) replacement.
    """
    if existing is None:
        return False
    return set(existing.memory_strategies) != set(spec.memory_strategies)


def _guard_allowed_tools(spec: HarnessSpec) -> None:
    """Refuse ``["*"]`` at the boundary too, not only inside the spec (ADR-0002).

    Defence in depth: the spec validates its own declared tools, but a caller could hand this
    module a spec built some other way, and ``*`` makes the in-VM shell builtin LLM-reachable.
    """
    if "*" in spec.allowed_tools:
        raise HarnessProvisionerError(
            f"{spec.name}: refusing allowedTools=['*'] — it makes the in-VM shell builtin "
            "reachable by the LLM (ADR-0002); use the @*/<tool> form"
        )


def create_harness(
    client: Any,
    spec: HarnessSpec,
    *,
    execution_role_arn: str,
    model_id: str | None = None,
    tags: dict[str, str] | None = None,
    error_log: Any | None = None,
) -> HarnessState:
    """Create the harness from ``spec``, capturing and re-raising any failure."""
    _guard_allowed_tools(spec)
    payload = spec.to_create_payload(
        execution_role_arn=execution_role_arn, model_id=model_id
    )
    if tags:
        payload["tags"] = dict(tags)
    try:
        response = client.create_harness(**payload)
    except Exception as exc:
        _capture(error_log, exc, name=spec.name)
        raise
    record = response.get("harness", response)
    return _state(record)


def update_harness(
    client: Any,
    spec: HarnessSpec,
    *,
    harness_id: str,
    model_id: str | None = None,
    error_log: Any | None = None,
) -> HarnessState:
    """Update everything that IS updatable — deliberately never the memory block."""
    _guard_allowed_tools(spec)
    payload = spec.to_update_payload(model_id=model_id)
    if "memory" in payload:
        # A real check, not an assert: asserts vanish under `python -O`, and this one guards a
        # call AWS would reject. If the update payload ever grows a memory key, failing here is
        # far clearer than a ValidationException from the service.
        raise HarnessProvisionerError(
            f"{spec.name}: the update payload carries a memory key, but UpdateHarness rejects "
            "the CreateHarness memory shape — memory is create-time only"
        )
    try:
        response = client.update_harness(harnessId=harness_id, **payload)
    except Exception as exc:
        _capture(error_log, exc, name=spec.name)
        raise
    record = response.get("harness", response)
    return _state(record)


def delete_harness(
    client: Any,
    *,
    harness_id: str,
    error_log: Any | None = None,
    name: str = "",
) -> DeleteOutcome:
    """Start deletion, cascading the managed Memory, and report that it is asynchronous.

    A harness already ``DELETING`` is left alone: a second delete would be rejected, and
    issuing it would only obscure the fact that the first one is still draining.
    """
    try:
        current = client.get_harness(harnessId=harness_id).get("harness") or {}
    except Exception:
        current = {}
    if str(current.get("status") or "") == DELETING:
        return DeleteOutcome(harness_id=harness_id, requested=False, already_deleting=True)

    try:
        client.delete_harness(harnessId=harness_id, deleteManagedMemory=True)
    except Exception as exc:
        _capture(error_log, exc, name=name or str(current.get("harnessName") or harness_id))
        raise
    return DeleteOutcome(harness_id=harness_id, requested=True)


def provision_harness(
    client: Any,
    spec: HarnessSpec,
    *,
    execution_role_arn: str,
    model_id: str | None = None,
    tags: dict[str, str] | None = None,
    allow_replace: bool = False,
    error_log: Any | None = None,
) -> HarnessState:
    """Bring harness ``spec.name`` to the declared configuration.

    Decision order, and why each branch exists:

    1. **``DELETING``** → :class:`HarnessDeletingError`. Neither updatable nor deletable, and
       creating over the name collides while it drains.
    2. **absent** → create.
    3. **memory differs** → REPLACEMENT is required, because memory is create-time only. This
       raises unless ``allow_replace=True``: replacement destroys the accumulated long-term
       memory and takes >5 minutes, so it must be a deliberate choice rather than a silent
       consequence of editing a strategy list.
    4. **otherwise** → update in place, with no memory key.

    Args:
        allow_replace: Permit the destructive delete-then-recreate path in case 3.

    Returns:
        The resulting state. For a started replacement the status is
        ``DELETE_STARTED_REPLACE_PENDING``: the old harness is draining and the caller must
        re-invoke once it is gone, since the new one cannot take the name until then.
    """
    _guard_allowed_tools(spec)
    existing = find_harness(client, spec.name)

    if existing is not None and existing.is_deleting:  # nosemgrep: is-function-without-parentheses
        raise HarnessDeletingError(
            f"{spec.name}: harness is {DELETING} — it can be neither updated nor deleted, and "
            f"the name is unavailable until it drains (asynchronous, "
            f">{DELETE_MINIMUM_SECONDS}s, cascades to the managed Memory)"
        )

    if existing is None:
        return create_harness(
            client,
            spec,
            execution_role_arn=execution_role_arn,
            model_id=model_id,
            tags=tags,
            error_log=error_log,
        )

    if memory_differs(spec, existing):
        if not allow_replace:
            raise HarnessProvisionerError(
                f"{spec.name}: memory strategies changed "
                f"{sorted(existing.memory_strategies)} → {sorted(spec.memory_strategies)}, but "
                "memory is create-time only (UpdateHarness rejects the CreateHarness memory "
                "shape). Replacing the harness DESTROYS its accumulated long-term memory and "
                "takes >5 minutes — pass allow_replace=True to accept that."
            )
        delete_harness(
            client, harness_id=existing.harness_id, error_log=error_log, name=spec.name
        )
        # Deletion is asynchronous, so the replacement cannot be created in this call: the name
        # stays taken until the old harness finishes draining.
        return HarnessState(
            name=spec.name,
            harness_id=existing.harness_id,
            arn=existing.arn,
            status="DELETE_STARTED_REPLACE_PENDING",
            version=existing.version,
            memory_strategies=existing.memory_strategies,
        )

    return update_harness(
        client,
        spec,
        harness_id=existing.harness_id,
        model_id=model_id,
        error_log=error_log,
    )


__all__ = [
    "DELETE_MINIMUM_SECONDS",
    "DELETING",
    "DeleteOutcome",
    "HarnessDeletingError",
    "HarnessProvisionerError",
    "HarnessState",
    "create_harness",
    "delete_harness",
    "find_harness",
    "memory_differs",
    "provision_harness",
    "update_harness",
]
