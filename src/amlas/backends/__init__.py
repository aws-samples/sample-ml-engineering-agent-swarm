"""The pluggable backend-abstraction seam (doc 05 §4; roadmap M0).

This package introduces the **ports-and-adapters** layer that fixes RECON §10.1
("no backend-abstraction / portability layer"). Public surface:

  * ``ports``     — the 19 domain-typed ``Protocol`` ports (the seam).
  * ``types``     — the domain value types crossing the seam (``JobHandle`` …).
  * ``registry``  — ``BackendRegistry.resolve(port, …)`` + the ``REGISTRY`` singleton.
  * ``hooks``     — the GuardrailHooks ``PreToolUse``/``PostToolUse`` layer (bronze-write deny).
  * ``adapters``  — the concrete adapters (self-registering; imported lazily by the registry).

A tool becomes backend-neutral by calling ``REGISTRY.resolve(SomePort, …).adapter``
instead of boto3 directly (doc 05 §4.4). Today's behavior is preserved as the
default/local adapter (ADR-001); the offline/test default resolves to the local
adapter so the the offline AWS API emulator suite + offline offline validation never touch AWS.
"""

from __future__ import annotations

from .hooks import GuardrailHooks, HookContext, HookEvent, default_guardrail_hooks
from .ports import ALL_PORTS, port_key
from .registry import REGISTRY, AdapterUnavailableError, BackendRegistry, ResolvedBackend, register_adapter

__all__ = [
    "ALL_PORTS",
    "REGISTRY",
    "AdapterUnavailableError",
    "BackendRegistry",
    "GuardrailHooks",
    "HookContext",
    "HookEvent",
    "ResolvedBackend",
    "default_guardrail_hooks",
    "port_key",
    "register_adapter",
]
