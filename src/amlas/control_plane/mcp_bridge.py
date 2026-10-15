"""MCP bridge (consume side) — connect to external MCP servers + surface their tools (ADR-016).

The Claude-Code MCP analog (consume direction): frame the ports/registry as the MCP-style
pluggable seam, and add a thin **MCP client bridge** that connects to an external MCP server and
surfaces its tools to the agent (the Strands/AgentCore MCP-client analog). M6 is **consume-only**;
exposing OUR surface AS an MCP server is M8 (doc 05 §4 / roadmap M8).

Design (offline-safe, no network in tests):
* ``MCPServerSpec`` — a config-driven server descriptor (name + transport + command/url + env).
  The list of servers is read from project memory (``.agenticml.yaml`` ``mcp_servers``) or an env
  list, so adding a server is configuration, not code.
* ``MCPClient`` — a thin client Protocol: ``list_tools()`` + ``call_tool(name, args)``. A real
  Strands ``MCPClient`` (stdio/HTTP) plugs in; the bridge accepts an injected ``client_factory``
  so callers may pass a **fake in-memory MCP server** (no subprocess / no network).
* ``MCPBridge`` — connects to the configured servers, aggregates their tools into a namespaced
  registry (``<server>.<tool>``), and exposes ``list_tools`` / ``call_tool`` to the agent.

Pure-Python + a lazy optional ``mcp``/Strands import — importing the module is network-free.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    # M8 expose side (ADR-016) — expose OUR flows/tools AS an MCP server.
    "ExposedTool",
    "MCPBridge",
    "MCPClient",
    "MCPServer",
    "MCPServerSpec",
    "MCPToolRef",
    "load_mcp_server_specs",
]


# ──────────────────────────────────────────────────────────────────────────────
# Config — the MCP server descriptors (project-memory / env driven)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MCPServerSpec:
    """A config-driven external MCP server descriptor (consume side).

    ``transport`` is ``stdio`` (a ``command`` to launch) or ``http`` (a ``url``); ``env`` carries
    any per-server environment. This is the ``.agenticml.yaml`` ``mcp_servers`` entry shape, so a
    new server is a config edit (ADR-016).
    """

    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


def load_mcp_server_specs(memory: dict[str, Any] | None = None) -> list[MCPServerSpec]:
    """Load MCP server specs from project memory (``mcp_servers``) — config-driven (ADR-016).

    Reads the ``mcp_servers`` block of the merged ``.agenticml.yaml`` project memory (a list of
    ``{name, transport, command/url, args, env, enabled}`` dicts). Returns the enabled specs.
    Missing/empty config → ``[]`` (no servers; never raises).
    """
    block: list[Any] = []
    if isinstance(memory, dict):
        block = memory.get("mcp_servers") or []
    specs: list[MCPServerSpec] = []
    if isinstance(block, list):
        for entry in block:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            specs.append(MCPServerSpec(
                name=str(entry["name"]),
                transport=str(entry.get("transport", "stdio")),
                command=str(entry.get("command", "")),
                args=[str(a) for a in entry.get("args", [])],
                url=str(entry.get("url", "")),
                env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
                enabled=bool(entry.get("enabled", True)),
            ))
    return [s for s in specs if s.enabled]


# ──────────────────────────────────────────────────────────────────────────────
# Client Protocol — what the bridge needs from an MCP client (fake-able)
# ──────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class MCPClient(Protocol):
    """The minimal MCP client surface the bridge consumes (structural; fake-able in tests).

    A real Strands ``MCPClient`` (stdio/HTTP transport) satisfies this; a test fake satisfies it in
    memory with no subprocess/network. ``list_tools`` returns ``[{name, description, input_schema}]``;
    ``call_tool`` invokes a remote tool and returns its result dict.
    """

    def list_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MCPToolRef:
    """A bridged external MCP tool, namespaced by its server (``<server>.<tool>``)."""

    server: str
    tool: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        return f"{self.server}.{self.tool}"


# ──────────────────────────────────────────────────────────────────────────────
# The bridge — connect, aggregate, surface
# ──────────────────────────────────────────────────────────────────────────────
def _default_client_factory(spec: MCPServerSpec) -> MCPClient:
    """Build a real Strands MCP client for ``spec`` (lazy import; network only when called).

    Raised import/connection errors are surfaced to the bridge, which records the server as
    unavailable (fail-soft) rather than crashing the whole surface — exactly the adapter pattern.
    """
    try:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        from strands.tools.mcp import MCPClient as StrandsMCPClient
    except Exception as e:
        raise RuntimeError(f"MCP client deps unavailable for {spec.name!r}: {e}") from e

    if spec.transport == "stdio":
        params = StdioServerParameters(command=spec.command, args=list(spec.args),
                                       env={**os.environ, **spec.env})
        return StrandsMCPClient(lambda: stdio_client(params))  # type: ignore[return-value]
    raise RuntimeError(f"unsupported MCP transport {spec.transport!r} for {spec.name!r} (M6: stdio)")


class MCPBridge:
    """Consume-side MCP bridge — connect to external servers + surface their tools (ADR-016).

    Config-driven: the server list comes from project memory. Connection is fail-soft (a server
    that won't connect is recorded as unavailable, not fatal). ``client_factory`` is injectable so
    callers may pass a fake in-memory MCP server (no subprocess/network).
    """

    def __init__(self, specs: list[MCPServerSpec] | None = None,
                 client_factory: Callable[[MCPServerSpec], MCPClient] | None = None) -> None:
        self._specs = list(specs or [])
        self._factory = client_factory or _default_client_factory
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPToolRef] = {}
        self._errors: dict[str, str] = {}

    @classmethod
    def from_memory(cls, memory: dict[str, Any] | None,
                    client_factory: Callable[[MCPServerSpec], MCPClient] | None = None) -> MCPBridge:
        """Build a bridge from the project-memory ``mcp_servers`` config (the config-driven entry)."""
        return cls(specs=load_mcp_server_specs(memory), client_factory=client_factory)

    def connect(self) -> dict[str, Any]:
        """Connect to all configured servers + aggregate their tools (fail-soft per server).

        Returns ``{connected, failed, tool_count, tools}`` — ``connected`` lists the servers whose
        tools were surfaced; ``failed`` maps a server name → error; ``tools`` is the namespaced
        ``<server>.<tool>`` list. A server that fails to connect/list is recorded in ``failed`` and
        skipped (the bridge never crashes the surface).
        """
        connected: list[str] = []
        for spec in self._specs:
            try:
                client = self._factory(spec)
                tools = client.list_tools()
                self._clients[spec.name] = client
                for t in tools:
                    ref = MCPToolRef(server=spec.name, tool=str(t.get("name", "")),
                                     description=str(t.get("description", "")),
                                     input_schema=t.get("input_schema", t.get("inputSchema", {})) or {})
                    if ref.tool:
                        self._tools[ref.qualified_name] = ref
                connected.append(spec.name)
            except Exception as e:
                self._errors[spec.name] = str(e)
        return {"connected": connected, "failed": dict(self._errors),
                "tool_count": len(self._tools), "tools": sorted(self._tools)}

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the aggregated external MCP tools (namespaced ``<server>.<tool>``)."""
        return [
            {"qualified_name": ref.qualified_name, "server": ref.server, "tool": ref.tool,
             "description": ref.description, "input_schema": ref.input_schema}
            for ref in self._tools.values()
        ]

    def call_tool(self, qualified_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a bridged external MCP tool by its ``<server>.<tool>`` qualified name.

        Routes to the owning server's client. Returns ``{ok, server, tool, result}`` on success, or
        ``{ok: False, error}`` for an unknown tool / a server-call failure (fail-soft).
        """
        ref = self._tools.get(qualified_name)
        if ref is None:
            return {"ok": False, "error": f"unknown MCP tool {qualified_name!r}",
                    "available": sorted(self._tools)}
        client = self._clients.get(ref.server)
        if client is None:
            return {"ok": False, "error": f"server {ref.server!r} not connected"}
        try:
            result = client.call_tool(ref.tool, dict(arguments or {}))
            return {"ok": True, "server": ref.server, "tool": ref.tool, "result": result}
        except Exception as e:
            return {"ok": False, "server": ref.server, "tool": ref.tool, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Expose side (M8, ADR-016) — expose OUR flows/tools AS an MCP server
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExposedTool:
    """One amlas capability exposed as an MCP tool (the expose-side analog of MCPToolRef).

    ``name`` is the MCP tool name (our flow id, e.g. ``"data_quality_flow"``); ``description`` +
    ``input_schema`` come from the flow registry so an external consumer sees a self-describing
    tool. ``gated`` flags whether invoking it routes through the approval gate (spend/irreversible).
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    gated: bool = False
    milestone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema, "gated": self.gated, "milestone": self.milestone}


class MCPServer:
    """Expose OUR surface's flows/tools AS an MCP server (M8 / ADR-016 — the expose direction).

    The bidirectional counterpart of :class:`MCPBridge`: where the bridge CONSUMES external MCP
    tools, this SERVER lists our registered flows as MCP tools (:meth:`list_tools`) and invokes them
    through the SAME M7 capability funnel (:meth:`call_tool` → ``invoke_capability``). An external
    Claude-Code / agent therefore consumes the amlas capabilities with the IDENTICAL gated +
    approved behavior every internal surface gets (ADR-010 parity) — a spend/irreversible flow still
    routes through ``plan_approve_execute`` (and a non-interactive caller is safely DENIED).

    Config-driven + thin: which flows are exposed is filtered by ``expose`` (an allowlist from
    project memory ``mcp_expose``) — default exposes every runnable capability. No network/transport
    here: this is the in-process MCP-server *surface* a real stdio/HTTP MCP transport wraps; tests
    drive it with a fake in-memory client (``list_tools``/``call_tool``), no subprocess.
    """

    def __init__(self, *, name: str = "amlas", expose: list[str] | None = None,
                 invoker: Callable[..., Any] | None = None) -> None:
        self._name = name
        self._expose = list(expose) if expose else None  # None → expose all runnable
        self._invoker = invoker  # injectable for isolated validation; default → capability.invoke_capability

    @classmethod
    def from_memory(cls, memory: dict[str, Any] | None, *, name: str = "amlas",
                    invoker: Callable[..., Any] | None = None) -> MCPServer:
        """Build an exposing server from project-memory ``mcp_expose`` (config-driven allowlist)."""
        expose = None
        if isinstance(memory, dict):
            block = memory.get("mcp_expose")
            if isinstance(block, list) and block:
                expose = [str(x) for x in block]
        return cls(name=name, expose=expose, invoker=invoker)

    def _capabilities(self) -> list[dict[str, Any]]:
        """Pull the capability descriptors from the M7 funnel (lazy import; no cycle)."""
        from .capability import list_capabilities

        return list_capabilities()

    def list_tools(self) -> list[dict[str, Any]]:
        """List our flows as MCP tools (the server-side ``list_tools``), filtered by ``expose``.

        Returns ``[{name, description, input_schema, gated, milestone}]`` for each exposed runnable
        capability — the self-describing tool set an external MCP client discovers. The
        ``input_schema`` is a minimal ``{params: object}`` shape (our flows take kwargs).
        """
        tools: list[ExposedTool] = []
        for cap in self._capabilities():
            cid = cap["capability_id"]
            if not cap.get("runnable"):
                continue
            if self._expose is not None and cid not in self._expose:
                continue
            tools.append(ExposedTool(
                name=cid,
                description=cap.get("description", ""),
                input_schema={"type": "object", "properties": {"params": {"type": "object"}}},
                gated=bool(cap.get("gated")),
                milestone=str(cap.get("milestone", "")),
            ))
        return [t.to_dict() for t in sorted(tools, key=lambda t: t.name)]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None,
                  policy: Any = None) -> dict[str, Any]:
        """Invoke an exposed flow through the M7 capability funnel (the server-side ``call_tool``).

        Routes the external call to ``invoke_capability`` so the SAME gating/approval applies
        (ADR-010): a spend/irreversible flow goes through ``plan_approve_execute`` and a
        non-interactive caller is DENIED (safe-by-default). Returns ``{ok, tool, status, approved,
        gated, result}`` — the MCP-tool result envelope. An unknown / non-exposed tool → ``ok:False``.
        """
        args = dict(arguments or {})
        exposed = {t["name"] for t in self.list_tools()}
        if name not in exposed:
            return {"ok": False, "tool": name, "error": f"tool {name!r} not exposed by {self._name}",
                    "available": sorted(exposed)}
        params = args.get("params", args)
        invoker = self._invoker
        if invoker is None:
            from .capability import invoke_capability

            invoker = invoke_capability
        try:
            res = invoker(capability_id=name, params=params, policy=policy, surface="mcp")
        except Exception as e:
            return {"ok": False, "tool": name, "error": str(e)}
        # CapabilityResult → MCP envelope (support both the dataclass + a dict invoker).
        status = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else "completed")
        approved = getattr(res, "approved", None)
        gated = getattr(res, "gated", None)
        result = getattr(res, "result", None) if not isinstance(res, dict) else res.get("result")
        if isinstance(res, dict):
            approved = res.get("approved", approved)
            gated = res.get("gated", gated)
        return {"ok": status != "error", "tool": name, "status": status,
                "approved": approved, "gated": gated, "result": result, "server": self._name}
