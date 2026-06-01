# Copyright (c) Microsoft. All rights reserved.

"""A sandboxed :class:`~agent_framework.ToolCodeCompiler` backed by Monty.

This is the recommended secure substrate for the core *LLM-defined dynamic
tools* feature (``agent_framework.make_define_tool``). The body authored by the
model never runs in-process: it is executed inside the Monty interpreter, which
blocks filesystem and network access and enforces resource limits.

See ``docs/decisions/0027-llm-defined-dynamic-tools.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from functools import partial
from json import dumps
from typing import TYPE_CHECKING, Any, cast

from agent_framework import CompiledDynamicTool, DynamicToolError, DynamicToolSpec

from ._monty_bridge import InlineCodeBridge

if TYPE_CHECKING:
    from agent_framework import FunctionTool

__all__ = ["MontySandboxToolCompiler"]

_COMPILER_ID = "monty-sandbox"

# Cap the serialized size of arguments embedded into the generated script,
# bounding host-side work before Monty's own resource limits take effect.
_MAX_ARGS_BYTES = 64 * 1024


def _make_host_tool_callback(tool_obj: FunctionTool) -> Callable[..., Any]:
    """Return an async callable that invokes ``tool_obj`` and returns its raw value."""
    return partial(copy(tool_obj).invoke, skip_parsing=True)


def _indent(body: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in body.splitlines()) or f"{pad}pass"


def _build_script(spec: DynamicToolSpec, args: dict[str, Any]) -> str:
    """Build a Monty script that runs the tool body with ``args`` bound.

    The body becomes the suite of an async function whose parameters are the
    schema's property names. Arguments are embedded as Python literals via
    ``repr`` (safe for validated JSON primitives), and the function's awaited
    result becomes the script output (the value of the last top-level
    expression).
    """
    properties: Any = spec.parameters.get("properties", {})
    param_names: list[str] = (
        [str(key) for key in cast("dict[str, Any]", properties)] if isinstance(properties, dict) else []
    )
    signature = ", ".join(param_names)
    return (
        f"async def __dynamic_tool__({signature}):\n"
        f"{_indent(spec.body)}\n"
        f"__args__ = {args!r}\n"
        "await __dynamic_tool__(**__args__)"
    )


class MontySandboxToolCompiler:
    """Compile :class:`~agent_framework.DynamicToolSpec` bodies into sandboxed callables.

    Each compiled tool runs its body inside a fresh :class:`InlineCodeBridge`.
    By default the body has access to no host tools; pass ``allowed_host_tools``
    to expose specific, already-approved tools to generated bodies.

    Keyword Args:
        resource_limits: Forwarded to Monty's ``ResourceLimits`` (CPU, memory,
            output size, recursion depth, GC frequency).
        allowed_host_tools: Host tools that generated bodies may call as
            ``await tool_name(...)``. Empty by default (no ambient authority).
    """

    compiler_id: str = _COMPILER_ID

    def __init__(
        self,
        *,
        resource_limits: dict[str, Any] | None = None,
        allowed_host_tools: Sequence[FunctionTool] = (),
    ) -> None:
        self._resource_limits = dict(resource_limits) if resource_limits else None
        for tool_obj in allowed_host_tools:
            # Sandbox bodies call host tools directly (FunctionTool.invoke), which
            # bypasses the agent loop's approval gate. Refuse to expose any tool
            # that is supposed to require approval.
            if getattr(tool_obj, "approval_mode", "never_require") != "never_require":
                raise DynamicToolError(
                    f"Host tool {tool_obj.name!r} requires approval and cannot be exposed to dynamic tool bodies, "
                    "because sandbox calls bypass the approval gate."
                )
        self._tool_map: dict[str, Callable[..., Any]] = {
            tool_obj.name: _make_host_tool_callback(tool_obj) for tool_obj in allowed_host_tools
        }

    def compile(
        self,
        spec: DynamicToolSpec,
        *,
        granted_capabilities: Sequence[str] = (),
    ) -> CompiledDynamicTool:
        """Compile ``spec`` into a :class:`~agent_framework.CompiledDynamicTool`."""
        resource_limits = self._resource_limits
        tool_map = dict(self._tool_map)

        async def _run(**kwargs: Any) -> Any:
            if len(dumps(kwargs, default=str).encode("utf-8")) > _MAX_ARGS_BYTES:
                raise DynamicToolError(
                    f"Arguments for dynamic tool {spec.name!r} exceed the maximum size of {_MAX_ARGS_BYTES} bytes."
                )
            script = _build_script(spec, kwargs)
            bridge = InlineCodeBridge(tool_map, resource_limits=resource_limits)
            try:
                result = await bridge.run(script)
            except Exception as exc:
                raise DynamicToolError(
                    f"Sandboxed execution of dynamic tool {spec.name!r} failed: {type(exc).__name__}: {exc}"
                ) from exc
            return result.get("output")

        return CompiledDynamicTool(
            func=_run,
            compiler_id=self.compiler_id,
            capabilities=tuple(granted_capabilities),
            resource_limits=resource_limits,
        )
