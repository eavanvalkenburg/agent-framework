# Copyright (c) Microsoft. All rights reserved.

"""LLM-defined dynamic tools (self-extending agents).

This module lets a model *define a brand-new tool at runtime* and expose it to
itself through the progressive tool exposure mechanism
(:meth:`FunctionInvocationContext.add_tools`).

Security model (see ``docs/decisions/0027-llm-defined-dynamic-tools.md``):

- The mechanism here is **substrate-agnostic**. It never executes
  model-authored code itself. Running a tool body is delegated to a
  :class:`ToolCodeCompiler`, which is expected to run the body in an isolated
  sandbox (for example ``agent_framework_monty.MontySandboxToolCompiler``).
- The feature is **default-deny**: nothing happens unless a
  :class:`DynamicToolPolicy` with ``enabled=True`` is supplied.
- Newly defined tools default to ``approval_mode="always_require"`` and the
  ``define_tool`` meta-tool itself defaults to requiring approval.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from ._feature_stage import ExperimentalFeature, experimental
from .exceptions import AgentException

if TYPE_CHECKING:
    from ._middleware import FunctionInvocationContext
    from ._tools import FunctionTool

logger = logging.getLogger("agent_framework")

__all__ = [
    "CompiledDynamicTool",
    "DynamicToolError",
    "DynamicToolPolicy",
    "DynamicToolSpec",
    "ToolCodeCompiler",
    "make_define_tool",
    "rehydrate_dynamic_tool",
]

# Marker attributes set on dynamically generated FunctionTools so the framework
# can recognize, count, and (on approval resume) rehydrate them.
DYNAMIC_TOOL_MARKER = "__dynamic_tool__"
DYNAMIC_TOOL_SPEC_ATTR = "__dynamic_tool_spec__"
DYNAMIC_TOOL_COMPILER_ATTR = "__dynamic_tool_compiler_id__"

_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")

# Parameter names that would collide with the sandbox wrapper's own locals.
_RESERVED_PARAM_NAMES = frozenset({"__args__", "__dynamic_tool__"})

# JSON Schema keywords we accept for dynamic-tool parameter schemas. Anything
# outside this subset is rejected so a model cannot smuggle in expensive or
# ambiguous constructs (``$ref``, ``patternProperties``, regex ``pattern`` that
# could enable ReDoS, ``oneOf``/``allOf``/``anyOf`` combinator explosions, ...).
_ALLOWED_SCHEMA_KEYS = frozenset({
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "description",
    "title",
    "default",
    "additionalProperties",
})
_ALLOWED_SCHEMA_TYPES = frozenset({
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
})


class DynamicToolError(AgentException):
    """Raised when a dynamic tool definition violates policy or fails to compile."""


@experimental(feature_id=ExperimentalFeature.DYNAMIC_TOOL_DEFINITION)
class DynamicToolSpec(BaseModel):
    """A model-authored description of a tool to be created at runtime.

    Only ``name``, ``description``, ``parameters`` and ``body`` come from the
    model. Provenance and depth are computed by the framework, never taken from
    model input.
    """

    name: str = Field(description="The tool name. Must match ``^[a-zA-Z_][a-zA-Z0-9_]{0,63}$``.")
    description: str = Field(default="", description="A short description of what the tool does.")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="A JSON Schema (restricted subset) describing the tool parameters.",
    )
    body: str = Field(description="The implementation body of the tool, executed by a sandboxed compiler.")

    @property
    def spec_hash(self) -> str:
        """A stable content hash of this spec (used for idempotency and audit)."""
        canonical = json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "body": self.body,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CompiledDynamicTool:
    """The result of compiling a :class:`DynamicToolSpec`.

    A compiler returns this descriptor (not a bare callable) so the framework
    has the metadata it needs for auditing and serialization.
    """

    func: Callable[..., Awaitable[Any]] | Callable[..., Any]
    compiler_id: str
    capabilities: tuple[str, ...] = ()
    resource_limits: dict[str, Any] | None = None


@runtime_checkable
class ToolCodeCompiler(Protocol):
    """Pluggable substrate that turns a :class:`DynamicToolSpec` into a callable.

    Implementations are expected to execute the tool body in an isolated
    sandbox. Core deliberately ships no in-process (``exec``-based) compiler.
    """

    compiler_id: str

    def compile(
        self,
        spec: DynamicToolSpec,
        *,
        granted_capabilities: Sequence[str] = (),
    ) -> CompiledDynamicTool:
        """Compile ``spec`` into a :class:`CompiledDynamicTool`."""
        ...


@dataclass
class DynamicToolPolicy:
    """Policy controlling whether and how a model may define tools at runtime.

    The feature is default-deny: ``enabled`` must be set to ``True`` explicitly.
    """

    enabled: bool = False
    max_dynamic_tools: int = 16
    # Reserved for the (future) composition substrate where a defined tool can
    # itself define tools. The sandbox substrate cannot define tools, so depth
    # is naturally bounded to one generation today.
    max_definition_depth: int = 1
    reserved_names: frozenset[str] = field(default_factory=lambda: frozenset({"define_tool"}))
    require_define_approval: bool = True
    require_invoke_approval: bool = True
    max_schema_bytes: int = 8_192
    max_schema_depth: int = 5
    allowed_capabilities: frozenset[str] = field(default_factory=lambda: frozenset[str]())


def _validate_name(name: str, policy: DynamicToolPolicy) -> None:
    if not _NAME_PATTERN.match(name):
        raise DynamicToolError(
            f"Invalid dynamic tool name {name!r}. Names must match '^[a-zA-Z_][a-zA-Z0-9_]{{0,63}}$'."
        )
    if name in policy.reserved_names:
        raise DynamicToolError(f"Dynamic tool name {name!r} is reserved and cannot be used.")


def _validate_schema(parameters: dict[str, Any], policy: DynamicToolPolicy) -> None:
    encoded = json.dumps(parameters, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > policy.max_schema_bytes:
        raise DynamicToolError(f"Parameter schema exceeds the maximum size of {policy.max_schema_bytes} bytes.")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise DynamicToolError("Parameter schema must be a JSON Schema object with 'type': 'object'.")

    def _walk(node: dict[str, Any], depth: int) -> None:
        if depth > policy.max_schema_depth:
            raise DynamicToolError(f"Parameter schema nesting exceeds the maximum depth of {policy.max_schema_depth}.")
        for key in node:
            if key not in _ALLOWED_SCHEMA_KEYS:
                raise DynamicToolError(f"Unsupported JSON Schema keyword {key!r} in dynamic tool parameters.")
        node_type: Any = node.get("type")
        if node_type is not None and node_type not in _ALLOWED_SCHEMA_TYPES:
            raise DynamicToolError(f"Unsupported JSON Schema type {node_type!r} in dynamic tool parameters.")
        properties: Any = node.get("properties")
        if properties is not None:
            if not isinstance(properties, dict):
                raise DynamicToolError("'properties' must be a JSON object.")
            for prop_name, prop_schema in cast("dict[str, Any]", properties).items():
                # Top-level property names become Python parameters in the
                # generated tool body, so they must be safe identifiers.
                if depth == 1:
                    _validate_property_name(str(prop_name))
                if not isinstance(prop_schema, dict):
                    raise DynamicToolError("Each schema node must be a JSON object.")
                _walk(cast("dict[str, Any]", prop_schema), depth + 1)
        items: Any = node.get("items")
        if isinstance(items, dict):
            _walk(cast("dict[str, Any]", items), depth + 1)
        additional: Any = node.get("additionalProperties")
        if isinstance(additional, dict):
            _walk(cast("dict[str, Any]", additional), depth + 1)
        elif additional is not None and not isinstance(additional, bool):
            raise DynamicToolError("'additionalProperties' must be a boolean or a JSON Schema object.")

    _walk(parameters, 1)


def _validate_property_name(name: str) -> None:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise DynamicToolError(
            f"Invalid dynamic tool parameter name {name!r}. Parameter names must be valid Python identifiers."
        )
    if name in _RESERVED_PARAM_NAMES:
        raise DynamicToolError(f"Dynamic tool parameter name {name!r} is reserved.")


def _is_dynamic_tool(tool: Any) -> bool:
    return bool(getattr(tool, DYNAMIC_TOOL_MARKER, False))


def _build_dynamic_function_tool(
    spec: DynamicToolSpec,
    compiled: CompiledDynamicTool,
    *,
    approval_mode: str,
) -> FunctionTool:
    """Wrap a compiled body into a :class:`FunctionTool`.

    The wrapper receives only validated arguments. It is never given the
    :class:`FunctionInvocationContext`, so a model-authored body cannot reach the
    live tool list, session, or runtime kwargs.
    """
    import inspect

    from ._tools import FunctionTool

    compiled_func = compiled.func

    async def _runner(**kwargs: Any) -> Any:
        result = compiled_func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    tool = FunctionTool(
        name=spec.name,
        description=spec.description,
        input_model=spec.parameters,
        func=_runner,
        approval_mode=approval_mode,  # type: ignore[arg-type]
    )
    setattr(tool, DYNAMIC_TOOL_MARKER, True)
    setattr(tool, DYNAMIC_TOOL_SPEC_ATTR, spec)
    setattr(tool, DYNAMIC_TOOL_COMPILER_ATTR, compiled.compiler_id)
    return tool


@experimental(feature_id=ExperimentalFeature.DYNAMIC_TOOL_DEFINITION)
def rehydrate_dynamic_tool(
    spec: DynamicToolSpec,
    compiler: ToolCodeCompiler,
    *,
    policy: DynamicToolPolicy | None = None,
    granted_capabilities: Sequence[str] = (),
) -> FunctionTool:
    """Recreate a dynamic tool from its persisted spec.

    Used to restore a dynamic tool after an approval pause/resume or a
    checkpoint round-trip, where the original closure no longer exists.
    """
    policy = policy or DynamicToolPolicy(enabled=True)
    approval_mode = "always_require" if policy.require_invoke_approval else "never_require"
    compiled = compiler.compile(spec, granted_capabilities=granted_capabilities)
    return _build_dynamic_function_tool(spec, compiled, approval_mode=approval_mode)


@experimental(feature_id=ExperimentalFeature.DYNAMIC_TOOL_DEFINITION)
def make_define_tool(
    *,
    compiler: ToolCodeCompiler,
    policy: DynamicToolPolicy | None = None,
    granted_capabilities: Sequence[str] = (),
) -> FunctionTool:
    """Build the ``define_tool`` meta-tool that lets a model define new tools.

    Add the returned tool to an agent. When the model calls it, the framework
    validates the request against ``policy``, compiles the body with
    ``compiler``, and exposes the new tool via
    :meth:`FunctionInvocationContext.add_tools` so it becomes callable on the
    next iteration of the function-calling loop.

    Keyword Args:
        compiler: The sandboxed substrate that compiles tool bodies. Core ships
            no in-process compiler; use a CodeAct provider's compiler (for
            example ``MontySandboxToolCompiler``).
        policy: The :class:`DynamicToolPolicy`. Defaults to a default-deny
            policy, so you must pass one with ``enabled=True`` to use the
            feature.
        granted_capabilities: Capabilities forwarded to the compiler for every
            tool it compiles (substrate-specific).

    Returns:
        A ``define_tool`` :class:`FunctionTool`.
    """
    effective_policy = policy or DynamicToolPolicy()
    invalid = set(granted_capabilities) - effective_policy.allowed_capabilities
    if invalid:
        raise DynamicToolError(f"Capabilities not allowed by policy: {sorted(invalid)}.")

    define_approval = "always_require" if effective_policy.require_define_approval else "never_require"
    invoke_approval = "always_require" if effective_policy.require_invoke_approval else "never_require"

    def define_tool(
        name: str,
        description: str,
        parameters: dict[str, Any],
        body: str,
        ctx: FunctionInvocationContext,
    ) -> str:
        """Define a new tool and make it available for subsequent steps.

        Args:
            name: The new tool's name.
            description: What the new tool does.
            parameters: A JSON Schema object describing the tool's parameters.
            body: The Python body of the tool. It runs in a sandbox; parameters
                are available as local variables and the body should return a
                JSON-serializable value.
            ctx: The function-invocation context (injected by the framework; not
                supplied by the model).
        """
        if not effective_policy.enabled:
            raise DynamicToolError(
                "Dynamic tool definition is disabled. Enable it with a DynamicToolPolicy(enabled=True)."
            )
        if ctx.tools is None:
            raise DynamicToolError("define_tool is only available within an agent's function-calling loop.")

        spec = DynamicToolSpec(name=name, description=description, parameters=parameters, body=body)
        _validate_name(spec.name, effective_policy)
        _validate_schema(spec.parameters, effective_policy)

        existing_dynamic: dict[str, Any] = {}
        for existing in ctx.tools:
            existing_name = getattr(existing, "name", None)
            if existing_name is None:
                continue
            if existing_name == spec.name and not _is_dynamic_tool(existing):
                raise DynamicToolError(
                    f"A non-dynamic tool named {spec.name!r} already exists and cannot be overridden."
                )
            if _is_dynamic_tool(existing):
                existing_dynamic[existing_name] = existing

        if spec.name in existing_dynamic:
            prior_spec = getattr(existing_dynamic[spec.name], DYNAMIC_TOOL_SPEC_ATTR, None)
            if isinstance(prior_spec, DynamicToolSpec) and prior_spec.spec_hash == spec.spec_hash:
                logger.info("define_tool: %r already defined (idempotent, hash=%s)", spec.name, spec.spec_hash[:12])
                return f"Tool '{spec.name}' is already defined and available."
            raise DynamicToolError(f"A different dynamic tool named {spec.name!r} already exists in this run.")

        if len(existing_dynamic) >= effective_policy.max_dynamic_tools:
            raise DynamicToolError(
                f"Maximum number of dynamic tools ({effective_policy.max_dynamic_tools}) reached for this run."
            )

        try:
            compiled = compiler.compile(spec, granted_capabilities=granted_capabilities)
        except DynamicToolError:
            raise
        except Exception as exc:
            raise DynamicToolError(f"Failed to compile dynamic tool {spec.name!r}: {exc}") from exc

        new_tool = _build_dynamic_function_tool(spec, compiled, approval_mode=invoke_approval)
        ctx.add_tools([new_tool])
        # Audit log stores hashes/sizes only, never the raw body.
        logger.info(
            "define_tool: defined %r (compiler=%s, hash=%s, body_bytes=%d, schema_bytes=%d)",
            spec.name,
            compiled.compiler_id,
            spec.spec_hash[:12],
            len(spec.body.encode("utf-8")),
            len(json.dumps(spec.parameters, separators=(",", ":")).encode("utf-8")),
        )
        return f"Tool '{spec.name}' defined and available on the next step."

    from ._tools import FunctionTool

    return FunctionTool(
        name="define_tool",
        description=(
            "Define a brand-new tool at runtime that you can call on subsequent steps. "
            "Provide a name, description, a JSON Schema for the parameters, and a Python "
            "body that returns a JSON-serializable value."
        ),
        func=define_tool,
        approval_mode=define_approval,  # type: ignore[arg-type]
    )
