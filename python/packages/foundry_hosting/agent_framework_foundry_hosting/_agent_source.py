# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeAlias, TypeGuard, TypeVar, cast, runtime_checkable

from agent_framework import SupportsAgentRun, Workflow

AgentSource: TypeAlias = SupportsAgentRun | Callable[[], SupportsAgentRun | Awaitable[SupportsAgentRun]]


@runtime_checkable
class SupportsWorkflowBuild(Protocol):
    """A workflow builder with a zero-argument build method."""

    def build(self) -> Workflow: ...


RequestT = TypeVar("RequestT")
WorkflowSource: TypeAlias = Workflow | SupportsWorkflowBuild | Callable[[RequestT], Workflow | Awaitable[Workflow]]


def is_agent(value: object) -> TypeGuard[SupportsAgentRun]:
    return not inspect.isclass(value) and isinstance(value, SupportsAgentRun)


def validate_agent_source(source: object) -> None:
    if is_agent(source):
        return
    if not callable(source):
        raise TypeError("agent must be an agent instance or a zero-argument callable that creates one.")
    try:
        inspect.signature(source).bind()
    except (TypeError, ValueError) as exc:
        raise TypeError("agent callable must accept no arguments.") from exc


async def resolve_agent(source: AgentSource) -> SupportsAgentRun:
    """Resolve an agent instance or request-scoped agent factory."""
    if is_agent(source):
        return source

    factory = cast(Callable[[], SupportsAgentRun | Awaitable[SupportsAgentRun]], source)
    result = factory()
    agent = await result if inspect.isawaitable(result) else result
    if not is_agent(agent):
        raise TypeError("The agent factory must return an object implementing SupportsAgentRun.")
    return agent


def validate_workflow_source(source: object) -> None:
    """Require an already-built workflow or a factory that can build one."""
    if isinstance(source, Workflow):
        return
    if not inspect.isclass(source) and isinstance(source, SupportsWorkflowBuild):
        factory: Callable[..., object] = source.build
        args: tuple[object, ...] = ()
    elif callable(source) and not inspect.isclass(source):
        factory = source
        args = (object(),)
    else:
        raise TypeError("workflow must be a Workflow, builder, or one-request-argument callable.")
    try:
        inspect.signature(factory).bind(*args)
    except (TypeError, ValueError) as exc:
        raise TypeError("workflow factory must accept exactly one request (builders use build()).") from exc


async def resolve_workflow(source: WorkflowSource[RequestT], request: RequestT) -> Workflow:
    """Build a fresh workflow using the approved request-aware source."""
    if isinstance(source, Workflow):
        return source
    result = source.build() if isinstance(source, SupportsWorkflowBuild) else source(request)
    workflow = await result if inspect.isawaitable(result) else result
    if not isinstance(workflow, Workflow):
        raise TypeError("The workflow factory must return a built Workflow, not a WorkflowBuilder.")
    return workflow
