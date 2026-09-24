# Copyright (c) Microsoft. All rights reserved.

"""Request models and option mapping shared by the Foundry protocol hosts."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from agent_framework import AgentRunInputs, Message, WorkflowInvocationKwargs
from azure.ai.agentserver.responses import ResponseContext
from azure.ai.agentserver.responses.models import CreateResponse, Item

from ._scope import FoundryRequestScope

UnsupportedOptions: TypeAlias = Literal["ignore", "warn", "error"]

_PROTOCOL_FIELDS = frozenset({
    "agent",
    "agent_reference",
    "agent_session_id",
    "background",
    "call_id",
    "conversation",
    "conversation_id",
    "continuation_token",
    "input",
    "previous_response_id",
    "response_id",
    "service_session_id",
    "session_id",
    "store",
    "stream",
    "user_id",
})
_OPTION_NAMES = {"max_output_tokens": "max_tokens", "parallel_tool_calls": "allow_multiple_tool_calls"}
_NATIVE_FIELDS = frozenset(CreateResponse.__annotations__)


def response_run_options(request: CreateResponse) -> dict[str, Any]:
    """Map native generation fields before overlaying arbitrary extra-body fields."""
    native: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for name, value in request.items():
        if name in _PROTOCOL_FIELDS or value is None or (name == "model" and value == ""):
            continue
        if name in _NATIVE_FIELDS:
            native[_OPTION_NAMES.get(name, name)] = value
        else:
            extra[name] = value
    return {**native, **extra}


class HostedResponseRequest:
    """Expose only the current turn, effective runtime options, and trusted scope."""

    def __init__(
        self,
        request: CreateResponse,
        context: ResponseContext,
        scope: FoundryRequestScope,
        options: Mapping[str, Any],
        *,
        input_messages: Callable[[Sequence[Item]], Awaitable[list[Message]]] | None = None,
        workflow_responses: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.request = request
        self.context = context
        self.scope = scope
        self._options = MappingProxyType(dict(options))
        self._input_messages = input_messages
        self._workflow_responses = workflow_responses

    @property
    def options(self) -> Mapping[str, Any]:
        """Return native and extra-body options after the developer hook."""
        return self._options

    def set_options(self, options: Mapping[str, Any]) -> None:
        """Replace this request's runtime options after the developer hook."""
        self._options = MappingProxyType(dict(options))

    def set_workflow_responses(self, loader: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        """Bind approval decoding after the host has resolved a scoped checkpoint."""
        self._workflow_responses = loader

    def set_input_messages(self, converter: Callable[[Sequence[Item]], Awaitable[list[Message]]]) -> None:
        """Bind conversion to the approval store for this request's lineage."""
        self._input_messages = converter

    async def get_input_items(self) -> list[Item]:
        """Load only the current request's input items, not prior history."""
        return list(await self.context.get_input_items())

    async def get_input_text(self) -> str | None:
        """Get the current request's text, if any."""
        return await self.context.get_input_text()

    async def get_input_messages(self) -> list[Message]:
        """Convert the current request's items to MAF messages."""
        if self._input_messages is None:
            raise RuntimeError("This host does not support Responses input-message conversion.")
        return await self._input_messages(await self.get_input_items())

    async def get_workflow_responses(self) -> dict[str, Any]:
        """Load validated replies to this workflow's pending user-input requests."""
        if self._workflow_responses is None:
            raise RuntimeError("Workflow response conversion requires a restored checkpoint.")
        return await self._workflow_responses()


InputT = TypeVar("InputT")


@dataclass(frozen=True)
class WorkflowTurn(Generic[InputT]):
    """One workflow input or a reply to pending workflow requests."""

    input: InputT | None = None
    responses: Mapping[str, Any] | None = None
    client_kwargs: WorkflowInvocationKwargs | Mapping[str, Any] | None = None
    function_invocation_kwargs: WorkflowInvocationKwargs | Mapping[str, Any] | None = None
    stream: bool = False

    def __post_init__(self) -> None:
        if (self.input is None and not self.responses) or (self.input is not None and self.responses is not None):
            raise ValueError("WorkflowTurn requires exactly one of input or non-empty responses.")
        if not isinstance(self.stream, bool):
            raise TypeError("WorkflowTurn.stream must be a boolean.")


@dataclass(frozen=True)
class InvocationRun:
    """A parsed application request for an Invocations-hosted agent."""

    messages: AgentRunInputs
    options: Mapping[str, Any] = field(default_factory=lambda: dict[str, Any]())
    stream: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.options, Mapping):
            raise TypeError("InvocationRun.options must be a mapping.")
        if not isinstance(self.stream, bool):
            raise TypeError("InvocationRun.stream must be a boolean.")


OptionsHook: TypeAlias = Callable[
    [HostedResponseRequest, dict[str, Any]],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]


async def prepare_response_options(request: HostedResponseRequest, hook: OptionsHook | None) -> None:
    """Apply the developer hook without mutating agent-level defaults."""
    if hook is None:
        return
    result = hook(request, dict(request.options))
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, Mapping):
        raise TypeError("prepare_options must return a mapping of MAF run options.")
    request.set_options(result)


def validate_unsupported_options(mode: str) -> UnsupportedOptions:
    """Reject misspelled policy values at host construction."""
    if mode not in ("ignore", "warn", "error"):
        raise ValueError("unsupported_options must be 'ignore', 'warn', or 'error'.")
    return mode
