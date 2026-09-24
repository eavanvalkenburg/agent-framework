# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import ipaddress
import json
import logging
import os
import re
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, AsyncExitStack, aclosing, suppress
from dataclasses import asdict, dataclass, is_dataclass
from typing import Generic, Literal, TypeGuard, TypeVar, cast
from urllib.parse import urlparse

from agent_framework import (
    AgentExecutor,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    ChatOptions,
    CheckpointStorage,
    Content,
    ContextProvider,
    HistoryProvider,
    InMemoryHistoryProvider,
    Message,
    RawAgent,
    ResponseStream,
    SessionStore,
    SupportsAgentRun,
    UsageDetails,
    WorkflowAgent,
    WorkflowEvent,
    WorkflowRunState,
    add_usage_details,
)
from agent_framework._telemetry import mark_feature_used
from agent_framework.exceptions import AgentFrameworkException
from azure.ai.agentserver.core import AgentConfig, get_request_context
from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseProviderProtocol,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses._id_generator import IdGenerator
from azure.ai.agentserver.responses.aio import ResponseEventStream
from azure.ai.agentserver.responses.hosting import ResponsesAgentServerHost
from azure.ai.agentserver.responses.models import (
    CreateResponse,
    FunctionShellAction,
    FunctionShellCallOutputContent,
    FunctionShellCallOutputExitOutcome,
    Item,
    ItemReasoningItem,
    LocalEnvironmentResource,
    MessageContent,
    OAuthConsentRequestOutputItem,
    OutputItem,
    OutputItemReasoningItem,
    OutputMessageContent,
    ResponseIncompleteReason,
    ResponseStreamEvent,
    ResponseUsage,
    ResponseUsageInputTokensDetails,
    ResponseUsageOutputTokensDetails,
)
from azure.ai.agentserver.responses.streaming._builders import (
    OutputItemBuilder,
    OutputItemFunctionCallBuilder,
    OutputItemMcpCallBuilder,
    OutputItemMessageBuilder,
    ReasoningSummaryPartBuilder,
    RefusalContentBuilder,
    TextContentBuilder,
)
from azure.ai.agentserver.responses.streaming._checkpoint import ResponseCheckpointEvent
from mcp import McpError
from typing_extensions import Any

from ._agent_source import (
    AgentSource,
    WorkflowSource,
    is_agent,
    resolve_agent,
    resolve_workflow,
    validate_agent_source,
    validate_workflow_source,
)
from ._feature_usage import FeatureIndex
from ._request import (
    HostedResponseRequest,
    OptionsHook,
    UnsupportedOptions,
    WorkflowTurn,
    prepare_response_options,
    response_run_options,
    validate_unsupported_options,
)
from ._scope import FoundryRequestScope
from ._state_store import (
    AgentSessionStoreProvider,
    CheckpointStoreProvider,
    ContextScopedStoreProvider,
    FunctionApprovalStore,
    FunctionApprovalStoreProvider,
    StoreProvider,
)
from ._workflow_state import FoundryWorkflowBindingStore, WorkflowApproval, WorkflowBinding

logger = logging.getLogger(__name__)

_MODEL_OUTPUT_KIND_KEY = "model_output_kind"
_MODEL_OUTPUT_REFUSAL = "refusal"
_HOSTED_RESPONSES_HISTORY_SOURCE_ID = "_foundry_responses_history"
_HOSTED_PROVIDER_STATE_KEY = "_foundry_provider_background"
_HOSTED_SOURCE_CONVERSATION_KEY = "_foundry_source_conversation"


def _is_refusal_text_content(content: Content) -> bool:
    return content.type == "text" and content.additional_properties.get(_MODEL_OUTPUT_KIND_KEY) == _MODEL_OUTPUT_REFUSAL


def _validate_checkpoint_context_id(context_id: str) -> None:
    """Validate that a checkpoint context ID is a single safe path component in case file-based storage is used."""
    if (
        not context_id
        or "/" in context_id
        or "\\" in context_id
        or "\x00" in context_id
        or context_id.strip(".") == ""
        or os.path.isabs(context_id)
        or os.path.splitdrive(context_id)[0]
    ):
        raise RuntimeError(f"Invalid context id: {context_id!r}")


def _is_hosted_responses_history_sentinel(provider: ContextProvider) -> bool:
    """Return whether ``provider`` is the host's transient history buffer."""
    return (
        isinstance(provider, InMemoryHistoryProvider)
        and provider.source_id == _HOSTED_RESPONSES_HISTORY_SOURCE_ID
        and provider.load_messages
        and provider.store_inputs
        and not provider.store_context_messages
        and provider.store_outputs
    )


def _create_response_event_stream(context: ResponseContext) -> ResponseEventStream:
    """Create a response stream seeded from recovery state when available."""
    if context.is_recovery:
        persisted_response = context.persisted_response
        if persisted_response is not None:
            return ResponseEventStream(response=persisted_response, response_id=context.response_id)
    return ResponseEventStream(response_id=context.response_id)


def _workflow_event_updates(event: WorkflowEvent[Any], response_id: str) -> list[AgentResponseUpdate]:
    """Project public workflow outputs and requests onto the existing Responses content writer."""
    data = event.data
    if event.type in ("output", "intermediate"):
        if isinstance(data, AgentResponseUpdate):
            return [data]
        if isinstance(data, AgentResponse):
            updates = [
                AgentResponseUpdate(
                    contents=list(message.contents),
                    role=message.role,
                    author_name=message.author_name or event.executor_id,
                    response_id=response_id,
                    message_id=message.message_id or uuid.uuid4().hex,
                )
                for message in data.messages
            ]
            if data.usage_details is not None:
                updates.append(AgentResponseUpdate(contents=[Content.from_usage(data.usage_details)]))
            return updates
        if isinstance(data, Message):
            return [
                AgentResponseUpdate(
                    contents=list(data.contents),
                    role=data.role,
                    response_id=response_id,
                    message_id=data.message_id or uuid.uuid4().hex,
                )
            ]
        if isinstance(data, list):
            updates: list[AgentResponseUpdate] = []
            for item in cast(list[Any], data):
                updates.extend(_workflow_event_updates(WorkflowEvent("output", data=item), response_id))
            return updates
        return [
            AgentResponseUpdate(
                contents=[data if isinstance(data, Content) else Content.from_text(_json_safe_to_str(data))],
                role="assistant",
                response_id=response_id,
                message_id=uuid.uuid4().hex,
            )
        ]

    if event.type != "request_info":
        return []
    if not event.request_id:
        raise ValueError("Workflow request_info event must have a request ID.")
    if isinstance(data, Content) and data.user_input_request and data.type != "text":
        content = data
    else:
        content = Content.from_function_call(
            event.request_id,
            "request_info",
            arguments={"request_id": event.request_id, "request": _json_safe_to_str(data)},
        )
    return [
        AgentResponseUpdate(
            contents=[content],
            role="assistant",
            response_id=response_id,
            message_id=uuid.uuid4().hex,
        )
    ]


def _agent_response_updates(response: AgentResponse[Any], response_id: str) -> list[AgentResponseUpdate]:
    """Convert a completed inner provider response without exposing its continuation token."""
    updates = [
        AgentResponseUpdate(
            contents=list(message.contents),
            role=message.role,
            author_name=message.author_name,
            response_id=response_id,
            message_id=message.message_id or uuid.uuid4().hex,
        )
        for message in response.messages
    ]
    if response.usage_details is not None:
        updates.append(AgentResponseUpdate(contents=[Content.from_usage(response.usage_details)]))
    return updates


_T = TypeVar("_T")

# Sentinel put on the internal queue by _SignalledIterator's driver task to signal that the
# wrapped iterator is exhausted (distinct from `None`, which is a valid item value).
_STOP_SENTINEL: Any = object()


class _SignalledIterator(Generic[_T]):
    """Wraps an async iterator, stopping early as soon as any of ``events`` fires.

    Plain ``async for update in agent.run(...): if event.is_set(): break`` only observes ``event``
    once ``run()`` actually yields an item -- if it's suspended on a single slow model or tool call
    with no intermediate item, the signal is invisible until that call resolves. This drives the
    wrapped iterator from a single persistent background task and races each produced item against
    ``events`` via ``asyncio.wait`` instead, so a signal is observed immediately.

    The background task (``_drive``) is required (rather than spawning a fresh task per step)
    because some cleanup run by the wrapped iterator (e.g. observability span teardown) resets a
    contextvar token set on an earlier call and requires every call against it to share the same
    async context.

    If an event and a new item becomes ready at the same time, the event takes priority and the item
    is discarded. Cancelling the background task while it's mid-call interrupts a suspended model or
    tool call, then the driver closes the underlying stream in ``finally``.

    Callers MUST drive this through ``contextlib.aclosing`` (or an equivalent try/finally calling
    ``aclose()``): ``__anext__`` only cancels the driver task on its own signalled/exhausted paths, so
    if the consumer of ``async for`` raises instead (e.g. while processing a yielded item), the driver
    task -- and the real agent/workflow run it's pumping -- would otherwise be silently abandoned.
    """

    def __init__(
        self,
        iterator: AsyncIterator[_T],
        *events: asyncio.Event,
        stamp: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        """Wrap an async iterator, stopping early if any of ``events`` fires.

        Args:
            iterator: The async iterator to wrap.
            events: One or more asyncio.Event objects to watch for. If any of them is set, iteration stops early.
            stamp: Optional coroutine function the driver awaits right after the wrapped iterator produces an
                item and before it is advanced again. Its result is exposed as :attr:`stamp` while that item
                is the current one, which lets a consumer observe state (e.g. the latest persisted workflow
                checkpoint) as it was when the item was produced rather than when it is consumed: the driver
                runs one item ahead, so by consumption time the wrapped iterator may already have moved on.
        """
        self._iterator = iterator
        self._events = events
        self._stamp_fn = stamp
        self._stamp: Any = None
        self._signalled = False
        # The queue is used to communicate items from the background driver task to the main iteration loop.
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        # The background task that drives the wrapped iterator.
        self._driver: asyncio.Task[None] | None = None

    @property
    def signalled(self) -> bool:
        """Whether iteration stopped early due to an event being set.

        ``signalled`` is only set when iteration stopped early because of an event -- never on ordinary
        exhaustion -- so callers can tell "the agent/workflow finished" apart from "we gave up waiting".
        """
        return self._signalled

    @property
    def stamp(self) -> Any:
        """The ``stamp`` result taken when the current item was produced (``None`` without a ``stamp``)."""
        return self._stamp

    def __aiter__(self) -> _SignalledIterator[_T]:
        return self

    async def _drive(self) -> None:
        """Pull items from the wrapped iterator into ``self._queue`` for the object's lifetime."""
        try:
            while True:
                try:
                    item: Any = await self._iterator.__anext__()
                    stamp = await self._stamp_fn() if self._stamp_fn is not None else None
                except StopAsyncIteration:
                    await self._queue.put(_STOP_SENTINEL)
                    return
                except Exception as exc:
                    await self._queue.put(exc)
                    return
                await self._queue.put((item, stamp))
        finally:
            iterator: AsyncIterator[_T] = self._iterator
            if isinstance(iterator, ResponseStream):
                await cast(ResponseStream[_T, Any], iterator).close()
            else:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()

    async def __anext__(self) -> _T:
        if self._driver is None:
            self._driver = asyncio.ensure_future(self._drive())

        # Create the background tasks for monitoring the events and the queue.
        waiters = [asyncio.ensure_future(event.wait()) for event in self._events]
        get_task = asyncio.ensure_future(self._queue.get())
        try:
            # Waits until at least one of the tasks is completed.
            await asyncio.wait([get_task, *waiters], return_when=asyncio.FIRST_COMPLETED)
            if any(waiter.done() for waiter in waiters):
                self._signalled = True
                self._driver.cancel()
                with suppress(BaseException):
                    await self._driver
                get_task.cancel()
                with suppress(BaseException):
                    await get_task
                raise StopAsyncIteration
            item = get_task.result()
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            for waiter in waiters:
                with suppress(BaseException):
                    await waiter
        if item is _STOP_SENTINEL:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        item, self._stamp = item
        return cast(_T, item)

    async def aclose(self) -> None:
        """Cancel the background driver task, if any, and wait for it to finish.

        Safe to call unconditionally: a no-op if the driver was never started, and cancelling an
        already-finished task (normal exhaustion or a prior signalled stop) is also a no-op.
        """
        if self._driver is None:
            return
        self._driver.cancel()
        with suppress(BaseException):
            await self._driver


# Reserved response metadata key pinning the workflow checkpoint that was current at the moment of
# the last successfully persisted response-stream checkpoint. Recovery MUST resume from this specific
# checkpoint if it exists, not simply the latest one in checkpoint_storage: the workflow may have
# saved further checkpoints after it but before a crash, without their output ever being durably
# recorded in response.output. If this key is missing, the workflow will resume from the latest
# checkpoint in storage (if any), or replay the original input if none exists as no output was ever
# durably persisted.
_LATEST_CHECKPOINT_ID_KEY = "_last_checkpoint_id"
# ``internal_metadata`` key carrying a truncating finish reason across resilient checkpoints, so a
# turn cut short before a crash still ends ``incomplete`` after recovery.
_INCOMPLETE_REASON_KEY = "_incomplete_reason"


# Foundry Toolbox Auth integration
# Consent-URL error code returned by the Foundry MCP gateway when calling `/list`
CONSENT_ERROR_CODE = -32006

_OAUTH_HOST_PATTERN = re.compile(r"^[A-Za-z0-9._~-]+$")


@dataclass
class ConsentError:
    name: str
    consent_url: str


def _is_safe_oauth_consent_link(consent_link: object) -> TypeGuard[str]:
    """Return whether a consent link is an absolute HTTPS URL safe to expose as an action."""
    if not isinstance(consent_link, str) or not consent_link:
        return False
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in consent_link):
        return False

    try:
        parsed = urlparse(consent_link)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False

    if parsed.scheme.lower() != "https" or not hostname or parsed.username is not None or parsed.password is not None:
        return False

    if "%" in hostname:
        return False
    authority = parsed.netloc
    if authority.startswith("["):
        closing_bracket = authority.find("]")
        if closing_bracket == -1:
            return False
        ipv6_literal = authority[1:closing_bracket]
        suffix = authority[closing_bracket + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return False
        try:
            ipaddress.IPv6Address(ipv6_literal)
        except ValueError:
            return False
        return True
    if "[" in authority or "]" in authority or ":" in hostname:
        return False
    return _OAUTH_HOST_PATTERN.fullmatch(hostname) is not None


def consent_url_from_error(exc: BaseException) -> list[ConsentError] | None:
    """Return the consent URLs when ``exc`` wraps Foundry MCP gateway consent errors.

    Args:
        exc: The exception to inspect.

    Returns:
        The consent URL(s) extracted from the error, or ``None`` if no consent error was found.
    """
    inner_exception = next((arg for arg in exc.args if isinstance(arg, McpError)), None)
    if inner_exception is not None and inner_exception.error.code == CONSENT_ERROR_CODE:
        # Parse the error message
        # The error message is structured with the following format:
        # "tools/list failed for 1 tool source(s), succeeded for 0 tool source(s) {"errors":[{"name": ..."
        # where the second part is a JSON string that can be deserialized into an object with the following shape:
        # ruff: disable[commented-out-code]
        # {
        #   "errors" : [
        #       {
        #           "name": "Name of the MCP tool that requires consent",
        #           "type" : "mcp" | "a2a_preview",
        #           "error": {
        #               "code": "CONSENT_REQUIRED",
        #               "message": consent_url,
        #           }
        #       }
        #   ]
        # }
        # ruff: enable[commented-out-code]
        try:
            consent_errors: list[ConsentError] = []
            error_message_start = inner_exception.error.message.find("{")
            if error_message_start == -1:
                logger.warning("Consent error message does not contain JSON: %s", inner_exception.error.message)
                return None
            consent_details_json = inner_exception.error.message[error_message_start:]
            consent_details = json.loads(consent_details_json)
            if "errors" not in consent_details or not isinstance(consent_details["errors"], list):
                logger.warning("Consent error message JSON does not contain 'errors' list: %s", consent_details_json)
                return None
            for error in consent_details["errors"]:
                if (
                    isinstance(error, dict)
                    and error.get("type") in ("mcp", "a2a_preview")  # type: ignore
                    and "error" in error
                    and isinstance(error["error"], dict)
                    and error["error"].get("code") == "CONSENT_REQUIRED"  # type: ignore
                    and "message" in error["error"]
                ):
                    consent_url = error["error"]["message"]  # type: ignore
                    if isinstance(consent_url, str):
                        consent_errors.append(ConsentError(name=error.get("name", "Unknown"), consent_url=consent_url))  # type: ignore
                    else:
                        logger.warning("Consent URL in error message is not a valid URL: %s", consent_url)  # type: ignore
            if consent_errors:
                return consent_errors
        except json.JSONDecodeError:
            logger.warning("Failed to parse consent details JSON: %s", inner_exception.error.message)
    return None


# endregion Foundry Toolbox Auth integration


@dataclass(frozen=True)
class _AgentConfiguration:
    workflow: bool
    agent_server_history: bool
    client_stores_by_default: bool
    hosted_history: bool


def _validate_agent_configuration(
    agent: SupportsAgentRun,
    history_source: Literal["agent_server", "agent"],
    options: ResponsesServerOptions | None,
    *,
    inner_background: Literal["host", "provider"] = "host",
) -> _AgentConfiguration:
    is_workflow_agent = isinstance(agent, WorkflowAgent)
    if is_workflow_agent and agent.workflow._runner_context.has_checkpointing():  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError(
            "There should not be a checkpoint storage already present in the workflow agent. "
            "The hosting infrastructure will manage checkpoints instead."
        )

    resilient_background = bool(options and options.resilient_background)
    if resilient_background and not is_workflow_agent and inner_background != "provider":
        raise RuntimeError(
            "resilient_background=True is only supported for workflow agents. "
            "Crash recovery cannot be provided for non-workflow agents."
        )
    if inner_background == "provider" and (
        not isinstance(agent, RawAgent) or getattr(cast(Any, agent).client, "STORES_BY_DEFAULT", None) is not True
    ):
        raise RuntimeError("Provider background requires a RawAgent with a storing, resumable Responses client.")
    if options and options.steerable_conversations and is_workflow_agent:
        raise RuntimeError(
            "steerable_conversations=True is only supported for non-workflow agents. "
            "Steering cannot be provided reliably for workflow agents."
        )

    uses_agent_server_history = history_source == "agent_server"
    client_stores_by_default = False
    if uses_agent_server_history and not is_workflow_agent:
        if not isinstance(agent, RawAgent):
            raise RuntimeError(
                "history_source='agent_server' requires a RawAgent so hosting can enforce downstream "
                "storage options. Construct ResponsesHostServer with history_source='agent' for a custom "
                "SupportsAgentRun implementation."
            )
        for provider in agent.context_providers:
            if isinstance(provider, HistoryProvider) and provider.load_messages:
                if _is_hosted_responses_history_sentinel(provider):
                    continue
                raise RuntimeError(
                    "AgentServer response history is enabled, but the agent has a HistoryProvider "
                    "with load_messages=True. Remove that provider or construct ResponsesHostServer "
                    "with history_source='agent' to use the agent's regular history setup."
                )
        service_continuation_options = [
            name
            for name in ("conversation_id", "previous_response_id", "conversation")
            if agent.default_options.get(name) is not None
        ]
        if service_continuation_options:
            raise RuntimeError(
                "AgentServer response history is enabled, but the agent has downstream service continuation "
                f"option(s): {', '.join(service_continuation_options)}. Remove them or construct "
                "ResponsesHostServer with history_source='agent' to resume the downstream service conversation."
            )
        stores_by_default = getattr(cast(Any, agent).client, "STORES_BY_DEFAULT", None)
        if not isinstance(stores_by_default, bool):
            raise RuntimeError(
                "history_source='agent_server' requires the agent's chat client to declare "
                "STORES_BY_DEFAULT so hosting can enforce downstream storage behavior."
            )
        client_stores_by_default = stores_by_default
        if not client_stores_by_default and agent.default_options.get("store") is not None:
            raise RuntimeError(
                "The chat client does not store by default, but the agent sets a downstream store option. "
                "Remove that developer-owned default rather than letting hosting change it."
            )

    return _AgentConfiguration(
        workflow=is_workflow_agent,
        agent_server_history=uses_agent_server_history,
        client_stores_by_default=client_stores_by_default,
        hosted_history=uses_agent_server_history and not is_workflow_agent,
    )


def _initialize_agent_history(agent: SupportsAgentRun, configuration: _AgentConfiguration) -> None:
    if not configuration.hosted_history or not isinstance(agent, RawAgent):
        return
    if not any(
        _is_hosted_responses_history_sentinel(provider)
        for provider in cast(Sequence[ContextProvider], agent.context_providers)
    ):
        agent.context_providers.append(InMemoryHistoryProvider(source_id=_HOSTED_RESPONSES_HISTORY_SOURCE_ID))


# region ResponsesHostServer
class ResponsesHostServer(ResponsesAgentServerHost):
    """A Foundry Responses server for an agent or native workflow."""

    def __init__(
        self,
        agent: AgentSource | None = None,
        *,
        workflow: WorkflowSource[HostedResponseRequest] | None = None,
        parse_response: Callable[[HostedResponseRequest], WorkflowTurn[Any] | Awaitable[WorkflowTurn[Any]]]
        | None = None,
        prefix: str = "",
        options: ResponsesServerOptions | None = None,
        store: ResponseProviderProtocol | None = None,
        response_store: ResponseProviderProtocol | None = None,
        agent_session_store_provider: StoreProvider[SessionStore] | None = None,
        checkpoint_store_provider: ContextScopedStoreProvider[CheckpointStorage] | None = None,
        function_approval_store_provider: StoreProvider[FunctionApprovalStore] | None = None,
        inner_history: Literal["host", "service", "agent"] | None = None,
        history_source: Literal["agent_server", "agent"] | None = None,
        inner_background: Literal["host", "provider"] = "host",
        prepare_options: OptionsHook | None = None,
        unsupported_options: UnsupportedOptions = "warn",
        **kwargs: Any,
    ) -> None:
        """Configure a Foundry Responses host.

        Args:
            agent: The agent to handle responses for, or a zero-argument sync or async callable that creates one for
                each request. Use a callable for agents that keep mutable state outside `AgentSession`.
            workflow: A built native workflow, a builder, or a request-aware factory returning a built workflow.
            parse_response: Required for native workflows; maps Responses input into a `WorkflowTurn`.
            prefix: The URL prefix for the server.
            options: AgentServer configuration for background work, steering, and other protocol features.
            store: Deprecated alias for `response_store`.
            response_store: Backend for caller-visible Responses storage and history.
            agent_session_store_provider: Optional provider for MAF agent session storage.
            checkpoint_store_provider: Optional provider for workflow checkpoint storage.
            function_approval_store_provider: Optional provider for function approval storage.
            inner_history: Whether outer Responses history, downstream service storage, or agent history is used.
            history_source: Deprecated alias for host or agent history selection.
            inner_background: Keep provider background off by default; `"provider"` opts a supporting client in.
            prepare_options: Optional hook to edit caller runtime options before `Agent.run`.
            unsupported_options: Whether to ignore, warn, or reject options an agent cannot accept.
            **kwargs: Additional AgentServer constructor arguments.
        """
        if history_source is not None and history_source not in ("agent_server", "agent"):
            raise ValueError("history_source must be either 'agent_server' or 'agent'.")
        if inner_history is not None and inner_history not in ("host", "service", "agent"):
            raise ValueError("inner_history must be 'host', 'service', or 'agent'.")
        if inner_history is not None and history_source is not None:
            raise ValueError("inner_history and history_source cannot be combined.")
        if inner_background not in ("host", "provider"):
            raise ValueError("inner_background must be 'host' or 'provider'.")
        if store is not None and response_store is not None:
            raise ValueError("Pass response_store instead of store; they cannot be combined.")
        if agent is None and workflow is None:
            raise TypeError("agent must be an agent instance or a zero-argument callable, or provide workflow.")
        if agent is not None and workflow is not None:
            raise TypeError("Provide exactly one of agent or workflow.")

        resolved_history: Literal["host", "service", "agent"] = inner_history or (
            "agent" if history_source == "agent" else "host"
        )
        if workflow is not None:
            validate_workflow_source(workflow)
            if parse_response is None:
                raise TypeError("parse_response is required when hosting a native Workflow.")
            if options and options.steerable_conversations:
                raise ValueError("Steering a native Workflow is not supported.")
            if inner_background != "host":
                raise ValueError("Provider background cannot be enabled for a native Workflow.")
        else:
            validate_agent_source(agent)
            if parse_response is not None:
                raise ValueError("parse_response is only valid for a native Workflow.")
            if inner_background == "provider" and resolved_history != "service":
                raise ValueError("Provider background requires inner_history='service'.")

        if workflow is not None and not callable(workflow) and AgentConfig.from_env().is_hosted:
            raise ValueError("Hosted workflows require a request-aware factory that creates fresh executors.")

        resolved_agent = agent if is_agent(agent) else None
        configuration = (
            _validate_agent_configuration(
                resolved_agent,
                "agent_server" if resolved_history == "host" else "agent",
                options,
                inner_background=inner_background,
            )
            if resolved_agent is not None
            else None
        )

        # No caller-owned agent state is mutated until all validation and base-host construction succeed.
        super().__init__(
            prefix=prefix,
            options=options,
            store=response_store if response_store is not None else store,
            **kwargs,
        )
        if options and options.steerable_conversations:
            # AgentServer needs its task manager even when crash recovery is not enabled.
            from azure.ai.agentserver.core.tasks import set_resilient_tasks_enabled

            set_resilient_tasks_enabled(True)

        self._agent_source = agent
        self._agent = resolved_agent
        self._configuration = configuration
        self._workflow_source = workflow
        self._parse_response = parse_response
        self._inner_history = resolved_history
        self._inner_background: Literal["host", "provider"] = inner_background
        self._prepare_options = prepare_options
        self._unsupported_options = validate_unsupported_options(unsupported_options)
        self._history_source: Literal["agent_server", "agent"] = (
            "agent_server" if resolved_history == "host" else "agent"
        )
        self._host_options = options
        self._uses_agent_server_history = (
            configuration.agent_server_history if configuration is not None else resolved_history == "host"
        )
        self._resilient_background = bool(options and options.resilient_background)
        if resolved_agent is not None and configuration is not None:
            _initialize_agent_history(resolved_agent, configuration)

        # Storage providers
        self._checkpoint_storage_provider = (
            CheckpointStoreProvider() if checkpoint_store_provider is None else checkpoint_store_provider
        )
        self._session_storage_provider = (
            AgentSessionStoreProvider() if agent_session_store_provider is None else agent_session_store_provider
        )
        self._function_approval_storage_provider = (
            FunctionApprovalStoreProvider()
            if function_approval_store_provider is None
            else function_approval_store_provider
        )

        # Lazy agent lifecycle: the agent (and any MCP tools it owns) is entered on
        # the first request rather than at server startup, so that authentication
        # failures during MCP connect can be surfaced to the client as an
        # `oauth_consent_request` stream event instead of crashing the server.
        self._agent_stack: AsyncExitStack | None = None
        self._agent_init_lock = asyncio.Lock()

        self.shutdown_handler(self._cleanup_agent)
        self.response_handler(self._handle_response)

        mark_feature_used(FeatureIndex.FOUNDRY_HOSTING)

    async def _ensure_agent_ready(self) -> None:
        """Lazily enter the agent's async context exactly once.

        On failure the partial exit stack is closed and ``_agent_stack`` is left
        as ``None`` so a subsequent request (e.g. after the user completes OAuth
        consent) can retry the connection.
        """
        if self._agent_stack is not None:
            return
        async with self._agent_init_lock:
            if self._agent_stack is not None:
                return
            agent = self._agent
            if agent is None:
                raise RuntimeError("A request-scoped agent cannot use the server-lifetime initialization path.")
            stack = AsyncExitStack()
            try:
                if isinstance(agent, AbstractAsyncContextManager):
                    await stack.enter_async_context(cast(AbstractAsyncContextManager[Any], agent))
            except BaseException:
                await stack.aclose()
                raise
            self._agent_stack = stack

    async def _cleanup_agent(self) -> None:
        """Close the agent's async context. Registered as the server shutdown handler."""
        stack = self._agent_stack
        if stack is not None:
            self._agent_stack = None
            await stack.aclose()

    async def _handle_response(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncIterable[ResponseStreamEvent | ResponseCheckpointEvent]:
        """Handle the creation of a response."""
        response_event_stream = _create_response_event_stream(context)
        if context.is_steered_turn:
            logger.debug("Serving steered turn (pending_input_count=%d)", context.pending_input_count)
        yield response_event_stream.emit_created()
        yield response_event_stream.emit_in_progress()

        terminal_event: ResponseStreamEvent | None = None
        try:
            scope = FoundryRequestScope.from_context(
                self.config,
                get_request_context(),
                local_session_id=context.response_id,
            )
            hosted_request = HostedResponseRequest(
                request,
                context,
                scope,
                response_run_options(request),
                input_messages=lambda items: _items_to_messages(items, approval_storage=None),
            )
            await prepare_response_options(hosted_request, self._prepare_options)
        except Exception as exc:
            logger.exception("Failed to prepare hosted Responses request")
            for event in self._emit_failure(response_event_stream, None, exc):
                yield event
            return

        if self._workflow_source is not None:
            async for event in self._handle_native_workflow_response(
                hosted_request,
                response_event_stream,
                cancellation_signal,
            ):
                yield event
            return

        if self._agent_source is None:
            raise RuntimeError("No agent or workflow source is configured.")
        agent = await resolve_agent(self._agent_source)
        configuration = self._configuration or _validate_agent_configuration(
            agent, self._history_source, self._host_options, inner_background=self._inner_background
        )
        if self._configuration is None:
            _initialize_agent_history(agent, configuration)

        async with AsyncExitStack() as resources:
            inner = self._handle_prepared_response(
                request,
                context,
                cancellation_signal,
                response_event_stream,
                agent,
                configuration,
                resources,
                hosted_request,
            )
            try:
                async for event in inner:
                    if isinstance(event, Mapping) and event.get("type") in (
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    ):
                        terminal_event = event
                    else:
                        yield event
            finally:
                await inner.aclose()
        if terminal_event is not None:
            yield terminal_event

    async def _load_workflow_responses(
        self,
        hosted_request: HostedResponseRequest,
        *,
        checkpoint_storage: CheckpointStorage | None,
        checkpoint_id: str | None,
        lineage_id: str,
        bindings: FoundryWorkflowBindingStore,
        approval_storage: FunctionApprovalStore | None,
    ) -> dict[str, Any]:
        """Bind caller decisions to requests recorded by the exact pause checkpoint."""
        items = [cast(Mapping[str, Any], item) for item in await hosted_request.get_input_items()]
        replies = [item for item in items if item.get("type") in ("mcp_approval_response", "function_call_output")]
        if not replies:
            return {}
        if checkpoint_storage is None or checkpoint_id is None or approval_storage is None:
            raise ValueError("Workflow approval and user-input continuation requires store=true.")
        if len(replies) != len(items):
            raise ValueError("Pending workflow replies cannot be mixed with new user input.")

        checkpoint = await checkpoint_storage.load(checkpoint_id)
        pending = checkpoint.pending_request_info_events
        responses: dict[str, Any] = {}
        approvals_to_consume: list[str] = []
        for item in replies:
            if item["type"] == "mcp_approval_response":
                wire_id = item.get("approval_request_id")
                decision = item.get("approve")
                if not isinstance(wire_id, str) or type(decision) is not bool:
                    raise ValueError("A workflow approval requires its request ID and a boolean decision.")
                approval = await bindings.get_approval(wire_id, lineage_id=lineage_id)
                pending_request = pending.get(approval.request_id)
                if (
                    pending_request is None
                    or not isinstance(pending_request.data, Content)
                    or pending_request.data.type != "function_approval_request"
                ):
                    raise ValueError("No matching pending workflow approval exists in the checkpoint.")
                if approval.checkpoint_id != checkpoint_id:
                    original = await checkpoint_storage.load(approval.checkpoint_id)
                    recorded = original.pending_request_info_events.get(approval.request_id)
                    if recorded is None or recorded.to_dict() != pending_request.to_dict():
                        raise PermissionError("This approval no longer matches the pending workflow request.")
                if approval.request_id in responses:
                    raise ValueError("A pending workflow approval was answered more than once.")
                approved_request = await approval_storage.load_approval_request(wire_id)
                responses[approval.request_id] = approved_request.to_function_approval_response(decision)
                approvals_to_consume.append(wire_id)
                continue

            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Workflow function results require a call_id.")
            matches = [
                pending_id
                for pending_id, event in pending.items()
                if pending_id == call_id
                or (
                    isinstance(event.data, Content)
                    and event.data.type == "function_call"
                    and event.data.call_id == call_id
                )
            ]
            if len(matches) != 1:
                raise ValueError("A workflow function result must match exactly one pending request.")
            request_id = matches[0]
            if request_id in responses:
                raise ValueError("A pending workflow request was answered more than once.")
            pending_request = pending[request_id]
            if pending_request.response_type is Content:
                responses[request_id] = Content.from_function_result(
                    call_id,
                    result=_json_safe_to_str(item["output"]),
                )
            else:
                responses[request_id] = item["output"]
        for wire_id in approvals_to_consume:
            approval = await bindings.get_approval(wire_id, lineage_id=lineage_id)
            await bindings.consume_approval(wire_id, lineage_id=lineage_id, checkpoint_id=approval.checkpoint_id)
        return responses

    async def _handle_native_workflow_response(
        self,
        hosted_request: HostedResponseRequest,
        response_event_stream: ResponseEventStream,
        cancellation_signal: asyncio.Event,
    ) -> AsyncGenerator[ResponseStreamEvent | ResponseCheckpointEvent]:
        """Run a native Workflow without constructing or calling WorkflowAgent."""
        request = hosted_request.request
        context = hosted_request.context
        source = self._workflow_source
        parser = self._parse_response
        if source is None or parser is None:
            raise RuntimeError("A native workflow source and parser are required.")

        tracker = _OutputItemTracker(response_event_stream)
        stored = request.get("store") is not False
        resilient = stored and self._resilient_background and request.get("background") is True
        pending_request_ids: set[str] = set()
        wire_approvals: dict[str, str] = {}

        try:
            if not stored and (context.conversation_id is not None or request.get("previous_response_id") is not None):
                raise ValueError("Workflow continuation requires store=true.")

            workflow = await resolve_workflow(source, hosted_request)
            if workflow._runner_context.has_checkpointing():  # pyright: ignore[reportPrivateUsage]
                raise ValueError("Configure workflow checkpoints on the host, not on WorkflowBuilder.")

            platform_context = get_request_context()
            bindings = FoundryWorkflowBindingStore(hosted_request.scope)
            previous_response_id = request.get("previous_response_id")
            if context.conversation_id is not None and previous_response_id is not None:
                raise ValueError("conversation and previous_response_id cannot be combined.")

            prior: WorkflowBinding | None = None
            expected_etag: str | None = None
            if stored and context.conversation_id is not None:
                prior, expected_etag = await bindings.get_conversation_head(context.conversation_id)
            elif stored and previous_response_id is not None:
                prior = await bindings.get_response(previous_response_id)
                if prior is None:
                    raise ValueError("The previous workflow response has no checkpoint in this Foundry session.")
                if prior.conversation_id is not None:
                    raise ValueError("Branching a workflow from a Responses conversation is not supported.")
                lineage_head, expected_etag = await bindings.get_lineage_head(prior.lineage_id)
                if lineage_head is None or lineage_head.response_id != prior.response_id:
                    raise ValueError("Branching a workflow from an earlier response is not supported.")

            if prior is not None and (
                prior.workflow_name != workflow.name or prior.graph_hash != workflow.graph_signature_hash
            ):
                raise ValueError("The stored workflow checkpoint is incompatible with the current workflow.")
            lineage_id = prior.lineage_id if prior is not None else context.conversation_id or context.response_id
            checkpoint_storage = None
            approval_storage = None
            if stored:
                context_id = hashlib.sha256(f"{workflow.name}\0{lineage_id}".encode()).hexdigest()
                checkpoint_storage = self._checkpoint_storage_provider.get_store(
                    config=self.config,
                    context_id=context_id,
                    platform_context=platform_context,
                )
                approval_storage = self._function_approval_storage_provider.get_store(
                    config=self.config,
                    platform_context=platform_context,
                )
            hosted_request.set_input_messages(
                lambda items: _items_to_messages(items, approval_storage=approval_storage)
            )
            hosted_request.set_workflow_responses(
                lambda: self._load_workflow_responses(
                    hosted_request,
                    checkpoint_storage=checkpoint_storage,
                    checkpoint_id=prior.checkpoint_id if prior is not None else None,
                    lineage_id=lineage_id,
                    bindings=bindings,
                    approval_storage=approval_storage,
                )
            )

            def snapshot_response(
                checkpoint_id: str | None,
            ) -> Generator[ResponseStreamEvent | ResponseCheckpointEvent]:
                if not resilient or checkpoint_id is None:
                    return
                if checkpoint_id == response_event_stream.internal_metadata.get(_LATEST_CHECKPOINT_ID_KEY):
                    return
                yield from tracker.close()
                response_event_stream.internal_metadata[_LATEST_CHECKPOINT_ID_KEY] = checkpoint_id
                yield response_event_stream.checkpoint()

            async def checkpoint_stamp() -> str | None:  # ruff: ignore[unused-async]
                return workflow.get_last_checkpoint_id()

            async def forward(
                run: AsyncIterator[WorkflowEvent[Any]],
                *,
                already_pending: set[str] | None = None,
            ) -> AsyncGenerator[ResponseStreamEvent | ResponseCheckpointEvent]:
                iterator = _SignalledIterator(
                    run,
                    context.shutdown,
                    cancellation_signal,
                    stamp=checkpoint_stamp if resilient else None,
                )
                async with aclosing(iterator):
                    async for workflow_event in iterator:
                        if resilient:
                            stamp = iterator.stamp
                            if stamp is not None and not isinstance(stamp, str):
                                raise TypeError("A workflow checkpoint stamp must be a string.")
                            for output_event in snapshot_response(stamp):
                                yield output_event
                        if workflow_event.type == "request_info":
                            request_id = workflow_event.request_id
                            if not request_id:
                                raise ValueError("A workflow user-input request has no ID.")
                            if already_pending is not None and request_id in already_pending:
                                continue
                            if not stored:
                                raise ValueError("Workflow approval and user input require store=true.")
                            pending_request_ids.add(request_id)
                        for update in _workflow_event_updates(workflow_event, context.response_id):
                            async for output_event in tracker.handle_update(update, approval_storage=approval_storage):
                                if workflow_event.type == "request_info" and isinstance(output_event, Mapping):
                                    wire_event = cast(Mapping[str, Any], output_event)
                                    output_item = wire_event.get("item")
                                    item_data: Mapping[str, Any] = (
                                        cast(Mapping[str, Any], output_item)
                                        if isinstance(output_item, Mapping)
                                        else dict[str, Any]()
                                    )
                                    wire_id = item_data.get("id")
                                    if (
                                        wire_event.get("type") == "response.output_item.added"
                                        and item_data.get("type") == "mcp_approval_request"
                                        and isinstance(wire_id, str)
                                    ):
                                        wire_approvals[wire_id] = workflow_event.request_id
                                yield output_event
                if iterator.signalled and context.shutdown.is_set() and resilient:
                    await context.exit_for_recovery()

            recovery_from: str | None = None
            async with AsyncExitStack() as resources:
                entered: set[int] = set()
                for executor in workflow.executors.values():
                    if isinstance(executor, AgentExecutor) and id(executor.agent) not in entered:
                        entered.add(id(executor.agent))
                        if isinstance(executor.agent, AbstractAsyncContextManager):
                            await resources.enter_async_context(executor.agent)

                if context.is_recovery:
                    if not resilient or checkpoint_storage is None:
                        raise RuntimeError("Workflow recovery requires a stored resilient background response.")
                    checkpoint_id = response_event_stream.internal_metadata.get(_LATEST_CHECKPOINT_ID_KEY)
                    if checkpoint_id is None:
                        current = await bindings.get_response(context.response_id)
                        checkpoint_id = current.checkpoint_id if current is not None else None
                    if not isinstance(checkpoint_id, str):
                        raise RuntimeError("Cannot recover a workflow response without a paired checkpoint.")
                    recovery_from = checkpoint_id
                    async for event in forward(
                        workflow.run(stream=True, checkpoint_id=checkpoint_id, checkpoint_storage=checkpoint_storage)
                    ):
                        yield event
                else:
                    prior_pending: set[str] = set()
                    if prior is not None:
                        if checkpoint_storage is None:
                            raise RuntimeError("A stored workflow continuation requires checkpoint storage.")
                        checkpoint = await checkpoint_storage.load(prior.checkpoint_id)
                        prior_pending = set(checkpoint.pending_request_info_events)
                        async for event in forward(
                            workflow.run(
                                stream=True,
                                checkpoint_id=prior.checkpoint_id,
                                checkpoint_storage=checkpoint_storage,
                            ),
                            already_pending=prior_pending,
                        ):
                            yield event
                    turn = parser(hosted_request)
                    if inspect.isawaitable(turn):
                        turn = await turn
                    if not isinstance(turn, WorkflowTurn):
                        raise TypeError("parse_response must return a WorkflowTurn.")
                    if turn.responses is not None:
                        if prior is None or checkpoint_storage is None:
                            raise ValueError("Workflow approval and user-input continuation requires store=true.")
                        if not set(turn.responses) <= prior_pending:
                            raise ValueError("A workflow response does not match the pending checkpoint.")
                        run = workflow.run(
                            responses=turn.responses,
                            stream=True,
                            checkpoint_storage=checkpoint_storage,
                            client_kwargs=turn.client_kwargs,
                            function_invocation_kwargs=turn.function_invocation_kwargs,
                        )
                    else:
                        if workflow.status == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
                            raise ValueError("Answer pending workflow requests before sending new input.")
                        run = workflow.run(
                            message=turn.input,
                            stream=True,
                            checkpoint_storage=checkpoint_storage,
                            client_kwargs=turn.client_kwargs,
                            function_invocation_kwargs=turn.function_invocation_kwargs,
                        )
                    carried_pending = prior_pending - set(turn.responses) if turn.responses is not None else None
                    async for event in forward(run, already_pending=carried_pending):
                        yield event

                if cancellation_signal.is_set() and context.client_cancelled:
                    return
                checkpoint_id = workflow.get_last_checkpoint_id()
                if resilient:
                    for event in snapshot_response(checkpoint_id):
                        yield event
                if stored:
                    if checkpoint_id is None or checkpoint_storage is None:
                        raise RuntimeError("A stored workflow response must finish with a checkpoint.")
                    if pending_request_ids:
                        pause_id = await workflow.resolve_pause_checkpoint_id(
                            pending_request_ids, checkpoint_storage=checkpoint_storage
                        )
                        if pause_id is None:
                            raise RuntimeError("Cannot expose workflow user input without a durable pause checkpoint.")
                        for wire_id, request_id in wire_approvals.items():
                            await bindings.save_approval(
                                WorkflowApproval(
                                    wire_id=wire_id,
                                    request_id=request_id,
                                    response_id=context.response_id,
                                    lineage_id=lineage_id,
                                    checkpoint_id=pause_id,
                                    session_id=hosted_request.scope.session_id,
                                )
                            )
                    binding = WorkflowBinding(
                        response_id=context.response_id,
                        checkpoint_id=checkpoint_id,
                        lineage_id=lineage_id,
                        workflow_name=workflow.name,
                        graph_hash=workflow.graph_signature_hash,
                        session_id=hosted_request.scope.session_id,
                        conversation_id=context.conversation_id,
                    )
                    await bindings.save_response(binding, recovery_from=recovery_from)
                    should_advance = True
                    if context.is_recovery:
                        head, expected_etag = (
                            await bindings.get_conversation_head(context.conversation_id)
                            if context.conversation_id is not None
                            else await bindings.get_lineage_head(lineage_id)
                        )
                        if head is not None:
                            if head.response_id == context.response_id:
                                if head.checkpoint_id == checkpoint_id:
                                    should_advance = False
                                elif head.checkpoint_id != recovery_from:
                                    raise RuntimeError("Another response advanced this workflow during recovery.")
                            elif prior is None or head.response_id != prior.response_id:
                                raise RuntimeError("Another turn advanced this workflow during recovery.")
                    if should_advance:
                        if context.conversation_id is not None:
                            await bindings.advance_conversation(
                                context.conversation_id, binding, expected_etag=expected_etag
                            )
                        else:
                            await bindings.advance_lineage(lineage_id, binding, expected_etag=expected_etag)

            for event in tracker.close():
                yield event
            if cancellation_signal.is_set() and context.client_cancelled:
                return
            if tracker.oauth_consent_requested or tracker.incomplete_reason is not None:
                yield response_event_stream.emit_incomplete(reason=tracker.incomplete_reason, usage=tracker.usage)
            else:
                yield response_event_stream.emit_completed(usage=tracker.usage)
        except Exception as exc:
            logger.error("Failed to produce response for workflow", exc_info=(type(exc), exc, exc.__traceback__))
            for event in self._emit_failure(response_event_stream, tracker, exc):
                yield event

    async def _handle_prepared_response(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
        response_event_stream: ResponseEventStream,
        agent: SupportsAgentRun,
        configuration: _AgentConfiguration,
        resources: AsyncExitStack,
        hosted_request: HostedResponseRequest,
    ) -> AsyncGenerator[ResponseStreamEvent | ResponseCheckpointEvent]:
        # Lazy-enter the agent (and any MCP tools it owns). The MCP client wraps gateway
        # consent failures (and other connection-time errors) in AgentFrameworkException; if
        # one of those is a consent error we surface the consent link to the client through
        # the already-opened response stream instead of failing the request. Other exception
        # types fall through to the outer handler below and become ``response.failed``.
        try:
            if self._configuration is not None:
                await self._ensure_agent_ready()
            elif isinstance(agent, AbstractAsyncContextManager):
                await resources.enter_async_context(agent)
        except AgentFrameworkException as ex:
            consent_errors_to_emit = consent_url_from_error(ex)
            if consent_errors_to_emit is None or len(consent_errors_to_emit) == 0:
                logger.error("Failed to prepare agent: %s", ex, exc_info=(type(ex), ex, ex.__traceback__))
                for event in self._emit_failure(response_event_stream, None, ex):
                    yield event
                return

            invalid_consent = next(
                (
                    consent_error
                    for consent_error in consent_errors_to_emit
                    if not _is_safe_oauth_consent_link(consent_error.consent_url)
                ),
                None,
            )
            if invalid_consent is not None:
                validation_error = ValueError(
                    f"OAuth consent request for tool '{invalid_consent.name}' must include a safe HTTPS consent link."
                )
                logger.error("%s", validation_error)
                for event in self._emit_failure(response_event_stream, None, validation_error):
                    yield event
                return

            if request.get("store") is False:
                for event in self._emit_failure(
                    response_event_stream,
                    None,
                    ValueError("OAuth consent continuation requires store=true."),
                ):
                    yield event
                return

            if not configuration.workflow:
                try:
                    request_context = get_request_context()
                    session_storage = self._session_storage_provider.get_store(
                        config=self.config, platform_context=request_context
                    )
                    previous_response_id = request.get("previous_response_id")
                    session_load_id = context.conversation_id or previous_response_id
                    session = await session_storage.get(session_load_id) if session_load_id is not None else None
                    if session is None:
                        if previous_response_id is not None and context.conversation_id is None:
                            raise RuntimeError(
                                "Cannot find an existing agent session for "
                                f"previous_response_id={previous_response_id}."
                            )
                        session = agent.create_session()
                    if previous_response_id is not None and context.conversation_id is None:
                        if session.service_session_id is not None and session.state.get(
                            _HOSTED_SOURCE_CONVERSATION_KEY
                        ):
                            raise ValueError("A service-managed downstream conversation cannot be forked.")
                        session.state.pop(_HOSTED_SOURCE_CONVERSATION_KEY, None)
                    if context.conversation_id is not None:
                        session.state[_HOSTED_SOURCE_CONVERSATION_KEY] = context.conversation_id
                    await session_storage.set(context.response_id, session)
                    if context.conversation_id is not None:
                        await session_storage.set(context.conversation_id, session)
                except Exception as save_error:
                    logger.error(
                        "Failed to persist the Agent Framework session for OAuth consent",
                        exc_info=(type(save_error), save_error, save_error.__traceback__),
                    )
                    for event in self._emit_failure(response_event_stream, None, save_error):
                        yield event
                    return

            for consent_error in consent_errors_to_emit:
                logger.warning("Consent URL for tool '%s': %s", consent_error.name, consent_error.consent_url)
                oauth_item = OAuthConsentRequestOutputItem(
                    id=IdGenerator.new_id("oacr"),
                    response_id=context.response_id,
                    type="oauth_consent_request",
                    consent_link=consent_error.consent_url,
                    server_label=consent_error.name,
                )
                builder = response_event_stream.add_output_item(oauth_item["id"])
                yield builder.emit_added(oauth_item)
                yield builder.emit_done(oauth_item)

            yield response_event_stream.emit_incomplete()
            return

        tracker = _OutputItemTracker(response_event_stream)
        try:
            if configuration.workflow:
                inner = self._handle_inner_workflow(
                    request,
                    context,
                    response_event_stream,
                    tracker,
                    cancellation_signal,
                    cast(WorkflowAgent, agent),
                )
            else:
                inner = self._handle_inner_agent(
                    request,
                    context,
                    response_event_stream,
                    tracker,
                    cancellation_signal,
                    agent,
                    configuration,
                    hosted_request,
                )

            try:
                async for event in inner:
                    yield event
            except BaseException:
                await inner.aclose()
                raise

            if cancellation_signal.is_set() and context.client_cancelled:
                # A cancelled run drains the inner generator without raising (both
                # ``_handle_inner_workflow`` and ``_handle_inner_agent`` stop their
                # ``_SignalledIterator`` loop and return normally once the signal fires).
                # Emit nothing here so a caller cannot mistake this for a normal
                # completion; the host server's cancel-aware layer synthesizes the
                # cancelled terminal when the handler returns without one. Gated on
                # ``client_cancelled`` (not just the signal) because steering pressure
                # also sets ``cancellation_signal`` without that cause flag; a steered
                # turn must still drain ``tracker.close()`` and emit its normal terminal
                # below so its partial output is not misreported as a failure.
                return

            for event in tracker.close():
                yield event

            if cancellation_signal.is_set() and context.client_cancelled:
                # Draining ``tracker.close()`` yields events one at a time, and each
                # ``yield`` above suspends this handler until the caller resumes it.
                # A cancellation can arrive during that window, after the earlier check
                # already passed, so it must be rechecked here, immediately before
                # selecting the terminal event. Same ``client_cancelled`` gate as above.
                return

            incomplete_reason = tracker.incomplete_reason
            if tracker.oauth_consent_requested or incomplete_reason is not None:
                yield response_event_stream.emit_incomplete(reason=incomplete_reason, usage=tracker.usage)
            else:
                yield response_event_stream.emit_completed(usage=tracker.usage)
        except Exception as ex:
            logger.error("Failed to produce response for agent", exc_info=(type(ex), ex, ex.__traceback__))
            for event in tracker.close():
                yield event

            for event in self._emit_failure(response_event_stream, tracker, ex):
                yield event

    async def _load_request_messages(
        self,
        context: ResponseContext,
        *,
        approval_storage: FunctionApprovalStore | None,
        configuration: _AgentConfiguration | None = None,
    ) -> list[Message]:
        """Load the request's input and prior history concurrently, assembled for the run.

        The caller's input items and the conversation history are independent
        storage round-trips with no data dependency, so they are fetched in
        parallel to remove serial latency from the request critical path. The
        history read is only issued when AgentServer is the history source; in
        stateless single-turn requests it short-circuits without a round-trip.

        Returns the messages already ordered as model input (history precedes
        input), so the message-ordering rule lives only here and callers do not
        need to know the storage-result ordering. If either read fails, the
        sibling task is cancelled and drained so no storage read is orphaned.
        """
        uses_agent_server_history = (
            configuration.agent_server_history if configuration is not None else self._uses_agent_server_history
        )

        async def _load_input() -> list[Message]:
            input_items = await context.get_input_items()
            return await _items_to_messages(input_items, approval_storage=approval_storage)

        async def _load_history() -> list[Message]:
            if not uses_agent_server_history:
                return []
            history = await context.get_history()
            return await _output_items_to_messages(history, approval_storage=approval_storage)

        input_task = asyncio.ensure_future(_load_input())
        history_task = asyncio.ensure_future(_load_history())
        try:
            input_messages, history_messages = await asyncio.gather(input_task, history_task)
        except BaseException:
            # gather surfaces the first failure without cancelling the sibling, and a
            # cancellation of this coroutine must not leave either read running. Cancel
            # both and await them so no storage operation is orphaned after we unwind.
            input_task.cancel()
            history_task.cancel()
            await asyncio.gather(input_task, history_task, return_exceptions=True)
            raise
        return [*history_messages, *input_messages]

    async def _handle_inner_agent(
        self,
        request: CreateResponse,
        context: ResponseContext,
        response_event_stream: ResponseEventStream,
        tracker: _OutputItemTracker,
        cancellation_signal: asyncio.Event,
        agent: SupportsAgentRun,
        configuration: _AgentConfiguration,
        hosted_request: HostedResponseRequest,
    ) -> AsyncGenerator[ResponseStreamEvent]:
        """Handle a regular (non-workflow) agent.

        The response stream, tracker, and opening lifecycle events are produced
        by :meth:`_handle_response`, which also converts any raised exception
        into a terminal ``response.failed`` event (draining the tracker so the
        SSE stream stays well-formed).
        """
        provider_background = self._inner_background == "provider" and request.get("background") is True
        if context.is_recovery and not provider_background:
            raise RuntimeError("A non-resumable agent cannot be replayed after a process crash.")

        stored = request.get("store") is not False
        request_messages_task: asyncio.Task[list[Message]] | None = None
        try:
            request_context = get_request_context()
            approval_storage = (
                self._function_approval_storage_provider.get_store(config=self.config, platform_context=request_context)
                if stored
                else None
            )
            session_storage = (
                self._session_storage_provider.get_store(config=self.config, platform_context=request_context)
                if stored
                else None
            )

            # Load the caller's input items and prior conversation history concurrently with the
            # session load below. These are independent storage round-trips with no data dependency
            # between them, so overlapping them removes serial latency from the request critical path.
            request_messages_task = asyncio.ensure_future(
                self._load_request_messages(
                    context,
                    approval_storage=approval_storage,
                    configuration=configuration,
                )
            )

            previous_response_id = request.get("previous_response_id")
            session_load_id = (
                context.response_id
                if context.is_recovery and provider_background
                else context.conversation_id or previous_response_id
            )
            session = (
                await session_storage.get(session_load_id)
                if session_storage is not None and session_load_id is not None
                else None
            )
            if session is None:
                if context.is_recovery and provider_background:
                    raise RuntimeError("Provider background was interrupted before its continuation token was stored.")
                if stored and previous_response_id is not None and context.conversation_id is None:
                    raise RuntimeError(
                        f"Cannot find an existing agent session for previous_response_id={previous_response_id}."
                    )
                session = agent.create_session()
            if not stored:
                if not isinstance(agent, RawAgent):
                    raise RuntimeError("A custom agent cannot guarantee store=false for its inner model.")
                continuation_defaults = [
                    name
                    for name in ("conversation_id", "previous_response_id", "conversation")
                    if agent.default_options.get(name) is not None
                ]
                if continuation_defaults:
                    raise RuntimeError(
                        "store=false cannot use developer defaults for downstream service continuation: "
                        + ", ".join(continuation_defaults)
                    )
            if previous_response_id is not None and context.conversation_id is None:
                if session.service_session_id is not None and session.state.get(_HOSTED_SOURCE_CONVERSATION_KEY):
                    raise ValueError("A service-managed downstream conversation cannot be forked.")
                session.state.pop(_HOSTED_SOURCE_CONVERSATION_KEY, None)
        except BaseException as ex:
            # Session preparation failed (or the request was cancelled / the stream closed —
            # neither of which is an Exception). Cancel and drain the in-flight message-loading
            # task so it is not orphaned, and log only ordinary failures.
            if request_messages_task is not None:
                request_messages_task.cancel()
                with suppress(BaseException):
                    await request_messages_task
            if isinstance(ex, Exception):
                logger.error("Failed to prepare state storage: %s", ex, exc_info=(type(ex), ex, ex.__traceback__))
            raise

        request_failure: Exception | None = None
        save_failure: Exception | None = None
        request_interrupted = False

        try:
            if configuration.agent_server_history:
                session.state.pop(_HOSTED_RESPONSES_HISTORY_SOURCE_ID, None)
                # A restored service ID belongs to the downstream model service. Replaying the
                # AgentServer transcript while resuming that service history would duplicate every
                # prior turn, so AgentServer-history mode always starts the model call statelessly.
                session.service_session_id = None

            messages = await request_messages_task
            run_kwargs: dict[str, Any] = {
                "messages": messages,
                "session": session,
            }
            chat_options = cast(ChatOptions[Any], dict(hosted_request.options))
            are_options_set = bool(chat_options)
            if configuration.agent_server_history:
                if configuration.client_stores_by_default:
                    # The response provider already owns the transcript used for this run. Keep a
                    # storing downstream service stateless so it cannot become a second history source.
                    chat_options["store"] = False
                else:
                    # Do not pass a storage option to clients that do not advertise support for it.
                    chat_options.pop("store", None)
            elif self._inner_history == "service":
                chat_options["store"] = stored
                if not stored:
                    session.service_session_id = None
            else:
                chat_options["store"] = False
                session.service_session_id = None

            if isinstance(agent, RawAgent):
                run_kwargs["options"] = chat_options
            elif are_options_set:
                if self._unsupported_options == "error":
                    raise TypeError("The hosted agent does not accept caller runtime options.")
                if self._unsupported_options == "warn":
                    logger.warning("Agent doesn't support runtime options. They will be ignored.")

            if provider_background:
                if session_storage is None or not isinstance(agent, RawAgent):
                    raise RuntimeError("Provider background requires a stored MAF agent session.")
                updates = self._provider_background_updates(
                    agent=cast(RawAgent[ChatOptions[Any]], agent),
                    messages=messages,
                    session=session,
                    session_storage=session_storage,
                    options=chat_options,
                    context=context,
                    cancellation_signal=cancellation_signal,
                )
            else:
                updates = _SignalledIterator(
                    agent.run(stream=True, **run_kwargs),  # type: ignore[reportUnknownMemberType]
                    context.shutdown,
                    cancellation_signal,
                )
            last_continuation_token: object | None = None
            async with aclosing(updates):
                async for update in updates:
                    if not provider_background:
                        last_continuation_token = update.continuation_token
                    if not stored and any(
                        content.type in ("function_approval_request", "oauth_consent_request")
                        or content.user_input_request
                        for content in update.contents
                    ):
                        raise ValueError("Approval and user-input continuation requires store=true.")
                    async for event in tracker.handle_update(update, approval_storage=approval_storage):
                        yield event
            if (
                isinstance(updates, _SignalledIterator)
                and not updates.signalled
                and last_continuation_token is not None
            ):
                raise RuntimeError(
                    "The inner agent returned an unfinished provider response; "
                    "configure inner_background='provider' to resume it."
                )
        except (asyncio.CancelledError, GeneratorExit):
            request_interrupted = True
            raise
        except Exception as ex:
            request_failure = ex
            logger.error(
                "Failed to produce response for agent",
                exc_info=(type(ex), ex, ex.__traceback__),
            )
        finally:
            if configuration.hosted_history:
                session.state.pop(_HOSTED_RESPONSES_HISTORY_SOURCE_ID, None)

            # A service ID here means the client stored the turn despite the forced `store=False`.
            # Do not persist a session that could resume that unreconciled history on a later turn.
            stored_output_violation = configuration.agent_server_history and session.service_session_id is not None
            if stored_output_violation:
                misconfigured = RuntimeError(
                    "The agent's chat client stored this turn server-side while AgentServer response history "
                    "is supplying the conversation. Configure the client to honor store=False, or construct "
                    "ResponsesHostServer with history_source='agent' to use the agent's regular history setup."
                )
                logger.error("%s", misconfigured)
                if request_failure is None and not request_interrupted:
                    request_failure = misconfigured
            try:
                final_provider_state = not provider_background or (
                    request_failure is None
                    and not request_interrupted
                    and _HOSTED_PROVIDER_STATE_KEY not in session.state
                )
                superseded_by_steering = bool(self._host_options and self._host_options.steerable_conversations) and (
                    cancellation_signal.is_set() and not context.client_cancelled and not context.shutdown.is_set()
                )
                if session_storage is not None and not stored_output_violation and final_provider_state:
                    if context.conversation_id is not None:
                        session.state[_HOSTED_SOURCE_CONVERSATION_KEY] = context.conversation_id
                    await session_storage.set(context.response_id, session)
                    if context.conversation_id is not None and not superseded_by_steering:
                        await session_storage.set(context.conversation_id, session)
            except Exception as save_error:
                save_failure = save_error
                if request_interrupted:
                    message = "Failed to persist the Agent Framework session while unwinding an interrupted request"
                elif request_failure is not None:
                    message = "Failed to persist the Agent Framework session after an agent failure"
                else:
                    message = "Failed to persist the Agent Framework session after a successful request"
                logger.error(message, exc_info=(type(save_error), save_error, save_error.__traceback__))

        if request_failure is not None and save_failure is not None:
            raise RuntimeError(
                f"Agent request failed: {str(request_failure) or type(request_failure).__name__}; "
                f"session persistence also failed: {str(save_failure) or type(save_failure).__name__}"
            )
        elif request_failure is not None:
            raise request_failure
        elif save_failure is not None:
            raise save_failure

    async def _provider_background_updates(
        self,
        *,
        agent: RawAgent[ChatOptions[Any]],
        messages: list[Message],
        session: AgentSession,
        session_storage: SessionStore,
        options: ChatOptions[Any],
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncGenerator[AgentResponseUpdate]:
        """Poll an inner provider token while AgentServer owns the outer background response."""
        if context.is_recovery:
            saved = session.state.get(_HOSTED_PROVIDER_STATE_KEY)
            if not isinstance(saved, Mapping):
                raise RuntimeError("Cannot recover a provider background job without its stored continuation token.")
            saved_payload = cast(Mapping[str, Any], saved)
            if saved_payload.get("outer_response_id") != context.response_id:
                raise RuntimeError("Cannot recover a provider background job without its stored continuation token.")
            token = saved_payload.get("continuation_token")
            if not isinstance(token, Mapping):
                raise RuntimeError("The stored provider continuation token is invalid.")
            continuation_token: Mapping[str, Any] = cast(Mapping[str, Any], token)
        else:
            first = await agent.run(
                messages,
                session=session,
                options=cast(ChatOptions[Any], {**options, "background": True}),
            )
            if first.continuation_token is None:
                for update in _agent_response_updates(first, context.response_id):
                    yield update
                return
            first_token = first.continuation_token
            if not isinstance(first_token, Mapping):
                raise RuntimeError("The provider returned a continuation token that cannot be persisted.")
            continuation_token = cast(Mapping[str, Any], first_token)
            session.state[_HOSTED_PROVIDER_STATE_KEY] = {
                "outer_response_id": context.response_id,
                "continuation_token": dict(continuation_token),
            }
            await session_storage.set(context.response_id, session)

        while True:
            if context.shutdown.is_set() and self._resilient_background:
                await context.exit_for_recovery()
            if cancellation_signal.is_set() and context.client_cancelled:
                return
            await asyncio.sleep(2)
            current = await agent.run(
                session=session,
                options=cast(ChatOptions[Any], {"continuation_token": continuation_token, "store": True}),
            )
            if current.continuation_token is None:
                session.state.pop(_HOSTED_PROVIDER_STATE_KEY, None)
                await session_storage.set(context.response_id, session)
                for update in _agent_response_updates(current, context.response_id):
                    yield update
                return
            if not isinstance(current.continuation_token, Mapping):
                raise RuntimeError("The provider returned a continuation token that cannot be persisted.")
            continuation_token = cast(Mapping[str, Any], current.continuation_token)
            session.state[_HOSTED_PROVIDER_STATE_KEY] = {
                "outer_response_id": context.response_id,
                "continuation_token": dict(continuation_token),
            }
            await session_storage.set(context.response_id, session)

    async def _handle_inner_workflow(
        self,
        request: CreateResponse,
        context: ResponseContext,
        response_event_stream: ResponseEventStream,
        tracker: _OutputItemTracker,
        cancellation_signal: asyncio.Event,
        agent: WorkflowAgent,
    ) -> AsyncGenerator[ResponseStreamEvent | ResponseCheckpointEvent]:
        """Handle the creation of a response for a workflow agent."""
        try:
            request_context = get_request_context()
            approval_storage = self._function_approval_storage_provider.get_store(
                config=self.config, platform_context=request_context
            )
            input_items = await context.get_input_items()
            input_messages = await _items_to_messages(input_items, approval_storage=approval_storage)

            _, are_options_set = _to_chat_options(request)
            if are_options_set:
                logger.warning("Workflow agent doesn't support runtime options. They will be ignored.")

            # Determine the checkpoint storage for this request. The checkpoint
            # storage is keyed by the conversation ID (if present) or the response
            # ID (if no conversation ID is present). On a subsequent turn, the same
            # conversation ID or a `previous_response_id` can be used to resume the
            # workflow from the last checkpoint.
            checkpoint_save_id = context.conversation_id or context.response_id
            _validate_checkpoint_context_id(checkpoint_save_id)
            checkpoint_storage = self._checkpoint_storage_provider.get_store(
                config=self.config,
                context_id=checkpoint_save_id,
                platform_context=request_context,
            )

            if context.is_recovery:
                if not self._resilient_background:
                    raise RuntimeError("Recovery mode is only supported when resilient_background=True.")
                # Resume from the workflow checkpoint durably paired with the last persisted response
                # snapshot (recorded in that snapshot's own metadata) -- NOT simply the latest workflow
                # checkpoint in storage, which may be ahead of what response.output actually reflects if
                # the crash happened between two response-stream checkpoint() calls.
                checkpoint_id = response_event_stream.internal_metadata.get(_LATEST_CHECKPOINT_ID_KEY)
                if checkpoint_id is not None:
                    logger.debug("Serving recovery request from workflow checkpoint %s", checkpoint_id)
                    run_stream = self._resume_workflow_from_checkpoint(
                        checkpoint_id, checkpoint_storage, context.response_id, agent
                    )
                else:
                    raise RuntimeError("Cannot recover a workflow without a checkpoint paired with persisted output.")
            else:
                # Determine the latest checkpoint (if any) so we can resume the
                # workflow's prior state for this turn. The directory is keyed by
                # the conversation id or the previous response id.
                previous_response_id = request.get("previous_response_id")
                if previous_response_id is not None and context.conversation_id is not None:
                    raise RuntimeError("Previous response ID cannot be used in conjunction with conversation ID.")
                checkpoint_load_id = context.conversation_id or previous_response_id
                restore_checkpoint_storage = checkpoint_storage
                if checkpoint_load_id is not None:
                    _validate_checkpoint_context_id(checkpoint_load_id)
                    if checkpoint_load_id != checkpoint_save_id:
                        restore_checkpoint_storage = self._checkpoint_storage_provider.get_store(
                            config=self.config,
                            context_id=checkpoint_load_id,
                            platform_context=request_context,
                        )
                latest_checkpoint = await restore_checkpoint_storage.get_latest(workflow_name=agent.workflow.name)

                if latest_checkpoint is None and previous_response_id is not None:
                    # A previous_response_id must have a prior workflow checkpoint to resume from
                    raise RuntimeError(
                        f"Cannot find an existing workflow checkpoint for previous_response_id={previous_response_id}."
                    )

                if latest_checkpoint is not None:
                    # If we have a prior checkpoint, restore it first (drive the workflow
                    # back to idle with prior state intact), then make a separate call that
                    # delivers the new user input. The restore-only call may yield events
                    # from any pending in-flight work in the checkpoint; we consume those
                    # internally here so they don't surface to the response stream as duplicates.
                    #
                    # If the restored checkpoint had pending request_info events, the
                    # restore-only call replays them through
                    # ``WorkflowAgent._convert_workflow_event_to_agent_response_updates``
                    # and populates ``agent.pending_requests``. That is the correct
                    # state: those requests are genuinely outstanding, and the next
                    # ``run(input_messages, ...)`` call may contain ``function_call_output``
                    # items (carried as FunctionResult/FunctionApprovalResponse content)
                    # that fulfill them via :meth:`WorkflowAgent._process_pending_requests`.
                    restore_iter = _SignalledIterator(
                        agent.run(
                            stream=True,
                            checkpoint_id=latest_checkpoint.checkpoint_id,
                            checkpoint_storage=restore_checkpoint_storage,
                        ),
                        context.shutdown,
                        cancellation_signal,
                    )
                    async with aclosing(restore_iter):
                        async for _ in restore_iter:
                            pass
                    if restore_iter.signalled:
                        if context.shutdown.is_set():
                            await context.exit_for_recovery()
                        if cancellation_signal.is_set():
                            return

                # A cancel signal that fired after the restore-only replay finished (or was never
                # entered) must still preempt starting a brand new workflow run below.
                if cancellation_signal.is_set():
                    return

                run_stream = agent.run(
                    input_messages,
                    stream=True,
                    checkpoint_storage=checkpoint_storage,
                )

            workflow_name = agent.workflow.name

            async def latest_checkpoint_id() -> str | None:
                latest = await checkpoint_storage.get_latest(workflow_name=workflow_name)
                return latest.checkpoint_id if latest is not None else None

            def snapshot_response(
                checkpoint_id: str | None,
            ) -> Generator[ResponseStreamEvent | ResponseCheckpointEvent]:
                # Pair the response output emitted so far with the workflow checkpoint it corresponds
                # to, so recovery from that checkpoint replays exactly the updates that came after it.
                if checkpoint_id is None or checkpoint_id == response_event_stream.internal_metadata.get(
                    _LATEST_CHECKPOINT_ID_KEY
                ):
                    return
                yield from tracker.close()
                response_event_stream.internal_metadata[_LATEST_CHECKPOINT_ID_KEY] = checkpoint_id
                yield response_event_stream.checkpoint()

            main_iter = _SignalledIterator(
                run_stream,
                context.shutdown,
                cancellation_signal,
                # The runner creates a checkpoint at the end of each superstep, inside the generator
                # that produces the updates (see RunnerImpl.run_until_convergence). Stamping each
                # update with the latest checkpoint as it was produced tells which checkpoint the
                # update follows; the driver runs one update ahead, so by the time an update is
                # consumed the workflow may already have checkpointed past it.
                stamp=latest_checkpoint_id if self._resilient_background else None,
            )
            async with aclosing(main_iter):
                async for update in main_iter:
                    if self._resilient_background:
                        # Every update before this one belongs to the stamped checkpoint (or an
                        # earlier one), so the output so far can be snapshotted against it. If the
                        # workflow crashes before any update is produced, no snapshot is taken and
                        # recovery still resumes from the latest workflow checkpoint.
                        for event in snapshot_response(main_iter.stamp):
                            yield event

                    async for event in tracker.handle_update(update, approval_storage=approval_storage):
                        yield event
            # Cancellation needs no extra action here (the loop above already stopped); shutdown
            # does, but only if it's what actually stopped the loop, not a natural completion.
            if main_iter.signalled and context.shutdown.is_set():
                await context.exit_for_recovery()
            elif self._resilient_background and not main_iter.signalled:
                # The workflow ran to completion: pair its final checkpoint with the full output, so
                # recovery after this point does not replay the last superstep.
                for event in snapshot_response(await latest_checkpoint_id()):
                    yield event
        except Exception:
            logger.exception("Failed to produce response for workflow agent")
            raise

    async def _resume_workflow_from_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_storage: CheckpointStorage,
        response_id: str,
        agent: WorkflowAgent,
    ) -> AsyncGenerator[AgentResponseUpdate]:
        """Resume a crashed background workflow run, forwarding every event it produces.

        ``WorkflowAgent.run(checkpoint_id=..., messages=None)`` treats a message-less resume as
        "restore only": it drives the workflow with the checkpoint's own already-queued internal
        messages, but silently discards every event produced while doing so, on the assumption
        that the workflow merely settles back to idle awaiting the next turn's input. That
        assumption doesn't hold for crash recovery: the countdown (and any other self-driving
        workflow) genuinely continues -- and may run to completion -- from its own queued
        messages, and that output must not be lost. Drive the underlying ``Workflow`` directly so
        none of it is discarded, converting each event the same way ``WorkflowAgent.run`` does.

        TODO(@taochen): #7677
        """
        async for event in agent.workflow.run(
            stream=True,
            checkpoint_id=checkpoint_id,
            checkpoint_storage=checkpoint_storage,
        ):
            for update in agent._convert_workflow_event_to_agent_response_updates(  # pyright: ignore[reportPrivateUsage]
                response_id, event
            ):
                yield update

    @staticmethod
    def _emit_failure(
        response_event_stream: ResponseEventStream,
        tracker: _OutputItemTracker | None,
        ex: BaseException,
    ) -> Generator[ResponseStreamEvent]:
        """Yield a terminal ``response.failed`` event for ``ex``.

        Drains any in-progress streaming output item first so the resulting
        SSE stream stays well-formed, then emits ``response.failed`` carrying
        the exception's message (falling back to the exception type name when
        ``str(ex)`` is empty). Any error raised while draining the tracker is
        logged and otherwise ignored so that the original failure is always
        what the client sees.
        """
        if tracker is not None:
            try:
                yield from tracker.close()
            except Exception:
                logger.exception("Error while closing streaming tracker after failure")
        message = str(ex) or type(ex).__name__
        yield response_event_stream.emit_failed(message=message, usage=tracker.usage if tracker is not None else None)


# endregion ResponsesHostServer

# region Active Builder State


class _OutputItemTracker:
    """Converts a stream of agent ``Content`` into ``ResponseStreamEvent``s for one response.

    For content types that arrive as a series of deltas (text, reasoning, function calls, MCP
    calls) it tracks the single currently-open output item builder, merging consecutive same-item
    deltas and closing the builder (emitting its `*_done` events) as soon as a different item
    starts. All other content types (function results, image generation, shell calls/results,
    approval requests, etc.) are emitted in one shot, closing any still-open streaming item first.
    """

    def __init__(self, stream: ResponseEventStream) -> None:
        self._stream = stream
        self._usage_details: UsageDetails | None = None
        self._active_type: str | None = None
        self._active_id: str | None = None
        # message_id of the update that opened the active text item, used to detect a new
        # logical message (e.g. a fresh workflow yield_output call) even when the content
        # type doesn't change, so it isn't silently merged into the still-open item.
        self._active_message_id: str | None = None
        # Accumulated delta text for the current active builder
        self._accumulated: list[str] = []
        # Builder state — only one is active at a time
        self._message_item: OutputItemMessageBuilder | None = None
        self._text_content: TextContentBuilder | None = None
        self._refusal_content: RefusalContentBuilder | None = None
        self._reasoning_item: OutputItemBuilder | None = None
        self._summary_part: ReasoningSummaryPartBuilder | None = None
        self._reasoning_encrypted_content: str | None = None
        self._fc_builder: OutputItemFunctionCallBuilder | None = None
        self._mcp_builder: OutputItemMcpCallBuilder | None = None
        self._outstanding_function_calls: dict[str, str | None] = {}
        self._oauth_consent_requests: set[tuple[str, str]] = set()
        # Set when an agent update reports the model stopped early (content filter, token
        # limit); the response then ends as ``incomplete`` instead of ``completed`` so callers
        # can tell a cut-short turn from a successful one. Mirrored into the stream's
        # ``internal_metadata`` so it survives a resilient checkpoint/recovery cycle, which
        # rebuilds this tracker from the persisted response.
        self._incomplete_reason: ResponseIncompleteReason | None = None
        persisted_reason = stream.internal_metadata.get(_INCOMPLETE_REASON_KEY)
        if isinstance(persisted_reason, str):
            with suppress(ValueError):
                self._incomplete_reason = ResponseIncompleteReason(persisted_reason)
        for item in stream.response.get("output", []):
            if not isinstance(item, Mapping):
                continue
            persisted_item = cast(Mapping[str, Any], item)
            if persisted_item.get("type") != "oauth_consent_request":
                continue
            consent_link = persisted_item.get("consent_link")
            server_label = persisted_item.get("server_label")
            if isinstance(consent_link, str) and isinstance(server_label, str):
                self._oauth_consent_requests.add((consent_link, server_label))

    @property
    def usage(self) -> ResponseUsage | None:
        """Return accumulated usage in the Responses API schema."""
        if self._usage_details is None:
            return None

        input_tokens = int(self._usage_details.get("input_token_count") or 0)
        output_tokens = int(self._usage_details.get("output_token_count") or 0)
        total_tokens = self._usage_details.get("total_token_count")
        return ResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=ResponseUsageInputTokensDetails(
                cached_tokens=int(self._usage_details.get("cache_read_input_token_count") or 0),
                cache_write_tokens=int(self._usage_details.get("cache_creation_input_token_count") or 0),
            ),
            output_tokens=output_tokens,
            output_tokens_details=ResponseUsageOutputTokensDetails(
                reasoning_tokens=int(self._usage_details.get("reasoning_output_token_count") or 0)
            ),
            total_tokens=int(total_tokens) if total_tokens is not None else input_tokens + output_tokens,
        )

    @property
    def oauth_consent_requested(self) -> bool:
        """Return whether this response emitted an OAuth consent request."""
        return bool(self._oauth_consent_requests)

    @property
    def incomplete_reason(self) -> ResponseIncompleteReason | None:
        """Return why the turn was cut short, if any update reported a truncating finish reason."""
        return self._incomplete_reason

    def record_finish_reason(self, finish_reason: str | None) -> None:
        """Note the finish reason of an agent update.

        Only finish reasons that mean the model stopped early are retained, mapped onto the
        Responses ``incomplete_details.reason`` vocabulary. A content filter is kept in
        preference to a token limit if both are seen during a multi-step turn, since it is the
        more actionable signal for the caller.
        """
        if finish_reason == "content_filter":
            self._incomplete_reason = ResponseIncompleteReason.CONTENT_FILTER
        elif finish_reason == "length" and self._incomplete_reason is None:
            self._incomplete_reason = ResponseIncompleteReason.MAX_OUTPUT_TOKENS
        else:
            return
        self._stream.internal_metadata[_INCOMPLETE_REASON_KEY] = self._incomplete_reason.value

    async def handle_update(
        self,
        update: AgentResponseUpdate,
        *,
        approval_storage: FunctionApprovalStore | None = None,
    ) -> AsyncGenerator[ResponseStreamEvent]:
        """Process one agent update: note its finish reason, then handle each of its contents.

        This is the single entry point for both the plain-agent and the workflow loops, so the
        finish reason cannot be forgotten on one of them.
        """
        self.record_finish_reason(update.finish_reason)
        for content in update.contents:
            async for event in self.handle(content, message_id=update.message_id, approval_storage=approval_storage):
                yield event

    async def handle(
        self,
        content: Content,
        message_id: str | None = None,
        *,
        approval_storage: FunctionApprovalStore | None = None,
    ) -> AsyncGenerator[ResponseStreamEvent]:
        """Process a content item, yielding its events.

        Args:
            content: The content item to process.
            message_id: The ``message_id`` of the update ``content`` came from, if any. A
                change in ``message_id`` across otherwise same-typed text content marks a new
                logical message and forces the previous output item closed, rather than being
                merged into it.
            approval_storage: Used for content types that fall back to one-shot emission
                (anything not recognized as a streaming delta type) to save/load approval requests.
        """
        if _is_refusal_text_content(content) and content.text is not None:
            for event in self._ensure_message_content("refusal", message_id):
                yield event
            self._active_message_id = message_id
            self._accumulated.append(content.text)
            if self._refusal_content is not None:
                yield self._refusal_content.emit_delta(content.text)

        elif content.type == "text" and content.text is not None:
            for event in self._ensure_message_content("text", message_id):
                yield event
            self._active_message_id = message_id
            self._accumulated.append(content.text)
            if self._text_content is not None:
                yield self._text_content.emit_delta(content.text)

        elif content.type == "text_reasoning":
            if self._active_type != "text_reasoning" or (content.id is not None and content.id != self._active_id):
                for event in self._close():
                    yield event
                for event in self._open_reasoning(content):
                    yield event
            if encrypted_content := _reasoning_encrypted_content(content):
                self._reasoning_encrypted_content = encrypted_content
            if content.text:
                self._accumulated.append(content.text)
                if self._summary_part is not None:
                    yield self._summary_part.emit_text_delta(content.text)

        elif content.type == "function_call" and content.call_id is not None:
            # Declaration-only calls replay request metadata after the streamed call. Scope suppression to the
            # outstanding occurrence because a call_id may be reused after its terminal result.
            if (
                content.user_input_request
                and content.arguments is None
                and content.call_id in self._outstanding_function_calls
                and self._outstanding_function_calls[content.call_id] == content.name
            ):
                return
            if self._active_type != "function_call" or self._active_id != content.call_id:
                for event in self._close():
                    yield event
                for event in self._open_function_call(content):
                    yield event
            args_str = _json_safe_to_str(content.arguments)
            self._accumulated.append(args_str)
            if self._fc_builder is not None:
                yield self._fc_builder.emit_arguments_delta(args_str)

        elif content.type == "function_result":
            for event in self._close():
                yield event
            async for event in self._stream.output_item_function_call_output(
                content.call_id,  # type: ignore[arg-type]
                _json_safe_to_str(content.result),
            ):
                yield event
            if content.call_id is not None:
                self._outstanding_function_calls.pop(content.call_id, None)

        elif content.type == "mcp_server_tool_call" and content.tool_name:
            key = content.call_id or f"{content.server_name or 'default'}::{content.tool_name}"
            if self._active_type != "mcp_server_tool_call" or self._active_id != key:
                for event in self._close():
                    yield event
                for event in self._open_mcp_call(content):
                    yield event
            args_str = _json_safe_to_str(content.arguments)
            self._accumulated.append(args_str)
            if self._mcp_builder is not None:
                yield self._mcp_builder.emit_arguments_delta(args_str)

        elif (
            content.type == "mcp_server_tool_result"
            and self._active_type == "mcp_server_tool_call"
            and self._mcp_builder is not None
            and content.call_id is not None
            and content.call_id == self._mcp_builder.item_id
        ):
            accumulated = "".join(self._accumulated)
            yield self._mcp_builder.emit_arguments_done(accumulated)
            yield self._mcp_builder.emit_completed()
            yield self._mcp_builder.emit_done(output=_stringify_mcp_output(content.output))
            self._mcp_builder = None
            self._active_type = None
            self._active_id = None
            self._accumulated.clear()
            return

        elif content.type == "image_generation_tool_result" and content.outputs is not None:
            for event in self._close():
                yield event
            async for event in self._stream.output_item_image_gen_call(str(content.outputs)):
                yield event

        elif content.type == "mcp_server_tool_call":
            # Reached only when `content.tool_name` is falsy (the streaming branch above didn't match).
            for event in self._close():
                yield event
            mcp_call = self._stream.add_output_item_mcp_call(
                server_label=content.server_name or "default",
                name=content.tool_name or "",
                item_id=content.call_id,
            )
            yield mcp_call.emit_added()
            async for event in mcp_call.arguments(_json_safe_to_str(content.arguments)):
                yield event
            yield mcp_call.emit_completed()
            yield mcp_call.emit_done()

        elif content.type == "mcp_server_tool_result":
            # Reached when there's no correlated in-progress mcp_server_tool_call to close against.
            for event in self._close():
                yield event
            output = _stringify_mcp_output(content.output)
            async for event in self._stream.output_item_custom_tool_call_output(content.call_id or "", output):
                yield event

        elif content.type == "shell_tool_call":
            for event in self._close():
                yield event
            action = FunctionShellAction(
                commands=content.commands or [],
                timeout_ms=content.timeout_ms,
                max_output_length=content.max_output_length,
            )
            async for event in self._stream.output_item_function_shell_call(
                content.call_id or "",
                action,
                LocalEnvironmentResource(type="local"),
                status=content.status or "completed",
            ):
                yield event

        elif content.type == "shell_tool_result":
            for event in self._close():
                yield event
            output_items: list[FunctionShellCallOutputContent] = []
            if content.outputs:
                for out in content.outputs:
                    exit_code = getattr(out, "exit_code", None)
                    output_items.append(
                        FunctionShellCallOutputContent(
                            stdout=getattr(out, "stdout", "") or "",
                            stderr=getattr(out, "stderr", "") or "",
                            outcome=FunctionShellCallOutputExitOutcome(
                                type="exit",
                                exit_code=exit_code if exit_code is not None else 0,
                            ),
                        )
                    )
            async for event in self._stream.output_item_function_shell_call_output(
                content.call_id or "",
                output_items,
                status=content.status or "completed",
                max_output_length=content.max_output_length,
            ):
                yield event

        elif content.type == "function_approval_request":
            for event in self._close():
                yield event
            function_call: Content = content.function_call  # type: ignore
            server_label = function_call.additional_properties.get("server_label", "agent_framework")
            request_saved = False
            async for event in self._stream.output_item_mcp_approval_request(
                server_label,
                function_call.name,  # type: ignore
                _json_safe_to_str(function_call.arguments),
            ):
                if approval_storage is not None and not request_saved:
                    # Extract the approval request ID generated by the infrastructure when the
                    # approval request item is added to the stream, and save it to approval
                    # storage so it can be retrieved later for round trips.
                    item = event.get("item") if isinstance(event, Mapping) else getattr(event, "item", None)
                    approval_request_id = (
                        cast(Mapping[str, Any], item).get("id")
                        if isinstance(item, Mapping)
                        else getattr(item, "id", None)
                    )
                    if isinstance(approval_request_id, str):
                        await approval_storage.save_approval_request(approval_request_id, content)
                        request_saved = True
                yield event
            if approval_storage is not None and not request_saved:
                logger.warning(
                    "Approval request was not saved to approval storage because the approval request ID "
                    "could not be extracted from the stream event."
                )

        elif content.type == "oauth_consent_request":
            for event in self._close():
                yield event

            consent_link = content.consent_link
            if not _is_safe_oauth_consent_link(consent_link):
                raise ValueError("OAuth consent request content must include a safe HTTPS consent link.")

            server_label = content.additional_properties.get("server_label")
            if not isinstance(server_label, str) or not server_label:
                server_label = getattr(content.raw_representation, "server_label", None)
            if not isinstance(server_label, str) or not server_label:
                server_label = "agent_framework"

            consent_key = (consent_link, server_label)
            if consent_key in self._oauth_consent_requests:
                return
            self._oauth_consent_requests.add(consent_key)

            oauth_item = OAuthConsentRequestOutputItem(
                id=IdGenerator.new_id("oacr"),
                response_id=str(self._stream.response["id"]),
                type="oauth_consent_request",
                consent_link=consent_link,
                server_label=server_label,
            )
            builder = self._stream.add_output_item(oauth_item["id"])
            yield builder.emit_added(oauth_item)
            yield builder.emit_done(oauth_item)

        elif content.type == "usage":
            self._usage_details = add_usage_details(self._usage_details, content.usage_details)

        else:
            for event in self._close():
                yield event
            # Defensive: covers content types not recognized above (e.g. "text"/"text_reasoning"/
            # "function_call" with missing required fields), logged instead of raised so the
            # response stream isn't broken by one unsupported content item.
            logger.warning(f"Content type '{content.type}' is not supported yet. This is usually safe to ignore.")

    def close(self) -> Generator[ResponseStreamEvent]:
        """Close any remaining active builder."""
        yield from self._close()

    # -- Private open/close helpers --

    def _ensure_message_content(
        self,
        content_type: Literal["text", "refusal"],
        message_id: str | None,
    ) -> Generator[ResponseStreamEvent]:
        message_changed = (
            message_id is not None and self._active_message_id is not None and message_id != self._active_message_id
        )
        if self._active_type == content_type and not message_changed:
            return
        if self._message_item is not None and self._active_type in {"text", "refusal"} and not message_changed:
            yield from self._close_message_content()
            yield from self._open_message_content(content_type)
            return
        yield from self._close()
        yield from self._open_message(content_type)

    def _open_message(self, content_type: Literal["text", "refusal"]) -> Generator[ResponseStreamEvent]:
        self._message_item = self._stream.add_output_item_message()
        yield self._message_item.emit_added()
        yield from self._open_message_content(content_type)

    def _open_message_content(
        self,
        content_type: Literal["text", "refusal"],
    ) -> Generator[ResponseStreamEvent]:
        if self._message_item is None:
            raise RuntimeError("Cannot open message content without an active message")
        self._active_type = content_type
        self._active_id = None
        if content_type == "refusal":
            self._refusal_content = self._message_item.add_refusal_content()
            yield self._refusal_content.emit_added()
        else:
            self._text_content = self._message_item.add_text_content()
            yield self._text_content.emit_added()

    def _open_reasoning(self, content: Content) -> Generator[ResponseStreamEvent]:
        item_id = content.id
        if not item_id or not IdGenerator.is_valid(item_id)[0]:
            item_id = IdGenerator.new_id("rs")
        self._reasoning_item = self._stream.add_output_item(item_id)
        self._summary_part = ReasoningSummaryPartBuilder(
            self._stream,
            self._reasoning_item.output_index,
            0,
            item_id,
        )
        self._reasoning_encrypted_content = _reasoning_encrypted_content(content)
        self._active_type = "text_reasoning"
        self._active_id = item_id
        yield self._reasoning_item.emit_added(
            _reasoning_output_item(
                item_id=item_id,
                summary_texts=[],
                encrypted_content=None,
                status="in_progress",
            )
        )
        yield self._summary_part.emit_added()

    def _open_function_call(self, content: Content) -> Generator[ResponseStreamEvent]:
        self._fc_builder = self._stream.add_output_item_function_call(
            name=content.name or "",
            call_id=content.call_id or "",
        )
        self._active_type = "function_call"
        self._active_id = content.call_id
        self._outstanding_function_calls[content.call_id or ""] = content.name
        yield self._fc_builder.emit_added()

    def _open_mcp_call(self, content: Content) -> Generator[ResponseStreamEvent]:
        self._mcp_builder = self._stream.add_output_item_mcp_call(
            server_label=content.server_name or "default",
            name=content.tool_name or "",
            item_id=content.call_id,
        )
        self._active_type = "mcp_server_tool_call"
        self._active_id = content.call_id or f"{content.server_name or 'default'}::{content.tool_name}"
        yield self._mcp_builder.emit_added()

    def _close(self) -> Generator[ResponseStreamEvent]:
        if self._active_type in {"text", "refusal"}:
            yield from self._close_message_content()
            if self._message_item is not None:
                yield self._message_item.emit_done()
            self._message_item = None

        elif self._active_type == "text_reasoning" and self._summary_part and self._reasoning_item:
            accumulated = "".join(self._accumulated)
            yield self._summary_part.emit_text_done(accumulated)
            yield self._summary_part.emit_done()
            yield self._reasoning_item.emit_done(
                _reasoning_output_item(
                    item_id=self._reasoning_item.item_id,
                    summary_texts=[accumulated],
                    encrypted_content=self._reasoning_encrypted_content,
                    status="completed",
                )
            )
            self._summary_part = None
            self._reasoning_item = None
            self._reasoning_encrypted_content = None

        elif self._active_type == "function_call" and self._fc_builder:
            accumulated = "".join(self._accumulated)
            yield self._fc_builder.emit_arguments_done(accumulated)
            yield self._fc_builder.emit_done()
            self._fc_builder = None

        elif self._active_type == "mcp_server_tool_call" and self._mcp_builder:
            accumulated = "".join(self._accumulated)
            yield self._mcp_builder.emit_arguments_done(accumulated)
            yield self._mcp_builder.emit_completed()
            yield self._mcp_builder.emit_done()
            self._mcp_builder = None

        self._active_type = None
        self._active_id = None
        self._active_message_id = None
        self._accumulated.clear()

    def _close_message_content(self) -> Generator[ResponseStreamEvent]:
        accumulated = "".join(self._accumulated)
        if self._active_type == "text" and self._text_content is not None:
            yield self._text_content.emit_text_done(accumulated)
            yield self._text_content.emit_done()
            self._text_content = None
        elif self._active_type == "refusal" and self._refusal_content is not None:
            yield self._refusal_content.emit_refusal_done(accumulated)
            yield self._refusal_content.emit_done()
            self._refusal_content = None
        self._active_type = None
        self._active_id = None
        self._accumulated.clear()


# endregion


# region Option Conversion


def _to_chat_options(request: CreateResponse) -> tuple[ChatOptions, bool]:
    """Converts a CreateResponse request to ChatOptions.

    Args:
        request (CreateResponse): The request to convert.

    Returns:
        ChatOptions: The converted ChatOptions.
        bool: Whether any options were set.

    """
    chat_options = ChatOptions()
    are_options_set = False

    if (temperature := request.get("temperature")) is not None:
        chat_options["temperature"] = temperature
        are_options_set = True
    if (top_p := request.get("top_p")) is not None:
        chat_options["top_p"] = top_p
        are_options_set = True
    if (max_output_tokens := request.get("max_output_tokens")) is not None:
        chat_options["max_tokens"] = max_output_tokens
        are_options_set = True
    if (parallel_tool_calls := request.get("parallel_tool_calls")) is not None:
        chat_options["allow_multiple_tool_calls"] = parallel_tool_calls
        are_options_set = True

    return chat_options, are_options_set


# endregion


# region Input Message Conversion


async def _items_to_messages(
    input_items: Sequence[Item], *, approval_storage: FunctionApprovalStore | None = None
) -> list[Message]:
    """Converts a sequence of input items to a list of Messages, one per item.

    Args:
        input_items: The input items to convert.
        approval_storage: An optional ApprovalStorage instance used to look up
            approval requests when converting MCP approval response items.

    Returns:
        A list of Messages, one per supported input item.
    """
    messages: list[Message] = []
    for item in input_items:
        messages.append(await _item_to_message(item, approval_storage=approval_storage))
    return messages


def _reasoning_item_to_contents(reasoning: ItemReasoningItem | OutputItemReasoningItem) -> list[Content]:
    """Convert a hosted reasoning item without losing its stateless replay metadata."""
    encrypted_content = reasoning.get("encrypted_content")
    if summary_parts := reasoning.get("summary"):
        return [
            Content.from_text_reasoning(
                id=reasoning["id"],
                text=summary["text"],
                protected_data=encrypted_content if index == 0 else None,
            )
            for index, summary in enumerate(summary_parts)
        ]
    return [Content.from_text_reasoning(id=reasoning["id"], protected_data=encrypted_content)]


async def _item_to_message(
    item: Item,
    *,
    approval_storage: FunctionApprovalStore | None = None,
    _item_type_name: Literal["Item", "OutputItem"] = "Item",
) -> Message:
    """Converts an Item to a Message.

    Args:
        item: The Item to convert.
        approval_storage: An optional ApprovalStorage instance used to look up
            approval requests when converting MCP approval response items.
        _item_type_name: The item type name to include in unsupported-type errors.

    Returns:
        The converted Message.

    Raises:
        ValueError: If the Item type is not supported.
    """
    if item["type"] == "message":
        if isinstance(item["content"], str):
            return Message(role=item["role"], contents=[Content.from_text(item["content"])])
        return Message(role=item["role"], contents=[_convert_message_content(part) for part in item["content"]])

    if item["type"] == "output_message":
        return Message(role=item["role"], contents=[_convert_output_message_content(part) for part in item["content"]])

    if item["type"] == "function_call":
        return Message(
            role="assistant",
            contents=[
                Content.from_function_call(
                    item["call_id"],
                    item["name"],
                    arguments=item["arguments"],
                )
            ],
        )

    if item["type"] == "function_call_output":
        call_id = item.get("call_id")
        if call_id is None:
            raise ValueError("Function call output item is missing a call_id.")
        return Message(
            role="tool",
            contents=[Content.from_function_result(call_id, result=_json_safe_to_str(item["output"]))],
        )

    if item["type"] == "reasoning":
        return Message(role="assistant", contents=_reasoning_item_to_contents(item))

    if item["type"] == "mcp_call":
        contents = [
            Content.from_mcp_server_tool_call(
                item["id"],
                item["name"],
                server_name=item["server_label"],
                arguments=item["arguments"],
            )
        ]
        if (output := item.get("output")) is not None:
            contents.append(Content.from_mcp_server_tool_result(call_id=item["id"], output=output))
        return Message(
            role="assistant",
            contents=contents,
        )

    if item["type"] == "mcp_approval_request":
        if approval_storage is not None:
            function_approval_request_content = await approval_storage.load_approval_request(item["id"])
        else:
            raise ValueError("ApprovalStorage is required to load approval request.")
        return Message(
            role="assistant",
            contents=[function_approval_request_content],
        )

    if item["type"] == "mcp_approval_response":
        if approval_storage is not None:
            function_approval_request_content = await approval_storage.load_approval_request(
                item["approval_request_id"]
            )
        else:
            raise ValueError("ApprovalStorage is required to load approval request.")
        return Message(
            role="user",
            contents=[function_approval_request_content.to_function_approval_response(item["approve"])],
        )

    if item["type"] == "code_interpreter_call":
        return Message(
            role="assistant",
            contents=[Content.from_code_interpreter_tool_call(call_id=item["id"])],
        )

    if item["type"] == "image_generation_call":
        return Message(
            role="assistant",
            contents=[Content.from_image_generation_tool_call(image_id=item["id"])],
        )

    if item["type"] == "shell_call":
        return Message(
            role="assistant",
            contents=[
                Content.from_shell_tool_call(
                    call_id=item["call_id"],
                    commands=item["action"]["commands"],
                    timeout_ms=item["action"].get("timeout_ms"),
                    max_output_length=item["action"].get("max_output_length"),
                    status=str(item.get("status")),
                )
            ],
        )

    if item["type"] == "shell_call_output":
        outputs = [
            Content.from_shell_command_output(
                stdout=out["stdout"] or "",
                stderr=out["stderr"] or "",
                exit_code=out["outcome"].get("exit_code"),
            )
            for out in (item["output"] or [])
        ]
        return Message(
            role="tool",
            contents=[
                Content.from_shell_tool_result(
                    call_id=item["call_id"],
                    outputs=outputs,
                    max_output_length=item.get("max_output_length"),
                )
            ],
        )

    if item["type"] == "local_shell_call":
        commands = item["action"].get("command") or []
        return Message(
            role="assistant",
            contents=[
                Content.from_shell_tool_call(
                    call_id=item["call_id"],
                    commands=commands,
                    timeout_ms=item["action"].get("timeout_ms"),
                    status=str(item["status"]),
                )
            ],
        )

    if item["type"] == "local_shell_call_output":
        return Message(
            role="tool",
            contents=[
                Content.from_shell_tool_result(
                    call_id=item["id"],
                    outputs=[Content.from_shell_command_output(stdout=item["output"])],
                )
            ],
        )

    if item["type"] == "file_search_call":
        return Message(
            role="assistant",
            contents=[
                Content.from_function_call(
                    item["id"],
                    "file_search",
                    arguments=_json_safe_to_str({"queries": item["queries"]}),
                    informational_only=True,
                )
            ],
        )

    if item["type"] == "web_search_call":
        return Message(
            role="assistant",
            contents=[Content.from_function_call(item["id"], "web_search", informational_only=True)],
        )

    if item["type"] == "computer_call":
        return Message(
            role="assistant",
            contents=[
                Content.from_function_call(
                    item["call_id"],
                    "computer_use",
                    arguments=_json_safe_to_str(item.get("action")),
                    informational_only=True,
                )
            ],
        )

    if item["type"] == "computer_call_output":
        return Message(
            role="tool",
            contents=[Content.from_function_result(item["call_id"], result=_json_safe_to_str(item["output"]))],
        )

    if item["type"] == "custom_tool_call":
        return Message(
            role="assistant",
            contents=[
                Content.from_function_call(
                    item["call_id"],
                    item["name"],
                    arguments=item["input"],
                    informational_only=True,
                )
            ],
        )

    if item["type"] == "custom_tool_call_output":
        output = _json_safe_to_str(item["output"])
        # Hosted-MCP results land here because the host writes them via
        # `aoutput_item_custom_tool_call_output` (see `_OutputItemTracker.handle` for
        # `mcp_server_tool_result`). The persisted `call_id` keeps its
        # `mcp_*` prefix; on read, route those back to a hosted-MCP result
        # Content so the chat-client serialize layer can coalesce them
        # onto a single `mcp_call` input item with `output` populated.
        # Issue #5546.
        if item["call_id"] and item["call_id"].startswith("mcp_"):
            return Message(
                role="tool",
                contents=[Content.from_mcp_server_tool_result(call_id=item["call_id"], output=output)],
            )
        return Message(
            role="tool",
            contents=[Content.from_function_result(item["call_id"], result=output)],
        )

    if item["type"] == "apply_patch_call":
        return Message(
            role="assistant",
            contents=[
                Content.from_function_call(
                    item["call_id"],
                    "apply_patch",
                    arguments=_json_safe_to_str(item["operation"]),
                    informational_only=True,
                )
            ],
        )

    if item["type"] == "apply_patch_call_output":
        return Message(
            role="tool",
            contents=[Content.from_function_result(item["call_id"], result=_json_safe_to_str(item.get("output")))],
        )

    raise ValueError(f"Unsupported {_item_type_name} type: {item['type']}")


async def _output_items_to_messages(
    history: Sequence[OutputItem],
    *,
    approval_storage: FunctionApprovalStore | None = None,
) -> list[Message]:
    """Converts a sequence of OutputItem objects to a list of Message objects.

    Args:
        history (Sequence[OutputItem]): The sequence of OutputItem objects to convert.
        approval_storage (ApprovalStorage | None, optional): The approval storage to use for
            resolving MCP approval requests. Defaults to None.

    Returns:
        list[Message]: The list of Message objects.
    """
    messages: list[Message] = []
    for item in history:
        messages.append(await _output_item_to_message(item, approval_storage=approval_storage))
    return messages


async def _output_item_to_message(
    item: OutputItem, *, approval_storage: FunctionApprovalStore | None = None
) -> Message:
    """Converts an OutputItem to a Message.

    Args:
        item (OutputItem): The OutputItem to convert.
        approval_storage (ApprovalStorage | None, optional): The approval storage to use for
            resolving MCP approval requests. Defaults to None.

    Returns:
        Message: The converted Message.

    Raises:
        ValueError: If the OutputItem type is not supported.
    """
    if item["type"] == "oauth_consent_request":
        return Message(
            role="assistant",
            contents=[Content.from_oauth_consent_request(item["consent_link"])],
        )

    if item["type"] == "structured_outputs":
        return Message(role="assistant", contents=[Content.from_text(_json_safe_to_str(item["output"]))])

    return await _item_to_message(
        cast(Item, item),
        approval_storage=approval_storage,
        _item_type_name="OutputItem",
    )


def _convert_output_message_content(content: OutputMessageContent) -> Content:
    """Converts an OutputMessageContent to a Content object.

    Args:
        content (OutputMessageContent): The OutputMessageContent to convert.

    Returns:
        Content: The converted Content object.

    Raises:
        ValueError: If the OutputMessageContent type is not supported.
    """
    if content["type"] == "output_text":
        return Content.from_text(content["text"])
    if content["type"] == "refusal":
        return Content.from_text(
            content["refusal"],
            additional_properties={_MODEL_OUTPUT_KIND_KEY: _MODEL_OUTPUT_REFUSAL},
        )

    # Defensive: `OutputMessageContent` currently only supports `output_text` and `refusal`,
    # but if new types are added in the future, this will catch them.
    raise ValueError(f"Unsupported OutputMessageContent type: {content['type']}")


def _convert_file_data(data_uri: str, filename: str | None = None) -> Content:
    """Convert a file_data data URI to a Content object.

    For text/* MIME types, decodes the base64 content and returns it as text.
    For other types, returns a URI-based Content with the filename preserved.
    """
    # Parse data URI: data:<media_type>;base64,<data>
    if data_uri.startswith("data:") and ";base64," in data_uri:
        header, encoded = data_uri.split(";base64,", 1)
        media_type = header[len("data:") :]
        if media_type.startswith("text/"):
            try:
                decoded_text = base64.b64decode(encoded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                logger.warning(
                    "Failed to decode text/* file_data as UTF-8, falling through to URI passthrough.",
                    exc_info=True,
                )
            else:
                prefix = f"[File: {filename}]\n" if filename else ""
                return Content.from_text(f"{prefix}{decoded_text}")
    additional_properties = {"filename": filename} if filename else None
    return Content.from_uri(data_uri, additional_properties=additional_properties)


def _convert_message_content(content: MessageContent) -> Content:
    """Converts a MessageContent to a Content object.

    Args:
        content (MessageContent): The MessageContent to convert.

    Returns:
        Content: The converted Content object.

    Raises:
        ValueError: If the MessageContent type is not supported.
    """
    if content["type"] == "input_text":
        return Content.from_text(content["text"])
    if content["type"] == "output_text":
        return Content.from_text(content["text"])
    if content["type"] == "text":
        return Content.from_text(content["text"])
    if content["type"] == "summary_text":
        return Content.from_text(content["text"])
    if content["type"] == "refusal":
        return Content.from_text(
            content["refusal"],
            additional_properties={_MODEL_OUTPUT_KIND_KEY: _MODEL_OUTPUT_REFUSAL},
        )
    if content["type"] == "reasoning_text":
        return Content.from_text_reasoning(text=content["text"])
    if content["type"] == "input_image":
        if image_url := content.get("image_url"):
            if image_url.startswith("data:"):
                return Content.from_uri(image_url)
            return Content.from_uri(image_url, media_type="image/*")
        if file_id := content.get("file_id"):
            return Content.from_hosted_file(file_id)
    if content["type"] == "input_file":
        if file_url := content.get("file_url"):
            return Content.from_uri(file_url)
        if file_id := content.get("file_id"):
            return Content.from_hosted_file(file_id, name=content.get("filename"))
        if file_data := content.get("file_data"):
            return _convert_file_data(file_data, content.get("filename"))
    if content["type"] == "computer_screenshot":
        if image_url := content.get("image_url"):
            return Content.from_uri(image_url)
        if file_id := content.get("file_id"):
            return Content.from_hosted_file(file_id, name=content.get("filename"))

    raise ValueError(f"Unsupported MessageContent type: {content['type']}")


# endregion

# region Output Item Conversion


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return str(value)


def _json_safe_to_str(value: Any | None) -> str:
    """Convert an argument or result value to a JSON-safe string.

    Args:
        value: The value to convert, which can be a string, JSON-like object, or None.

    Returns:
        The value as a JSON string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=_json_default)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _reasoning_encrypted_content(content: Content) -> str | None:
    """Return the opaque reasoning payload used for stateless replay."""
    encrypted_content = content.protected_data or content.additional_properties.get("encrypted_content")
    return encrypted_content if isinstance(encrypted_content, str) else None


def _reasoning_output_item(
    *,
    item_id: str,
    summary_texts: Sequence[str],
    encrypted_content: str | None,
    status: Literal["in_progress", "completed"],
) -> OutputItemReasoningItem:
    """Build a hosted reasoning item while retaining provider replay metadata."""
    return OutputItemReasoningItem({
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": text} for text in summary_texts],
        "encrypted_content": encrypted_content,
        "status": status,
    })


def _mcp_mapping_text(output: Mapping[Any, Any]) -> str | None:
    """Extract text only from a recognized MCP text-content mapping."""
    text = output.get("text")
    if not isinstance(text, str):
        return None
    if output.get("type") == "text" or set(output) == {"text"}:
        return text
    return None


def _stringify_mcp_output(output: Any) -> str:
    """Convert hosted MCP output payloads into the string shape expected by mcp_call.output."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        mapping = cast(Mapping[Any, Any], output)
        if (text := _mcp_mapping_text(mapping)) is not None:
            return text
        return _json_safe_to_str(mapping)
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        parts: list[str] = []
        entries = cast(Sequence[Any], output)
        for entry in entries:
            if isinstance(entry, str):
                parts.append(entry)
                continue
            if isinstance(entry, Content) and entry.type == "text":
                parts.append(entry.text or "")
                continue
            if isinstance(entry, Mapping) and (text := _mcp_mapping_text(cast(Mapping[Any, Any], entry))) is not None:
                parts.append(text)
                continue
            return _json_safe_to_str(entries)
        return "".join(parts)
    return _json_safe_to_str(output)


# endregion
