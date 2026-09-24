# Copyright (c) Microsoft. All rights reserved.

"""Foundry Invocations protocol hosting for agents and native workflows."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from agent_framework import (
    AgentExecutor,
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    CheckpointStorage,
    ResponseStream,
    SessionStore,
    SupportsAgentRun,
    WorkflowEvent,
    WorkflowRunState,
)
from agent_framework._telemetry import mark_feature_used
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    get_request_context,
    reset_request_context,
    set_request_context,
)
from azure.ai.agentserver.invocations import InvocationAgentServerHost
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

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
from ._request import InvocationRun, UnsupportedOptions, WorkflowTurn, validate_unsupported_options
from ._scope import FoundryRequestScope
from ._state_store import AgentSessionStoreProvider, CheckpointStoreProvider, ContextScopedStoreProvider, StoreProvider
from ._workflow_state import FoundryWorkflowBindingStore, WorkflowBinding

logger = logging.getLogger(__name__)

InvocationParser = Callable[[Request], InvocationRun | WorkflowTurn[Any] | Awaitable[InvocationRun | WorkflowTurn[Any]]]
InvocationOptionsHook = Callable[[Request, dict[str, Any]], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


def _sse(event: str, data: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _workflow_output(event: WorkflowEvent[Any]) -> Any:
    """Keep the application payload rather than assuming workflow outputs are text."""
    data = event.data
    if isinstance(data, AgentResponse):
        return data.text
    if isinstance(data, AgentResponseUpdate):
        return data.text
    if hasattr(data, "to_dict") and callable(data.to_dict):
        data = data.to_dict()
    elif is_dataclass(data) and not isinstance(data, type):
        data = asdict(data)
    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Workflow output type {type(event.data).__name__} is not JSON-serializable.") from exc
    return data


class InvocationsHostServer(InvocationAgentServerHost):
    """Serve a MAF agent or workflow with application-defined Invocations input."""

    def __init__(
        self,
        agent: AgentSource | None = None,
        *,
        workflow: WorkflowSource[Request] | None = None,
        parse_request: InvocationParser | None = None,
        prepare_options: InvocationOptionsHook | None = None,
        unsupported_options: UnsupportedOptions = "warn",
        agent_session_store_provider: StoreProvider[SessionStore] | None = None,
        checkpoint_store_provider: ContextScopedStoreProvider[CheckpointStorage] | None = None,
        openapi_spec: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if agent is None and workflow is None:
            raise TypeError("agent must be an agent instance or a zero-argument callable, or provide workflow.")
        if agent is not None and workflow is not None:
            raise TypeError("Provide exactly one of agent or workflow.")
        if workflow is not None:
            validate_workflow_source(workflow)
            if parse_request is None:
                raise TypeError("parse_request is required when hosting a native Workflow.")
        else:
            validate_agent_source(agent)
        if parse_request is not None and not callable(parse_request):
            raise TypeError("parse_request must be a callable.")

        super().__init__(openapi_spec=openapi_spec, **kwargs)
        if workflow is not None and not callable(workflow) and self.config.is_hosted:
            raise ValueError("Hosted workflows require a request-aware factory that creates fresh executors.")

        self._agent = agent
        self._workflow = workflow
        self._owns_request_agent = not is_agent(agent)
        self._parse_request = parse_request
        self._prepare_options = prepare_options
        self._unsupported_options = validate_unsupported_options(unsupported_options)
        self._session_storage_provider = agent_session_store_provider or AgentSessionStoreProvider()
        self._checkpoint_storage_provider = checkpoint_store_provider or CheckpointStoreProvider()
        self._locks: dict[str | tuple[str, str], asyncio.Lock] = {}
        self.invoke_handler(self._handle_invoke)
        mark_feature_used(FeatureIndex.FOUNDRY_HOSTING)

    def _partition_key(self) -> str | tuple[str, str]:
        """Trust only platform-provided user and sandbox IDs when hosted."""
        context = get_request_context()
        if self.config.is_hosted:
            if not context.session_id or not context.user_id or not context.call_id:
                raise RuntimeError("The hosted environment is missing session_id or user_id or call_id.")
            return context.session_id, context.user_id
        if not context.session_id:
            raise RuntimeError("The request context is missing session_id.")
        return context.session_id

    @staticmethod
    def _session_id(partition_key: str | tuple[str, str]) -> str:
        return json.dumps(partition_key, separators=(",", ":")) if isinstance(partition_key, tuple) else partition_key

    @staticmethod
    def _storage_key(partition_key: str | tuple[str, str]) -> str:
        return hashlib.sha256(InvocationsHostServer._session_id(partition_key).encode("utf-8")).hexdigest()

    async def _get_session(
        self,
        partition_key: str | tuple[str, str],
        platform_context: FoundryAgentRequestContext,
        agent: SupportsAgentRun,
    ) -> tuple[SessionStore, AgentSession]:
        store = self._session_storage_provider.get_store(config=self.config, platform_context=platform_context)
        session = await store.get(self._storage_key(partition_key))
        if session is None:
            session = agent.create_session(session_id=self._session_id(partition_key))
        return store, session

    @asynccontextmanager
    async def _request_agent(self) -> AsyncGenerator[SupportsAgentRun]:
        if self._agent is None:
            raise RuntimeError("No agent is configured for Invocations.")
        agent = await resolve_agent(self._agent)
        async with AsyncExitStack() as resources:
            if self._owns_request_agent and isinstance(agent, AbstractAsyncContextManager):
                await resources.enter_async_context(agent)
            yield agent

    async def _parse(self, request: Request) -> InvocationRun | WorkflowTurn[Any]:
        if self._parse_request is not None:
            result = self._parse_request(request)
            return await result if inspect.isawaitable(result) else result
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("The invocation must be a JSON object.")
        body = cast(Mapping[str, Any], payload)
        message = body.get("message")
        stream = body.get("stream", False)
        options = body.get("options", {})
        if not isinstance(message, str):
            raise ValueError("message must be a string.")
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean.")
        if not isinstance(options, dict):
            raise ValueError("options must be an object.")
        return InvocationRun(
            messages=message if stream else [message],
            options=cast(Mapping[str, Any], options),
            stream=stream,
        )

    async def _options(self, request: Request, parsed: InvocationRun) -> dict[str, Any]:
        options = dict(parsed.options)
        if self._prepare_options is not None:
            result = self._prepare_options(request, dict(options))
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, Mapping):
                raise TypeError("prepare_options must return a mapping of MAF run options.")
            options = dict(result)
        return options

    def _agent_kwargs(self, agent: SupportsAgentRun, options: dict[str, Any]) -> dict[str, Any]:
        if not options:
            return {}
        try:
            inspect.signature(agent.run).bind_partial(options=options)
        except (TypeError, ValueError):
            if self._unsupported_options == "error":
                raise TypeError("The hosted agent does not accept caller runtime options.") from None
            if self._unsupported_options == "warn":
                logger.warning("Agent doesn't support runtime options. They will be ignored.")
            return {}
        return {"options": options}

    async def _handle_invoke(self, request: Request) -> Response:
        """Parse input before touching agent or workflow state."""
        try:
            partition_key = self._partition_key()
            platform_context = get_request_context()
            scope = FoundryRequestScope.from_context(self.config, platform_context)
        except RuntimeError:
            logger.exception("Foundry platform session is missing")
            return JSONResponse({"error": "A valid Foundry session is required."}, status_code=500)

        try:
            parsed = await self._parse(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if self._workflow is not None:
            if not isinstance(parsed, WorkflowTurn):
                return JSONResponse({"error": "A workflow parser must return WorkflowTurn."}, status_code=400)
            return await self._handle_workflow(request, parsed, scope, platform_context, partition_key)
        if not isinstance(parsed, InvocationRun):
            return JSONResponse({"error": "An agent parser must return InvocationRun."}, status_code=400)
        try:
            options = await self._options(request, parsed)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if parsed.stream:

            async def stream_response() -> AsyncGenerator[str]:
                token = set_request_context(platform_context)
                try:
                    async with self._locks.setdefault(partition_key, asyncio.Lock()), self._request_agent() as agent:
                        store, session = await self._get_session(partition_key, platform_context, agent)
                        stream = agent.run(
                            parsed.messages,
                            session=session,
                            stream=True,
                            **self._agent_kwargs(agent, options),
                        )
                        stream_failure: Exception | None = None
                        save_failure: Exception | None = None
                        stream_completed = False
                        try:
                            async for update in stream:
                                if update.text:
                                    yield _sse("delta", {"text": update.text})
                            stream_completed = True
                        except Exception as exc:
                            stream_failure = exc
                            logger.exception("Invocations agent stream failed")
                        finally:
                            try:
                                if isinstance(stream, ResponseStream):
                                    if stream_completed:
                                        await stream.get_final_response()
                                    else:
                                        close = getattr(stream, "close", None)
                                        if close is not None:
                                            await close()
                                        else:
                                            logger.warning(
                                                "The installed core cannot close an interrupted agent stream."
                                            )
                                else:
                                    close = getattr(stream, "aclose", None)
                                    if close is not None:
                                        await close()
                            except Exception as exc:
                                logger.exception("Failed to close Invocations agent stream")
                                stream_failure = stream_failure or exc
                            try:
                                await store.set(self._storage_key(partition_key), session)
                            except Exception as exc:
                                logger.exception("Failed to persist Invocations session after streaming")
                                save_failure = exc
                        if stream_failure is not None or save_failure is not None:
                            yield _sse("error", {"message": "Agent invocation or session persistence failed."})
                            return
                        yield _sse("done", {"session_id": scope.session_id})
                except Exception:
                    logger.exception("Failed to prepare Invocations agent stream")
                    yield _sse("error", {"message": "Agent invocation failed."})
                finally:
                    reset_request_context(token)

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        try:
            async with self._locks.setdefault(partition_key, asyncio.Lock()), self._request_agent() as agent:
                store, session = await self._get_session(partition_key, platform_context, agent)
                response = await agent.run(
                    parsed.messages,
                    session=session,
                    **self._agent_kwargs(agent, options),
                )
                await store.set(self._storage_key(partition_key), session)
            return JSONResponse({"response": response.text})
        except Exception:
            logger.exception("Invocations agent request failed")
            return JSONResponse({"error": "Agent invocation failed."}, status_code=500)

    async def _handle_workflow(
        self,
        request: Request,
        parsed: WorkflowTurn[Any],
        scope: FoundryRequestScope,
        platform_context: FoundryAgentRequestContext,
        partition_key: str | tuple[str, str],
    ) -> Response:
        """Resume one workflow checkpoint lineage within a Foundry session."""
        if self._workflow is None:
            raise RuntimeError("No workflow is configured for Invocations.")

        if parsed.stream:

            async def stream_response() -> AsyncGenerator[str]:
                try:
                    async for output_event in self._workflow_events(
                        request, parsed, scope, platform_context, partition_key
                    ):
                        yield _sse(output_event["type"], output_event)
                except ValueError as exc:
                    yield _sse("error", {"message": str(exc)})
                except Exception:
                    logger.exception("Invocations workflow stream failed")
                    yield _sse("error", {"message": "Workflow invocation failed."})
                else:
                    yield _sse("done", {"session_id": scope.session_id})

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        try:
            output = [
                event async for event in self._workflow_events(request, parsed, scope, platform_context, partition_key)
            ]
            return JSONResponse({"output": output})
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            logger.exception("Invocations workflow request failed")
            return JSONResponse({"error": "Workflow invocation failed."}, status_code=500)

    async def _workflow_events(
        self,
        request: Request,
        parsed: WorkflowTurn[Any],
        scope: FoundryRequestScope,
        platform_context: FoundryAgentRequestContext,
        partition_key: str | tuple[str, str],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Deliver restore output and new output before advancing the scoped cursor."""
        if self._workflow is None:
            raise RuntimeError("No workflow is configured for Invocations.")
        token = set_request_context(platform_context)
        try:
            async with self._locks.setdefault(partition_key, asyncio.Lock()):
                workflow = await resolve_workflow(self._workflow, request)
                if workflow._runner_context.has_checkpointing():  # pyright: ignore[reportPrivateUsage]
                    raise ValueError("Configure workflow checkpoints on the host, not on WorkflowBuilder.")
                bindings = FoundryWorkflowBindingStore(scope)
                lineage_id = f"invocations:{workflow.name}"
                prior, etag = await bindings.get_lineage_head(lineage_id)
                if prior is not None and prior.graph_hash != workflow.graph_signature_hash:
                    raise ValueError("The previous workflow checkpoint is incompatible with this workflow.")
                checkpoint_storage = self._checkpoint_storage_provider.get_store(
                    config=self.config,
                    context_id=hashlib.sha256(lineage_id.encode()).hexdigest(),
                    platform_context=platform_context,
                )
                async with AsyncExitStack() as resources:
                    entered: set[int] = set()
                    for executor in workflow.executors.values():
                        if isinstance(executor, AgentExecutor) and id(executor.agent) not in entered:
                            entered.add(id(executor.agent))
                            if isinstance(executor.agent, AbstractAsyncContextManager):
                                await resources.enter_async_context(executor.agent)

                    pending_ids: set[str] = set()
                    if prior is not None:
                        checkpoint = await checkpoint_storage.load(prior.checkpoint_id)
                        pending_ids = set(checkpoint.pending_request_info_events)
                        restore = workflow.run(
                            stream=True, checkpoint_id=prior.checkpoint_id, checkpoint_storage=checkpoint_storage
                        )
                        async for event in restore:
                            if event.type == "request_info" and event.request_id in pending_ids:
                                continue
                            if event.type in ("output", "intermediate", "request_info"):
                                yield {
                                    "type": event.type,
                                    "data": _workflow_output(event),
                                    "request_id": event.request_id if event.type == "request_info" else None,
                                }
                    if parsed.responses is not None:
                        if not pending_ids or not set(parsed.responses) <= pending_ids:
                            raise ValueError("A workflow response must match the pending checkpoint.")
                        run = workflow.run(
                            responses=parsed.responses,
                            stream=True,
                            checkpoint_storage=checkpoint_storage,
                            client_kwargs=parsed.client_kwargs,
                            function_invocation_kwargs=parsed.function_invocation_kwargs,
                        )
                    else:
                        if workflow.status == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
                            raise ValueError("Answer pending workflow requests before sending new input.")
                        run = workflow.run(
                            message=parsed.input,
                            stream=True,
                            checkpoint_storage=checkpoint_storage,
                            client_kwargs=parsed.client_kwargs,
                            function_invocation_kwargs=parsed.function_invocation_kwargs,
                        )
                    async for event in run:
                        if event.type in ("output", "intermediate", "request_info"):
                            yield {
                                "type": event.type,
                                "data": _workflow_output(event),
                                "request_id": event.request_id if event.type == "request_info" else None,
                            }
                checkpoint_id = workflow.get_last_checkpoint_id()
                if checkpoint_id is None:
                    raise RuntimeError("An Invocations workflow must persist a checkpoint.")
                binding = WorkflowBinding(
                    response_id=platform_context.call_id or uuid.uuid4().hex,
                    checkpoint_id=checkpoint_id,
                    lineage_id=lineage_id,
                    workflow_name=workflow.name,
                    graph_hash=workflow.graph_signature_hash,
                    session_id=scope.session_id,
                    conversation_id=None,
                )
                await bindings.save_response(binding)
                await bindings.advance_lineage(lineage_id, binding, expected_etag=etag)
        finally:
            reset_request_context(token)
