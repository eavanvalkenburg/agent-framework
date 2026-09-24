# Copyright (c) Microsoft. All rights reserved.

"""Native workflows exercise the real AgentServer Responses route."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agent_framework import (
    AgentResponse,
    Content,
    Executor,
    Message,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
    response_handler,
)
from azure.ai.agentserver.responses import (
    FileResponseStore,
    InMemoryResponseProvider,
    ResponseContext,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import CreateResponse, ResponseObject
from azure.ai.agentserver.responses.streaming._checkpoint import ResponseCheckpointEvent

from agent_framework_foundry_hosting import HostedResponseRequest, ResponsesHostServer, WorkflowTurn
from agent_framework_foundry_hosting._responses import _LATEST_CHECKPOINT_ID_KEY


def build_workflow(request: HostedResponseRequest) -> Workflow:
    """Build a fresh executor so only checkpointed state survives a turn."""
    assert request.scope.session_id

    @executor(id="echo")
    async def echo(text: str, ctx: WorkflowContext[str, str]) -> None:
        count = ctx.get_state("count", 0) + 1
        ctx.set_state("count", count)
        await ctx.yield_output(f"{count}: {text}")

    return WorkflowBuilder(name="echo-workflow", start_executor=echo, output_from="all").build()


async def parse_response(request: HostedResponseRequest) -> WorkflowTurn[str]:
    text = await request.get_input_text()
    if not text:
        raise ValueError("A non-empty message is required.")
    return WorkflowTurn(input=text)


def _output_text(body: Mapping[str, Any]) -> str:
    return "".join(
        part["text"]
        for item in body["output"]
        if item["type"] == "message"
        for part in item.get("content", [])
        if part["type"] == "output_text"
    )


def _completed_body(events: Sequence[Any]) -> Mapping[str, Any]:
    terminal = [
        cast(Mapping[str, Any], event)
        for event in events
        if isinstance(event, Mapping) and event.get("type") == "response.completed"
    ]
    assert len(terminal) == 1, events
    body = terminal[0].get("response")
    assert isinstance(body, dict)
    return body


async def test_native_workflow_restores_the_correct_sandbox_checkpoint() -> None:
    server = ResponsesHostServer(
        workflow=build_workflow,
        parse_response=parse_response,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        first = await client.post("/responses", json={"input": "first", "store": True})
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "completed", first_body
        assert _output_text(first_body) == "1: first"

        second = await client.post(
            "/responses",
            json={
                "input": "second",
                "previous_response_id": first_body["id"],
                "agent_session_id": first_body["agent_session_id"],
                "store": True,
            },
        )
        assert second.status_code == 200
        assert second.json()["status"] == "completed", second.json()
        assert _output_text(second.json()) == "2: second"


async def test_unstored_workflow_cannot_leave_durable_continuation() -> None:
    server = ResponsesHostServer(
        workflow=build_workflow,
        parse_response=parse_response,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        with patch(
            "agent_framework_foundry_hosting._workflow_state.FoundryStateStore.get_or_create",
            new=AsyncMock(side_effect=AssertionError("no durable workflow store allowed")),
        ):
            response = await client.post("/responses", json={"input": "one shot", "store": False})
        body = response.json()
        assert body["status"] == "completed", body
        assert _output_text(body) == "1: one shot"
        assert (await client.get(f"/responses/{body['id']}")).status_code == 404


def test_native_workflow_requires_a_parser() -> None:
    with pytest.raises(TypeError, match="parse_response is required"):
        ResponsesHostServer(workflow=build_workflow, response_store=InMemoryResponseProvider())


async def test_native_workflow_conversation_uses_its_head_and_rejects_a_fork() -> None:
    server = ResponsesHostServer(
        workflow=build_workflow,
        parse_response=parse_response,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        first = await client.post(
            "/responses", json={"input": "first", "conversation": "conversation-1", "store": True}
        )
        assert first.json()["status"] == "completed", first.json()
        second = await client.post(
            "/responses", json={"input": "next", "conversation": "conversation-1", "store": True}
        )
        assert second.json()["status"] == "completed", second.json()
        assert _output_text(second.json()) == "2: next"

        branch = await client.post(
            "/responses",
            json={
                "input": "fork",
                "previous_response_id": first.json()["id"],
                "agent_session_id": first.json()["agent_session_id"],
                "store": True,
            },
        )
        assert branch.json()["status"] == "failed", branch.json()
        assert "Branching a workflow" in branch.json()["error"]["message"]


async def test_native_workflow_cannot_resume_into_another_sandbox() -> None:
    server = ResponsesHostServer(
        workflow=build_workflow,
        parse_response=parse_response,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        first = await client.post("/responses", json={"input": "private", "store": True})
        other = await client.post(
            "/responses",
            json={
                "input": "steal",
                "previous_response_id": first.json()["id"],
                "agent_session_id": "a-different-sandbox",
                "store": True,
            },
        )
        assert other.json()["status"] == "failed", other.json()
        assert "no checkpoint in this Foundry session" in other.json()["error"]["message"]


async def test_resilient_native_workflow_pairs_output_with_exact_checkpoint(tmp_path: Path) -> None:
    server = ResponsesHostServer(
        workflow=build_workflow,
        parse_response=parse_response,
        response_store=FileResponseStore(storage_dir=tmp_path),
        options=ResponsesServerOptions(resilient_background=True),
    )
    request = CreateResponse(input="first", store=True, background=True, stream=True)
    context = ResponseContext(response_id="native-background", mode_flags=MagicMock())
    with patch.object(ResponseContext, "get_input_text", new=AsyncMock(return_value="first")):
        events = [event async for event in server._handle_response(request, context, asyncio.Event())]

    snapshots = [event for event in events if isinstance(event, ResponseCheckpointEvent)]
    assert snapshots
    last = snapshots[-1].response
    raw_metadata = cast(Mapping[str, Any], last.get("metadata") or {})
    internal_metadata = raw_metadata.get("_internal_metadata")
    assert isinstance(internal_metadata, str)
    metadata = json.loads(internal_metadata)
    assert isinstance(metadata[_LATEST_CHECKPOINT_ID_KEY], str)
    assert _output_text(last) == "1: first"

    restarted = ResponsesHostServer(
        workflow=build_workflow,
        parse_response=parse_response,
        response_store=FileResponseStore(storage_dir=tmp_path),
        options=ResponsesServerOptions(resilient_background=True),
    )
    recovered = ResponseContext(response_id="native-background", mode_flags=MagicMock())
    recovered.is_recovery = True
    recovered.persisted_response = last
    with patch.object(ResponseContext, "get_input_text", new=AsyncMock(side_effect=AssertionError("input replayed"))):
        continuation = [event async for event in restarted._handle_response(request, recovered, asyncio.Event())]
    assert _output_text(_completed_body(continuation)) == "1: first"


async def test_resilient_native_workflow_recovers_after_partial_output(tmp_path: Path) -> None:
    def build_two_steps(_: HostedResponseRequest) -> Workflow:
        @executor(id="first")
        async def first(text: str, ctx: WorkflowContext[str, str]) -> None:
            await ctx.yield_output("first")
            await ctx.send_message(text)

        @executor(id="second")
        async def second(_: str, ctx: WorkflowContext[str, str]) -> None:
            await ctx.yield_output("second")

        return WorkflowBuilder(name="two-step", start_executor=first, output_from="all").add_edge(first, second).build()

    request = CreateResponse(input="go", store=True, background=True, stream=True)
    server = ResponsesHostServer(
        workflow=build_two_steps,
        parse_response=parse_response,
        response_store=FileResponseStore(storage_dir=tmp_path),
        options=ResponsesServerOptions(resilient_background=True),
    )
    context = ResponseContext(response_id="native-midflight", mode_flags=MagicMock())
    partial: ResponseObject | None = None
    with patch.object(ResponseContext, "get_input_text", new=AsyncMock(return_value="go")):
        stream = cast(AsyncGenerator[Any, None], server._handle_response(request, context, asyncio.Event()))
        async with aclosing(stream) as events:
            async for event in events:
                if isinstance(event, ResponseCheckpointEvent) and _output_text(event.response) == "first":
                    partial = event.response
                    break
    assert partial is not None, "The first step must have a durable output/checkpoint pair."

    restarted = ResponsesHostServer(
        workflow=build_two_steps,
        parse_response=parse_response,
        response_store=FileResponseStore(storage_dir=tmp_path),
        options=ResponsesServerOptions(resilient_background=True),
    )
    recovered = ResponseContext(response_id="native-midflight", mode_flags=MagicMock())
    recovered.is_recovery = True
    recovered.persisted_response = partial
    with patch.object(ResponseContext, "get_input_text", new=AsyncMock(side_effect=AssertionError("input replayed"))):
        continuation = [event async for event in restarted._handle_response(request, recovered, asyncio.Event())]
    assert _output_text(_completed_body(continuation)) == "firstsecond"


def build_approval_workflow(request: HostedResponseRequest) -> Workflow:
    """Produce one deterministic approval request without invoking a model."""
    assert request.scope.session_id
    function_call = Content.from_function_call(call_id="call-1", name="publish_note", arguments={"note": "hello"})
    approval = Content.from_function_approval_request(id="inner-approval-1", function_call=function_call)

    class ApprovalExecutor(Executor):
        @handler
        async def start(self, _: str, ctx: WorkflowContext) -> None:
            await ctx.request_info(approval, Content, request_id="inner-approval-1")

        @response_handler
        async def decide(
            self,
            original_request: Content,
            response: Content,
            ctx: WorkflowContext[str, AgentResponse],
        ) -> None:
            assert original_request.id == "inner-approval-1"
            assert response.type == "function_approval_response"
            await ctx.yield_output(
                AgentResponse(
                    messages=[
                        Message(
                            role="assistant",
                            contents=[Content.from_text(f"Approved: {response.approved}")],
                        )
                    ]
                )
            )

    executor_instance = ApprovalExecutor(id="approval")
    return WorkflowBuilder(name="approval-workflow", start_executor=executor_instance).build()


async def parse_approval(request: HostedResponseRequest) -> WorkflowTurn[str]:
    if replies := await request.get_workflow_responses():
        return WorkflowTurn(responses=replies)
    text = await request.get_input_text()
    if not text:
        raise ValueError("The workflow requires an input or approval decision.")
    return WorkflowTurn(input=text)


async def test_native_workflow_approval_resumes_once_in_the_original_sandbox() -> None:
    server = ResponsesHostServer(
        workflow=build_approval_workflow,
        parse_response=parse_approval,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        first = await client.post("/responses", json={"input": "publish the note", "store": True})
        body = first.json()
        assert body["status"] == "completed", body
        approvals = [item for item in body["output"] if item["type"] == "mcp_approval_request"]
        assert len(approvals) == 1

        follow_up = {
            "input": [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approvals[0]["id"],
                    "approve": True,
                }
            ],
            "previous_response_id": body["id"],
            "agent_session_id": body["agent_session_id"],
            "store": True,
        }
        other_sandbox = await client.post("/responses", json={**follow_up, "agent_session_id": "other-sandbox"})
        assert other_sandbox.json()["status"] == "failed", other_sandbox.json()
        forged = await client.post(
            "/responses",
            json={
                **follow_up,
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": "forged-approval",
                        "approve": True,
                    }
                ],
            },
        )
        assert forged.json()["status"] == "failed", forged.json()

        second = await client.post("/responses", json=follow_up)
        assert second.json()["status"] == "completed", second.json()
        assert _output_text(second.json()) == "Approved: True"

        duplicate = await client.post("/responses", json=follow_up)
        assert duplicate.json()["status"] == "failed", duplicate.json()


async def test_unstored_native_workflow_refuses_cross_turn_approval() -> None:
    server = ResponsesHostServer(
        workflow=build_approval_workflow,
        parse_response=parse_approval,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        response = await client.post("/responses", json={"input": "publish the note", "store": False})
        body = response.json()
        assert body["status"] == "failed", body
        assert "require store=true" in body["error"]["message"]
        assert all(item["type"] != "mcp_approval_request" for item in body["output"])


async def test_partial_approval_keeps_other_request_resumable_without_reissuing_it() -> None:
    def build_multi_approval_workflow(request: HostedResponseRequest) -> Workflow:
        assert request.scope.session_id

        class MultiApprovalExecutor(Executor):
            @handler
            async def start(self, _: str, ctx: WorkflowContext) -> None:
                for number in (1, 2):
                    request_id = f"approval-{number}"
                    call = Content.from_function_call(
                        call_id=f"call-{number}", name=f"publish_{number}", arguments="{}"
                    )
                    approval = Content.from_function_approval_request(id=request_id, function_call=call)
                    await ctx.request_info(approval, Content, request_id=request_id)

            @response_handler
            async def decide(
                self,
                original_request: Content,
                response: Content,
                ctx: WorkflowContext[str, str],
            ) -> None:
                await ctx.yield_output(f"{original_request.id}: {response.approved}")

        approval = MultiApprovalExecutor(id="two_approvals")
        return WorkflowBuilder(name="two-approvals", start_executor=approval, output_from="all").build()

    server = ResponsesHostServer(
        workflow=build_multi_approval_workflow,
        parse_response=parse_approval,
        response_store=InMemoryResponseProvider(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        first = (await client.post("/responses", json={"input": "publish", "store": True})).json()
        assert first["status"] == "completed", first
        approvals = [item for item in first["output"] if item["type"] == "mcp_approval_request"]
        assert len(approvals) == 2

        def follow_up(approval_id: str, previous_id: str) -> dict[str, Any]:
            return {
                "input": [{"type": "mcp_approval_response", "approval_request_id": approval_id, "approve": True}],
                "previous_response_id": previous_id,
                "agent_session_id": first["agent_session_id"],
                "store": True,
            }

        duplicate_decisions = follow_up(approvals[0]["id"], first["id"])
        duplicate_decisions["input"].append(duplicate_decisions["input"][0])
        invalid = (await client.post("/responses", json=duplicate_decisions)).json()
        assert invalid["status"] == "failed", invalid

        second = (await client.post("/responses", json=follow_up(approvals[0]["id"], first["id"]))).json()
        assert second["status"] == "completed", second
        assert "approval-1" in _output_text(second)
        assert not any(item["type"] == "mcp_approval_request" for item in second["output"])

        third = (await client.post("/responses", json=follow_up(approvals[1]["id"], second["id"]))).json()
        assert third["status"] == "completed", third
        assert "approval-2" in _output_text(third)
