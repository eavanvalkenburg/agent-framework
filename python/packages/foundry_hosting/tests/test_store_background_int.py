# Copyright (c) Microsoft. All rights reserved.

"""Live-model and HTTP integration coverage for Responses storage and background execution."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from agent_framework import Agent, SessionStore, Workflow, WorkflowBuilder, WorkflowContext, executor
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatOptions
from azure.ai.agentserver.core import AgentConfig, FoundryAgentRequestContext
from azure.ai.agentserver.responses import FileResponseStore, InMemoryResponseProvider, ResponsesServerOptions
from azure.identity import AzureCliCredential

from agent_framework_foundry_hosting import HostedResponseRequest, ResponsesHostServer, StoreProvider, WorkflowTurn

pytestmark = pytest.mark.integration

skip_without_foundry = pytest.mark.skipif(
    not os.getenv("FOUNDRY_PROJECT_ENDPOINT") or not os.getenv("FOUNDRY_MODEL"),
    reason="Live Foundry project endpoint and model are required.",
)


@pytest.fixture(autouse=True)
def isolated_state_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path / "agentserver"))


def _real_agent(*, store_by_default: bool = False) -> Agent[Any]:
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["FOUNDRY_MODEL"],
        credential=AzureCliCredential(),
    )
    return Agent(
        client=client,
        instructions="Be concise. Reply in one short sentence.",
        default_options=OpenAIChatOptions(store=True) if store_by_default else None,
    )


async def _post(server: ResponsesHostServer, payload: dict[str, Any]) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        return await client.post("/responses", json=payload, timeout=120)


async def _get(server: ResponsesHostServer, response_id: str) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test") as client:
        return await client.get(f"/responses/{response_id}", timeout=30)


async def _poll(server: ResponsesHostServer, response_id: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 120
    while asyncio.get_running_loop().time() < deadline:
        response = await _get(server, response_id)
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        if body["status"] in ("completed", "failed", "incomplete"):
            return body
        await asyncio.sleep(0.5)
    raise AssertionError(f"Background response {response_id} did not reach a terminal status.")


def _output_text(body: dict[str, Any]) -> str:
    return "".join(
        part["text"]
        for item in body["output"]
        if item["type"] == "message"
        for part in item.get("content", [])
        if part["type"] == "output_text"
    )


@pytest.mark.flaky
@skip_without_foundry
@pytest.mark.parametrize("store", [False, True])
async def test_foreground_store_controls_response_retrieval(store: bool) -> None:
    server = ResponsesHostServer(agent=_real_agent(), response_store=InMemoryResponseProvider())
    result = await _post(server, {"input": "Reply with the word ready.", "store": store})

    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "completed", body.get("error")
    assert _output_text(body)
    retrieved = await _get(server, body["id"])
    assert retrieved.status_code == (200 if store else 404)
    if store:
        assert retrieved.json()["id"] == body["id"]


@pytest.mark.flaky
@skip_without_foundry
async def test_azd_session_id_is_not_forwarded_as_a_model_option() -> None:
    server = ResponsesHostServer(agent=_real_agent(), response_store=InMemoryResponseProvider())
    result = await _post(
        server,
        {"input": "Reply with the word ready.", "store": True, "session_id": "caller-platform-session"},
    )

    assert result.status_code == 200
    assert result.json()["status"] == "completed", result.json().get("error")


@pytest.mark.flaky
@skip_without_foundry
async def test_unstored_stream_returns_terminal_output_without_retrievable_history() -> None:
    server = ResponsesHostServer(agent=_real_agent(), response_store=InMemoryResponseProvider())
    result = await _post(server, {"input": "Reply with the word ready.", "store": False, "stream": True})

    assert result.status_code == 200
    assert "text/event-stream" in result.headers["content-type"]
    events = [json.loads(line.removeprefix("data: ")) for line in result.text.splitlines() if line.startswith("data: ")]
    completed = [event["response"] for event in events if event.get("type") == "response.completed"]
    assert len(completed) == 1
    assert _output_text(completed[0])
    assert (await _get(server, completed[0]["id"])).status_code == 404


@pytest.mark.flaky
@skip_without_foundry
async def test_background_requires_outer_storage() -> None:
    server = ResponsesHostServer(agent=_real_agent(), response_store=InMemoryResponseProvider())
    result = await _post(server, {"input": "Reply with the word ready.", "store": False, "background": True})
    assert result.status_code == 400


@pytest.mark.flaky
@skip_without_foundry
async def test_outer_background_uses_outer_response_id_for_polling() -> None:
    server = ResponsesHostServer(agent=_real_agent(), response_store=InMemoryResponseProvider())
    result = await _post(server, {"input": "Reply with the word ready.", "store": True, "background": True})

    assert result.status_code == 200
    pending = result.json()
    assert pending["status"] in ("queued", "in_progress", "completed"), pending
    final = await _poll(server, pending["id"])
    assert final["status"] == "completed", final.get("error")
    assert final["id"] == pending["id"]
    assert _output_text(final)


@pytest.mark.flaky
@skip_without_foundry
async def test_provider_background_poll_does_not_expose_inner_id(tmp_path: Path) -> None:
    agent = _real_agent()
    server = ResponsesHostServer(
        agent=agent,
        inner_history="service",
        inner_background="provider",
        options=ResponsesServerOptions(resilient_background=True),
        response_store=FileResponseStore(storage_dir=tmp_path / "responses"),
    )
    result = await _post(server, {"input": "Reply with the word ready.", "store": True, "background": True})

    assert result.status_code == 200
    pending = result.json()
    final = await _poll(server, pending["id"])
    assert final["status"] == "completed", final.get("error")
    assert final["id"] == pending["id"]
    assert _output_text(final)
    assert "continuation_token" not in final


@pytest.mark.flaky
@skip_without_foundry
async def test_service_history_uses_outer_id_for_continuation() -> None:
    sessions = SessionStore()

    class SessionProvider(StoreProvider[SessionStore]):
        def get_store(self, *, config: AgentConfig, platform_context: FoundryAgentRequestContext) -> SessionStore:
            return sessions

    agent = _real_agent(store_by_default=True)
    server = ResponsesHostServer(
        agent=agent,
        inner_history="service",
        response_store=InMemoryResponseProvider(),
        agent_session_store_provider=SessionProvider(),
    )
    first = await _post(server, {"input": "My code word is VERDANT42. Acknowledge it briefly.", "store": True})
    first_body = first.json()
    assert first_body["status"] == "completed", first_body.get("error")
    first_session = await sessions.get(first_body["id"])
    assert first_session is not None
    assert first_session.service_session_id is not None
    assert str(first_session.service_session_id) != first_body["id"]

    second = await _post(
        server,
        {
            "input": "What code word did I give you? Reply with the word only.",
            "store": True,
            "previous_response_id": first_body["id"],
            "agent_session_id": first_body["agent_session_id"],
        },
    )
    body = second.json()
    assert body["status"] == "completed", body.get("error")
    assert _output_text(body)
    second_session = await sessions.get(body["id"])
    assert second_session is not None
    assert second_session.service_session_id is not None


@pytest.mark.flaky
@skip_without_foundry
async def test_native_workflow_background_restores_its_checkpoint(tmp_path: Path) -> None:
    release = asyncio.Event()

    def build(_: HostedResponseRequest) -> Workflow:
        @executor(id="stateful")
        async def stateful(text: str, ctx: WorkflowContext[str, str]) -> None:
            await release.wait()
            count = ctx.get_state("count", 0) + 1
            ctx.set_state("count", count)
            await ctx.yield_output(f"{count}: {text}")

        return WorkflowBuilder(name="stored-background", start_executor=stateful, output_from="all").build()

    async def parse(request: HostedResponseRequest) -> WorkflowTurn[str]:
        text = await request.get_input_text()
        if not text:
            raise ValueError("Workflow input is required.")
        return WorkflowTurn(input=text)

    server = ResponsesHostServer(
        workflow=build,
        parse_response=parse,
        options=ResponsesServerOptions(resilient_background=True),
        response_store=FileResponseStore(storage_dir=tmp_path / "responses"),
    )
    try:
        result = await asyncio.wait_for(
            _post(server, {"input": "first", "store": True, "background": True}),
            timeout=10,
        )
        assert result.status_code == 200
        pending = result.json()
        assert pending["status"] in ("queued", "in_progress"), pending
    finally:
        release.set()

    first = await _poll(server, pending["id"])
    assert first["status"] == "completed", first.get("error")
    assert _output_text(first) == "1: first"
    second = await _post(
        server,
        {
            "input": "second",
            "store": True,
            "previous_response_id": first["id"],
            "agent_session_id": first["agent_session_id"],
        },
    )
    second_body = second.json()
    assert second_body["status"] == "completed", second_body.get("error")
    assert _output_text(second_body) == "2: second"


@pytest.mark.flaky
@skip_without_foundry
async def test_native_workflow_without_storage_is_one_shot() -> None:
    def build(_: HostedResponseRequest) -> Workflow:
        @executor(id="echo")
        async def echo(text: str, ctx: WorkflowContext[str, str]) -> None:
            await ctx.yield_output(f"one: {text}")

        return WorkflowBuilder(name="one-shot", start_executor=echo, output_from="all").build()

    async def parse(request: HostedResponseRequest) -> WorkflowTurn[str]:
        text = await request.get_input_text()
        if not text:
            raise ValueError("Input is required.")
        return WorkflowTurn(input=text)

    server = ResponsesHostServer(
        workflow=build,
        parse_response=parse,
        response_store=InMemoryResponseProvider(),
    )
    response = await _post(server, {"input": "hello", "store": False})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed", body.get("error")
    assert _output_text(body) == "one: hello"
    assert (await _get(server, body["id"])).status_code == 404
