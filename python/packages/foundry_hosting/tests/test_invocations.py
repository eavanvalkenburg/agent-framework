# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for InvocationsHostServer.

These tests exercise ``InvocationsHostServer`` directly by constructing the
host, driving ``_partition_key`` and ``_handle_invoke`` with a fake agent and
mock requests. The Foundry request context is injected via the public
``set_request_context`` / ``reset_request_context`` helpers rather than by
patching, matching the style used in ``test_toolbox.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from itertools import product
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    Content,
    Message,
    ResponseStream,
    ServiceSessionId,
    SessionStore,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from typing_extensions import Any

from agent_framework_foundry_hosting import InvocationRun, InvocationsHostServer, WorkflowTurn

# region Helpers


@pytest.fixture(autouse=True)
def _durable_test_sessions(monkeypatch: pytest.MonkeyPatch) -> SessionStore:
    """Simulate persistence without calling the hosted Azure state-store endpoint."""
    store = SessionStore()

    def get_store(*_args: Any, **_kwargs: Any) -> SessionStore:
        return store

    monkeypatch.setattr(
        "agent_framework_foundry_hosting._invocations.AgentSessionStoreProvider.get_store",
        get_store,
    )
    return store


class _FakeAgent:
    """Minimal agent implementing the ``SupportsAgentRun`` protocol.

    ``run`` returns an awaitable when ``stream`` is ``False`` and an async
    iterator when ``stream`` is ``True``. Call arguments are recorded on
    ``calls`` for assertions.
    """

    def __init__(
        self,
        *,
        response: AgentResponse | None = None,
        stream_updates: list[AgentResponseUpdate] | None = None,
    ) -> None:
        self.id = "fake-agent"
        self.name: str | None = "Fake Agent"
        self.description: str | None = "A fake agent for testing"
        self._response = response
        self._stream_updates = stream_updates or []
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        messages: Any = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append({
            "messages": messages,
            "stream": stream,
            "session": session,
            "options": kwargs.get("options"),
        })
        if stream:

            async def _gen() -> AsyncIterator[AgentResponseUpdate]:
                for update in self._stream_updates:
                    yield update

            return _gen()

        async def _run() -> AgentResponse:
            assert self._response is not None
            return self._response

        return _run()

    def create_session(self, *, session_id: str | None = None) -> AgentSession:
        return AgentSession(session_id=session_id)

    def get_session(
        self,
        service_session_id: str | ServiceSessionId,
        *,
        session_id: str | None = None,
    ) -> AgentSession:
        return AgentSession(service_session_id=service_session_id, session_id=session_id)


class _ContextAgent(_FakeAgent):
    def __init__(
        self,
        events: list[str],
        *,
        response: AgentResponse | None = None,
        stream_updates: list[AgentResponseUpdate] | None = None,
    ) -> None:
        super().__init__(response=response, stream_updates=stream_updates)
        self._events = events

    async def __aenter__(self) -> _ContextAgent:
        self._events.append("enter")
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._events.append("exit")

    def run(
        self,
        messages: Any = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Any:
        self._events.append("run")
        result = super().run(messages, stream=stream, session=session, **kwargs)
        if not stream:
            return result

        async def _gen() -> AsyncIterator[AgentResponseUpdate]:
            try:
                async for update in result:
                    yield update
            finally:
                self._events.append("stream_close")

        return _gen()


def _make_agent(
    *,
    response_text: str | None = None,
    stream_texts: list[str] | None = None,
) -> _FakeAgent:
    """Build a ``_FakeAgent`` from plain text for non-streaming/streaming runs."""
    response = None
    if response_text is not None:
        response = AgentResponse(messages=[Message(role="assistant", contents=[Content.from_text(response_text)])])
    stream_updates = None
    if stream_texts is not None:
        stream_updates = [AgentResponseUpdate(contents=[Content.from_text(t)]) for t in stream_texts]
    return _FakeAgent(response=response, stream_updates=stream_updates)


def _make_request(payload: dict[str, Any]) -> Request:
    """Build a mock Starlette request whose ``json()`` returns ``payload``."""
    request = MagicMock(spec=Request)
    request.json = AsyncMock(return_value=payload)
    return request


@contextmanager
def _request_context(
    *,
    call_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[None]:
    """Install a Foundry request context for the duration of the block."""
    token = set_request_context(FoundryAgentRequestContext(call_id=call_id, user_id=user_id, session_id=session_id))
    try:
        yield
    finally:
        reset_request_context(token)


async def _collect_stream(response: StreamingResponse) -> str:
    """Concatenate the string chunks produced by a StreamingResponse."""
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else bytes(chunk).decode())
    return "".join(chunks)


# endregion


# region Initialization


class TestInit:
    def test_accepts_supports_agent_run(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        assert server._agent is not None  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize("agent", [None, 42])
    def test_rejects_invalid_agent_source(self, agent: Any) -> None:
        with pytest.raises(TypeError, match="agent must be an agent instance or a zero-argument callable"):
            InvocationsHostServer(agent)

    def test_rejects_agent_class_requiring_constructor_arguments(self) -> None:
        with pytest.raises(TypeError, match="agent callable must accept no arguments"):
            InvocationsHostServer(cast(Any, _ContextAgent))

    def test_rejects_factory_requiring_arguments(self) -> None:
        def create_agent(name: str) -> _FakeAgent:
            return _make_agent(response_text=name)

        with pytest.raises(TypeError, match="agent callable must accept no arguments"):
            InvocationsHostServer(cast(Any, create_agent))


# endregion


# region Partition key


class TestPartitionKey:
    def test_local_returns_session_id(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        with _request_context(session_id="sess-1"):
            assert server._partition_key() == "sess-1"  # pyright: ignore[reportPrivateUsage]

    def test_local_missing_session_id_raises(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        with _request_context(), pytest.raises(RuntimeError, match="missing session_id"):
            server._partition_key()  # pyright: ignore[reportPrivateUsage]

    def test_local_ignores_user_id(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        with _request_context(session_id="sess-1", user_id="user-1"):
            assert server._partition_key() == "sess-1"  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize(
        ("session_id", "user_id"),
        [(None, "user-1"), ("", "user-1"), ("sess-1", None), ("sess-1", ""), (None, None)],
    )
    def test_hosted_requires_both_identifiers(self, session_id: str | None, user_id: str | None) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        server.config.is_hosted = True
        with (
            _request_context(call_id="call-1", session_id=session_id, user_id=user_id),
            pytest.raises(RuntimeError, match="missing session_id or user_id"),
        ):
            server._partition_key()  # pyright: ignore[reportPrivateUsage]

    def test_hosted_returns_composite_key(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        server.config.is_hosted = True
        with _request_context(call_id="call-1", session_id="sess-1", user_id="user-1"):
            assert server._partition_key() == ("sess-1", "user-1")  # pyright: ignore[reportPrivateUsage]

    async def test_hosted_keys_and_session_ids_preserve_identifier_values(self) -> None:
        agent = _make_agent(response_text="hi")
        server = InvocationsHostServer(agent)
        server.config.is_hosted = True
        identifiers = ["part", "part:part", "part,part", "[part]", 'part"\\', "part\n\t", "\u00e9", r"\u00e9", " part "]
        keys: set[tuple[str, str]] = set()
        request = _make_request({"message": "Hi"})

        for session_id, user_id in product(identifiers, repeat=2):
            with _request_context(call_id="call-1", session_id=session_id, user_id=user_id):
                key = server._partition_key()  # pyright: ignore[reportPrivateUsage]
                response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]

            assert isinstance(key, tuple)
            assert key == (session_id, user_id)
            assert key not in keys
            keys.add(key)
            assert response.status_code == 200
            session = agent.calls[-1]["session"]
            assert isinstance(session, AgentSession)
            expected_id = json.dumps([session_id, user_id], separators=(",", ":"))
            assert session.session_id == expected_id
            assert session.to_dict()["session_id"] == expected_id


# endregion


# region Handle invoke


class TestHandleInvoke:
    async def test_instance_context_lifetime_remains_caller_owned(self) -> None:
        events: list[str] = []
        response = AgentResponse(messages=[Message(role="assistant", contents=[Content.from_text("ok")])])
        server = InvocationsHostServer(_ContextAgent(events, response=response))

        with _request_context(session_id="sess-1"):
            await server._handle_invoke(_make_request({"message": "one"}))  # pyright: ignore[reportPrivateUsage]

        assert events == ["run"]

    async def test_factory_agent_context_lifetime_non_streaming(self) -> None:
        events: list[str] = []
        response = AgentResponse(messages=[Message(role="assistant", contents=[Content.from_text("ok")])])
        server = InvocationsHostServer(lambda: _ContextAgent(events, response=response))

        with _request_context(session_id="sess-1"):
            result = await server._handle_invoke(_make_request({"message": "one"}))  # pyright: ignore[reportPrivateUsage]

        assert json.loads(bytes(result.body)) == {"response": "ok"}
        assert events == ["enter", "run", "exit"]

    async def test_factory_agent_context_lifetime_until_stream_closes(self) -> None:
        events: list[str] = []
        updates = [
            AgentResponseUpdate(contents=[Content.from_text("one")]),
            AgentResponseUpdate(contents=[Content.from_text("two")]),
        ]
        server = InvocationsHostServer(lambda: _ContextAgent(events, stream_updates=updates))

        with _request_context(session_id="sess-1"):
            response = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                _make_request({"message": "one", "stream": True})
            )

        assert isinstance(response, StreamingResponse)
        iterator = cast(Any, response.body_iterator)
        assert '"text": "one"' in await anext(iterator)
        await iterator.aclose()

        assert events == ["enter", "run", "stream_close", "exit"]

    async def test_agent_callable_is_resolved_for_each_request(self) -> None:
        agents: list[_FakeAgent] = []

        def create_agent() -> _FakeAgent:
            agent = _make_agent(response_text=f"agent-{len(agents) + 1}")
            agents.append(agent)
            return agent

        server = InvocationsHostServer(create_agent)

        with _request_context(session_id="sess-1"):
            first = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                _make_request({"message": "one"})
            )
            second = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                _make_request({"message": "two"})
            )

        assert json.loads(bytes(first.body)) == {"response": "agent-1"}
        assert json.loads(bytes(second.body)) == {"response": "agent-2"}
        assert len(agents) == 2
        assert agents[0] is not agents[1]
        assert agents[0].calls[0]["session"].session_id == agents[1].calls[0]["session"].session_id

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("hosted", [False, True])
    async def test_reusing_session_persists_a_single_agent_session(
        self, hosted: bool, stream: bool, _durable_test_sessions: SessionStore
    ) -> None:
        agent = _make_agent(response_text="ok", stream_texts=["ok"])
        server = InvocationsHostServer(agent)
        server.config.is_hosted = hosted
        request = _make_request({"message": "Hi", "stream": stream})
        expected_id = '["sess-1","user-1"]' if hosted else "sess-1"

        with _request_context(call_id="call-1", session_id="sess-1", user_id="user-1"):
            for _ in range(2):
                response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]
                if isinstance(response, StreamingResponse):
                    assert '"text": "ok"' in await _collect_stream(response)
                else:
                    assert json.loads(bytes(response.body)) == {"response": "ok"}

        assert agent.calls[0]["session"].session_id == expected_id
        assert agent.calls[1]["session"].session_id == expected_id
        persisted = await _durable_test_sessions.get(server._storage_key(("sess-1", "user-1") if hosted else "sess-1"))
        assert persisted is not None
        assert persisted.session_id == expected_id

    async def test_missing_message_returns_400(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        request = _make_request({"stream": False})
        with _request_context(session_id="sess-1"):
            response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(response, Response)
        assert response.status_code == 400

    async def test_missing_message_streaming_returns_400(self) -> None:
        server = InvocationsHostServer(_make_agent(stream_texts=["a"]))
        request = _make_request({"stream": True})
        with _request_context(session_id="sess-1"):
            response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(response, Response)
        assert response.status_code == 400

    async def test_partition_key_failure_returns_500(self) -> None:
        server = InvocationsHostServer(_make_agent(response_text="hi"))
        request = _make_request({"message": "Hi"})
        # No session_id in the (local) context -> _partition_key raises -> 500.
        with _request_context():
            response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(response, Response)
        assert response.status_code == 500

    async def test_non_streaming_returns_agent_text(self) -> None:
        agent = _make_agent(response_text="Hello!")
        server = InvocationsHostServer(agent)
        request = _make_request({"message": "Hi", "stream": False})
        with _request_context(session_id="sess-1"):
            response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]

        assert isinstance(response, Response)
        assert response.status_code == 200
        assert json.loads(bytes(response.body)) == {"response": "Hello!"}
        # Agent is called with the message wrapped in a list and a scoped session.
        assert agent.calls[0]["messages"] == ["Hi"]
        assert agent.calls[0]["stream"] is False
        assert agent.calls[0]["session"].session_id == "sess-1"

    async def test_streaming_yields_update_text(self) -> None:
        agent = _make_agent(stream_texts=["Hel", "lo", "!"])
        server = InvocationsHostServer(agent)
        request = _make_request({"message": "Hi", "stream": True})
        with _request_context(session_id="sess-1"):
            response = await server._handle_invoke(request)  # pyright: ignore[reportPrivateUsage]

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        events = await _collect_stream(response)
        assert all(f'"text": "{part}"' in events for part in ("Hel", "lo", "!"))
        assert "event: done" in events
        assert agent.calls[0]["messages"] == "Hi"
        assert agent.calls[0]["stream"] is True

    async def test_stream_finalizes_with_released_core_that_lacks_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def updates() -> AsyncIterator[AgentResponseUpdate]:
            yield AgentResponseUpdate(contents=[Content.from_text("ready")], role="assistant")

        agent = _make_agent()
        stream = ResponseStream(updates(), finalizer=AgentResponse.from_updates)
        monkeypatch.setattr(agent, "run", MagicMock(return_value=stream))
        monkeypatch.delattr(ResponseStream, "close")
        server = InvocationsHostServer(agent)
        with _request_context(session_id="sandbox"):
            response = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                _make_request({"message": "Hi", "stream": True})
            )
            assert isinstance(response, StreamingResponse)
            events = await _collect_stream(response)

        assert "event: delta" in events
        assert "event: done" in events
        assert "event: error" not in events

    async def test_streaming_agent_factory_failure_yields_an_error_event(self) -> None:
        async def unavailable_agent() -> _FakeAgent:
            raise RuntimeError("Cannot initialize this agent.")

        server = InvocationsHostServer(unavailable_agent)
        with _request_context(session_id="sess-1"):
            response = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                _make_request({"message": "Hi", "stream": True})
            )

        assert isinstance(response, StreamingResponse)
        events = await _collect_stream(response)
        assert "event: error" in events
        assert "event: done" not in events

    async def test_session_is_loaded_across_requests(self) -> None:
        agent = _make_agent(response_text="ok")
        server = InvocationsHostServer(agent)

        with _request_context(session_id="sess-1"):
            await server._handle_invoke(_make_request({"message": "one"}))  # pyright: ignore[reportPrivateUsage]
            await server._handle_invoke(_make_request({"message": "two"}))  # pyright: ignore[reportPrivateUsage]

        assert agent.calls[0]["session"].session_id == agent.calls[1]["session"].session_id == "sess-1"

    async def test_two_hosts_observe_the_latest_persisted_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class SerializingStore(SessionStore):
            async def get(self, session_id: str) -> AgentSession | None:
                session = await super().get(session_id)
                return AgentSession.from_dict(session.to_dict()) if session is not None else None

            async def set(self, session_id: str, session: AgentSession) -> None:
                await super().set(session_id, AgentSession.from_dict(session.to_dict()))

        class CountingAgent(_FakeAgent):
            def run(
                self,
                messages: Any = None,
                *,
                stream: bool = False,
                session: AgentSession | None = None,
                **kwargs: Any,
            ) -> Any:
                assert session is not None
                session.state["turns"] = session.state.get("turns", 0) + 1
                return super().run(messages, stream=stream, session=session, **kwargs)

        store = SerializingStore()
        monkeypatch.setattr(
            "agent_framework_foundry_hosting._invocations.AgentSessionStoreProvider.get_store",
            lambda *_args, **_kwargs: store,
        )
        first_agent = CountingAgent(response=_make_agent(response_text="ok")._response)
        second_agent = CountingAgent(response=_make_agent(response_text="ok")._response)
        first = InvocationsHostServer(first_agent)
        second = InvocationsHostServer(second_agent)

        with _request_context(session_id="sandbox"):
            for host in (first, second, first):
                response = await host._handle_invoke(_make_request({"message": "next"}))  # pyright: ignore[reportPrivateUsage]
                assert response.status_code == 200

        assert [first_agent.calls[0]["session"].state["turns"], second_agent.calls[0]["session"].state["turns"]] == [
            1,
            2,
        ]
        assert first_agent.calls[1]["session"].state["turns"] == 3

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("first_session_id", "first_user_id", "second_session_id", "second_user_id"),
        [
            ("session:segment", "user", "session", "segment:user"),
            ("session,segment", "user", "session", "segment,user"),
            ("session", "first-user", "session", "second-user"),
            ("first-session", "user", "second-session", "user"),
        ],
    )
    async def test_hosted_sessions_preserve_identifier_boundaries(
        self,
        stream: bool,
        first_session_id: str,
        first_user_id: str,
        second_session_id: str,
        second_user_id: str,
    ) -> None:
        class PersistingAgent(_FakeAgent):
            def run(
                self,
                messages: Any = None,
                *,
                stream: bool = False,
                session: AgentSession | None = None,
                **kwargs: Any,
            ) -> Any:
                assert session is not None
                session.state.setdefault("turn", session.session_id)
                return super().run(messages, stream=stream, session=session, **kwargs)

        agent = PersistingAgent(
            response=_make_agent(response_text="ok")._response,
            stream_updates=[AgentResponseUpdate(contents=[Content.from_text("ok")])],
        )
        server = InvocationsHostServer(agent)
        server.config.is_hosted = True
        identifiers = [(first_session_id, first_user_id), (second_session_id, second_user_id)]
        sessions: list[AgentSession] = []

        for session_id, user_id in identifiers:
            with _request_context(call_id="call-1", session_id=session_id, user_id=user_id):
                response = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                    _make_request({"message": "Hi", "stream": stream})
                )
                if isinstance(response, StreamingResponse):
                    assert '"text": "ok"' in await _collect_stream(response)
                else:
                    assert json.loads(bytes(response.body)) == {"response": "ok"}
                assert response.status_code == 200

            session = agent.calls[-1]["session"]
            assert isinstance(session, AgentSession)
            assert session.state == {"turn": session.session_id}
            sessions.append(session)

        assert sessions[0] is not sessions[1]
        assert sessions[0].session_id != sessions[1].session_id

        for (session_id, user_id), session in zip(identifiers, sessions):
            with _request_context(call_id="call-2", session_id=session_id, user_id=user_id):
                response = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
                    _make_request({"message": "Continue", "stream": stream})
                )
                if isinstance(response, StreamingResponse):
                    assert '"text": "ok"' in await _collect_stream(response)
                else:
                    assert json.loads(bytes(response.body)) == {"response": "ok"}
                assert response.status_code == 200

            assert agent.calls[-1]["session"].session_id == session.session_id
            assert agent.calls[-1]["session"].state == {"turn": session.session_id}


async def test_invocations_recovers_agent_session_after_host_restart(
    _durable_test_sessions: SessionStore,
) -> None:
    class CountingAgent(_FakeAgent):
        def run(
            self,
            messages: Any = None,
            *,
            stream: bool = False,
            session: AgentSession | None = None,
            **kwargs: Any,
        ) -> Any:
            assert session is not None
            session.state["turns"] = session.state.get("turns", 0) + 1
            return super().run(messages, stream=stream, session=session, **kwargs)

    first_agent = CountingAgent(response=_make_agent(response_text="ok")._response)
    second_agent = CountingAgent(response=_make_agent(response_text="ok")._response)
    with _request_context(session_id="durable-session"):
        first = await InvocationsHostServer(first_agent)._handle_invoke(_make_request({"message": "first"}))  # pyright: ignore[reportPrivateUsage]
        second = await InvocationsHostServer(second_agent)._handle_invoke(_make_request({"message": "second"}))  # pyright: ignore[reportPrivateUsage]

    assert json.loads(bytes(first.body)) == json.loads(bytes(second.body)) == {"response": "ok"}
    assert second_agent.calls[0]["session"].state["turns"] == 2
    persisted = await _durable_test_sessions.get(
        InvocationsHostServer._storage_key("durable-session")  # pyright: ignore[reportPrivateUsage]
    )
    assert persisted is not None
    assert persisted.state["turns"] == 2


async def test_invocations_parser_and_hook_use_maf_runtime_options() -> None:
    agent = _make_agent(response_text="ok")

    async def parse(request: Request) -> InvocationRun:
        body = await request.json()
        return InvocationRun(messages=body["prompt"], options={"temperature": 0.8, "store": True})

    def prepare(_request: Request, options: dict[str, Any]) -> dict[str, Any]:
        options.pop("store")
        return options

    server = InvocationsHostServer(agent, parse_request=parse, prepare_options=prepare)
    with _request_context(session_id="sandbox"):
        result = await server._handle_invoke(_make_request({"prompt": "hello"}))  # pyright: ignore[reportPrivateUsage]

    assert result.status_code == 200
    assert agent.calls[0]["messages"] == "hello"
    assert agent.calls[0]["options"] == {"temperature": 0.8}


async def test_invocations_can_resume_a_native_workflow() -> None:
    def build_workflow(request: Request) -> Workflow:
        assert request is not None

        @executor(id="counter")
        async def counter(text: str, ctx: WorkflowContext[Any, str]) -> None:
            count = ctx.get_state("count", 0) + 1
            ctx.set_state("count", count)
            await ctx.yield_output(f"{count}: {text}")

        return WorkflowBuilder(name="invocations-counter", start_executor=counter, output_from="all").build()

    async def parse(request: Request) -> WorkflowTurn[str]:
        body = await request.json()
        return WorkflowTurn(input=body["message"], stream=body.get("stream", False))

    server = InvocationsHostServer(workflow=build_workflow, parse_request=parse)
    with _request_context(session_id="workflow-sandbox"):
        first = await server._handle_invoke(_make_request({"message": "first"}))  # pyright: ignore[reportPrivateUsage]
        second = await server._handle_invoke(_make_request({"message": "second"}))  # pyright: ignore[reportPrivateUsage]
    assert json.loads(bytes(first.body))["output"][0]["data"] == "1: first"
    assert json.loads(bytes(second.body))["output"][0]["data"] == "2: second"
    with _request_context(session_id="workflow-sandbox"):
        streamed = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
            _make_request({"message": "third", "stream": True})
        )
        assert isinstance(streamed, StreamingResponse)
        events = await _collect_stream(streamed)
    assert "event: output" in events
    assert "3: third" in events
    assert "event: done" in events


async def test_interrupted_stream_persists_partial_maf_session(_durable_test_sessions: SessionStore) -> None:
    class CountingStreamAgent(_FakeAgent):
        def run(
            self,
            messages: Any = None,
            *,
            stream: bool = False,
            session: AgentSession | None = None,
            **kwargs: Any,
        ) -> Any:
            assert session is not None
            session.state["turns"] = session.state.get("turns", 0) + 1
            return super().run(messages, stream=stream, session=session, **kwargs)

    agent = CountingStreamAgent(stream_updates=[AgentResponseUpdate(contents=[Content.from_text("part")])])
    server = InvocationsHostServer(agent)
    with _request_context(session_id="sandbox"):
        streamed = await server._handle_invoke(  # pyright: ignore[reportPrivateUsage]
            _make_request({"message": "hello", "stream": True})
        )
        assert isinstance(streamed, StreamingResponse)
        iterator = cast(Any, streamed.body_iterator)
        assert '"text": "part"' in await anext(iterator)
        await iterator.aclose()

    persisted = await _durable_test_sessions.get(server._storage_key("sandbox"))  # pyright: ignore[reportPrivateUsage]
    assert persisted is not None
    assert persisted.state["turns"] == 1


# endregion
