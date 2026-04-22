# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agent_framework import AgentResponseUpdate, AgentSession
from agent_framework.exceptions import AgentException
from pydantic import BaseModel

from agent_framework_codex import CodexAgent, CodexAgentOptions, CodexAgentSettings


class TestCodexAgentSettings:
    """Tests for CodexAgentSettings."""

    def test_default_values(self) -> None:
        """Test default settings values are None."""
        from agent_framework import load_settings

        settings = load_settings(CodexAgentSettings, env_prefix="CODEX_AGENT_")
        assert settings["codex_path"] is None
        assert settings["model"] is None
        assert settings["cwd"] is None
        assert settings["approval_policy"] is None

    def test_explicit_values(self) -> None:
        """Test explicit values override defaults."""
        from agent_framework import load_settings

        settings = load_settings(
            CodexAgentSettings,
            env_prefix="CODEX_AGENT_",
            codex_path="/usr/local/bin/codex",
            model="gpt-5.4",
            cwd="/tmp/project",
            approval_policy="on-request",
        )
        assert settings["codex_path"] == "/usr/local/bin/codex"
        assert settings["model"] == "gpt-5.4"
        assert settings["cwd"] == "/tmp/project"
        assert settings["approval_policy"] == "on-request"

    def test_env_variable_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test settings load from CODEX_AGENT_* environment variables."""
        from agent_framework import load_settings

        monkeypatch.setenv("CODEX_AGENT_MODEL", "gpt-5.4")
        settings = load_settings(CodexAgentSettings, env_prefix="CODEX_AGENT_")
        assert settings["model"] == "gpt-5.4"


class TestCodexAgentInit:
    """Tests for CodexAgent initialization."""

    def test_default_initialization(self) -> None:
        """Test the agent initializes with defaults."""
        agent = CodexAgent()
        assert agent.id is not None
        assert agent.name is None
        assert agent.description is None

    def test_with_name_and_description(self) -> None:
        """Test the agent stores name and description."""
        agent = CodexAgent(name="codex-agent", description="A test agent")
        assert agent.name == "codex-agent"
        assert agent.description == "A test agent"

    def test_with_instructions_parameter(self) -> None:
        """Test the instructions parameter maps to the system prompt option."""
        agent = CodexAgent(instructions="You are a helpful coding assistant.")
        assert agent._default_options.get("system_prompt") == "You are a helpful coding assistant."  # type: ignore[reportPrivateUsage]

    def test_with_default_options(self) -> None:
        """Test thread settings are loaded from default options."""
        options: CodexAgentOptions = {
            "model": "gpt-5.4",
            "approval_policy": "on-request",
        }
        agent = CodexAgent(default_options=options)
        assert agent._settings["model"] == "gpt-5.4"  # type: ignore[reportPrivateUsage]
        assert agent._settings["approval_policy"] == "on-request"  # type: ignore[reportPrivateUsage]

    def test_tools_are_not_supported_in_constructor(self) -> None:
        """Test constructor tools raise a clear error."""

        def greet(name: str) -> str:
            return f"Hello, {name}"

        with pytest.raises(ValueError, match="does not currently support Agent Framework tools"):
            CodexAgent(tools=[greet])

    def test_default_options_maps_system_prompt_to_instructions(self) -> None:
        """Test telemetry-facing default_options uses instructions."""
        agent = CodexAgent(instructions="Be concise")
        default_options = agent.default_options
        assert default_options["instructions"] == "Be concise"
        assert "system_prompt" not in default_options


def _make_mock_thread(events: list[Any], *, thread_id: str = "thread-123") -> MagicMock:
    """Create a mock Codex thread that yields the provided events."""

    async def _stream_events(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        for event in events:
            yield event

    thread = MagicMock()
    thread.id = thread_id
    thread.run_streamed_events = _stream_events
    return thread


def _make_mock_codex(mock_thread: MagicMock) -> MagicMock:
    """Create a mock Codex client returning the provided thread."""
    codex = MagicMock()
    codex.start_thread.return_value = mock_thread
    codex.resume_thread.return_value = mock_thread
    return codex


class TestCodexAgentLifecycle:
    """Tests for Codex lifecycle helpers."""

    async def test_start_creates_client(self) -> None:
        """Test start creates a Codex client with the configured options."""
        with patch("agent_framework_codex._agent.Codex") as codex_type:
            agent = CodexAgent(
                default_options={
                    "base_url": "https://example.invalid/v1",
                    "api_key": "secret",
                    "env": {"CODEX_API_KEY": "from-env"},
                }
            )
            await agent.start()

            codex_type.assert_called_once()
            options = codex_type.call_args.kwargs["options"]
            assert options.base_url == "https://example.invalid/v1"
            assert options.api_key == "secret"
            assert options.env == {"CODEX_API_KEY": "from-env"}

    async def test_stop_resets_thread_state(self) -> None:
        """Test stop clears thread state and owned clients."""
        agent = CodexAgent()
        agent._client = MagicMock()  # type: ignore[reportPrivateUsage]
        agent._owns_client = True  # type: ignore[reportPrivateUsage]
        agent._current_thread = MagicMock()  # type: ignore[reportPrivateUsage]
        agent._current_session_id = "thread-123"  # type: ignore[reportPrivateUsage]
        agent._current_thread_config = {"model": "gpt-5.4"}  # type: ignore[reportPrivateUsage]

        await agent.stop()

        assert agent._client is None  # type: ignore[reportPrivateUsage]
        assert agent._current_thread is None  # type: ignore[reportPrivateUsage]
        assert agent._current_session_id is None  # type: ignore[reportPrivateUsage]
        assert agent._current_thread_config is None  # type: ignore[reportPrivateUsage]


class TestCodexAgentRun:
    """Tests for the CodexAgent run method."""

    async def test_run_with_string_message(self) -> None:
        """Test run returns the streamed assistant text."""
        from codex_sdk.events import ItemUpdatedEvent, TurnCompletedEvent, Usage
        from codex_sdk.items import AgentMessageItem

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text="Hello!"),
            ),
            TurnCompletedEvent(
                type="turn.completed",
                usage=Usage(input_tokens=10, cached_input_tokens=0, output_tokens=5),
            ),
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            response = await agent.run("Hello")
            assert response.text == "Hello!"
            assert response.usage_details is not None
            assert response.usage_details["input_token_count"] == 10
            assert response.usage_details["output_token_count"] == 5
            assert response.usage_details["total_token_count"] == 15

    async def test_run_stream_yields_text_deltas(self) -> None:
        """Test streaming emits only the newly appended text."""
        from codex_sdk.events import ItemUpdatedEvent, TurnCompletedEvent, Usage
        from codex_sdk.items import AgentMessageItem

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text="Hello"),
            ),
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text="Hello, world"),
            ),
            TurnCompletedEvent(
                type="turn.completed",
                usage=Usage(input_tokens=2, cached_input_tokens=0, output_tokens=2),
            ),
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            updates: list[AgentResponseUpdate] = []
            async for update in agent.run("Hello", stream=True):
                updates.append(update)

            assert [update.text for update in updates[:2]] == ["Hello", ", world"]

    async def test_run_stream_yields_reasoning_deltas(self) -> None:
        """Test streaming yields reasoning updates as reasoning text content."""
        from codex_sdk.events import ItemUpdatedEvent, TurnCompletedEvent, Usage
        from codex_sdk.items import AgentMessageItem, ReasoningItem

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=ReasoningItem(id="reason-1", type="reasoning", text="Thinking"),
            ),
            ItemUpdatedEvent(
                type="item.updated",
                item=ReasoningItem(id="reason-1", type="reasoning", text="Thinking through it"),
            ),
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text="Done"),
            ),
            TurnCompletedEvent(
                type="turn.completed",
                usage=Usage(input_tokens=4, cached_input_tokens=0, output_tokens=1),
            ),
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            updates = [update async for update in agent.run("Hello", stream=True)]
            assert updates[0].contents[0].type == "text_reasoning"
            assert updates[0].contents[0].text == "Thinking"
            assert updates[1].contents[0].text == " through it"
            assert updates[2].text == "Done"

    async def test_run_with_existing_session_resumes_thread(self) -> None:
        """Test run resumes an existing thread when service_session_id is present."""
        from codex_sdk.events import ItemUpdatedEvent, TurnCompletedEvent, Usage
        from codex_sdk.items import AgentMessageItem

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text="Response"),
            ),
            TurnCompletedEvent(
                type="turn.completed",
                usage=Usage(input_tokens=10, cached_input_tokens=0, output_tokens=5),
            ),
        ]

        mock_thread = _make_mock_thread(events, thread_id="existing-thread")
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            session = AgentSession(service_session_id="existing-thread")
            await agent.run("Hello", session=session)

            mock_codex.resume_thread.assert_called_once()
            assert mock_codex.resume_thread.call_args.args[0] == "existing-thread"

    async def test_run_updates_session_with_thread_id(self) -> None:
        """Test new sessions capture the Codex thread ID."""
        from codex_sdk.events import ItemUpdatedEvent, TurnCompletedEvent, Usage
        from codex_sdk.items import AgentMessageItem

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text="Response"),
            ),
            TurnCompletedEvent(
                type="turn.completed",
                usage=Usage(input_tokens=10, cached_input_tokens=0, output_tokens=5),
            ),
        ]

        mock_thread = _make_mock_thread(events, thread_id="thread-xyz")
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            session = agent.create_session()
            await agent.run("Hello", session=session)
            assert session.service_session_id == "thread-xyz"

    async def test_response_format_is_parsed_into_response_value(self) -> None:
        """Test response_format is forwarded for structured output parsing."""
        from codex_sdk.events import ItemUpdatedEvent, TurnCompletedEvent, Usage
        from codex_sdk.items import AgentMessageItem

        class StructuredAnswer(BaseModel):
            answer: str

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=AgentMessageItem(id="msg-1", type="agent_message", text='{"answer":"pytest"}'),
            ),
            TurnCompletedEvent(
                type="turn.completed",
                usage=Usage(input_tokens=4, cached_input_tokens=0, output_tokens=2),
            ),
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            response = await agent.run("Hello", options={"response_format": StructuredAnswer})
            assert isinstance(response.value, StructuredAnswer)
            assert response.value.answer == "pytest"

    async def test_tools_are_not_supported_at_run_time(self) -> None:
        """Test run rejects Agent Framework tools."""
        with patch("agent_framework_codex._agent.Codex"):
            agent = CodexAgent()

            def greet(name: str) -> str:
                return f"Hello, {name}"

            with pytest.raises(AgentException, match="does not currently support Agent Framework tools"):
                await agent.run("Hello", tools=[greet])

    async def test_background_is_not_supported(self) -> None:
        """Test background execution raises a clear error."""
        with patch("agent_framework_codex._agent.Codex"):
            agent = CodexAgent()
            with pytest.raises(AgentException, match="does not currently support background execution"):
                await agent.run("Hello", background=True)


class TestCodexAgentErrorHandling:
    """Tests for CodexAgent error handling."""

    async def test_run_stream_raises_on_error_item(self) -> None:
        """Test ErrorItem raises AgentException."""
        from codex_sdk.events import ItemUpdatedEvent
        from codex_sdk.items import ErrorItem

        events = [
            ItemUpdatedEvent(
                type="item.updated",
                item=ErrorItem(id="err-1", type="error", message="API rate limit exceeded"),
            )
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            with pytest.raises(AgentException, match="API rate limit exceeded"):
                async for _ in agent.run("Hello", stream=True):
                    pass

    async def test_run_stream_raises_on_turn_failed(self) -> None:
        """Test TurnFailedEvent raises AgentException."""
        from codex_sdk.events import ThreadError, TurnFailedEvent

        events = [
            TurnFailedEvent(
                type="turn.failed",
                error=ThreadError(message="Model not found"),
            )
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            with pytest.raises(AgentException, match="Model not found"):
                async for _ in agent.run("Hello", stream=True):
                    pass

    async def test_run_stream_raises_on_thread_error(self) -> None:
        """Test ThreadErrorEvent raises AgentException."""
        from codex_sdk.events import ThreadErrorEvent

        events = [
            ThreadErrorEvent(
                type="error",
                message="Connection lost",
            )
        ]

        mock_thread = _make_mock_thread(events)
        mock_codex = _make_mock_codex(mock_thread)

        with patch("agent_framework_codex._agent.Codex", return_value=mock_codex):
            agent = CodexAgent()
            with pytest.raises(AgentException, match="Connection lost"):
                async for _ in agent.run("Hello", stream=True):
                    pass
