# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ClassVar, Generic, Literal, cast, overload

from agent_framework import (
    AgentMiddlewareLayer,
    AgentMiddlewareTypes,
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    AgentSession,
    BaseAgent,
    Content,
    ContextProvider,
    ResponseStream,
    ToolTypes,
    UsageDetails,
    load_settings,
    normalize_messages,
)
from agent_framework.exceptions import AgentException
from agent_framework.observability import AgentTelemetryLayer
from codex_sdk import (
    ApprovalMode,
    Codex,
    CodexOptions,
    ModelReasoningEffort,
    SandboxMode,
    Thread,
    ThreadOptions,
    TurnOptions,
)
from codex_sdk.events import (
    ItemCompletedEvent,
    ItemUpdatedEvent,
    ThreadErrorEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)
from codex_sdk.items import AgentMessageItem, ErrorItem, ReasoningItem
from pydantic import BaseModel

if sys.version_info >= (3, 13):
    from typing import TypeVar  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypeVar  # type: ignore # pragma: no cover
if sys.version_info >= (3, 11):
    from typing import TypedDict  # pragma: no cover
else:
    from typing_extensions import TypedDict  # pragma: no cover

logger = logging.getLogger("agent_framework.codex")

StructuredResponseFormatT = type[BaseModel] | Mapping[str, Any] | None


class CodexAgentSettings(TypedDict, total=False):
    """Codex Agent settings.

    Settings are resolved in this order: explicit keyword arguments, values from an
    explicitly provided .env file, then environment variables with the prefix
    ``CODEX_AGENT_``.

    Keys:
        codex_path: The path to the Codex CLI executable.
        model: The model to use for new threads.
        cwd: The working directory for Codex.
        approval_policy: Approval policy for Codex tool execution.
    """

    codex_path: str | None
    model: str | None
    cwd: str | None
    approval_policy: str | None


class CodexAgentOptions(TypedDict, total=False):
    """Codex Agent-specific options."""

    system_prompt: str
    """System prompt for the agent."""

    response_format: type[BaseModel] | Mapping[str, Any]
    """Structured output schema for the response text."""

    codex_path: str | Path
    """Path to the Codex CLI executable. Defaults to the CLI found on ``PATH``."""

    base_url: str
    """Optional base URL override for the Codex backend."""

    api_key: str
    """Optional API key override for the Codex process."""

    cwd: str | Path
    """Working directory for Codex. Defaults to the current working directory."""

    env: Mapping[str, str]
    """Environment variables to pass to the Codex CLI process."""

    model: str
    """Model to use for new threads."""

    sandbox_mode: str | SandboxMode
    """Sandbox mode for code execution."""

    model_reasoning_effort: str | ModelReasoningEffort
    """Reasoning effort preset for the model."""

    approval_policy: str | ApprovalMode
    """Approval policy for Codex tool execution."""

    additional_directories: list[str | Path]
    """Additional directories to expose to Codex."""

    config_overrides: Mapping[str, Any]
    """Thread-level Codex configuration overrides."""


OptionsT = TypeVar(
    "OptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="CodexAgentOptions",
    covariant=True,
)


def _merge_options(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge two options dictionaries, preferring non-None override values."""
    result = dict(base)
    if override is None:
        return result

    for key, value in override.items():
        if value is not None:
            result[key] = value
    return result


def _has_tools_configured(
    tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None,
) -> bool:
    """Return whether a tools argument contains any configured tools."""
    if tools is None:
        return False
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        return len(cast(Sequence[Any], tools)) > 0
    return True


def _normalize_path(value: str | Path | None) -> str | None:
    """Convert ``Path`` values to strings."""
    if value is None:
        return None
    return str(value)


def _normalize_paths(values: Sequence[str | Path] | None) -> list[str] | None:
    """Convert path sequences to string lists."""
    if values is None:
        return None
    return [str(value) for value in values]


def _response_format_to_schema(response_format: StructuredResponseFormatT) -> Mapping[str, Any] | None:
    """Convert Agent Framework response formats to Codex output schemas."""
    if response_format is None:
        return None
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return response_format.model_json_schema()
    if isinstance(response_format, Mapping):
        return dict(response_format)

    msg = f"Unsupported response_format type: {type(response_format)}"
    raise TypeError(msg)


def _usage_details_from_event_usage(usage: Any) -> UsageDetails | None:
    """Convert a Codex ``Usage`` object into Agent Framework ``UsageDetails``."""
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    cached_input_tokens = int(getattr(usage, "cached_input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    total_input_tokens = input_tokens + cached_input_tokens
    total_tokens = total_input_tokens + output_tokens
    if total_tokens == 0:
        return None

    return UsageDetails(
        input_token_count=total_input_tokens,
        output_token_count=output_tokens,
        total_token_count=total_tokens,
    )


def _text_delta(text_offsets: dict[str, int], *, item_id: str, text: str) -> str:
    """Return only the newly appended text for an updated item."""
    previous_offset = text_offsets.get(item_id, 0)
    if previous_offset >= len(text):
        return ""

    text_offsets[item_id] = len(text)
    return text[previous_offset:]


class RawCodexAgent(BaseAgent, Generic[OptionsT]):
    """OpenAI Codex Agent using Codex without middleware or telemetry layers.

    This is the core Codex agent implementation without Agent Framework telemetry
    or middleware composition. For most use cases, prefer :class:`CodexAgent`.

    The wrapper currently supports streaming, non-streaming responses,
    thread-backed session reuse, and structured output through ``response_format``.
    It intentionally does not yet expose Agent Framework custom ``tools=`` because
    the published Codex SDK does not provide a clean equivalent to the Claude and
    GitHub Copilot SDK host-managed tool bridges.
    """

    AGENT_PROVIDER_NAME: ClassVar[str] = "openai"

    def __init__(
        self,
        instructions: str | None = None,
        *,
        client: Codex | None = None,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        default_options: OptionsT | Mapping[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a RawCodexAgent instance.

        Args:
            instructions: System prompt for the agent.

        Keyword Args:
            client: Optional pre-configured Codex instance. If not provided, a new
                client will be created lazily when the agent starts.
            id: Unique identifier for the agent.
            name: Name of the agent.
            description: Description of the agent.
            context_providers: Context providers for the agent.
            middleware: Optional agent middleware configuration to preserve on the
                base agent. RawCodexAgent itself does not compose middleware.
            tools: Agent Framework tools. These are not currently supported by
                CodexAgent and will raise ``ValueError`` when configured.
            default_options: Default CodexAgentOptions including the system prompt,
                model, sandbox mode, and structured response format.
            env_file_path: Optional path to a ``.env`` file.
            env_file_encoding: Optional encoding for the ``.env`` file.
        """
        if _has_tools_configured(tools):
            msg = "CodexAgent does not currently support Agent Framework tools."
            raise ValueError(msg)

        super().__init__(
            id=id,
            name=name,
            description=description,
            context_providers=context_providers,
            middleware=middleware,
        )

        self._client = client
        self._owns_client = client is None

        opts: dict[str, Any] = dict(default_options) if default_options else {}
        if instructions is not None:
            opts["system_prompt"] = instructions

        codex_path = opts.pop("codex_path", None)
        model = opts.pop("model", None)
        cwd = opts.pop("cwd", None)
        approval_policy = opts.pop("approval_policy", None)
        self._client_option_overrides: dict[str, Any] = {
            "base_url": opts.pop("base_url", None),
            "api_key": opts.pop("api_key", None),
            "env": opts.pop("env", None),
        }

        self._settings = load_settings(
            CodexAgentSettings,
            env_prefix="CODEX_AGENT_",
            codex_path=codex_path,
            model=model,
            cwd=cwd,
            approval_policy=approval_policy,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )

        self._default_options = opts
        self._current_thread: Thread | None = None
        self._current_session_id: str | None = None
        self._current_thread_config: dict[str, Any] | None = None
        self._instruction_file_path: Path | None = None
        self._instruction_file_prompt: str | None = None

    async def __aenter__(self) -> RawCodexAgent[OptionsT]:
        """Start the agent when entering an async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the agent when exiting an async context manager."""
        await self.stop()

    async def start(self) -> None:
        """Create a Codex client if one is not already configured."""
        if self._client is not None:
            return

        try:
            self._client = self._create_codex_client()
            self._owns_client = True
        except Exception as ex:
            raise AgentException(f"Failed to create Codex client: {ex}") from ex

    async def stop(self) -> None:
        """Reset local Codex thread state and release temporary resources."""
        self._current_thread = None
        self._current_session_id = None
        self._current_thread_config = None
        self._cleanup_instruction_file()

        if self._owns_client:
            self._client = None

    def _cleanup_instruction_file(self) -> None:
        """Delete the temporary instructions file, if any."""
        if self._instruction_file_path is None:
            return

        with contextlib.suppress(FileNotFoundError):
            self._instruction_file_path.unlink()
        self._instruction_file_path = None
        self._instruction_file_prompt = None

    def _ensure_instruction_file(self, system_prompt: str | None) -> str | None:
        """Write the current instructions to a temporary file for Codex."""
        if not system_prompt:
            self._cleanup_instruction_file()
            return None

        if self._instruction_file_prompt == system_prompt and self._instruction_file_path is not None:
            return str(self._instruction_file_path)

        self._cleanup_instruction_file()
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            prefix="codex_instructions_",
            delete=False,
        ) as instructions_file:
            instructions_file.write(system_prompt)
            self._instruction_file_path = Path(instructions_file.name)
            self._instruction_file_prompt = system_prompt

        return str(self._instruction_file_path)

    def _create_codex_client(self) -> Codex:
        """Create a configured Codex client instance."""
        return Codex(
            options=CodexOptions(
                codex_path_override=self._settings.get("codex_path"),
                base_url=self._client_option_overrides.get("base_url"),
                api_key=self._client_option_overrides.get("api_key"),
                env=self._client_option_overrides.get("env"),
            )
        )

    def _build_thread_config(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Build the effective thread configuration for the current invocation."""
        config: dict[str, Any] = {}

        if model := self._settings.get("model"):
            config["model"] = model
        if cwd := self._settings.get("cwd"):
            config["working_directory"] = cwd
        if approval_policy := self._settings.get("approval_policy"):
            config["approval_policy"] = approval_policy

        if options.get("model") is not None:
            config["model"] = options["model"]
        if options.get("cwd") is not None:
            config["working_directory"] = _normalize_path(options["cwd"])
        if options.get("approval_policy") is not None:
            config["approval_policy"] = options["approval_policy"]
        if options.get("sandbox_mode") is not None:
            config["sandbox_mode"] = options["sandbox_mode"]
        if options.get("model_reasoning_effort") is not None:
            config["model_reasoning_effort"] = options["model_reasoning_effort"]
        if options.get("additional_directories") is not None:
            config["additional_directories"] = _normalize_paths(options["additional_directories"])
        if options.get("config_overrides") is not None:
            config["config_overrides"] = dict(options["config_overrides"])
        if options.get("system_prompt") is not None:
            config["system_prompt"] = options["system_prompt"]

        return config

    def _build_thread_options(self, config: Mapping[str, Any]) -> ThreadOptions:
        """Convert a normalized thread configuration into ``ThreadOptions``."""
        options: dict[str, Any] = {}

        if config.get("model") is not None:
            options["model"] = config["model"]
        if config.get("working_directory") is not None:
            options["working_directory"] = config["working_directory"]
        if config.get("approval_policy") is not None:
            options["approval_policy"] = config["approval_policy"]
        if config.get("sandbox_mode") is not None:
            options["sandbox_mode"] = config["sandbox_mode"]
        if config.get("model_reasoning_effort") is not None:
            options["model_reasoning_effort"] = config["model_reasoning_effort"]
        if config.get("additional_directories") is not None:
            options["additional_directories"] = config["additional_directories"]
        if config.get("config_overrides") is not None:
            options["config_overrides"] = config["config_overrides"]
        if instruction_file := self._ensure_instruction_file(cast(str | None, config.get("system_prompt"))):
            options["model_instructions_file"] = instruction_file

        return ThreadOptions(**options)

    def _get_or_create_thread(self, session: AgentSession, options: Mapping[str, Any]) -> Thread:
        """Get the current thread or create/resume one for the session."""
        if self._client is None:
            raise RuntimeError("Codex client not initialized. Call start() first.")

        session_id = session.service_session_id
        thread_config = self._build_thread_config(options)

        if (
            self._current_thread is not None
            and session_id is not None
            and session_id == self._current_session_id
            and thread_config == self._current_thread_config
        ):
            return self._current_thread

        thread_options = self._build_thread_options(thread_config)
        if session_id:
            thread = self._client.resume_thread(session_id, options=thread_options)
        else:
            thread = self._client.start_thread(options=thread_options)

        effective_session_id = session_id or thread.id
        if effective_session_id is not None:
            session.service_session_id = effective_session_id

        self._current_thread = thread
        self._current_session_id = effective_session_id
        self._current_thread_config = thread_config
        return thread

    @staticmethod
    def _format_prompt(messages: list[Any] | None) -> str:
        """Format framework messages into a single Codex prompt string."""
        if not messages:
            return ""

        rendered_messages: list[str] = []
        for message in messages:
            text = message.text.strip() if message.text else ""
            if not text:
                continue
            if len(messages) == 1 and message.role == "user":
                rendered_messages.append(text)
            else:
                rendered_messages.append(f"{message.role.capitalize()}: {text}")

        return "\n\n".join(rendered_messages)

    @property
    def default_options(self) -> dict[str, Any]:
        """Expose default options with ``instructions`` for telemetry compatibility."""
        options = dict(self._default_options)
        system_prompt = options.pop("system_prompt", None)
        if system_prompt is not None:
            options["instructions"] = system_prompt
        return options

    @staticmethod
    def _finalize_response(
        updates: Sequence[AgentResponseUpdate],
        *,
        response_format: StructuredResponseFormatT = None,
    ) -> AgentResponse[Any]:
        """Build the final AgentResponse from collected updates."""
        return AgentResponse.from_updates(updates, output_format_type=response_format)

    @overload
    def run(  # type: ignore[override]
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = None,
        options: OptionsT | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...

    @overload
    def run(  # type: ignore[override]
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        options: OptionsT | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        options: OptionsT | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Run the agent with the given messages.

        Args:
            messages: The messages to process.

        Keyword Args:
            stream: If True, returns a streaming response. Otherwise returns the
                final AgentResponse.
            session: The conversation session. Reusing a session reuses the same
                Codex thread via ``service_session_id``.
            options: Per-run Codex options. Thread-level options such as ``model``
                or ``sandbox_mode`` apply when starting or resuming a thread.
                ``response_format`` is translated into Codex ``output_schema``.
            tools: Agent Framework tools. These are not currently supported by
                CodexAgent and will raise ``AgentException`` when configured.
            function_invocation_kwargs: Accepted for interface compatibility but
                not used by CodexAgent.
            client_kwargs: Accepted for interface compatibility but not used by
                CodexAgent.
            kwargs: Additional compatibility arguments. ``background`` and
                ``continuation_token`` are rejected because the current Codex SDK
                does not expose a detached execution model.
        """
        del function_invocation_kwargs, client_kwargs

        if _has_tools_configured(tools):
            raise AgentException("CodexAgent does not currently support Agent Framework tools.")
        if kwargs.get("background"):
            raise AgentException("CodexAgent does not currently support background execution.")
        if kwargs.get("continuation_token") is not None:
            raise AgentException("CodexAgent does not currently support continuation tokens.")

        merged_options = _merge_options(self._default_options, dict(options) if options else None)
        response_format = cast(StructuredResponseFormatT, merged_options.get("response_format"))
        response = ResponseStream(
            self._get_stream(messages, session=session, options=merged_options),
            finalizer=lambda updates: self._finalize_response(updates, response_format=response_format),
        )
        if stream:
            return response
        return response.get_final_response()

    async def _get_stream(
        self,
        messages: AgentRunInputs | None = None,
        *,
        session: AgentSession | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterable[AgentResponseUpdate]:
        """Internal streaming implementation."""
        session = session or self.create_session()
        merged_options = _merge_options(self._default_options, options)

        if self._client is None:
            await self.start()

        thread = self._get_or_create_thread(session, merged_options)
        turn_options = TurnOptions(output_schema=_response_format_to_schema(merged_options.get("response_format")))
        prompt = self._format_prompt(normalize_messages(messages))
        text_offsets: dict[str, int] = {}

        async for event in thread.run_streamed_events(prompt, turn_options):
            if isinstance(event, ThreadErrorEvent):
                raise AgentException(f"Codex thread error: {event.message}")

            if isinstance(event, TurnFailedEvent):
                error_message = getattr(event.error, "message", str(event.error))
                raise AgentException(f"Codex turn failed: {error_message}")

            if isinstance(event, TurnCompletedEvent):
                if usage_details := _usage_details_from_event_usage(event.usage):
                    yield AgentResponseUpdate(
                        role="assistant",
                        contents=[Content.from_usage(usage_details=usage_details, raw_representation=event)],
                        raw_representation=event,
                    )
                continue

            if not isinstance(event, (ItemUpdatedEvent, ItemCompletedEvent)):
                continue

            item = event.item
            if isinstance(item, AgentMessageItem):
                if delta := _text_delta(text_offsets, item_id=item.id, text=item.text):
                    yield AgentResponseUpdate(
                        role="assistant",
                        contents=[Content.from_text(text=delta, raw_representation=event)],
                        raw_representation=event,
                    )
            elif isinstance(item, ReasoningItem):
                if delta := _text_delta(text_offsets, item_id=item.id, text=item.text):
                    yield AgentResponseUpdate(
                        role="assistant",
                        contents=[Content.from_text_reasoning(text=delta, raw_representation=event)],
                        raw_representation=event,
                    )
            elif isinstance(item, ErrorItem):
                raise AgentException(f"Codex error: {item.message}")


class CodexAgent(AgentMiddlewareLayer, AgentTelemetryLayer, RawCodexAgent[OptionsT], Generic[OptionsT]):
    """OpenAI Codex Agent with middleware and OpenTelemetry instrumentation."""

    def __init__(
        self,
        instructions: str | None = None,
        *,
        client: Codex | None = None,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        default_options: OptionsT | Mapping[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a CodexAgent with middleware and telemetry support.

        Args:
            instructions: System prompt for the agent.

        Keyword Args:
            client: Optional pre-configured Codex instance. If not provided, a new
                client will be created lazily when the agent starts.
            id: Unique identifier for the agent.
            name: Name of the agent.
            description: Description of the agent.
            context_providers: Context providers for the agent.
            middleware: Optional per-agent middleware for wrapping Codex runs.
            tools: Agent Framework tools. These are not currently supported by
                CodexAgent and will raise ``ValueError`` when configured.
            default_options: Default CodexAgentOptions including thread settings
                and structured response format.
            env_file_path: Optional path to a ``.env`` file.
            env_file_encoding: Optional encoding for the ``.env`` file.
        """
        super().__init__(
            instructions=instructions,
            client=client,
            id=id,
            name=name,
            description=description,
            context_providers=context_providers,
            middleware=middleware,
            tools=tools,
            default_options=default_options,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )

    @overload  # type: ignore[override]
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        options: OptionsT | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        continuation_token: Any | None = None,
        background: bool = False,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...

    @overload  # type: ignore[override]
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        options: OptionsT | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        continuation_token: Any | None = None,
        background: bool = False,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(  # pyright: ignore[reportIncompatibleMethodOverride]  # type: ignore[override]
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        options: OptionsT | None = None,
        tools: ToolTypes | Callable[..., Any] | str | Sequence[ToolTypes | Callable[..., Any] | str] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        continuation_token: Any | None = None,
        background: bool = False,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Run the Codex agent with middleware and telemetry enabled."""
        if background:
            raise AgentException("CodexAgent does not currently support background execution.")
        if continuation_token is not None:
            raise AgentException("CodexAgent does not currently support continuation tokens.")

        super_run = cast(
            "Callable[..., Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]]",
            super().run,
        )
        return super_run(
            messages=messages,
            stream=stream,
            session=session,
            middleware=middleware,
            options=options,
            tools=tools,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            function_invocation_kwargs=function_invocation_kwargs,
            client_kwargs=client_kwargs,
            **kwargs,
        )
