# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Union, cast

import openai
from dotenv import dotenv_values
from openai import (
    AsyncOpenAI,
    AsyncStream,
    _legacy_response,  # type: ignore
)
from openai.types import Completion
from openai.types.audio import Transcription
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.images_response import ImagesResponse
from openai.types.responses.response import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent
from packaging.version import parse

from .._serialization import SerializationMixin
from .._settings import SecretString, load_settings
from .._telemetry import AGENT_FRAMEWORK_USER_AGENT, APP_INFO, USER_AGENT_KEY, prepend_agent_framework_to_user_agent
from .._tools import FunctionTool
from ..azure._entra_id_authentication import AzureCredentialTypes, AzureTokenProvider

if TYPE_CHECKING:
    from ..azure._shared import AzureOpenAISettings

logger: logging.Logger = logging.getLogger("agent_framework.openai")


RESPONSE_TYPE = Union[
    ChatCompletion,
    Completion,
    AsyncStream[ChatCompletionChunk],
    AsyncStream[Completion],
    list[Any],
    ImagesResponse,
    Response,
    AsyncStream[ResponseStreamEvent],
    Transcription,
    _legacy_response.HttpxBinaryResponseContent,
]

OPTION_TYPE = dict[str, Any]
OpenAIChatBackend = Literal["openai", "azure_openai"]
OpenAIResponsesBackend = Literal["openai", "azure_openai", "foundry", "foundry_hosted_agent"]

if sys.version_info >= (3, 11):
    from typing import TypedDict  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypedDict  # type: ignore # pragma: no cover


def _check_openai_version_for_callable_api_key() -> None:
    """Check if OpenAI version supports callable API keys.

    Callable API keys require OpenAI >= 1.106.0.
    If the version is too old, raise a ValueError with helpful message.
    """
    try:
        current_version = parse(openai.__version__)
        min_required_version = parse("1.106.0")

        if current_version < min_required_version:
            raise ValueError(
                f"Callable API keys require OpenAI SDK >= 1.106.0, but you have {openai.__version__}. "
                f"Please upgrade with 'pip install openai>=1.106.0' or provide a string API key instead. "
                f"Note: If you're using mem0ai, you may need to upgrade to mem0ai>=1.0.0 "
                f"to allow newer OpenAI versions."
            )
    except ValueError:
        raise  # Re-raise our own exception
    except Exception as e:
        logger.warning(f"Could not check OpenAI version for callable API key support: {e}")


def _resolve_env_value(
    *env_var_names: str,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> str | None:
    """Resolve the first matching environment variable from dotenv or process env."""
    if env_file_path:
        loaded_values = dotenv_values(dotenv_path=env_file_path, encoding=env_file_encoding or "utf-8")
        for env_var_name in env_var_names:
            dotenv_value = loaded_values.get(env_var_name)
            if dotenv_value is not None:
                return str(dotenv_value)

    for env_var_name in env_var_names:
        env_value = os.getenv(env_var_name)
        if env_value is not None:
            return env_value
    return None


def resolve_responses_model_id(
    *,
    model_id: str | None,
    backend: OpenAIResponsesBackend,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> str | None:
    """Resolve the responses model/deployment identifier for the selected backend."""
    env_names = (
        ("OPENAI_RESPONSES_MODEL_ID",)
        if backend == "openai"
        else ("RESPONSES_DEPLOYMENT_NAME", "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
    )
    settings = load_settings(
        _ModelIdSetting,
        model_id=model_id,
        env_var_names={"model_id": env_names},
        env_file_path=env_file_path,
        env_file_encoding=env_file_encoding,
    )
    return settings.get("model_id")


def load_azure_openai_settings(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    token_endpoint: str | None = None,
    chat_deployment_name: str | None = None,
    responses_deployment_name: str | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
    required_fields: Sequence[str | tuple[str, ...]] | None = None,
) -> AzureOpenAISettings:
    """Load Azure OpenAI settings with support for SDK-native env var aliases."""
    from ..azure._shared import AzureOpenAISettings

    if api_version is None:
        sdk_api_version = _resolve_env_value(
            "OPENAI_API_VERSION",
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        legacy_api_version = _resolve_env_value(
            "AZURE_OPENAI_API_VERSION",
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        if sdk_api_version is None and legacy_api_version is not None:
            # TODO(Copilot): Delete once ``AZURE_OPENAI_API_VERSION`` compatibility is removed.
            warnings.warn(
                "Using 'AZURE_OPENAI_API_VERSION' is deprecated; prefer 'OPENAI_API_VERSION'.",
                DeprecationWarning,
                stacklevel=3,
            )

    return load_settings(
        AzureOpenAISettings,
        env_prefix="AZURE_OPENAI_",
        api_key=api_key,
        base_url=base_url,
        endpoint=endpoint,
        chat_deployment_name=chat_deployment_name,
        responses_deployment_name=responses_deployment_name,
        token_endpoint=token_endpoint,
        api_version=api_version,
        env_var_names={"api_version": ("OPENAI_API_VERSION", "AZURE_OPENAI_API_VERSION")},
        required_fields=required_fields,
        env_file_path=env_file_path,
        env_file_encoding=env_file_encoding,
    )


class OpenAISettings(TypedDict, total=False):
    """OpenAI environment settings.

    Settings are resolved in this order: explicit keyword arguments, values from an
    explicitly provided .env file, then environment variables with the prefix
    'OPENAI_'. If settings are missing after resolution, validation will fail.

    Keyword Args:
        api_key: OpenAI API key, see https://platform.openai.com/account/api-keys.
            Can be set via environment variable OPENAI_API_KEY.
        base_url: The base URL for the OpenAI API.
            Can be set via environment variable OPENAI_BASE_URL.
        org_id: This is usually optional unless your account belongs to multiple organizations.
            Can be set via environment variable OPENAI_ORG_ID.
        chat_model_id: The OpenAI chat model ID to use, for example, gpt-3.5-turbo or gpt-4.
            Can be set via environment variable OPENAI_CHAT_MODEL_ID.
        responses_model_id: The OpenAI responses model ID to use, for example, gpt-4o or o1.
            Can be set via environment variable OPENAI_RESPONSES_MODEL_ID.
        embedding_model_id: The OpenAI embedding model ID to use, for example, text-embedding-3-small.
            Can be set via environment variable OPENAI_EMBEDDING_MODEL_ID.

    Examples:
        .. code-block:: python

            from agent_framework.openai import OpenAISettings

            # Using environment variables
            # Set OPENAI_API_KEY=sk-...
            # Set OPENAI_CHAT_MODEL_ID=gpt-4
            settings = load_settings(OpenAISettings, env_prefix="OPENAI_")

            # Or passing parameters directly
            settings = load_settings(OpenAISettings, env_prefix="OPENAI_", api_key="sk-...", chat_model_id="gpt-4")

            # Or loading from a .env file
            settings = load_settings(OpenAISettings, env_prefix="OPENAI_", env_file_path="path/to/.env")
    """

    api_key: SecretString | Callable[[], str | Awaitable[str]] | None
    base_url: str | None
    org_id: str | None
    chat_model_id: str | None
    responses_model_id: str | None
    embedding_model_id: str | None


class _ModelIdSetting(TypedDict, total=False):
    model_id: str | None


class OpenAIBase(SerializationMixin):
    """Base class for OpenAI Clients."""

    INJECTABLE: ClassVar[set[str]] = {"client"}

    def __init__(self, *, model_id: str | None = None, client: AsyncOpenAI | None = None, **kwargs: Any) -> None:
        """Initialize OpenAIBase.

        Keyword Args:
            client: The AsyncOpenAI client instance.
            model_id: The AI model ID to use.
            **kwargs: Additional keyword arguments.
        """
        self.client = client
        self.model_id = None
        if model_id:
            self.model_id = model_id.strip()

        # Call super().__init__() to continue MRO chain (e.g., RawChatClient)
        # Extract known kwargs that belong to other base classes
        additional_properties = kwargs.pop("additional_properties", None)
        middleware = kwargs.pop("middleware", None)
        instruction_role = kwargs.pop("instruction_role", None)
        function_invocation_configuration = kwargs.pop("function_invocation_configuration", None)

        # Build super().__init__() args
        super_kwargs = {}
        if additional_properties is not None:
            super_kwargs["additional_properties"] = additional_properties
        if middleware is not None:
            super_kwargs["middleware"] = middleware
        if function_invocation_configuration is not None:
            super_kwargs["function_invocation_configuration"] = function_invocation_configuration

        # Call super().__init__() with filtered kwargs
        super().__init__(**super_kwargs)

        # Store instruction_role and any remaining kwargs as instance attributes
        if instruction_role is not None:
            self.instruction_role = instruction_role
        for key, value in kwargs.items():
            setattr(self, key, value)

    async def _initialize_client(self) -> None:
        """Initialize OpenAI client asynchronously.

        Override in subclasses to initialize the OpenAI client asynchronously.
        """
        pass

    async def _ensure_client(self) -> AsyncOpenAI:
        """Ensure OpenAI client is initialized."""
        await self._initialize_client()
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized")

        return self.client


def normalize_openai_api_key(
    api_key: str | SecretString | Callable[[], str | Awaitable[str]] | None,
) -> str | Callable[[], str | Awaitable[str]] | None:
    """Normalize OpenAI API-key input for SDK client construction."""
    if isinstance(api_key, SecretString):
        return api_key.get_secret_value()

    if callable(api_key):
        _check_openai_version_for_callable_api_key()

    return api_key


def serialize_openai_default_headers(default_headers: Mapping[str, str] | None) -> dict[str, Any] | None:
    """Serialize default headers while omitting generated user-agent values."""
    if default_headers:
        return {k: v for k, v in default_headers.items() if k != USER_AGENT_KEY}
    return None


def create_openai_client(
    *,
    api_key: str | Callable[[], str | Awaitable[str]] | None,
    org_id: str | None,
    default_headers: Mapping[str, str] | None,
    client: AsyncOpenAI | None,
    base_url: str | None,
) -> AsyncOpenAI:
    """Create or reuse an AsyncOpenAI client with Agent Framework headers applied."""
    merged_headers = dict(copy(default_headers)) if default_headers else {}
    if APP_INFO:
        merged_headers.update(APP_INFO)
        merged_headers = prepend_agent_framework_to_user_agent(merged_headers)

    if client is not None:
        return client

    if not api_key:
        raise ValueError("Please provide an api_key")

    args: dict[str, Any] = {"api_key": api_key, "default_headers": merged_headers}
    if org_id:
        args["organization"] = org_id
    if base_url:
        args["base_url"] = base_url
    return AsyncOpenAI(**args)


def create_openai_client_from_project(
    *,
    project_client: Any | None,
    project_endpoint: str | None,
    credential: AzureCredentialTypes | AzureTokenProvider | None,
    allow_preview: bool | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from an Azure AI Foundry project."""
    from azure.ai.projects.aio import AIProjectClient

    if project_client is not None:
        return project_client.get_openai_client()

    if not project_endpoint:
        raise ValueError("Azure AI project endpoint is required when project_client is not provided.")
    if not credential:
        raise ValueError("Azure credential is required when using project_endpoint without a project_client.")

    project_client_kwargs: dict[str, Any] = {
        "endpoint": project_endpoint,
        "credential": credential,  # type: ignore[arg-type]
        "user_agent": AGENT_FRAMEWORK_USER_AGENT,
    }
    if allow_preview is not None:
        project_client_kwargs["allow_preview"] = allow_preview

    return AIProjectClient(**project_client_kwargs).get_openai_client()


class OpenAIConfigMixin(OpenAIBase):
    """Internal class for configuring a connection to an OpenAI service."""

    OTEL_PROVIDER_NAME: ClassVar[str] = "openai"  # type: ignore[reportIncompatibleVariableOverride, misc]

    def __init__(
        self,
        model_id: str,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a client for OpenAI services.

        This constructor sets up a client to interact with OpenAI's API, allowing for
        different types of AI model interactions, like chat or text completion.

        Args:
            model_id: OpenAI model identifier. Must be non-empty.
                Default to a preset value.
            api_key: OpenAI API key for authentication, or a callable that returns an API key.
                Must be non-empty. (Optional)
            org_id: OpenAI organization ID. This is optional
                unless the account belongs to multiple organizations.
            default_headers: Default headers
                for HTTP requests. (Optional)
            client: An existing OpenAI client, optional.
            instruction_role: The role to use for 'instruction'
                messages, for example, summarization prompts could use `developer` or `system`. (Optional)
            base_url: The optional base URL to use. If provided will override the standard value for a OpenAI connector.
                Will not be used when supplying a custom client.
            kwargs: Additional keyword arguments.

        """
        # Handle callable API key using base class method
        api_key_value = normalize_openai_api_key(api_key)
        client = create_openai_client(
            api_key=api_key_value,
            org_id=org_id,
            default_headers=default_headers,
            client=client,
            base_url=base_url,
        )

        # Store configuration as instance attributes for serialization
        self.org_id = org_id
        self.base_url = str(base_url)
        self.default_headers = serialize_openai_default_headers(default_headers)

        args = {
            "model_id": model_id,
            "client": client,
        }
        if instruction_role:
            args["instruction_role"] = instruction_role

        # Ensure additional_properties and middleware are passed through kwargs to RawChatClient
        # These are consumed by RawChatClient.__init__ via kwargs
        super().__init__(**args, **kwargs)


def to_assistant_tools(
    tools: Sequence[FunctionTool | MutableMapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert Agent Framework tools to OpenAI Assistants API format.

    Handles FunctionTool instances and dict-based tools from static factory methods.

    Args:
        tools: Sequence of Agent Framework tools.

    Returns:
        List of tool definitions for OpenAI Assistants API.
    """
    if not tools:
        return []

    tool_definitions: list[dict[str, Any]] = []

    for tool in tools:
        if isinstance(tool, FunctionTool):
            tool_definitions.append(tool.to_json_schema_spec())
        elif isinstance(tool, MutableMapping):
            # Pass through dict-based tools directly (from static factory methods)
            tool_definitions.append(dict(tool))

    return tool_definitions


def from_assistant_tools(
    assistant_tools: list[Any] | None,
) -> list[dict[str, Any]]:
    """Convert OpenAI Assistant tools to dict-based format.

    This converts hosted tools (code_interpreter, file_search) from an OpenAI
    Assistant definition back to dict-based tool definitions.

    Note: Function tools are skipped - user must provide implementations separately.

    Args:
        assistant_tools: Tools from OpenAI Assistant object (assistant.tools).

    Returns:
        List of dict-based tool definitions for hosted tools.
    """
    if not assistant_tools:
        return []

    tools: list[dict[str, Any]] = []

    for tool in assistant_tools:
        if hasattr(tool, "type"):
            tool_type = tool.type
        elif isinstance(tool, Mapping):
            typed_tool = cast(Mapping[str, Any], tool)
            tool_type_value: Any = typed_tool.get("type")
            tool_type = tool_type_value if isinstance(tool_type_value, str) else None
        else:
            tool_type = None

        if tool_type == "code_interpreter":
            tools.append({"type": "code_interpreter"})
        elif tool_type == "file_search":
            tools.append({"type": "file_search"})
        # Skip function tools - user must provide implementations

    return tools
