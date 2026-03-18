# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, cast

from openai.lib.azure import AsyncAzureOpenAI

from agent_framework import FunctionInvocationConfiguration
from agent_framework.openai._chat_client import (
    AzureOpenAIChatOptions as _AzureOpenAIChatOptions,
)
from agent_framework.openai._chat_client import (
    AzureOpenAIChatOptionsT,
    OpenAIChatClient,
)
from agent_framework.openai._chat_client import (
    AzureUserSecurityContext as _AzureUserSecurityContext,
)

from ._entra_id_authentication import AzureCredentialTypes, AzureTokenProvider

if sys.version_info >= (3, 13):
    from warnings import deprecated  # pragma: no cover
else:
    from typing_extensions import (
        deprecated,  # pragma: no cover
    )
if TYPE_CHECKING:
    from agent_framework._middleware import MiddlewareTypes

AzureOpenAIChatOptions = _AzureOpenAIChatOptions
AzureUserSecurityContext = _AzureUserSecurityContext


@deprecated(
    "'AzureOpenAIChatClient' is deprecated; use 'OpenAIChatClient(backend=\"azure_openai\")' instead.",
)
class AzureOpenAIChatClient(  # type: ignore[misc]
    OpenAIChatClient[AzureOpenAIChatOptionsT],
    Generic[AzureOpenAIChatOptionsT],
):
    """Azure OpenAI Chat completion class with middleware, telemetry, and function invocation support."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        deployment_name: str | None = None,
        endpoint: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        token_endpoint: str | None = None,
        credential: AzureCredentialTypes | AzureTokenProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncAzureOpenAI | None = None,
        additional_properties: dict[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        instruction_role: str | None = None,
        middleware: Sequence[MiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    ) -> None:
        """Initialize an Azure OpenAI Chat completion client.

        Keyword Args:
            api_key: The API key. If provided, will override the value in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_API_KEY.
            deployment_name: The deployment name. If provided, will override the value
                (chat_deployment_name) in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_CHAT_DEPLOYMENT_NAME.
            endpoint: The deployment endpoint. If provided will override the value
                in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_ENDPOINT.
            base_url: The deployment base URL. If provided will override the value
                in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_BASE_URL.
            api_version: The deployment API version. If provided will override the value
                in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_API_VERSION.
            token_endpoint: The token endpoint to request an Azure token.
                Can also be set via environment variable AZURE_OPENAI_TOKEN_ENDPOINT.
            credential: Azure credential or token provider for authentication. Accepts a
                ``TokenCredential``, ``AsyncTokenCredential``, or a callable that returns a
                bearer token string (sync or async), for example from
                ``azure.identity.get_bearer_token_provider()``.
            default_headers: The default headers mapping of string keys to
                string values for HTTP requests.
            async_client: An existing client to use.
            additional_properties: Additional properties stored on the client instance.
            env_file_path: Use the environment settings file as a fallback to using env vars.
            env_file_encoding: The encoding of the environment settings file, defaults to 'utf-8'.
            instruction_role: The role to use for 'instruction' messages, for example, summarization
                prompts could use `developer` or `system`.
            middleware: Optional sequence of middleware to apply to requests.
            function_invocation_configuration: Optional configuration for function invocation behavior.

        Examples:
            .. code-block:: python

                from agent_framework.azure import AzureOpenAIChatClient

                # Using environment variables
                # Set AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
                # Set AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<model name>
                # Set AZURE_OPENAI_API_KEY=your-key
                client = AzureOpenAIChatClient()

                # Or passing parameters directly
                client = AzureOpenAIChatClient(
                    endpoint="https://your-endpoint.openai.azure.com",
                    deployment_name="<model name>",
                    api_key="your-key",
                )

                # Or loading from a .env file
                client = AzureOpenAIChatClient(env_file_path="path/to/.env")

                # Using custom ChatOptions with type safety:
                from typing import TypedDict
                from agent_framework.openai import AzureOpenAIChatOptions


                class MyOptions(AzureOpenAIChatOptions, total=False):
                    my_custom_option: str


                client: AzureOpenAIChatClient[MyOptions] = AzureOpenAIChatClient()
                response = await client.get_response("Hello", options={"my_custom_option": "value"})
        """
        # TODO(Copilot): Delete once ``AzureOpenAIChatClient`` is removed in favor of ``OpenAIChatClient``.
        super().__init__(
            model_id=deployment_name,
            backend="azure_openai",
            api_key=api_key,
            base_url=base_url,
            endpoint=endpoint,
            api_version=api_version,
            token_endpoint=token_endpoint,
            credential=credential,
            default_headers=default_headers,
            async_client=async_client,
            additional_properties=additional_properties,
            instruction_role=instruction_role,
            middleware=cast(Any, middleware),
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
            function_invocation_configuration=function_invocation_configuration,
        )
