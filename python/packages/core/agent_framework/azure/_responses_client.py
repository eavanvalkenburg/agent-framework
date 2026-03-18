# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, cast

from openai import AsyncOpenAI

from .._tools import FunctionInvocationConfiguration
from ..exceptions import SettingNotFoundError
from ..openai._responses_client import OpenAIResponsesClient
from ..openai._shared import create_openai_client_from_project
from ._entra_id_authentication import AzureCredentialTypes, AzureTokenProvider

if sys.version_info >= (3, 13):
    from typing import TypeVar  # type: ignore # pragma: no cover
    from warnings import deprecated  # pragma: no cover
else:
    from typing_extensions import (
        TypeVar,  # type: ignore # pragma: no cover
        deprecated,  # pragma: no cover
    )
if sys.version_info >= (3, 11):
    from typing import TypedDict  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypedDict  # type: ignore # pragma: no cover

if TYPE_CHECKING:
    from .._middleware import MiddlewareTypes
    from ..openai._responses_client import OpenAIResponsesOptions


AzureOpenAIResponsesOptionsT = TypeVar(
    "AzureOpenAIResponsesOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="OpenAIResponsesOptions",
    covariant=True,
)


@deprecated(
    "'AzureOpenAIResponsesClient' is deprecated; use "
    "'OpenAIResponsesClient(backend=\"azure_openai\")' or "
    "'OpenAIResponsesClient(backend=\"foundry\")' instead.",
)
class AzureOpenAIResponsesClient(  # type: ignore[misc]
    OpenAIResponsesClient[AzureOpenAIResponsesOptionsT],
    Generic[AzureOpenAIResponsesOptionsT],
):
    """Azure Responses completion class with middleware, telemetry, and function invocation support."""

    @staticmethod
    def _create_client_from_project(
        *,
        project_client: Any | None,
        project_endpoint: str | None,
        credential: AzureCredentialTypes | AzureTokenProvider | None,
        allow_preview: bool | None = None,
    ) -> AsyncOpenAI:
        """Create an AsyncOpenAI client from an Azure AI Foundry project."""
        return create_openai_client_from_project(
            project_client=project_client,
            project_endpoint=project_endpoint,
            credential=credential,
            allow_preview=allow_preview,
        )

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
        async_client: AsyncOpenAI | None = None,
        project_client: Any | None = None,
        project_endpoint: str | None = None,
        allow_preview: bool | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        instruction_role: str | None = None,
        middleware: Sequence[MiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an Azure OpenAI Responses client.

        The client can be created in two ways:

        1. **Direct Azure OpenAI** (default): Provide endpoint, api_key, or credential
           to connect directly to an Azure OpenAI deployment.
        2. **Foundry project endpoint**: Provide a ``project_client`` or ``project_endpoint``
           (with ``credential``) to create the client via an Azure AI Foundry project.
           This requires the ``azure-ai-projects`` package to be installed.

        Keyword Args:
            api_key: The API key. If provided, will override the value in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_API_KEY.
            deployment_name: The deployment name. If provided, will override the value
                (responses_deployment_name) in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME.
            endpoint: The deployment endpoint. If provided will override the value
                in the env vars or .env file.
                Can also be set via environment variable AZURE_OPENAI_ENDPOINT.
            base_url: The deployment base URL. If provided will override the value
                in the env vars or .env file. Currently, the base_url must end with "/openai/v1/".
                Can also be set via environment variable AZURE_OPENAI_BASE_URL.
            api_version: The deployment API version. If provided will override the value
                in the env vars or .env file. Currently, the api_version must be "preview".
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
            project_client: An existing ``AIProjectClient`` (from ``azure.ai.projects.aio``) to use.
                The OpenAI client will be obtained via ``project_client.get_openai_client()``.
                Requires the ``azure-ai-projects`` package.
            project_endpoint: The Azure AI Foundry project endpoint URL.
                When provided with ``credential``, an ``AIProjectClient`` will be created
                and used to obtain the OpenAI client. Requires the ``azure-ai-projects`` package.
            allow_preview: Enables preview opt-in on internally-created ``AIProjectClient``.
            env_file_path: Use the environment settings file as a fallback to using env vars.
            env_file_encoding: The encoding of the environment settings file, defaults to 'utf-8'.
            instruction_role: The role to use for 'instruction' messages, for example, summarization
                prompts could use `developer` or `system`.
            middleware: Optional sequence of middleware to apply to requests.
            function_invocation_configuration: Optional configuration for function invocation behavior.
            kwargs: Additional keyword arguments.

        Examples:
            .. code-block:: python

                from agent_framework.azure import AzureOpenAIResponsesClient

                # Using environment variables
                # Set AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
                # Set AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-4o
                # Set AZURE_OPENAI_API_KEY=your-key
                client = AzureOpenAIResponsesClient()

                # Or passing parameters directly
                client = AzureOpenAIResponsesClient(
                    endpoint="https://your-endpoint.openai.azure.com", deployment_name="gpt-4o", api_key="your-key"
                )

                # Or loading from a .env file
                client = AzureOpenAIResponsesClient(env_file_path="path/to/.env")

                # Using a Foundry project endpoint
                from azure.identity import DefaultAzureCredential

                client = AzureOpenAIResponsesClient(
                    project_endpoint="https://your-project.services.ai.azure.com",
                    deployment_name="gpt-4o",
                    credential=DefaultAzureCredential(),
                )

                # Or using an existing AIProjectClient
                from azure.ai.projects.aio import AIProjectClient

                project_client = AIProjectClient(
                    endpoint="https://your-project.services.ai.azure.com",
                    credential=DefaultAzureCredential(),
                )
                client = AzureOpenAIResponsesClient(
                    project_client=project_client,
                    deployment_name="gpt-4o",
                )

                # Using custom ChatOptions with type safety:
                from typing import TypedDict
                from agent_framework.azure import AzureOpenAIResponsesOptions


                class MyOptions(AzureOpenAIResponsesOptions, total=False):
                    my_custom_option: str


                client: AzureOpenAIResponsesClient[MyOptions] = AzureOpenAIResponsesClient()
                response = await client.get_response("Hello", options={"my_custom_option": "value"})
        """
        # TODO(Copilot): Delete once ``AzureOpenAIResponsesClient`` is removed in favor of ``OpenAIResponsesClient``.
        model_id = kwargs.pop("model_id", None)
        if project_client is not None or project_endpoint is not None:
            project_openai_client = self._create_client_from_project(
                project_client=project_client,
                project_endpoint=project_endpoint,
                credential=credential,
                allow_preview=allow_preview,
            )
            try:
                super().__init__(
                    "foundry",
                    model_id=deployment_name or model_id,
                    default_headers=default_headers,
                    async_client=project_openai_client,
                    instruction_role=instruction_role,
                    middleware=cast(Any, middleware),
                    env_file_path=env_file_path,
                    env_file_encoding=env_file_encoding,
                    function_invocation_configuration=function_invocation_configuration,
                    **kwargs,
                )
            except SettingNotFoundError as exc:
                raise ValueError(str(exc)) from exc
            return

        try:
            super().__init__(
                "azure_openai",
                model_id=deployment_name or model_id,
                api_key=api_key,
                base_url=base_url,
                endpoint=endpoint,
                api_version=api_version,
                token_endpoint=token_endpoint,
                credential=credential,
                default_headers=default_headers,
                async_client=async_client,
                instruction_role=instruction_role,
                middleware=cast(Any, middleware),
                env_file_path=env_file_path,
                env_file_encoding=env_file_encoding,
                function_invocation_configuration=function_invocation_configuration,
                **kwargs,
            )
        except SettingNotFoundError as exc:
            raise ValueError(str(exc)) from exc
