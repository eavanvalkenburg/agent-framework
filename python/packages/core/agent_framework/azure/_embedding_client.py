# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Generic

from openai.lib.azure import AsyncAzureOpenAI

from agent_framework._telemetry import APP_INFO, prepend_agent_framework_to_user_agent
from agent_framework.observability import EmbeddingTelemetryLayer
from agent_framework.openai import OpenAIEmbeddingOptions
from agent_framework.openai._embedding_client import RawOpenAIEmbeddingClient
from agent_framework.openai._shared import serialize_openai_default_headers

from .._settings import load_settings
from ._entra_id_authentication import AzureCredentialTypes, AzureTokenProvider
from ._shared import (
    AzureOpenAISettings,
    _apply_azure_defaults,  # pyright: ignore[reportPrivateUsage]
    resolve_credential_to_token_provider,
)

if sys.version_info >= (3, 13):
    from typing import TypeVar  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypeVar  # type: ignore # pragma: no cover
if sys.version_info >= (3, 11):
    from typing import TypedDict  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypedDict  # type: ignore # pragma: no cover


AzureOpenAIEmbeddingOptionsT = TypeVar(
    "AzureOpenAIEmbeddingOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="OpenAIEmbeddingOptions",
    covariant=True,
)


class AzureOpenAIEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], AzureOpenAIEmbeddingOptionsT],
    RawOpenAIEmbeddingClient[AzureOpenAIEmbeddingOptionsT],
    Generic[AzureOpenAIEmbeddingOptionsT],
):
    """Azure OpenAI embedding client with telemetry support.

    Keyword Args:
        api_key: The API key. If provided, will override the value in the env vars or .env file.
            Can also be set via environment variable AZURE_OPENAI_API_KEY.
        deployment_name: The deployment name. If provided, will override the value
            (embedding_deployment_name) in the env vars or .env file.
            Can also be set via environment variable AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME.
        endpoint: The deployment endpoint.
            Can also be set via environment variable AZURE_OPENAI_ENDPOINT.
        base_url: The deployment base URL.
            Can also be set via environment variable AZURE_OPENAI_BASE_URL.
        api_version: The deployment API version.
            Can also be set via environment variable AZURE_OPENAI_API_VERSION.
        token_endpoint: The token endpoint to request an Azure token.
            Can also be set via environment variable AZURE_OPENAI_TOKEN_ENDPOINT.
        credential: Azure credential or token provider for authentication.
        default_headers: Default headers for HTTP requests.
        async_client: An existing client to use.
        env_file_path: Path to .env file for settings.
        env_file_encoding: Encoding for .env file.

    Examples:
        .. code-block:: python

            from agent_framework.azure import AzureOpenAIEmbeddingClient

            # Using environment variables
            # Set AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
            # Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
            # Set AZURE_OPENAI_API_KEY=your-key
            client = AzureOpenAIEmbeddingClient()

            # Or passing parameters directly
            client = AzureOpenAIEmbeddingClient(
                endpoint="https://your-endpoint.openai.azure.com",
                deployment_name="text-embedding-3-small",
                api_key="your-key",
            )

            result = await client.get_embeddings(["Hello, world!"])
    """

    OTEL_PROVIDER_NAME = "azure.ai.openai"

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
        otel_provider_name: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize an Azure OpenAI embedding client."""
        azure_openai_settings = load_settings(
            AzureOpenAISettings,
            env_prefix="AZURE_OPENAI_",
            api_key=api_key,
            base_url=base_url,
            endpoint=endpoint,
            embedding_deployment_name=deployment_name,
            api_version=api_version,
            required_fields=["embedding_deployment_name"],
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
            token_endpoint=token_endpoint,
        )
        _apply_azure_defaults(azure_openai_settings)

        embedding_deployment_name: str = azure_openai_settings["embedding_deployment_name"]  # type: ignore[assignment,typeddict-item]

        api_key_secret = azure_openai_settings.get("api_key")
        api_key_value = api_key_secret.get_secret_value() if api_key_secret else None

        if async_client is None:
            ad_token_provider = None
            if api_key_value is None and credential is not None:
                ad_token_provider = resolve_credential_to_token_provider(
                    credential,
                    azure_openai_settings.get("token_endpoint"),
                )

            if api_key_value is None and ad_token_provider is None:
                raise ValueError("Please provide either api_key, credential, or a client.")

            resolved_base_url = azure_openai_settings.get("base_url")
            resolved_endpoint = azure_openai_settings.get("endpoint")
            if resolved_base_url is None and resolved_endpoint is None:
                raise ValueError("Please provide an endpoint or a base_url")

            merged_headers = dict(default_headers) if default_headers else {}
            if APP_INFO:
                merged_headers.update(APP_INFO)
                merged_headers = prepend_agent_framework_to_user_agent(merged_headers)

            client_kwargs: dict[str, Any] = {
                "default_headers": merged_headers,
                "azure_deployment": embedding_deployment_name,
            }
            if api_version_value := azure_openai_settings.get("api_version"):
                client_kwargs["api_version"] = api_version_value
            if ad_token_provider is not None:
                client_kwargs["azure_ad_token_provider"] = ad_token_provider
            if api_key_value is not None:
                client_kwargs["api_key"] = api_key_value
            if resolved_base_url is not None:
                client_kwargs["base_url"] = str(resolved_base_url)
            elif resolved_endpoint is not None:
                client_kwargs["azure_endpoint"] = str(resolved_endpoint)

            async_client = AsyncAzureOpenAI(**client_kwargs)

        super().__init__(
            model_id=embedding_deployment_name,
            client=async_client,
            otel_provider_name=otel_provider_name,
        )
        self.endpoint = str(azure_openai_settings.get("endpoint"))
        self.base_url = str(azure_openai_settings.get("base_url"))
        self.api_version = azure_openai_settings.get("api_version")
        self.deployment_name = embedding_deployment_name
        self.default_headers = serialize_openai_default_headers(default_headers)
