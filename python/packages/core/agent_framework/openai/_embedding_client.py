# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import base64
import struct
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, ClassVar, Generic, Literal, TypedDict

from openai import AsyncOpenAI

from .._clients import BaseEmbeddingClient
from .._settings import load_settings
from .._types import Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings, UsageDetails
from ..observability import EmbeddingTelemetryLayer
from ._shared import OpenAISettings, create_openai_client, normalize_openai_api_key, serialize_openai_default_headers

if sys.version_info >= (3, 13):
    from typing import TypeVar  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypeVar  # type: ignore # pragma: no cover


class OpenAIEmbeddingOptions(EmbeddingGenerationOptions, total=False):
    """OpenAI-specific embedding options.

    Extends EmbeddingGenerationOptions with OpenAI-specific fields.

    Examples:
        .. code-block:: python

            from agent_framework.openai import OpenAIEmbeddingOptions

            options: OpenAIEmbeddingOptions = {
                "model_id": "text-embedding-3-small",
                "dimensions": 1536,
                "encoding_format": "float",
            }
    """

    encoding_format: Literal["float", "base64"]
    user: str


OpenAIEmbeddingOptionsT = TypeVar(
    "OpenAIEmbeddingOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="OpenAIEmbeddingOptions",
    covariant=True,
)


class RawOpenAIEmbeddingClient(
    BaseEmbeddingClient[str, list[float], OpenAIEmbeddingOptionsT],
    Generic[OpenAIEmbeddingOptionsT],
):
    """Raw OpenAI embedding client without telemetry."""

    INJECTABLE: ClassVar[set[str]] = {"client"}

    def __init__(
        self,
        *,
        model_id: str | None = None,
        client: AsyncOpenAI,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the raw OpenAI embedding client."""
        self.client = client
        self.model_id = model_id.strip() if model_id else None
        super().__init__(additional_properties=additional_properties)

    def service_url(self) -> str:
        """Get the URL of the service."""
        return str(self.client.base_url) if self.client else "Unknown"

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: OpenAIEmbeddingOptionsT | None = None,
    ) -> GeneratedEmbeddings[list[float], OpenAIEmbeddingOptionsT]:
        """Call the OpenAI embeddings API.

        Args:
            values: The text values to generate embeddings for.
            options: Optional embedding generation options.

        Returns:
            Generated embeddings with usage metadata.

        Raises:
            ValueError: If model_id is not provided or values is empty.
        """
        if not values:
            return GeneratedEmbeddings([], options=options)  # type: ignore

        opts: dict[str, Any] = options or {}  # type: ignore
        model = opts.get("model_id") or self.model_id
        if not model:
            raise ValueError("model_id is required")

        kwargs: dict[str, Any] = {"input": list(values), "model": model}
        if dimensions := opts.get("dimensions"):
            kwargs["dimensions"] = dimensions
        if encoding_format := opts.get("encoding_format"):
            kwargs["encoding_format"] = encoding_format
        if user := opts.get("user"):
            kwargs["user"] = user

        response = await self.client.embeddings.create(**kwargs)

        encoding = kwargs.get("encoding_format", "float")
        embeddings: list[Embedding[list[float]]] = []
        for item in response.data:
            vector: list[float]
            if encoding == "base64" and isinstance(item.embedding, str):
                # Decode base64-encoded floats (little-endian IEEE 754)
                raw = base64.b64decode(item.embedding)
                vector = list(struct.unpack(f"<{len(raw) // 4}f", raw))
            else:
                vector = item.embedding  # type: ignore[assignment]
            embeddings.append(
                Embedding(
                    vector=vector,
                    dimensions=len(vector),
                    model_id=response.model,
                )
            )

        usage_dict: UsageDetails | None = None
        if response.usage:
            usage_dict = {
                "input_token_count": response.usage.prompt_tokens,
                "total_token_count": response.usage.total_tokens,
            }

        return GeneratedEmbeddings(embeddings, options=options, usage=usage_dict)


class OpenAIEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], OpenAIEmbeddingOptionsT],
    RawOpenAIEmbeddingClient[OpenAIEmbeddingOptionsT],
    Generic[OpenAIEmbeddingOptionsT],
):
    """OpenAI embedding client with telemetry support.

    Keyword Args:
        model_id: The embedding model ID (e.g. "text-embedding-3-small").
            Can also be set via environment variable OPENAI_EMBEDDING_MODEL_ID.
        api_key: OpenAI API key.
            Can also be set via environment variable OPENAI_API_KEY.
        org_id: OpenAI organization ID.
        default_headers: Additional HTTP headers.
        async_client: Pre-configured AsyncOpenAI client.
        base_url: Custom API base URL.
        otel_provider_name: Override the OpenTelemetry provider name for telemetry.
        env_file_path: Path to .env file for settings.
        env_file_encoding: Encoding for .env file.

    Examples:
        .. code-block:: python

            from agent_framework.openai import OpenAIEmbeddingClient

            # Using environment variables
            # Set OPENAI_API_KEY=sk-...
            # Set OPENAI_EMBEDDING_MODEL_ID=text-embedding-3-small
            client = OpenAIEmbeddingClient()

            # Or passing parameters directly
            client = OpenAIEmbeddingClient(
                model_id="text-embedding-3-small",
                api_key="sk-...",
            )

            # Generate embeddings
            result = await client.get_embeddings(["Hello, world!"])
            print(result[0].vector)
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        otel_provider_name: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize an OpenAI embedding client."""
        openai_settings = load_settings(
            OpenAISettings,
            env_prefix="OPENAI_",
            api_key=api_key,
            base_url=base_url,
            org_id=org_id,
            embedding_model_id=model_id,
            required_fields=["embedding_model_id", *([] if async_client else ["api_key"])],
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )

        api_key_value = openai_settings["api_key"] if async_client is None else openai_settings.get("api_key")  # type: ignore[typeddict-item]
        embedding_model_id: str = openai_settings["embedding_model_id"]  # type: ignore[assignment,typeddict-item]
        base_url_value = openai_settings.get("base_url")
        async_client = create_openai_client(
            api_key=normalize_openai_api_key(api_key_value),
            org_id=openai_settings.get("org_id"),
            default_headers=default_headers,
            client=async_client,
            base_url=base_url_value if base_url_value else None,
        )

        super().__init__(
            model_id=embedding_model_id,
            client=async_client,
            additional_properties=None,
            otel_provider_name=otel_provider_name,
        )
        self.org_id = openai_settings.get("org_id")
        self.base_url = str(base_url_value)
        self.default_headers = serialize_openai_default_headers(default_headers)
