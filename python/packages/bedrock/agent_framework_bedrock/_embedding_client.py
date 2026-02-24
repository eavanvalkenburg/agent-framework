# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, TypedDict

from agent_framework import (
    AGENT_FRAMEWORK_USER_AGENT,
    BaseEmbeddingClient,
    Embedding,
    EmbeddingGenerationOptions,
    GeneratedEmbeddings,
)
from agent_framework._settings import SecretString, load_settings
from agent_framework.observability import EmbeddingTelemetryLayer
from boto3.session import Session as Boto3Session
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig

if sys.version_info >= (3, 13):
    from typing import TypeVar  # type: ignore # pragma: no cover
else:
    from typing_extensions import TypeVar  # type: ignore # pragma: no cover


logger = logging.getLogger("agent_framework.bedrock")

DEFAULT_REGION = "us-east-1"


class BedrockEmbeddingOptions(EmbeddingGenerationOptions, total=False):
    """Bedrock-specific embedding options.

    Extends EmbeddingGenerationOptions with Bedrock-specific fields.

    Examples:
        .. code-block:: python

            from agent_framework_bedrock import BedrockEmbeddingOptions

            options: BedrockEmbeddingOptions = {
                "model_id": "amazon.titan-embed-text-v2:0",
                "dimensions": 1024,
                "normalize": True,
            }
    """

    normalize: bool


BedrockEmbeddingOptionsT = TypeVar(
    "BedrockEmbeddingOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="BedrockEmbeddingOptions",
    covariant=True,
)


class BedrockEmbeddingSettings(TypedDict, total=False):
    """Bedrock embedding settings."""

    region: str | None
    embedding_model_id: str | None
    access_key: SecretString | None
    secret_key: SecretString | None
    session_token: SecretString | None


class RawBedrockEmbeddingClient(
    BaseEmbeddingClient[str, list[float], BedrockEmbeddingOptionsT],
    Generic[BedrockEmbeddingOptionsT],
):
    """Raw Bedrock embedding client without telemetry.

    Keyword Args:
        model_id: The Bedrock embedding model ID (e.g. "amazon.titan-embed-text-v2:0").
            Can also be set via environment variable BEDROCK_EMBEDDING_MODEL_ID.
        region: AWS region. Defaults to "us-east-1".
            Can also be set via environment variable BEDROCK_REGION.
        access_key: AWS access key for manual credential injection.
        secret_key: AWS secret key paired with access_key.
        session_token: AWS session token for temporary credentials.
        client: Preconfigured Bedrock runtime client.
        boto3_session: Custom boto3 session used to build the runtime client.
        env_file_path: Path to .env file for settings.
        env_file_encoding: Encoding for .env file.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        model_id: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        client: BaseClient | None = None,
        boto3_session: Boto3Session | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a raw Bedrock embedding client."""
        settings = load_settings(
            BedrockEmbeddingSettings,
            env_prefix="BEDROCK_",
            required_fields=["embedding_model_id"],
            region=region,
            embedding_model_id=model_id,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        if not settings.get("region"):
            settings["region"] = DEFAULT_REGION

        if client is None:
            session = boto3_session or self._create_session(settings)
            client = session.client(
                "bedrock-runtime",
                region_name=settings["region"],
                config=BotoConfig(user_agent_extra=AGENT_FRAMEWORK_USER_AGENT),
            )

        self._bedrock_client = client
        self.model_id = settings["embedding_model_id"]
        self.region = settings["region"]
        super().__init__(**kwargs)

    @staticmethod
    def _create_session(settings: BedrockEmbeddingSettings) -> Boto3Session:
        """Create a boto3 session from settings."""
        session_kwargs: dict[str, Any] = {"region_name": settings.get("region") or DEFAULT_REGION}
        if settings.get("access_key") and settings.get("secret_key"):
            session_kwargs["aws_access_key_id"] = settings["access_key"].get_secret_value()  # type: ignore[union-attr]
            session_kwargs["aws_secret_access_key"] = settings["secret_key"].get_secret_value()  # type: ignore[union-attr]
        if settings.get("session_token"):
            session_kwargs["aws_session_token"] = settings["session_token"].get_secret_value()  # type: ignore[union-attr]
        return Boto3Session(**session_kwargs)

    def service_url(self) -> str:
        """Get the URL of the service."""
        meta = getattr(self._bedrock_client, "meta", None)
        if meta and hasattr(meta, "endpoint_url"):
            return str(meta.endpoint_url)
        return f"https://bedrock-runtime.{self.region}.amazonaws.com"

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: BedrockEmbeddingOptionsT | None = None,
    ) -> GeneratedEmbeddings[list[float]]:
        """Call the Bedrock invoke_model API for embeddings.

        Uses the Amazon Titan Embeddings model format. Each value is embedded
        individually since Titan's invoke_model API accepts one input at a time.

        Args:
            values: The text values to generate embeddings for.
            options: Optional embedding generation options.

        Returns:
            Generated embeddings with usage metadata.

        Raises:
            ValueError: If model_id is not provided or values is empty.
        """
        if not values:
            return GeneratedEmbeddings([], options=options)

        opts: dict[str, Any] = dict(options) if options else {}
        model = opts.get("model_id") or self.model_id
        if not model:
            raise ValueError("model_id is required")

        embeddings: list[Embedding[list[float]]] = []
        total_input_tokens = 0

        for text in values:
            body: dict[str, Any] = {"inputText": text}
            if dimensions := opts.get("dimensions"):
                body["dimensions"] = dimensions
            if (normalize := opts.get("normalize")) is not None:
                body["normalize"] = normalize

            response = await asyncio.to_thread(
                self._bedrock_client.invoke_model,
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            response_body = json.loads(response["body"].read())
            vector = response_body["embedding"]
            embeddings.append(
                Embedding(
                    vector=vector,
                    dimensions=len(vector),
                    model_id=model,
                )
            )
            total_input_tokens += response_body.get("inputTextTokenCount", 0)

        usage_dict: dict[str, Any] | None = None
        if total_input_tokens > 0:
            usage_dict = {"prompt_tokens": total_input_tokens}

        return GeneratedEmbeddings(embeddings, options=options, usage=usage_dict)


class BedrockEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], BedrockEmbeddingOptionsT],
    RawBedrockEmbeddingClient[BedrockEmbeddingOptionsT],
    Generic[BedrockEmbeddingOptionsT],
):
    """Bedrock embedding client with telemetry support.

    Uses the Amazon Titan Embeddings model via Bedrock's invoke_model API.

    Keyword Args:
        model_id: The Bedrock embedding model ID (e.g. "amazon.titan-embed-text-v2:0").
            Can also be set via environment variable BEDROCK_EMBEDDING_MODEL_ID.
        region: AWS region. Defaults to "us-east-1".
            Can also be set via environment variable BEDROCK_REGION.
        access_key: AWS access key for manual credential injection.
        secret_key: AWS secret key paired with access_key.
        session_token: AWS session token for temporary credentials.
        client: Preconfigured Bedrock runtime client.
        boto3_session: Custom boto3 session used to build the runtime client.
        env_file_path: Path to .env file for settings.
        env_file_encoding: Encoding for .env file.

    Examples:
        .. code-block:: python

            from agent_framework_bedrock import BedrockEmbeddingClient

            # Using default AWS credentials
            client = BedrockEmbeddingClient(
                model_id="amazon.titan-embed-text-v2:0",
            )

            # Generate embeddings
            result = await client.get_embeddings(["Hello, world!"])
            print(result[0].vector)
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "aws.bedrock"  # type: ignore[reportIncompatibleVariableOverride, misc]

    def __init__(
        self,
        *,
        region: str | None = None,
        model_id: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        client: BaseClient | None = None,
        boto3_session: Boto3Session | None = None,
        otel_provider_name: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Bedrock embedding client."""
        super().__init__(
            region=region,
            model_id=model_id,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            client=client,
            boto3_session=boto3_session,
            otel_provider_name=otel_provider_name,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
            **kwargs,
        )
