# Copyright (c) Microsoft. All rights reserved.

"""End-to-end storage and background checks against an explicitly selected hosted version."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest
from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from openai import BadRequestError, NotFoundError, OpenAI

pytestmark = pytest.mark.integration

skip_without_deployment = pytest.mark.skipif(
    not os.getenv("FOUNDRY_PROJECT_ENDPOINT") or not os.getenv("FOUNDRY_DEPLOYED_AGENT_NAME"),
    reason="Set FOUNDRY_DEPLOYED_AGENT_NAME to run tests against a deployed Responses agent.",
)


@pytest.fixture(scope="module")
def deployed_responses() -> Iterator[OpenAI]:
    with (
        AzureCliCredential() as credential,
        AIProjectClient(
            endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            credential=credential,
            allow_preview=True,
        ) as project,
    ):
        # ty: ignore[unresolved-attribute]
        client = project.get_openai_client(agent_name=os.environ["FOUNDRY_DEPLOYED_AGENT_NAME"])  # zuban: ignore
        try:
            yield client
        finally:
            client.close()


@pytest.mark.flaky
@skip_without_deployment
def test_stored_response_retrieval_and_conversation_continuation(deployed_responses: OpenAI) -> None:
    conversation = deployed_responses.conversations.create()
    first = deployed_responses.responses.create(
        input="Reply with the word ready.",
        conversation=conversation.id,
        store=True,
    )
    assert first.status == "completed", first.error
    assert first.output_text
    assert deployed_responses.responses.retrieve(first.id).id == first.id

    second = deployed_responses.responses.create(
        input="Reply with another short sentence.",
        conversation=conversation.id,
        store=True,
    )
    assert second.status == "completed", second.error
    assert second.output_text
    assert second.id != first.id


@pytest.mark.flaky
@skip_without_deployment
def test_unstored_response_cannot_be_retrieved(deployed_responses: OpenAI) -> None:
    response = deployed_responses.responses.create(input="Reply with the word ready.", store=False)
    assert response.status == "completed", response.error
    assert response.output_text
    with pytest.raises(NotFoundError):
        deployed_responses.responses.retrieve(response.id)


@pytest.mark.flaky
@skip_without_deployment
def test_background_response_uses_outer_id_for_polling(deployed_responses: OpenAI) -> None:
    pending = deployed_responses.responses.create(
        input="Reply with a short sentence.",
        store=True,
        background=True,
    )
    assert pending.status in ("queued", "in_progress", "completed"), pending.error
    deadline = time.monotonic() + 120
    current = pending
    while current.status in ("queued", "in_progress") and time.monotonic() < deadline:
        time.sleep(1)
        current = deployed_responses.responses.retrieve(pending.id)
    assert current.status == "completed", current.error
    assert current.id == pending.id
    assert current.output_text


@pytest.mark.flaky
@skip_without_deployment
def test_unstored_background_request_is_rejected(deployed_responses: OpenAI) -> None:
    with pytest.raises(BadRequestError) as error:
        deployed_responses.responses.create(
            input="Reply with the word ready.",
            store=False,
            background=True,
        )
    assert error.value.status_code == 400


@pytest.mark.flaky
@skip_without_deployment
def test_extra_body_reaches_host_generation_options(deployed_responses: OpenAI) -> None:
    response = deployed_responses.responses.create(
        input="Reply with the word ready.",
        store=True,
        max_output_tokens=200,
        extra_body={"max_tokens": 150},
    )
    assert response.status == "completed", response.error
    assert response.output_text
