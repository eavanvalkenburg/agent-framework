# Copyright (c) Microsoft. All rights reserved.

"""The caller-facing request view and option policy are request-scoped."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import WorkflowInvocationKwargs
from azure.ai.agentserver.core import AgentConfig, FoundryAgentRequestContext
from azure.ai.agentserver.responses import ResponseContext
from azure.ai.agentserver.responses.models import CreateResponse

from agent_framework_foundry_hosting import HostedResponseRequest, InvocationRun, WorkflowTurn
from agent_framework_foundry_hosting._request import (
    prepare_response_options,
    response_run_options,
    validate_unsupported_options,
)
from agent_framework_foundry_hosting._scope import FoundryRequestScope


def test_native_options_are_translated_before_extra_body_fields() -> None:
    request = cast(
        CreateResponse,
        {
            "input": "hello",
            "store": True,
            "background": False,
            "conversation": "outer-conversation",
            "agent_session_id": "sandbox",
            "session_id": "azd-session",
            "user_id": "forged-user",
            "call_id": "forged-call",
            "conversation_id": "inner-conversation",
            "service_session_id": "inner-service-session",
            "continuation_token": {"response_id": "inner-response"},
            "max_output_tokens": 300,
            "max_tokens": 150,
            "parallel_tool_calls": False,
            "reasoning": {"effort": "high"},
            "slogan_style": "retro",
        },
    )

    assert response_run_options(request) == {
        "max_tokens": 150,
        "allow_multiple_tool_calls": False,
        "reasoning": {"effort": "high"},
        "slogan_style": "retro",
    }


async def test_hook_can_remove_caller_option_without_mutating_source() -> None:
    context = MagicMock(spec=ResponseContext)
    context.get_input_text = AsyncMock(return_value="hello")
    scope = FoundryRequestScope(session_id="sandbox", user_id=None, call_id=None, is_hosted=False)
    request = HostedResponseRequest(
        cast(CreateResponse, {"input": "hello"}),
        context,
        scope,
        {"temperature": 0.8},
    )

    async def remove_temperature(_request: HostedResponseRequest, options: dict[str, Any]) -> dict[str, Any]:
        options.pop("temperature")
        return options

    await prepare_response_options(request, remove_temperature)

    assert dict(request.options) == {}
    assert await request.get_input_text() == "hello"


@pytest.mark.parametrize("mode", ["ignore", "warn", "error"])
def test_known_unsupported_option_modes(mode: str) -> None:
    assert validate_unsupported_options(mode) == mode


def test_unknown_unsupported_option_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported_options"):
        validate_unsupported_options("silent")


def test_workflow_turn_requires_exactly_one_input_shape() -> None:
    assert WorkflowTurn(input=0).input == 0
    assert WorkflowTurn(responses={"request-1": True}).responses == {"request-1": True}
    with pytest.raises(ValueError, match="exactly one"):
        WorkflowTurn[int]()
    with pytest.raises(ValueError, match="exactly one"):
        WorkflowTurn(input="new input", responses={"request-1": True})
    with pytest.raises(ValueError, match="exactly one"):
        WorkflowTurn(responses={})


def test_workflow_turn_accepts_existing_per_executor_kwargs() -> None:
    kwargs = WorkflowInvocationKwargs(executor_kwargs={"writer": {"timeout": 10}})
    assert WorkflowTurn(input="hello", client_kwargs=kwargs).client_kwargs is kwargs


def test_invocation_run_rejects_malformed_options() -> None:
    with pytest.raises(TypeError, match="options"):
        InvocationRun(messages="hello", options=cast(Any, "not a mapping"))
    with pytest.raises(TypeError, match="stream"):
        InvocationRun(messages="hello", stream=cast(Any, "true"))


def test_hosted_scope_rejects_missing_identity_without_guessing() -> None:
    config = MagicMock(spec=AgentConfig)
    config.is_hosted = True
    with pytest.raises(RuntimeError, match="session ID"):
        FoundryRequestScope.from_context(config, FoundryAgentRequestContext(call_id="call", user_id="user"))
    with pytest.raises(RuntimeError, match="trusted user ID and call ID"):
        FoundryRequestScope.from_context(config, FoundryAgentRequestContext(session_id="sandbox", user_id="user"))

    scope = FoundryRequestScope.from_context(
        config,
        FoundryAgentRequestContext(session_id="sandbox", user_id="user", call_id="call"),
    )
    assert scope.session_id == "sandbox"
    assert len(scope.storage_key) == 64
