# Copyright (c) Microsoft. All rights reserved.

"""Foundry Memory sample scope must derive from the trusted request context."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from agent_framework.foundry import FoundryMemoryProvider
from azure.ai.agentserver.core import FoundryAgentRequestContext

_sample = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="sample_memory_scope")
create_agent = cast(Callable[[], Any], _sample["create_agent"])


@pytest.fixture
def memory_agent(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | None, str | None, str], str]:
    globals_ = create_agent.__globals__
    client = MagicMock()
    client.project_client = MagicMock()
    monkeypatch.setitem(globals_, "FoundryChatClient", MagicMock(return_value=client))
    monkeypatch.setitem(globals_, "DefaultAzureCredential", MagicMock())
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://test.services.ai.azure.com/api/projects/test")
    monkeypatch.setenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "test-model")
    monkeypatch.setenv("MEMORY_STORE_NAME", "isolated-test-store")

    def scope(user_id: str | None, call_id: str | None, session_id: str) -> str:
        platform = FoundryAgentRequestContext(user_id=user_id, call_id=call_id, session_id=session_id)
        monkeypatch.setitem(globals_, "get_request_context", lambda: platform)
        agent = create_agent()
        providers = [provider for provider in agent.context_providers if isinstance(provider, FoundryMemoryProvider)]
        assert len(providers) == 1
        return providers[0].scope

    return scope


def test_same_user_shares_memory_across_sandboxes(memory_agent: Callable[[str | None, str | None, str], str]) -> None:
    first = memory_agent("user-1", "call-1", "sandbox-1")
    second = memory_agent("user-1", "call-2", "sandbox-2")
    assert first == second
    assert len(first) == 64
    assert "user-1" not in first


def test_different_users_never_share_memory_scope(memory_agent: Callable[[str | None, str | None, str], str]) -> None:
    assert memory_agent("user-1", "call-1", "sandbox") != memory_agent("user-2", "call-2", "sandbox")


def test_missing_hosted_user_fails_closed(memory_agent: Callable[[str | None, str | None, str], str]) -> None:
    with pytest.raises(RuntimeError, match="trusted user ID"):
        memory_agent(None, "call-1", "sandbox")


def test_local_runs_have_explicit_single_user_scope(memory_agent: Callable[[str | None, str | None, str], str]) -> None:
    assert memory_agent(None, None, "local") == "local-user"
